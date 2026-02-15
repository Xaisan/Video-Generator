"""
pipeline/sdpa.py — Chunked SDPA for GPUs without flash/mem-efficient attention
===============================================================================

On ROCm (AMD RDNA3/RDNA2), PyTorch's scaled_dot_product_attention
falls back to the naive "math" backend that materialises the FULL
[B, H, N, N] attention matrix.  For large sequence lengths (e.g.
131 K tokens at 832×480×81 frames) this can be 160+ GiB — instant OOM.

The function below processes attention **one head at a time** and
chunks the query dimension, keeping peak memory bounded to roughly:
  chunk_size × N × sizeof(dtype) per step  (~256 MB with defaults).

This is the same mathematical operation as regular SDPA; only the
memory access pattern changes.  Quality is bit-identical.

Public API:
  _ORIGINAL_SDPA              — reference to the real F.scaled_dot_product_attention
  _sdpa_backends_available()  — probe GPU for flash/mem-efficient support
  _chunked_sdpa()             — head-by-head query-chunked attention
  patched_sdpa()              — context manager to monkey-patch SDPA
  _resolve_chunked_attention() — auto-detect if chunking is needed

Dependencies: vram_utils.flush_vram
"""

import math
from contextlib import contextmanager

import torch
import torch.nn.functional as F

from pipeline.vram_utils import flush_vram


_ORIGINAL_SDPA = F.scaled_dot_product_attention   # keep a reference


def _sdpa_backends_available() -> dict:
    """Probe which SDPA backends PyTorch actually supports on this GPU."""
    result = {"flash": False, "mem_efficient": False, "math": True}
    if not torch.cuda.is_available():
        return result
    try:
        q = torch.randn(1, 1, 8, 64, device="cuda", dtype=torch.float16)
        k = torch.randn(1, 1, 8, 64, device="cuda", dtype=torch.float16)
        v = torch.randn(1, 1, 8, 64, device="cuda", dtype=torch.float16)
        with torch.backends.cuda.sdp_kernel(
            enable_flash=True, enable_mem_efficient=False, enable_math=False
        ):
            try:
                _ = _ORIGINAL_SDPA(q, k, v)
                result["flash"] = True
            except Exception:
                pass
        with torch.backends.cuda.sdp_kernel(
            enable_flash=False, enable_mem_efficient=True, enable_math=False
        ):
            try:
                _ = _ORIGINAL_SDPA(q, k, v)
                result["mem_efficient"] = True
            except Exception:
                pass
        del q, k, v
        flush_vram()
    except Exception:
        pass
    return result


def _chunked_sdpa(
    query, key, value,
    attn_mask=None, dropout_p=0.0, is_causal=False, scale=None,
    enable_gqa=False,
):
    """Memory-efficient SDPA via head-by-head + query-chunked computation.

    Processes one head at a time and chunks the query sequence so the
    intermediate attention-score tensor is never larger than
    [1, chunk_size, N] — roughly 256 MB instead of 160 GiB.
    """
    B, H, N, D = query.shape
    _, Hk, Nk, Dk = key.shape

    if scale is None:
        scale = 1.0 / math.sqrt(D)

    # For short sequences, just use normal SDPA (math backend is fine)
    # Threshold: attention matrix < 512 MB ≈ N*Nk*2 < 512M → N < ~16k
    if N * Nk * 2 < 512 * 1024 * 1024:
        return _ORIGINAL_SDPA(
            query, key, value, attn_mask=attn_mask,
            dropout_p=dropout_p, is_causal=is_causal, scale=scale,
            enable_gqa=enable_gqa,
        )

    # Adaptive chunk size: keep per-chunk attention matrix ≤ target_mb
    target_bytes = 256 * 1024 * 1024   # 256 MiB
    elem_size = 4 if query.dtype == torch.float32 else 2
    chunk_size = max(256, target_bytes // (Nk * elem_size))
    chunk_size = min(chunk_size, N)

    # GQA: repeat K/V heads to match Q head count
    heads_per_kv = H // Hk
    output = torch.empty_like(query)

    for h in range(H):
        kv_h = h // heads_per_kv
        q_h = query[:, h]            # [B, N, D]
        k_h = key[:, kv_h]           # [B, Nk, Dk]
        v_h = value[:, kv_h]         # [B, Nk, Dk]

        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            q_chunk = q_h[:, start:end]                          # [B, chunk, D]
            scores = torch.bmm(q_chunk, k_h.transpose(-2, -1))  # [B, chunk, Nk]
            scores = scores * scale

            if attn_mask is not None:
                if attn_mask.dim() == 4:
                    scores = scores + attn_mask[:, h, start:end, :Nk]
                elif attn_mask.dim() == 2:
                    scores = scores + attn_mask[start:end, :Nk]

            if is_causal:
                row_idx = torch.arange(start, end, device=scores.device).unsqueeze(1)
                col_idx = torch.arange(Nk, device=scores.device).unsqueeze(0)
                causal = col_idx > row_idx
                scores = scores.masked_fill(causal, float("-inf"))

            scores = scores.softmax(dim=-1)

            if dropout_p > 0 and query.requires_grad:
                scores = F.dropout(scores, p=dropout_p)

            out_chunk = torch.bmm(scores, v_h)                  # [B, chunk, D]
            output[:, h, start:end] = out_chunk

    return output


@contextmanager
def patched_sdpa(enabled: bool = True):
    """Context manager that replaces F.scaled_dot_product_attention with
    the chunked version for the duration of the block."""
    if not enabled:
        yield
        return
    original = torch.nn.functional.scaled_dot_product_attention
    torch.nn.functional.scaled_dot_product_attention = _chunked_sdpa
    print("  [SDPA] Chunked attention ENABLED (no flash/mem-efficient on this GPU)")
    try:
        yield
    finally:
        torch.nn.functional.scaled_dot_product_attention = original
        print("  [SDPA] Restored original SDPA")


def _resolve_chunked_attention(cfg: dict) -> bool:
    """Decide whether to enable chunked SDPA based on config + GPU probe.

    Returns True if chunked SDPA should be used.
    """
    mode = cfg.get("chunked_attention", "auto")
    if mode == "off":
        print("  [SDPA] Chunked attention: OFF (config)")
        return False
    if mode == "on":
        print("  [SDPA] Chunked attention: ON (forced by config)")
        return True
    # mode == "auto" — probe the GPU
    backends = _sdpa_backends_available()
    has_efficient = backends["flash"] or backends["mem_efficient"]
    print(f"  [SDPA] Backend probe: flash={backends['flash']}, "
          f"mem_efficient={backends['mem_efficient']}, math={backends['math']}")
    if has_efficient:
        print("  [SDPA] Chunked attention: OFF (flash/mem-efficient available)")
        return False
    else:
        print("  [SDPA] Chunked attention: ON (only math backend available — "
              "chunking prevents OOM)")
        return True
