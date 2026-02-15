#!/usr/bin/env python3
"""
amd_tune.py — AMD ROCm GPU detection, tuning, and benchmarking
==============================================================

Pure PyTorch utility — no project dependencies (leaf module).
Can be run standalone for GPU diagnostics: python amd_tune.py

Dependencies (project-internal): none
External: torch, os, sys

Public API:
  apply_amd_optimisations(cfg) — set all AMD env vars from config
  detect_amd_gpu()             — return GPU info dict
  benchmark_matmul()           — quick TFLOPS throughput test
  main()                       — CLI diagnostic runner

Environment variables set:
  HSA_OVERRIDE_GFX_VERSION     — GFX ISA override (e.g. "11.0.0" for gfx1100)
  PYTORCH_HIP_ALLOC_CONF      — expandable_segments:True
  HIP_VISIBLE_DEVICES          — "0" (single GPU)
  TOKENIZERS_PARALLELISM       — "false"

Used by: app.py (at startup), pipeline_engine.py (apply_amd_env)
"""

import os
import sys
import torch


def detect_amd_gpu() -> dict:
    """Return a dict of GPU info, or empty if not available."""
    info = {"available": False}
    if not torch.cuda.is_available():
        return info

    info["available"] = True
    info["device_name"] = torch.cuda.get_device_name(0)
    props = torch.cuda.get_device_properties(0)
    # PyTorch 2.x uses 'total_memory', older versions may use 'total_mem'
    total_mem = getattr(props, "total_memory", None) or getattr(props, "total_mem", 0)
    info["total_vram_gb"] = round(total_mem / 1024**3, 1)
    info["hip_version"] = getattr(torch.version, "hip", "N/A")
    info["pytorch_version"] = torch.__version__
    info["bf16_support"] = torch.cuda.is_bf16_supported()

    # Check gfx arch
    try:
        arch = torch.cuda.get_device_properties(0).gcnArchName
        info["gfx_arch"] = arch
    except AttributeError:
        info["gfx_arch"] = "unknown"

    return info


def apply_amd_optimisations(cfg: dict | None = None) -> None:
    """
    Apply all AMD-specific performance tweaks.
    Call before building the pipeline.
    """
    cfg = cfg or {}
    amd = cfg.get("amd", {})

    # 1. GFX version override
    gfx = amd.get("hsa_override_gfx_version", "11.0.0")
    os.environ["HSA_OVERRIDE_GFX_VERSION"] = gfx

    # 2. Expandable VRAM segments (avoids OOM from fragmentation)
    alloc = amd.get("pytorch_hip_alloc_conf", "expandable_segments:True")
    os.environ["PYTORCH_HIP_ALLOC_CONF"] = alloc

    # 3. Single GPU
    os.environ.setdefault("HIP_VISIBLE_DEVICES", "0")

    # 4. Reduce tokenizer threading overhead
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # 5. If bfloat16 is slow on this ROCm version, user can force fp16
    if amd.get("force_fp16", False):
        print("  ⚠  force_fp16=True: using float16 instead of bfloat16")

    print(f"  🔧 AMD tuning applied: gfx={gfx}, alloc={alloc}")


def benchmark_matmul(dtype=torch.bfloat16, size=4096, iters=50):
    """Quick matmul throughput test to verify ROCm is working."""
    device = torch.device("cuda")
    a = torch.randn(size, size, dtype=dtype, device=device)
    b = torch.randn(size, size, dtype=dtype, device=device)

    # Warmup
    for _ in range(5):
        _ = torch.mm(a, b)
    torch.cuda.synchronize()

    import time
    t0 = time.time()
    for _ in range(iters):
        _ = torch.mm(a, b)
    torch.cuda.synchronize()
    elapsed = time.time() - t0

    tflops = (2 * size**3 * iters) / elapsed / 1e12
    print(f"  Matmul {size}×{size} {dtype}: {tflops:.1f} TFLOPS "
          f"({elapsed/iters*1000:.1f} ms/iter)")
    return tflops


def main():
    """Run AMD GPU diagnostics."""
    print("=" * 50)
    print("  AMD ROCm GPU Diagnostics")
    print("=" * 50)

    info = detect_amd_gpu()
    if not info["available"]:
        print("❌ No AMD GPU detected via PyTorch/ROCm")
        print("   Check: HSA_OVERRIDE_GFX_VERSION, HIP_VISIBLE_DEVICES")
        sys.exit(1)

    print(f"  GPU:      {info['device_name']}")
    print(f"  VRAM:     {info['total_vram_gb']} GB")
    print(f"  Arch:     {info['gfx_arch']}")
    print(f"  HIP:      {info['hip_version']}")
    print(f"  PyTorch:  {info['pytorch_version']}")
    print(f"  BF16:     {'✅' if info['bf16_support'] else '❌'}")

    print("\n📊 Running matmul benchmark…")
    benchmark_matmul(torch.float16)
    if info["bf16_support"]:
        benchmark_matmul(torch.bfloat16)
    benchmark_matmul(torch.float32)

    # Memory info
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    print(f"\n  VRAM after bench: {allocated:.2f} GB alloc, "
          f"{reserved:.2f} GB reserved")
    print("\n✅ AMD GPU is working correctly!")


if __name__ == "__main__":
    main()
