"""
pipeline/offload.py — Group offloading setup/removal
=====================================================

Group offloading shuttles transformer blocks between CPU ↔ GPU during
inference, keeping only 1–2 blocks on GPU at a time.  This is the #1
trick for fitting a 14B model into 24 GB VRAM.

Public API:
  setup_offloading(module, cfg)  — apply group offloading hooks
  remove_offloading(module)      — remove hooks (safe to call if none)

CRITICAL: use_stream is hardcoded False for AMD/HIP compatibility.
          CUDA streams are broken on ROCm.
"""

import torch


def setup_offloading(module, cfg: dict):
    """Apply group offloading to a single module.

    Defaults to block_level for better GPU utilisation on AMD.
    use_stream is hardcoded False for AMD/HIP compatibility.
    """
    from diffusers.hooks import apply_group_offloading
    offload_type = cfg.get("offload_type", "block_level")
    num_blocks = cfg.get("num_blocks_per_group", 1)
    print(f"  Setting up {offload_type} offloading (num_blocks_per_group={num_blocks})…")
    kwargs = dict(
        offload_type=offload_type,
        offload_device=torch.device("cpu"),
        onload_device=torch.device("cuda"),
        use_stream=False,
    )
    if offload_type == "block_level":
        kwargs["num_blocks_per_group"] = num_blocks
    apply_group_offloading(module, **kwargs)


def remove_offloading(module):
    """Remove group offloading hooks from a module (safe to call if none)."""
    if hasattr(module, "_diffusers_hook"):
        try:
            module._diffusers_hook.remove_hook("group_offloading", recurse=True)
        except Exception:
            pass
