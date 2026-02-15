"""
pipeline/amd_env.py — AMD-specific environment variable setup
=============================================================

Sets HSA_OVERRIDE_GFX_VERSION, HIP_VISIBLE_DEVICES, PYTORCH_ALLOC_CONF,
etc. from config.yaml.  Must be called BEFORE any HIP/ROCm call.

Leaf module — no project-internal dependencies.
"""

import os


def apply_amd_env(cfg: dict) -> None:
    """Set AMD-specific environment variables from config."""
    amd = cfg.get("amd", {})
    os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION",
                          amd.get("hsa_override_gfx_version", "11.0.0"))
    os.environ.setdefault("HIP_VISIBLE_DEVICES", "0")
    alloc_conf = amd.get("pytorch_hip_alloc_conf", "expandable_segments:True")
    os.environ["PYTORCH_ALLOC_CONF"] = alloc_conf
    os.environ["PYTORCH_HIP_ALLOC_CONF"] = alloc_conf
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
