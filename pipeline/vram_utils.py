"""
pipeline/vram_utils.py — VRAM and RAM monitoring helpers
========================================================

Leaf module — no project-internal dependencies.
Used by almost every other module in the pipeline package.
"""

import gc

import torch


def flush_vram():
    """Aggressively free GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def ram_stats() -> dict:
    """Return current system RAM usage."""
    try:
        import psutil
        mem = psutil.virtual_memory()
        return {
            "total_gb": round(mem.total / 1024**3, 2),
            "used_gb": round(mem.used / 1024**3, 2),
            "available_gb": round(mem.available / 1024**3, 2),
            "percent": mem.percent,
        }
    except ImportError:
        return {"available": False}


def vram_stats() -> dict:
    """Return current VRAM usage stats."""
    if not torch.cuda.is_available():
        return {"available": False}
    return {
        "available": True,
        "name": torch.cuda.get_device_name(0),
        "allocated_gb": round(torch.cuda.memory_allocated() / 1024**3, 2),
        "reserved_gb": round(torch.cuda.memory_reserved() / 1024**3, 2),
        "total_gb": round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2),
    }
