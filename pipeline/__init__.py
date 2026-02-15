"""
pipeline/ — Core video generation engine (split from pipeline_engine.py)
========================================================================

This package contains all model loading, memory management, attention
workarounds, and the 5-step generation engine for Wan 2.2 I2V.

Submodules:
  vram_utils      — VRAM/RAM helpers (flush, stats)
  amd_env         — AMD environment variable setup
  sdpa            — Chunked SDPA attention (AMD workaround)
  config          — load_config(), image prep, snap_to_mod
  loaders         — Component loaders (text encoder, VAE, transformer, scheduler)
  lora            — LoRA adapter loading (per-transformer target)
  offload         — Group offloading setup/removal
  session_logger  — Per-session file logger (generation.log + vram.log)
  engine          — PipelineEngine class (5-step orchestration)

Backward-compatible: all public names are re-exported here, so
  ``from pipeline import PipelineEngine, vram_stats, load_config``
works the same as the old
  ``from pipeline_engine import PipelineEngine, vram_stats, load_config``

See also: AGENTS.md, ARCHITECTURE.md
"""

# ── Re-export everything for backward compatibility ─────────────────
from pipeline.vram_utils import flush_vram, vram_stats, ram_stats
from pipeline.amd_env import apply_amd_env
from pipeline.sdpa import (
    _ORIGINAL_SDPA,
    _sdpa_backends_available,
    _chunked_sdpa,
    patched_sdpa,
    _resolve_chunked_attention,
)
from pipeline.config import load_config, prepare_image, snap_to_mod
from pipeline.loaders import (
    load_text_encoder,
    load_vae,
    load_single_transformer,
    load_transformers,
    load_scheduler,
)
from pipeline.lora import apply_loras_to_transformer, apply_loras
from pipeline.offload import setup_offloading, remove_offloading
from pipeline.session_logger import SessionLogger
from pipeline.engine import PipelineEngine

__all__ = [
    # vram_utils
    "flush_vram", "vram_stats", "ram_stats",
    # amd_env
    "apply_amd_env",
    # sdpa
    "_ORIGINAL_SDPA", "_sdpa_backends_available", "_chunked_sdpa",
    "patched_sdpa", "_resolve_chunked_attention",
    # config
    "load_config", "prepare_image", "snap_to_mod",
    # loaders
    "load_text_encoder", "load_vae", "load_single_transformer",
    "load_transformers", "load_scheduler",
    # lora
    "apply_loras_to_transformer", "apply_loras",
    # offload
    "setup_offloading", "remove_offloading",
    # session_logger
    "SessionLogger",
    # engine
    "PipelineEngine",
]
