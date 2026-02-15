"""
pipeline_engine.py — Backward-compatibility shim
=================================================

This file re-exports everything from the ``pipeline`` package so that
existing imports like::

    from pipeline_engine import PipelineEngine, vram_stats, load_config

continue to work unchanged.

The actual code now lives in ``pipeline/`` (split into logical modules).
See ``pipeline/__init__.py`` for the full list of submodules.

DO NOT add new code here — put it in the appropriate pipeline submodule.
"""

# Re-export everything from the pipeline package
from pipeline import (  # noqa: F401
    # vram_utils
    flush_vram,
    vram_stats,
    ram_stats,
    # amd_env
    apply_amd_env,
    # sdpa
    _ORIGINAL_SDPA,
    _sdpa_backends_available,
    _chunked_sdpa,
    patched_sdpa,
    _resolve_chunked_attention,
    # config
    load_config,
    prepare_image,
    snap_to_mod,
    # loaders
    load_text_encoder,
    load_vae,
    load_single_transformer,
    load_transformers,
    load_scheduler,
    # lora
    apply_loras_to_transformer,
    apply_loras,
    # offload
    setup_offloading,
    remove_offloading,
    # session_logger
    SessionLogger,
    # engine
    PipelineEngine,
)
