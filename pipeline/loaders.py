"""
pipeline/loaders.py — Component loaders (load on demand, free after use)
========================================================================

Each loader creates ONE model component and returns it. The caller is
responsible for placing it on the correct device and freeing it after use.

Public API:
  load_text_encoder(cfg, device_strategy) — UMT5-XXL text encoder + tokenizer
  load_vae(cfg)                           — VAE in float32
  load_single_transformer(cfg, which)     — ONE GGUF transformer (high or low)
  load_transformers(cfg)                  — both GGUF transformers (legacy)
  load_scheduler(cfg)                     — UniPC scheduler from pretrained config

Dependencies: vram_utils.flush_vram
"""

from pathlib import Path

import torch

from pipeline.vram_utils import flush_vram


def load_text_encoder(cfg: dict, device_strategy: str = "auto"):
    """Load text encoder + tokenizer with smart device placement.

    device_strategy:
        "auto" — use accelerate device_map to split model across GPU + CPU,
                 putting as many layers on GPU as VRAM allows.
        "cpu"  — entire model on CPU (safest, slightly slower).
        "gpu"  — entire model on GPU (requires >12 GB free VRAM).
    """
    from transformers import AutoTokenizer, UMT5EncoderModel

    model_id = cfg["model_id"]
    amd = cfg.get("amd", {})
    force_fp16 = amd.get("force_fp16", False)
    dtype = torch.float16 if force_fp16 else torch.bfloat16

    print(f"  Loading tokenizer from {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, subfolder="tokenizer")

    print(f"  Loading text encoder from {model_id} (strategy={device_strategy})")

    if device_strategy == "auto" and torch.cuda.is_available():
        # ── Smart split: measure free VRAM, give GPU as much as fits ──
        headroom_gb = cfg.get("text_encoder_gpu_headroom_gb", 1.5)
        total_vram = torch.cuda.get_device_properties(0).total_memory
        used = max(torch.cuda.memory_allocated(), torch.cuda.memory_reserved())
        free_gb = (total_vram - used) / 1024**3
        usable_gb = max(0.5, free_gb - headroom_gb)

        max_memory = {0: f"{usable_gb:.1f}GiB", "cpu": "64GiB"}
        print(f"  Smart split: {free_gb:.2f} GiB free VRAM, "
              f"budget {usable_gb:.1f} GiB for encoder "
              f"(headroom={headroom_gb} GiB)")

        try:
            text_encoder = UMT5EncoderModel.from_pretrained(
                model_id, subfolder="text_encoder", torch_dtype=dtype,
                device_map="auto",
                max_memory=max_memory,
            )
            # Report how layers were placed
            if hasattr(text_encoder, "hf_device_map"):
                dm = text_encoder.hf_device_map
                on_gpu = sum(1 for v in dm.values() if str(v) == "0")
                on_cpu = sum(1 for v in dm.values() if str(v) == "cpu")
                print(f"  Device split: {on_gpu}/{len(dm)} modules on GPU, "
                      f"{on_cpu}/{len(dm)} on CPU")
        except Exception as e:
            print(f"  *** device_map='auto' failed ({e}), "
                  f"falling back to full CPU", flush=True)
            text_encoder = UMT5EncoderModel.from_pretrained(
                model_id, subfolder="text_encoder", torch_dtype=dtype,
            )
            text_encoder.to("cpu")
    else:
        # "cpu" or "gpu" — plain load, caller moves to device
        text_encoder = UMT5EncoderModel.from_pretrained(
            model_id, subfolder="text_encoder", torch_dtype=dtype,
        )

    return tokenizer, text_encoder


def load_vae(cfg: dict):
    """Load VAE in float32. Uses local safetensors if available.

    Supports cfg["vae_path"] override from presets.
    Falls back to models/vae/wan_2.1_vae.safetensors, then HuggingFace download.
    """
    from diffusers import AutoencoderKLWan
    model_id = cfg["model_id"]

    # Check for preset override first, then default local path
    vae_path = cfg.get("vae_path", "")
    if vae_path and Path(vae_path).exists():
        local_vae = Path(vae_path)
    else:
        local_vae = Path("models/vae/wan_2.1_vae.safetensors")

    if local_vae.exists():
        print(f"  Loading local VAE: {local_vae}")
        vae = AutoencoderKLWan.from_single_file(
            str(local_vae),
            config=model_id,
            subfolder="vae",
            torch_dtype=torch.float32,
        )
    else:
        print(f"  Downloading VAE from {model_id}")
        vae = AutoencoderKLWan.from_pretrained(
            model_id, subfolder="vae", torch_dtype=torch.float32,
        )
    return vae


def load_single_transformer(cfg: dict, which: str = "high"):
    """Load ONE GGUF transformer to halve peak RAM (~10 GB instead of ~20 GB).

    Args:
        which: "high" for high-noise transformer, "low" for low-noise.
    """
    from diffusers import WanTransformer3DModel, GGUFQuantizationConfig

    model_id = cfg["model_id"]
    amd = cfg.get("amd", {})
    force_fp16 = amd.get("force_fp16", False)
    dtype = torch.float16 if force_fp16 else torch.bfloat16
    quant_config = GGUFQuantizationConfig(compute_dtype=dtype)

    if which == "high":
        gguf_path = cfg["gguf_transformer_high"]
        subfolder = "transformer"
    else:
        gguf_path = cfg["gguf_transformer_low"]
        subfolder = "transformer_2"

    print(f"  Loading GGUF transformer ({which}): {gguf_path}")
    transformer = WanTransformer3DModel.from_single_file(
        gguf_path,
        quantization_config=quant_config,
        config=model_id,
        subfolder=subfolder,
        torch_dtype=dtype,
    )
    return transformer


def load_transformers(cfg: dict):
    """Load both GGUF transformers (legacy — kept for backward compat)."""
    high = load_single_transformer(cfg, "high")
    low = load_single_transformer(cfg, "low")
    return high, low


def load_scheduler(cfg: dict):
    """Load scheduler from pretrained config."""
    from diffusers.schedulers import UniPCMultistepScheduler
    from diffusers import WanImageToVideoPipeline

    model_id = cfg["model_id"]
    pipe_tmp = WanImageToVideoPipeline.from_pretrained(
        model_id,
        transformer=None, transformer_2=None,
        text_encoder=None, vae=None, image_encoder=None,
        torch_dtype=torch.float32,
    )
    sched_config = pipe_tmp.scheduler.config
    del pipe_tmp
    flush_vram()

    flow_shift = cfg.get("flow_shift", 8.0)
    scheduler = UniPCMultistepScheduler.from_config(sched_config, flow_shift=flow_shift)
    return scheduler
