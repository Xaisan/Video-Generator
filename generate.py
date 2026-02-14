#!/usr/bin/env python3
"""
Wan 2.2 Image-to-Video Generator — AMD ROCm Optimised
======================================================
Replaces ComfyUI with a lean, headless diffusers pipeline tuned for
Radeon RX 7900 XTX (gfx1100) + Ryzen 9 9950X.

Key design:
  • Transformers loaded from local GGUF files (Q6_K) via from_single_file()
  • Two-stage denoising:
      transformer   (high-noise stage, steps 0 … boundary)
      transformer_2 (low-noise stage,  boundary … end)
  • LoRAs applied UNFUSED (GGUF stores uint quantized weights; fusing
    requires bfloat16 weight shapes → incompatible).  LoRAs are loaded
    BEFORE group offloading is applied.
  • Group-offloading keeps peak VRAM ≈14 GB for the 14B model.
"""

import os
import sys
import time
import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

# ─── AMD env vars (set BEFORE any HIP call) ────────────────────────
def apply_amd_env(cfg: dict) -> None:
    """Set AMD-specific environment variables from config."""
    amd = cfg.get("amd", {})
    os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION",
                          amd.get("hsa_override_gfx_version", "11.0.0"))
    os.environ.setdefault("HIP_VISIBLE_DEVICES", "0")
    os.environ.setdefault("PYTORCH_HIP_ALLOC_CONF",
                          amd.get("pytorch_hip_alloc_conf",
                                  "expandable_segments:True"))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


# ─── Config loading ────────────────────────────────────────────────
def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ─── Image preparation ─────────────────────────────────────────────
def prepare_image(image_path: str, pipe, max_area: int | None = None):
    """Load, resize to VAE-compatible dims, return (PIL.Image, h, w)."""
    image = Image.open(image_path).convert("RGB")
    if max_area is None:
        max_area = 480 * 832  # default I2V area

    aspect_ratio = image.height / image.width
    patch = pipe.transformer.config.patch_size
    mod_value = pipe.vae_scale_factor_spatial * (
        patch[1] if isinstance(patch, (list, tuple)) else patch
    )
    height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
    width  = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
    image = image.resize((int(width), int(height)))
    return image, int(height), int(width)


# ─── LoRA loading ──────────────────────────────────────────────────
def load_loras(pipe, lora_cfgs: list) -> None:
    """
    Load LoRAs into the pipeline.  With GGUF transformers we CANNOT fuse
    (GGUF stores uint-quantized weights with different shapes), so LoRAs
    stay active during inference.

    Order of operations (critical for GGUF + group offloading):
      1. Load all LoRA adapters (high → transformer, low → transformer_2)
      2. Set active adapters + per-adapter weights
      3. Do NOT fuse / unload — must remain active for GGUF
      4. Group offloading is applied AFTER this function returns
    """
    high_adapters = []
    low_adapters = []
    high_scales = {}
    low_scales = {}

    for lora in lora_cfgs:
        path = lora["path"]
        target = lora.get("target", "transformer")
        name = lora["adapter_name"]
        scale = lora.get("scale", 1.0)

        if not os.path.isfile(path):
            print(f"  ⚠  LoRA not found, skipping: {path}")
            continue

        print(f"  📦 Loading LoRA: {name} → {target} (scale={scale})")

        if target == "transformer":
            try:
                pipe.load_lora_weights(path, adapter_name=name)
                high_adapters.append(name)
                high_scales[name] = scale
            except Exception as e:
                print(f"  ❌ Failed to load {name} into transformer: {e}")

        elif target == "transformer_2":
            if not hasattr(pipe, "transformer_2") or pipe.transformer_2 is None:
                print(f"  ⚠  No transformer_2 — skipping {name}")
                continue
            try:
                pipe.load_lora_weights(
                    path,
                    adapter_name=name,
                    load_into_transformer_2=True,
                )
                low_adapters.append(name)
                low_scales[name] = scale
            except Exception as e:
                print(f"  ❌ Failed to load {name} into transformer_2: {e}")

    # Set active adapters with per-adapter weights
    if high_adapters:
        try:
            pipe.transformer.set_adapters(
                high_adapters,
                weights=[high_scales[a] for a in high_adapters],
            )
        except Exception as e:
            print(f"  ⚠  Failed to set high-noise adapters: {e}")

    if low_adapters and hasattr(pipe, "transformer_2") and pipe.transformer_2:
        try:
            pipe.transformer_2.set_adapters(
                low_adapters,
                weights=[low_scales[a] for a in low_adapters],
            )
        except Exception as e:
            print(f"  ⚠  Failed to set low-noise adapters: {e}")

    # NOTE: We do NOT call pipe.fuse_lora() — GGUF quantized weights
    # are stored as uint tensors with different shapes from the original
    # bfloat16 weights.  Fusing will crash with a size mismatch error.
    # See: https://github.com/huggingface/diffusers/issues/12047

    print(f"  ✅ LoRAs loaded (unfused): {len(high_adapters)} high-noise, "
          f"{len(low_adapters)} low-noise")


# ─── Group offloading (AMD-optimised) ──────────────────────────────
def setup_offloading(pipe, cfg: dict) -> None:
    """
    Apply group-offloading to move model blocks between CPU ↔ GPU on-the-fly.
    This is the #1 trick for fitting a 14B model into 24 GB VRAM.
    """
    from diffusers.hooks import apply_group_offloading

    offload_type = cfg.get("offload_type", "leaf_level")
    use_stream = cfg.get("use_stream", True)
    onload = torch.device("cuda")
    offload = torch.device("cpu")

    print(f"  ⚡ Offloading: {offload_type}, streams={use_stream}")

    # Transformer(s) — heaviest
    apply_group_offloading(
        pipe.transformer,
        offload_type=offload_type,
        offload_device=offload,
        onload_device=onload,
        use_stream=use_stream,
    )
    if hasattr(pipe, "transformer_2") and pipe.transformer_2 is not None:
        apply_group_offloading(
            pipe.transformer_2,
            offload_type=offload_type,
            offload_device=offload,
            onload_device=onload,
            use_stream=use_stream,
        )

    # Text encoder
    apply_group_offloading(
        pipe.text_encoder,
        offload_type="leaf_level",
        offload_device=offload,
        onload_device=onload,
        use_stream=use_stream,
    )

    # VAE (keep on CPU mostly, pull per-tile)
    apply_group_offloading(
        pipe.vae,
        offload_type="leaf_level",
        offload_device=offload,
        onload_device=onload,
        use_stream=use_stream,
    )

    # Image encoder (CLIP)
    if hasattr(pipe, "image_encoder") and pipe.image_encoder is not None:
        apply_group_offloading(
            pipe.image_encoder,
            offload_type="leaf_level",
            offload_device=offload,
            onload_device=onload,
            use_stream=use_stream,
        )


# ─── Pipeline construction ─────────────────────────────────────────
def build_pipeline(cfg: dict):
    """Build the Wan 2.2 I2V pipeline with GGUF transformers + LoRAs."""
    from diffusers import (
        WanImageToVideoPipeline,
        AutoencoderKLWan,
        WanTransformer3DModel,
        GGUFQuantizationConfig,
    )
    from diffusers.schedulers import UniPCMultistepScheduler

    model_id = cfg["model_id"]
    amd = cfg.get("amd", {})
    force_fp16 = amd.get("force_fp16", False)
    dtype = torch.float16 if force_fp16 else torch.bfloat16

    print(f"🚀 Loading Wan 2.2 I2V pipeline (GGUF quantized)")
    print(f"   Base config: {model_id}")
    print(f"   dtype={dtype}, device=cuda (ROCm)")

    t0 = time.time()

    # ── Load GGUF quantized transformers ────────────────────────────
    gguf_high = cfg.get("gguf_transformer_high")
    gguf_low = cfg.get("gguf_transformer_low")

    if not gguf_high or not os.path.isfile(gguf_high):
        print(f"❌ High-noise GGUF not found: {gguf_high}")
        sys.exit(1)
    if not gguf_low or not os.path.isfile(gguf_low):
        print(f"❌ Low-noise GGUF not found: {gguf_low}")
        sys.exit(1)

    quant_config = GGUFQuantizationConfig(compute_dtype=dtype)

    print(f"   Loading high-noise transformer: {os.path.basename(gguf_high)}")
    transformer_high = WanTransformer3DModel.from_single_file(
        gguf_high,
        quantization_config=quant_config,
        config=model_id,
        subfolder="transformer",
        torch_dtype=dtype,
    )

    print(f"   Loading low-noise transformer:  {os.path.basename(gguf_low)}")
    transformer_low = WanTransformer3DModel.from_single_file(
        gguf_low,
        quantization_config=quant_config,
        config=model_id,
        subfolder="transformer_2",
        torch_dtype=dtype,
    )

    # ── VAE (float32 for decode quality) ────────────────────────────
    vae = AutoencoderKLWan.from_pretrained(
        model_id, subfolder="vae", torch_dtype=torch.float32,
    )

    # ── Assemble pipeline ───────────────────────────────────────────
    pipe = WanImageToVideoPipeline.from_pretrained(
        model_id,
        transformer=transformer_high,
        transformer_2=transformer_low,
        vae=vae,
        torch_dtype=dtype,
    )

    # Scheduler: UniPC (matches ComfyUI uni_pc sampler)
    flow_shift = cfg.get("flow_shift", 8.0)
    pipe.scheduler = UniPCMultistepScheduler.from_config(
        pipe.scheduler.config,
        flow_shift=flow_shift,
    )

    print(f"   Pipeline loaded in {time.time() - t0:.1f}s")

    # ── LoRAs (MUST be loaded BEFORE group offloading) ──────────────
    # See: https://github.com/huggingface/diffusers/issues/11791
    loras = cfg.get("loras", [])
    if loras:
        print(f"\n📎 Loading {len(loras)} LoRA(s) (unfused — GGUF mode)…")
        load_loras(pipe, loras)

    # ── Group offloading or full GPU ─────────────────────────────────
    if cfg.get("enable_group_offload", True):
        # With group offloading, keep model on CPU; hooks move blocks to GPU
        pipe.to("cpu")
        setup_offloading(pipe, cfg)
    else:
        # No offloading — move everything to GPU
        pipe.to("cuda")

    # ── VAE tiling / slicing ────────────────────────────────────────
    if cfg.get("vae_tiling", True):
        pipe.vae.enable_tiling()
        print("   VAE tiling: ON")
    if cfg.get("vae_slicing", True):
        pipe.vae.enable_slicing()
        print("   VAE slicing: ON")

    # ── torch.compile (experimental on ROCm) ────────────────────────
    # IMPORTANT: torch.compile must be applied AFTER group offloading.
    # Also: fullgraph=True is NOT compatible with group offloading hooks.
    if amd.get("enable_torch_compile", False):
        mode = amd.get("compile_mode", "reduce-overhead")
        print(f"   torch.compile: {mode} (fullgraph=False, applied after offloading)")
        pipe.transformer = torch.compile(
            pipe.transformer, mode=mode, fullgraph=False
        )
        if hasattr(pipe, "transformer_2") and pipe.transformer_2 is not None:
            pipe.transformer_2 = torch.compile(
                pipe.transformer_2, mode=mode, fullgraph=False
            )

    return pipe


# ─── Generation ─────────────────────────────────────────────────────
def generate(pipe, cfg: dict) -> str:
    """Run the two-stage denoising and export video."""
    from diffusers.utils import export_to_video

    # Prepare input image
    image_path = cfg["input_image"]
    if not os.path.isfile(image_path):
        print(f"❌ Input image not found: {image_path}")
        print("   Place your image in the input/ directory and set input_image in config.yaml")
        sys.exit(1)

    print(f"\n🖼  Input image: {image_path}")

    # Use config width/height if set, otherwise auto-compute from aspect ratio
    cfg_width = cfg.get("width")
    cfg_height = cfg.get("height")
    if cfg_width and cfg_height:
        image = Image.open(image_path).convert("RGB").resize((cfg_width, cfg_height))
        width, height = cfg_width, cfg_height
    else:
        max_area = (cfg_height or 480) * (cfg_width or 832)
        image, height, width = prepare_image(image_path, pipe, max_area)

    num_frames = cfg.get("num_frames", 81)

    print(f"   Resolution: {width}×{height}, {num_frames} frames")

    # Generator for reproducibility
    seed = cfg.get("seed", -1)
    if seed < 0:
        seed = torch.seed()
    generator = torch.Generator(device="cuda").manual_seed(seed)
    print(f"   Seed: {seed}")

    # ── Run pipeline ────────────────────────────────────────────────
    steps = cfg.get("num_inference_steps", 8)
    cfg_scale = cfg.get("guidance_scale", 2.0)
    cfg_scale_2 = cfg.get("guidance_scale_2", cfg_scale)
    neg_prompt = cfg.get("negative_prompt", "")
    prompt = cfg.get("prompt", "")

    print(f"\n🎬 Generating: {steps} steps, CFG={cfg_scale}/{cfg_scale_2}")
    print(f"   Prompt: {prompt[:80]}{'…' if len(prompt)>80 else ''}")

    t0 = time.time()

    output = pipe(
        image=image,
        prompt=prompt,
        negative_prompt=neg_prompt,
        height=height,
        width=width,
        num_frames=num_frames,
        num_inference_steps=steps,
        guidance_scale=cfg_scale,
        guidance_scale_2=cfg_scale_2,
        generator=generator,
    )

    gen_time = time.time() - t0
    frames = output.frames[0]
    print(f"   ✅ Generation done in {gen_time:.1f}s "
          f"({gen_time/num_frames:.2f}s/frame)")

    # ── Export video ────────────────────────────────────────────────
    output_path = cfg.get("output_path", "output/video.mp4")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fps = cfg.get("fps", 16)
    export_to_video(frames, output_path, fps=fps)
    print(f"   📹 Saved: {output_path} ({fps} FPS)")

    # ── Optional upscale ────────────────────────────────────────────
    if cfg.get("enable_upscale", False):
        from upscale import upscale_video
        up_path = output_path.replace(".mp4", "_upscaled.mp4")
        upscale_video(output_path, up_path, cfg)
        print(f"   🔍 Upscaled: {up_path}")

    return output_path


# ─── Memory report ─────────────────────────────────────────────────
def print_gpu_stats():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        total = torch.cuda.get_device_properties(0).total_mem / 1024**3
        name = torch.cuda.get_device_name(0)
        print(f"\n📊 GPU: {name}")
        print(f"   VRAM: {allocated:.1f} GB used / {reserved:.1f} GB reserved / {total:.1f} GB total")


# ─── CLI ────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Wan 2.2 I2V Video Generator — AMD ROCm Optimised"
    )
    parser.add_argument("-c", "--config", default="config.yaml",
                        help="Path to YAML config")
    parser.add_argument("-i", "--image", default=None,
                        help="Override input image path")
    parser.add_argument("-p", "--prompt", default=None,
                        help="Override prompt text")
    parser.add_argument("-o", "--output", default=None,
                        help="Override output video path")
    parser.add_argument("-s", "--steps", type=int, default=None,
                        help="Override inference steps")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override seed")
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--frames", type=int, default=None)

    args = parser.parse_args()

    # Load config
    cfg = load_config(args.config)

    # CLI overrides
    if args.image:
        cfg["input_image"] = args.image
    if args.prompt:
        cfg["prompt"] = args.prompt
    if args.output:
        cfg["output_path"] = args.output
    if args.steps:
        cfg["num_inference_steps"] = args.steps
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.width:
        cfg["width"] = args.width
    if args.height:
        cfg["height"] = args.height
    if args.frames:
        cfg["num_frames"] = args.frames

    # Apply AMD env
    apply_amd_env(cfg)

    print("=" * 60)
    print("  Wan 2.2 I2V — AMD ROCm Video Generator")
    print("=" * 60)

    # Verify GPU
    if not torch.cuda.is_available():
        print("❌ No GPU detected! Make sure ROCm is working.")
        print("   Run: rocminfo | grep gfx")
        sys.exit(1)

    print_gpu_stats()

    # Build & run
    pipe = build_pipeline(cfg)
    print_gpu_stats()

    output_path = generate(pipe, cfg)

    print_gpu_stats()
    print(f"\n🎉 Done! Video saved to: {output_path}")


if __name__ == "__main__":
    main()
