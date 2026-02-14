#!/usr/bin/env python3
"""
Pipeline Engine — step-by-step video generation with checkpointing.

The pipeline is decomposed into discrete steps that can be run independently:
  1. encode      — text + image encoding → embeddings
  2. denoise     — two-stage transformer denoising → latents
  3. vae_decode  — VAE decode latents → frames
  4. export      — frames → mp4 video
  5. upscale     — (optional) 4× upscale

Between each step, the previous stage's heavy models are freed from VRAM.
Intermediate results are checkpointed to the session directory so any
step can be resumed without re-running earlier ones.
"""

import gc
import os
import sys
import time
import traceback
from typing import Callable

import numpy as np
import torch
import yaml
from PIL import Image

from session_manager import SessionManager, StepStatus, STEP_ORDER


# ─── VRAM cleanup ──────────────────────────────────────────────────
def flush_vram():
    """Aggressively free GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def vram_stats() -> dict:
    """Return current VRAM usage stats."""
    if not torch.cuda.is_available():
        return {"available": False}
    return {
        "available": True,
        "name": torch.cuda.get_device_name(0),
        "allocated_gb": round(torch.cuda.memory_allocated() / 1024**3, 2),
        "reserved_gb": round(torch.cuda.memory_reserved() / 1024**3, 2),
        "total_gb": round(torch.cuda.get_device_properties(0).total_mem / 1024**3, 2),
    }


# ─── AMD env vars ──────────────────────────────────────────────────
def apply_amd_env(cfg: dict) -> None:
    amd = cfg.get("amd", {})
    os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION",
                          amd.get("hsa_override_gfx_version", "11.0.0"))
    os.environ.setdefault("HIP_VISIBLE_DEVICES", "0")
    os.environ.setdefault("PYTORCH_HIP_ALLOC_CONF",
                          amd.get("pytorch_hip_alloc_conf",
                                  "expandable_segments:True"))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ─── Image preparation ─────────────────────────────────────────────
def prepare_image(image_path: str, width: int, height: int) -> Image.Image:
    """Load and resize image to exact target dimensions."""
    image = Image.open(image_path).convert("RGB")
    image = image.resize((width, height), Image.LANCZOS)
    return image


def snap_to_mod(value: int, mod: int = 16) -> int:
    """Round down to nearest multiple of mod."""
    return (value // mod) * mod


# ─── Pipeline component loaders (load on demand, free after use) ───

def load_text_encoder(cfg: dict):
    """Load just the text encoder + tokenizer."""
    from transformers import AutoTokenizer, UMT5EncoderModel

    model_id = cfg["model_id"]
    amd = cfg.get("amd", {})
    force_fp16 = amd.get("force_fp16", False)
    dtype = torch.float16 if force_fp16 else torch.bfloat16

    tokenizer = AutoTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    text_encoder = UMT5EncoderModel.from_pretrained(
        model_id, subfolder="text_encoder", torch_dtype=dtype,
    )
    return tokenizer, text_encoder


def load_image_encoder(cfg: dict):
    """Load CLIP image encoder + processor."""
    from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor

    model_id = cfg["model_id"]
    amd = cfg.get("amd", {})
    force_fp16 = amd.get("force_fp16", False)
    dtype = torch.float16 if force_fp16 else torch.bfloat16

    image_processor = CLIPImageProcessor.from_pretrained(
        model_id, subfolder="image_processor"
    )
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        model_id, subfolder="image_encoder", torch_dtype=dtype,
    )
    return image_processor, image_encoder


def load_vae(cfg: dict):
    """Load VAE in float32."""
    from diffusers import AutoencoderKLWan
    model_id = cfg["model_id"]
    vae = AutoencoderKLWan.from_pretrained(
        model_id, subfolder="vae", torch_dtype=torch.float32,
    )
    return vae


def load_transformers(cfg: dict):
    """Load both GGUF transformers."""
    from diffusers import WanTransformer3DModel, GGUFQuantizationConfig

    model_id = cfg["model_id"]
    amd = cfg.get("amd", {})
    force_fp16 = amd.get("force_fp16", False)
    dtype = torch.float16 if force_fp16 else torch.bfloat16
    quant_config = GGUFQuantizationConfig(compute_dtype=dtype)

    gguf_high = cfg["gguf_transformer_high"]
    gguf_low = cfg["gguf_transformer_low"]

    transformer_high = WanTransformer3DModel.from_single_file(
        gguf_high,
        quantization_config=quant_config,
        config=model_id,
        subfolder="transformer",
        torch_dtype=dtype,
    )
    transformer_low = WanTransformer3DModel.from_single_file(
        gguf_low,
        quantization_config=quant_config,
        config=model_id,
        subfolder="transformer_2",
        torch_dtype=dtype,
    )
    return transformer_high, transformer_low


def load_scheduler(cfg: dict):
    """Load scheduler from pretrained config."""
    from diffusers.schedulers import UniPCMultistepScheduler
    from diffusers import WanImageToVideoPipeline

    model_id = cfg["model_id"]
    # Load scheduler config from HF hub
    pipe_tmp = WanImageToVideoPipeline.from_pretrained(
        model_id,
        transformer=None,
        transformer_2=None,
        text_encoder=None,
        vae=None,
        image_encoder=None,
        torch_dtype=torch.float32,
    )
    sched_config = pipe_tmp.scheduler.config
    del pipe_tmp
    flush_vram()

    flow_shift = cfg.get("flow_shift", 8.0)
    scheduler = UniPCMultistepScheduler.from_config(sched_config, flow_shift=flow_shift)
    return scheduler


# ─── LoRA loading ──────────────────────────────────────────────────
def apply_loras(pipe, cfg: dict, lora_overrides: list | None = None):
    """Load LoRAs into the pipeline (unfused for GGUF)."""
    lora_cfgs = cfg.get("loras", [])
    if not lora_cfgs:
        return

    # Apply scale overrides from session
    if lora_overrides:
        override_map = {o["adapter_name"]: o["scale"] for o in lora_overrides}
        for lora in lora_cfgs:
            if lora["adapter_name"] in override_map:
                lora["scale"] = override_map[lora["adapter_name"]]

    high_adapters, low_adapters = [], []
    high_scales, low_scales = {}, {}

    for lora in lora_cfgs:
        path = lora["path"]
        target = lora.get("target", "transformer")
        name = lora["adapter_name"]
        scale = lora.get("scale", 1.0)

        if not os.path.isfile(path):
            continue

        if target == "transformer":
            try:
                pipe.load_lora_weights(path, adapter_name=name)
                high_adapters.append(name)
                high_scales[name] = scale
            except Exception as e:
                print(f"  ❌ LoRA {name}: {e}")

        elif target == "transformer_2":
            if not hasattr(pipe, "transformer_2") or pipe.transformer_2 is None:
                continue
            try:
                pipe.load_lora_weights(path, adapter_name=name, load_into_transformer_2=True)
                low_adapters.append(name)
                low_scales[name] = scale
            except Exception as e:
                print(f"  ❌ LoRA {name}: {e}")

    if high_adapters:
        pipe.transformer.set_adapters(high_adapters,
                                      weights=[high_scales[a] for a in high_adapters])
    if low_adapters and hasattr(pipe, "transformer_2") and pipe.transformer_2:
        pipe.transformer_2.set_adapters(low_adapters,
                                        weights=[low_scales[a] for a in low_adapters])


# ─── Group offloading ──────────────────────────────────────────────
def setup_offloading(module, cfg: dict):
    """Apply group offloading to a single module."""
    from diffusers.hooks import apply_group_offloading
    apply_group_offloading(
        module,
        offload_type=cfg.get("offload_type", "leaf_level"),
        offload_device=torch.device("cpu"),
        onload_device=torch.device("cuda"),
        use_stream=cfg.get("use_stream", True),
    )


# ═══════════════════════════════════════════════════════════════════
#  Pipeline Engine — orchestrates step-by-step generation
# ═══════════════════════════════════════════════════════════════════

class PipelineEngine:
    """
    Runs the video generation pipeline step-by-step with checkpointing.

    The pipeline object is NOT kept alive between steps — each step loads
    only the components it needs, then frees them.  This keeps VRAM usage
    minimal between steps.
    """

    def __init__(self, config_path: str = "config.yaml"):
        self.cfg = load_config(config_path)
        self.config_path = config_path
        apply_amd_env(self.cfg)
        self.sm = SessionManager()
        self._progress_callback: Callable | None = None
        self._cancel_flag = False

    def set_progress_callback(self, cb: Callable):
        """Set callback: cb(session_id, step, progress_pct, message)"""
        self._progress_callback = cb

    def cancel(self):
        self._cancel_flag = True

    def _emit(self, session_id: str, step: str, pct: float, msg: str):
        if self._progress_callback:
            self._progress_callback(session_id, step, pct, msg)

    # ─── Step 1: Encode ─────────────────────────────────────────────
    def step_encode(self, session_id: str) -> bool:
        """Encode text prompt + input image → embeddings checkpoint."""
        info = self.sm.get_session(session_id)
        if not info:
            return False

        self.sm.update_step(session_id, "encode", StepStatus.RUNNING)
        self._emit(session_id, "encode", 0, "Loading text encoder…")
        t0 = time.time()

        try:
            cfg = self.cfg
            amd = cfg.get("amd", {})
            force_fp16 = amd.get("force_fp16", False)
            dtype = torch.float16 if force_fp16 else torch.bfloat16

            # ── Text encoding ───────────────────────────────────────
            self._emit(session_id, "encode", 10, "Encoding text prompt…")
            tokenizer, text_encoder = load_text_encoder(cfg)
            text_encoder.to("cuda")

            # Tokenize
            prompt_ids = tokenizer(
                info.prompt, max_length=512, padding="max_length",
                truncation=True, return_tensors="pt",
            )
            neg_ids = tokenizer(
                info.negative_prompt, max_length=512, padding="max_length",
                truncation=True, return_tensors="pt",
            )

            with torch.no_grad():
                prompt_embeds = text_encoder(
                    prompt_ids.input_ids.to("cuda")
                ).last_hidden_state
                neg_embeds = text_encoder(
                    neg_ids.input_ids.to("cuda")
                ).last_hidden_state

            # Free text encoder
            del text_encoder, tokenizer
            flush_vram()
            self._emit(session_id, "encode", 50, "Text encoded. Loading image encoder…")

            # ── Image encoding ──────────────────────────────────────
            image = prepare_image(info.input_image, info.width, info.height)
            image_processor, image_encoder = load_image_encoder(cfg)
            image_encoder.to("cuda")

            pixel_values = image_processor(
                images=image, return_tensors="pt"
            ).pixel_values.to("cuda", dtype=dtype)

            with torch.no_grad():
                image_embeds = image_encoder(pixel_values).image_embeds

            # Free image encoder
            del image_encoder, image_processor
            flush_vram()

            # ── Save checkpoint ─────────────────────────────────────
            checkpoint = {
                "prompt_embeds": prompt_embeds.cpu(),
                "negative_prompt_embeds": neg_embeds.cpu(),
                "image_embeds": image_embeds.cpu(),
            }
            self.sm.save_checkpoint(session_id, "embeddings", checkpoint)

            elapsed = time.time() - t0
            self.sm.update_step(session_id, "encode", StepStatus.DONE, elapsed)
            self._emit(session_id, "encode", 100,
                       f"Encoding done in {elapsed:.1f}s")
            flush_vram()
            return True

        except Exception as e:
            elapsed = time.time() - t0
            self.sm.update_step(session_id, "encode", StepStatus.FAILED,
                                elapsed, str(e))
            self._emit(session_id, "encode", -1, f"Encode failed: {e}")
            traceback.print_exc()
            flush_vram()
            return False

    # ─── Step 2: Denoise ────────────────────────────────────────────
    def step_denoise(self, session_id: str) -> bool:
        """Run two-stage transformer denoising → latents checkpoint."""
        info = self.sm.get_session(session_id)
        if not info:
            return False

        # Check prerequisite
        if not self.sm.has_checkpoint(session_id, "embeddings"):
            self._emit(session_id, "denoise", -1,
                       "Missing embeddings checkpoint — run encode first")
            return False

        self.sm.update_step(session_id, "denoise", StepStatus.RUNNING)
        self._emit(session_id, "denoise", 0, "Loading GGUF transformers…")
        t0 = time.time()

        try:
            from diffusers import (
                WanImageToVideoPipeline,
                AutoencoderKLWan,
                WanTransformer3DModel,
                GGUFQuantizationConfig,
            )
            from diffusers.schedulers import UniPCMultistepScheduler

            cfg = self.cfg
            amd = cfg.get("amd", {})
            force_fp16 = amd.get("force_fp16", False)
            dtype = torch.float16 if force_fp16 else torch.bfloat16

            # Load the full pipeline (needed for the denoise __call__)
            self._emit(session_id, "denoise", 5, "Loading transformers…")
            transformer_high, transformer_low = load_transformers(cfg)

            self._emit(session_id, "denoise", 15, "Assembling pipeline…")
            # We need a minimal VAE just for the pipeline structure
            # (it won't actually run — we intercept after latents)
            vae = load_vae(cfg)

            pipe = WanImageToVideoPipeline.from_pretrained(
                cfg["model_id"],
                transformer=transformer_high,
                transformer_2=transformer_low,
                vae=vae,
                torch_dtype=dtype,
            )

            flow_shift = cfg.get("flow_shift", 8.0)
            pipe.scheduler = UniPCMultistepScheduler.from_config(
                pipe.scheduler.config, flow_shift=flow_shift,
            )

            # LoRAs (before offloading)
            self._emit(session_id, "denoise", 20, "Loading LoRAs…")
            apply_loras(pipe, cfg, info.lora_scales)

            # Offloading
            if cfg.get("enable_group_offload", True):
                self._emit(session_id, "denoise", 25, "Setting up offloading…")
                pipe.to("cpu")
                setup_offloading(pipe.transformer, cfg)
                if pipe.transformer_2 is not None:
                    setup_offloading(pipe.transformer_2, cfg)
                setup_offloading(pipe.text_encoder, cfg)
                setup_offloading(pipe.vae, cfg)
                if pipe.image_encoder is not None:
                    setup_offloading(pipe.image_encoder, cfg)
            else:
                pipe.to("cuda")

            # VAE tiling/slicing
            if cfg.get("vae_tiling", True):
                pipe.vae.enable_tiling()
            if cfg.get("vae_slicing", True):
                pipe.vae.enable_slicing()

            # Prepare image
            image = prepare_image(info.input_image, info.width, info.height)

            # Seed
            seed = info.seed
            if seed < 0:
                seed = torch.seed() & 0xFFFFFFFF
            generator = torch.Generator(device="cuda").manual_seed(seed)

            # Progress callback for denoising steps
            total_steps = info.num_inference_steps
            def denoise_cb(pipe_obj, step_idx, timestep, cb_kwargs):
                if self._cancel_flag:
                    raise InterruptedError("Generation cancelled by user")
                pct = 30 + int(65 * (step_idx + 1) / total_steps)
                self._emit(session_id, "denoise", pct,
                           f"Denoising step {step_idx + 1}/{total_steps}")
                return cb_kwargs

            self._emit(session_id, "denoise", 30,
                       f"Starting denoising: {total_steps} steps…")

            output = pipe(
                image=image,
                prompt=info.prompt,
                negative_prompt=info.negative_prompt,
                height=info.height,
                width=info.width,
                num_frames=info.num_frames,
                num_inference_steps=info.num_inference_steps,
                guidance_scale=info.guidance_scale,
                guidance_scale_2=info.guidance_scale_2,
                generator=generator,
                output_type="latent",
                callback_on_step_end=denoise_cb,
            )

            latents = output.frames  # raw latent tensor when output_type="latent"

            # Save latents checkpoint
            self.sm.save_checkpoint(session_id, "latents", latents.cpu())

            # Free everything
            del pipe, transformer_high, transformer_low, vae, output
            flush_vram()

            elapsed = time.time() - t0
            self.sm.update_step(session_id, "denoise", StepStatus.DONE, elapsed)
            self._emit(session_id, "denoise", 100,
                       f"Denoising done in {elapsed:.1f}s")
            return True

        except InterruptedError:
            elapsed = time.time() - t0
            self.sm.update_step(session_id, "denoise", StepStatus.FAILED,
                                elapsed, "Cancelled by user")
            self._emit(session_id, "denoise", -1, "Cancelled")
            flush_vram()
            return False

        except Exception as e:
            elapsed = time.time() - t0
            self.sm.update_step(session_id, "denoise", StepStatus.FAILED,
                                elapsed, str(e))
            self._emit(session_id, "denoise", -1, f"Denoise failed: {e}")
            traceback.print_exc()
            flush_vram()
            return False

    # ─── Step 3: VAE Decode ─────────────────────────────────────────
    def step_vae_decode(self, session_id: str) -> bool:
        """Decode latents → video frames via VAE."""
        info = self.sm.get_session(session_id)
        if not info:
            return False

        if not self.sm.has_checkpoint(session_id, "latents"):
            self._emit(session_id, "vae_decode", -1,
                       "Missing latents checkpoint — run denoise first")
            return False

        self.sm.update_step(session_id, "vae_decode", StepStatus.RUNNING)
        self._emit(session_id, "vae_decode", 0, "Loading VAE…")
        t0 = time.time()

        try:
            cfg = self.cfg

            # Load VAE only
            vae = load_vae(cfg)

            if cfg.get("vae_tiling", True):
                vae.enable_tiling()
            if cfg.get("vae_slicing", True):
                vae.enable_slicing()

            if cfg.get("enable_group_offload", True):
                vae.to("cpu")
                setup_offloading(vae, cfg)
            else:
                vae.to("cuda")

            # Load latents
            self._emit(session_id, "vae_decode", 20, "Loading latents…")
            latents = self.sm.load_checkpoint(session_id, "latents")

            if cfg.get("enable_group_offload", True):
                latents = latents.to("cpu")
            else:
                latents = latents.to("cuda")

            # Decode
            self._emit(session_id, "vae_decode", 30, "Decoding latents → frames…")

            # Get the vae scaling factor from config
            latent_mean = torch.tensor(vae.config.get("latents_mean", None))
            latent_std = torch.tensor(vae.config.get("latents_std", None))

            # Unscale latents if needed
            if latent_std is not None and latent_mean is not None:
                latent_mean = latent_mean.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
                latent_std = latent_std.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
                latents = latents * latent_std + latent_mean
            else:
                scaling = getattr(vae.config, "scaling_factor", 1.0)
                latents = latents / scaling

            with torch.no_grad():
                frames_tensor = vae.decode(latents).sample

            # Clamp + convert to uint8 PIL frames
            frames_tensor = frames_tensor.clamp(-1, 1)
            frames_tensor = ((frames_tensor + 1) / 2 * 255).to(torch.uint8)
            # Shape: [B, C, T, H, W] → list of PIL images
            frames_tensor = frames_tensor[0].permute(1, 2, 3, 0).cpu()  # [T, H, W, C]

            # Save frames as numpy for easy video export
            frames_np = frames_tensor.numpy()
            self.sm.save_checkpoint(session_id, "frames", torch.from_numpy(frames_np))

            # Free VAE
            del vae, latents, frames_tensor
            flush_vram()

            # Generate preview video
            self._emit(session_id, "vae_decode", 90, "Saving preview…")
            self._export_preview(session_id, frames_np, info.fps)

            elapsed = time.time() - t0
            self.sm.update_step(session_id, "vae_decode", StepStatus.DONE, elapsed)
            self._emit(session_id, "vae_decode", 100,
                       f"VAE decode done in {elapsed:.1f}s")
            return True

        except Exception as e:
            elapsed = time.time() - t0
            self.sm.update_step(session_id, "vae_decode", StepStatus.FAILED,
                                elapsed, str(e))
            self._emit(session_id, "vae_decode", -1, f"VAE decode failed: {e}")
            traceback.print_exc()
            flush_vram()
            return False

    def _export_preview(self, session_id: str, frames_np, fps: int):
        """Save a quick preview mp4 from numpy frames."""
        import imageio
        sdir = self.sm.session_dir(session_id)
        preview_path = sdir / "preview.mp4"
        writer = imageio.get_writer(str(preview_path), fps=fps, codec="libx264",
                                    quality=6, pixelformat="yuv420p")
        for frame in frames_np:
            writer.append_data(frame)
        writer.close()

    # ─── Step 4: Export ─────────────────────────────────────────────
    def step_export(self, session_id: str) -> bool:
        """Export frames → final mp4 video."""
        info = self.sm.get_session(session_id)
        if not info:
            return False

        if not self.sm.has_checkpoint(session_id, "frames"):
            self._emit(session_id, "export", -1,
                       "Missing frames checkpoint — run VAE decode first")
            return False

        self.sm.update_step(session_id, "export", StepStatus.RUNNING)
        self._emit(session_id, "export", 0, "Loading frames…")
        t0 = time.time()

        try:
            import imageio

            frames_tensor = self.sm.load_checkpoint(session_id, "frames")
            frames_np = frames_tensor.numpy()

            # Export high-quality video
            sdir = self.sm.session_dir(session_id)
            video_path = sdir / "video.mp4"

            self._emit(session_id, "export", 30, "Encoding video…")

            writer = imageio.get_writer(
                str(video_path), fps=info.fps, codec="libx264",
                quality=8, pixelformat="yuv420p",
            )
            for frame in frames_np:
                writer.append_data(frame)
            writer.close()

            # Also copy to output dir with timestamp
            output_dir = Path("output")
            output_dir.mkdir(exist_ok=True)
            import shutil
            final_name = f"video_{info.session_id}.mp4"
            final_path = output_dir / final_name
            shutil.copy2(video_path, final_path)

            elapsed = time.time() - t0
            self.sm.update_step(session_id, "export", StepStatus.DONE, elapsed)
            self._emit(session_id, "export", 100,
                       f"Video exported in {elapsed:.1f}s")
            return True

        except Exception as e:
            elapsed = time.time() - t0
            self.sm.update_step(session_id, "export", StepStatus.FAILED,
                                elapsed, str(e))
            self._emit(session_id, "export", -1, f"Export failed: {e}")
            traceback.print_exc()
            return False

    # ─── Step 5: Upscale (optional) ─────────────────────────────────
    def step_upscale(self, session_id: str) -> bool:
        """Upscale the exported video 4×."""
        info = self.sm.get_session(session_id)
        if not info:
            return False

        if not info.enable_upscale:
            self.sm.update_step(session_id, "upscale", StepStatus.SKIPPED)
            self._emit(session_id, "upscale", 100, "Upscale skipped")
            return True

        video_path = self.sm.get_file_path(session_id, "video.mp4")
        if not video_path:
            self._emit(session_id, "upscale", -1,
                       "Missing video file — run export first")
            return False

        self.sm.update_step(session_id, "upscale", StepStatus.RUNNING)
        self._emit(session_id, "upscale", 0, "Starting upscale…")
        t0 = time.time()

        try:
            from upscale import upscale_video

            sdir = self.sm.session_dir(session_id)
            up_path = str(sdir / "video_upscaled.mp4")
            upscale_video(str(video_path), up_path, self.cfg)

            elapsed = time.time() - t0
            self.sm.update_step(session_id, "upscale", StepStatus.DONE, elapsed)
            self._emit(session_id, "upscale", 100,
                       f"Upscale done in {elapsed:.1f}s")
            flush_vram()
            return True

        except Exception as e:
            elapsed = time.time() - t0
            self.sm.update_step(session_id, "upscale", StepStatus.FAILED,
                                elapsed, str(e))
            self._emit(session_id, "upscale", -1, f"Upscale failed: {e}")
            traceback.print_exc()
            flush_vram()
            return False

    # ─── Run all steps (or from a given step) ───────────────────────
    def run_from_step(self, session_id: str, start_step: str = "encode") -> bool:
        """
        Run the pipeline starting from a specific step.
        Steps before start_step must have their checkpoints available.
        """
        self._cancel_flag = False
        info = self.sm.get_session(session_id)
        if not info:
            return False

        self.sm.update_status(session_id, "running")

        step_funcs = {
            "encode": self.step_encode,
            "denoise": self.step_denoise,
            "vae_decode": self.step_vae_decode,
            "export": self.step_export,
            "upscale": self.step_upscale,
        }

        start_idx = STEP_ORDER.index(start_step) if start_step in STEP_ORDER else 0

        for step_name in STEP_ORDER[start_idx:]:
            if self._cancel_flag:
                self.sm.update_status(session_id, "cancelled")
                return False

            success = step_funcs[step_name](session_id)
            if not success:
                # Step either failed or was skipped
                info = self.sm.get_session(session_id)
                if info and info.steps.get(step_name) == StepStatus.FAILED:
                    return False
                # If skipped, continue

        self.sm.update_status(session_id, "done")
        return True

    def get_resume_step(self, session_id: str) -> str:
        """Determine which step to resume from based on available checkpoints."""
        info = self.sm.get_session(session_id)
        if not info:
            return "encode"

        # Check checkpoints in reverse order to find last completed step
        if self.sm.has_checkpoint(session_id, "frames"):
            return "export"
        if self.sm.has_checkpoint(session_id, "latents"):
            return "vae_decode"
        if self.sm.has_checkpoint(session_id, "embeddings"):
            return "denoise"
        return "encode"
