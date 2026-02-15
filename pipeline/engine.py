"""
pipeline/engine.py — PipelineEngine: 5-step video generation with checkpointing
================================================================================

This is the main orchestration class. Each step loads only the models it
needs, runs inference, saves a checkpoint, then frees everything.

Pipeline steps:
  1. step_encode()      — text prompt → embeddings.pt
  2. step_denoise()     — two-stage transformer denoising → latents.pt
  3. step_vae_decode()  — VAE decode latents → frames.pt + preview.mp4
  4. step_export()      — frames → output.mp4
  5. step_upscale()     — (optional) 4× spatial upscale + RIFE interpolation

Dependencies (within pipeline package):
  -> vram_utils   (flush_vram, vram_stats, ram_stats)
  -> amd_env      (apply_amd_env)
  -> sdpa         (_resolve_chunked_attention, patched_sdpa)
  -> config       (load_config, prepare_image)
  -> loaders      (load_text_encoder, load_vae, load_single_transformer)
  -> lora         (apply_loras_to_transformer)
  -> offload      (remove_offloading)
  -> session_logger (SessionLogger)

Dependencies (project-level):
  -> session_manager (SessionManager, StepStatus, STEP_ORDER)
  -> upscale         (upscale_video) — lazily imported in step_upscale()
"""

import gc
import os
import time
import traceback
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from session_manager import SessionManager, StepStatus, STEP_ORDER

from pipeline.vram_utils import flush_vram, vram_stats, ram_stats
from pipeline.amd_env import apply_amd_env
from pipeline.sdpa import _resolve_chunked_attention, patched_sdpa
from pipeline.config import load_config, prepare_image
from pipeline.loaders import load_text_encoder, load_vae, load_single_transformer
from pipeline.lora import apply_loras_to_transformer
from pipeline.offload import remove_offloading
from pipeline.session_logger import SessionLogger


class PipelineEngine:
    """
    Runs the video generation pipeline step-by-step with checkpointing.

    The pipeline object is NOT kept alive between steps — each step loads
    only the components it needs, then frees them.
    """

    def __init__(self, config_path: str = "config.yaml"):
        self.cfg = load_config(config_path)
        self.config_path = config_path
        apply_amd_env(self.cfg)
        self.sm = SessionManager()
        self._progress_callback: Callable | None = None
        self._cancel_flag = False
        self._slog: SessionLogger | None = None  # per-generation session logger

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
        """Encode text prompt → embeddings checkpoint."""
        info = self.sm.get_session(session_id)
        if not info:
            return False

        # Start session logger (encode is always the first step)
        sdir = self.sm.session_dir(session_id)
        self._slog = SessionLogger(sdir)
        self._slog.header(info, self.cfg)

        self.sm.update_step(session_id, "encode", StepStatus.RUNNING)
        self._emit(session_id, "encode", 0, "Loading text encoder…")
        self._slog.step_start("encode")
        t0 = time.time()

        try:
            cfg = self.cfg
            amd = cfg.get("amd", {})
            force_fp16 = amd.get("force_fp16", False)
            dtype = torch.float16 if force_fp16 else torch.bfloat16

            # Determine device strategy (backward compat: force_text_encoder_cpu)
            if cfg.get("force_text_encoder_cpu", False):
                device_strategy = "cpu"
            else:
                device_strategy = cfg.get("text_encoder_device", "auto")

            self._emit(session_id, "encode", 10, "Loading tokenizer + text encoder…")
            self._slog.log("encode", f"Loading tokenizer + text encoder (dtype={dtype}, strategy={device_strategy})")
            self._slog.vram_event("encode", "before text_encoder load")
            tokenizer, text_encoder = load_text_encoder(cfg, device_strategy)

            # For non-auto strategies, move model to target device
            if device_strategy == "gpu":
                try:
                    text_encoder.to("cuda")
                except torch.cuda.OutOfMemoryError:
                    print("  *** GPU OOM — retrying with auto split", flush=True)
                    self._slog.log("encode", "GPU OOM, retrying with device_map='auto'")
                    del text_encoder
                    flush_vram()
                    tokenizer, text_encoder = load_text_encoder(cfg, "auto")
            elif device_strategy == "cpu":
                text_encoder.to("cpu")
            # "auto" — already placed across devices by device_map

            # Input tensors go to the device of the first parameter
            # (accelerate hooks handle cross-device routing during forward)
            encode_device = next(text_encoder.parameters()).device
            is_split = hasattr(text_encoder, "hf_device_map")
            self._slog.vram_event("encode", f"text_encoder ready (input_device={encode_device}, split={is_split})")
            if is_split:
                self._slog.log("encode", f"Device map: {text_encoder.hf_device_map}")

            self._emit(session_id, "encode", 30, "Encoding prompt…")
            self._slog.log("encode", "Tokenizing prompt…")
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
                    prompt_ids.input_ids.to(encode_device)
                ).last_hidden_state
                neg_embeds = text_encoder(
                    neg_ids.input_ids.to(encode_device)
                ).last_hidden_state

            self._slog.log_tensor("encode", "prompt_embeds", prompt_embeds)
            self._slog.log_tensor("encode", "negative_prompt_embeds", neg_embeds)
            self._slog.vram_event("encode", "after encoding")

            del text_encoder, tokenizer
            flush_vram()
            self._slog.vram_event("encode", "after cleanup")
            print(f"  === ENCODE CLEANUP === VRAM: {vram_stats()}", flush=True)
            self._emit(session_id, "encode", 90, "Text encoded, saving checkpoint…")

            checkpoint = {
                "prompt_embeds": prompt_embeds.cpu(),
                "negative_prompt_embeds": neg_embeds.cpu(),
            }
            self.sm.save_checkpoint(session_id, "embeddings", checkpoint)
            self._slog.log_checkpoint_size("encode",
                self.sm.session_dir(session_id) / "embeddings.pt")

            elapsed = time.time() - t0
            self.sm.update_step(session_id, "encode", StepStatus.DONE, elapsed)
            self._slog.step_end("encode", "done", elapsed)
            self._emit(session_id, "encode", 100,
                       f"Text encoding done in {elapsed:.1f}s")
            flush_vram()
            return True

        except Exception as e:
            elapsed = time.time() - t0
            self.sm.update_step(session_id, "encode", StepStatus.FAILED,
                                elapsed, str(e))
            if self._slog:
                self._slog.log("encode", f"EXCEPTION: {e}")
                self._slog.log("encode", traceback.format_exc())
                self._slog.step_end("encode", "FAILED", elapsed, str(e))
            self._emit(session_id, "encode", -1, f"Encode failed: {e}")
            traceback.print_exc()
            for v in list(locals().values()):
                if isinstance(v, torch.nn.Module):
                    try:
                        v.to("meta")
                    except Exception:
                        pass
            gc.collect()
            flush_vram()
            print(f"  === ENCODE CLEANUP === VRAM after error: {vram_stats()}", flush=True)
            return False

    # ─── Step 2: Denoise ────────────────────────────────────────────
    def step_denoise(self, session_id: str) -> bool:
        """Run two-pass transformer denoising → latents checkpoint.

        Architecture (matches reference WanImageToVideoPipeline):
        ──────────────────────────────────────────────────────────
        Uses a SINGLE scheduler with ONE set_timesteps() call to maintain
        UniPC multi-step solver state.  boundary_timestep splits timesteps
        between the two transformers:
          - timesteps >= boundary → high-noise transformer
          - timesteps < boundary  → low-noise transformer

        Sequential loading: only one transformer in memory at a time.
        Phase 1: VAE encode → latents + condition. VAE freed.
        Phase 2: High-noise transformer loaded, LoRAs, offload, run steps >= boundary. Freed.
        Phase 3: Low-noise transformer loaded, LoRAs, offload, run steps < boundary. Freed.

        Peak RAM: ~10 GB (one transformer at a time).
        Peak VRAM: ~14 GB (transformer blocks offloaded, latents + embeds on GPU).
        """
        info = self.sm.get_session(session_id)
        if not info:
            return False

        if not self.sm.has_checkpoint(session_id, "embeddings"):
            self._emit(session_id, "denoise", -1,
                       "Missing embeddings checkpoint — run encode first")
            return False

        self.sm.update_step(session_id, "denoise", StepStatus.RUNNING)
        self._emit(session_id, "denoise", 0, "Starting denoise…")
        if self._slog:
            self._slog.step_start("denoise")
        t0 = time.time()

        # Pre-clean
        gc.collect()
        flush_vram()
        print(f"  === DENOISE START === VRAM: {vram_stats()}  RAM: {ram_stats()}", flush=True)
        if self._slog:
            self._slog.vram_event("denoise", "pre-clean")

        try:
            from diffusers import WanImageToVideoPipeline
            from diffusers.schedulers import UniPCMultistepScheduler
            from diffusers.hooks import apply_group_offloading

            cfg = self.cfg
            amd = cfg.get("amd", {})
            force_fp16 = amd.get("force_fp16", False)
            dtype = torch.float16 if force_fp16 else torch.bfloat16
            offload_type = cfg.get("offload_type", "block_level")
            use_offload = cfg.get("enable_group_offload", True)
            force_vae_cpu = cfg.get("force_vae_cpu", False)

            # ── Mode detection & logging ──
            # Distill and Quality modes set recommended defaults in the UI,
            # but the user has full control.  We only LOG warnings here.
            distill_lora_mode = getattr(info, "distill_lora_mode", False)
            flow_shift = getattr(info, "flow_shift", None) or cfg.get("flow_shift", 8.0)

            if distill_lora_mode:
                warnings = []
                if info.guidance_scale != 1.0 or info.guidance_scale_2 != 1.0:
                    warnings.append(f"CFG={info.guidance_scale}/{info.guidance_scale_2} "
                                    f"(distill default: 1.0 — higher = more prompt adherence "
                                    f"but may introduce artefacts)")
                if info.num_inference_steps != 4:
                    warnings.append(f"steps={info.num_inference_steps} (distill default: 4)")
                if flow_shift != 5.0:
                    warnings.append(f"flow_shift={flow_shift} (distill default: 5.0)")

                if warnings:
                    print(f"  ⚡ DISTILL MODE: custom overrides — {', '.join(warnings)}")
                else:
                    print("  ⚡ DISTILL MODE: LightX2V enabled — default params")

                if self._slog:
                    self._slog.log("denoise", "⚡ DISTILL MODE ACTIVE")
                    if warnings:
                        for w in warnings:
                            self._slog.log("denoise", f"  ⚠ {w}")
                    self._slog.log("denoise", f"  CFG={info.guidance_scale}/{info.guidance_scale_2}, "
                                   f"steps={info.num_inference_steps}, flow_shift={flow_shift}")
            else:
                # QUALITY MODE — log settings, no enforcement
                warnings = []
                if info.guidance_scale < 2.0:
                    warnings.append(f"CFG_high={info.guidance_scale} is very low "
                                    f"(recommended ≥2.0 for quality mode)")
                if info.guidance_scale_2 < 2.0:
                    warnings.append(f"CFG_low={info.guidance_scale_2} is very low")
                if info.num_inference_steps < 8:
                    warnings.append(f"steps={info.num_inference_steps} is very low "
                                    f"(recommended ≥8 for quality mode)")
                if flow_shift < 3.0:
                    warnings.append(f"flow_shift={flow_shift} is very low")

                if warnings:
                    print(f"  🎨 QUALITY MODE: ⚠ {', '.join(warnings)}")
                    if self._slog:
                        self._slog.log("denoise", "🎨 QUALITY MODE — distill LoRAs will be skipped")
                        for w in warnings:
                            self._slog.log("denoise", f"  ⚠ {w}")
                else:
                    print("  🎨 QUALITY MODE: parameters look good")
                    if self._slog:
                        self._slog.log("denoise", "🎨 QUALITY MODE — distill LoRAs will be skipped")

                if self._slog:
                    self._slog.log("denoise", f"  CFG={info.guidance_scale}/{info.guidance_scale_2}, "
                                   f"steps={info.num_inference_steps}, flow_shift={flow_shift}")

            # ── Chunked SDPA resolution ──
            use_chunked_sdpa = _resolve_chunked_attention(cfg)
            if self._slog:
                self._slog.log("denoise", f"Chunked SDPA: {'ENABLED' if use_chunked_sdpa else 'disabled'}")

            # Load scheduler config
            sched = UniPCMultistepScheduler.from_pretrained(
                cfg["model_id"], subfolder="scheduler",
            )
            sched = UniPCMultistepScheduler.from_config(
                sched.config, flow_shift=flow_shift,
            )

            # ══════════════════════════════════════════════════════════
            #  PHASE 1: VAE encode input image → latents + condition
            # ══════════════════════════════════════════════════════════
            vae_device = "cpu" if force_vae_cpu else "cuda"
            self._emit(session_id, "denoise", 3, "Loading VAE…")
            if self._slog:
                self._slog.log("denoise", f"PHASE 1: VAE encode (device={vae_device}, tiling={cfg.get('vae_tiling', True)}, slicing={cfg.get('vae_slicing', True)})")
            vae = load_vae(cfg)

            if cfg.get("vae_tiling", True):
                vae.enable_tiling()
            if cfg.get("vae_slicing", True):
                vae.enable_slicing()
            vae.to(vae_device)
            print(f"  VAE on {vae_device}. VRAM: {vram_stats()}", flush=True)
            if self._slog:
                self._slog.vram_event("denoise", f"VAE loaded on {vae_device}")

            # Model config constants for Wan2.2-I2V-A14B:
            # These are fixed values from the HF config JSONs.  Previously we
            # loaded a full 12 GB GGUF transformer just to read them — that
            # wasted ~12 GB of system RAM and ~20s of I/O.
            #   vae_scale_factor = 2^(len(dim_mult)-1) = 2^3 = 8
            #   z_dim = 16  (from vae/config.json)
            #   image_dim = None  (no CLIP cross-attn in this model variant)
            vae_scale_factor = 8
            num_channels_latents = vae.config.z_dim   # 16
            image_dim = None   # checked: always None for Wan2.2-I2V-A14B GGUFs

            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                cfg["model_id"], subfolder="tokenizer"
            )

            pipe_vae = WanImageToVideoPipeline(
                tokenizer=tokenizer,
                text_encoder=None,
                transformer=None,
                vae=vae,
                scheduler=sched,
                image_processor=None,
            )

            print(f"  Model config: vae_scale_factor={vae_scale_factor}, "
                  f"z_dim={num_channels_latents}, image_dim={image_dim}  "
                  f"VRAM: {vram_stats()}  RAM: {ram_stats()}")
            if self._slog:
                self._slog.log("denoise", f"vae_scale_factor={vae_scale_factor}, z_dim={num_channels_latents}, image_dim={image_dim} (hardcoded — no config reader load)")

            # Prepare image
            image = prepare_image(info.input_image, info.width, info.height)

            # Load text embeddings → GPU ONCE (stay there for both passes)
            embeddings = self.sm.load_checkpoint(session_id, "embeddings")
            prompt_embeds = embeddings["prompt_embeds"].to(dtype).to("cuda")
            negative_prompt_embeds = embeddings["negative_prompt_embeds"].to(dtype).to("cuda")
            del embeddings

            # Seed
            seed = info.seed
            if seed < 0:
                seed = torch.seed() & 0xFFFFFFFF
            generator = torch.Generator(device="cpu").manual_seed(seed)

            # Prepare latents (VAE encode)
            self._emit(session_id, "denoise", 10, "VAE-encoding input image…")
            print(f"  VRAM before prepare_latents: {vram_stats()}")

            from diffusers.video_processor import VideoProcessor
            video_proc = VideoProcessor(vae_scale_factor=vae_scale_factor)
            image_tensor = video_proc.preprocess(image, height=info.height,
                                                  width=info.width)
            image_tensor = image_tensor.to(vae_device, dtype=torch.float32)

            try:
                with torch.no_grad():
                    latents, condition = pipe_vae.prepare_latents(
                        image_tensor,
                        batch_size=1,
                        num_channels_latents=num_channels_latents,
                        height=info.height,
                        width=info.width,
                        num_frames=info.num_frames,
                        dtype=torch.float32,
                        device=vae_device,
                        generator=generator,
                    )
            except torch.OutOfMemoryError:
                if vae_device == "cuda":
                    print("  ⚠ OOM during GPU VAE encode; retrying on CPU…", flush=True)
                    flush_vram()
                    pipe_vae.vae.to("cpu")
                    image_tensor = image_tensor.to("cpu")
                    with torch.no_grad():
                        latents, condition = pipe_vae.prepare_latents(
                            image_tensor,
                            batch_size=1,
                            num_channels_latents=num_channels_latents,
                            height=info.height,
                            width=info.width,
                            num_frames=info.num_frames,
                            dtype=torch.float32,
                            device="cpu",
                            generator=generator,
                        )
                else:
                    raise

            # Move to GPU in inference dtype
            latents = latents.to("cuda", dtype=dtype)
            condition = condition.to("cuda", dtype=dtype)
            print(f"  VRAM after prepare_latents: {vram_stats()}")
            if self._slog:
                self._slog.log_tensor("denoise", "latents", latents)
                self._slog.log_tensor("denoise", "condition", condition)
                self._slog.log_tensor("denoise", "prompt_embeds", prompt_embeds)
                self._slog.vram_event("denoise", "after prepare_latents")

            # Free VAE completely
            pipe_vae.vae.to("meta")
            pipe_vae.vae = None
            del vae, pipe_vae, image_tensor, video_proc
            gc.collect()
            flush_vram()
            print(f"  Freed VAE. VRAM: {vram_stats()}  RAM: {ram_stats()}")
            if self._slog:
                self._slog.vram_event("denoise", "VAE freed")

            # CLIP image embeddings (if transformer expects them)
            image_embeds = None
            if image_dim is not None:
                self._emit(session_id, "denoise", 16, "Encoding image (CLIP)…")
                from transformers import CLIPVisionModel, CLIPImageProcessor
                img_proc = CLIPImageProcessor.from_pretrained(
                    cfg["model_id"], subfolder="image_processor"
                )
                img_enc = CLIPVisionModel.from_pretrained(
                    cfg["model_id"], subfolder="image_encoder",
                    torch_dtype=torch.float32,
                )
                img_enc.to("cuda")
                img_input = img_proc(images=image, return_tensors="pt")
                img_input = {k: v.to("cuda") for k, v in img_input.items()}
                with torch.no_grad():
                    image_embeds = img_enc(**img_input,
                                          output_hidden_states=True).hidden_states[-2]
                image_embeds = image_embeds.to(dtype)
                del img_enc, img_proc, img_input
                flush_vram()
                print(f"  Freed CLIP encoder. VRAM: {vram_stats()}")
                if self._slog:
                    self._slog.log_tensor("denoise", "image_embeds", image_embeds)
                    self._slog.vram_event("denoise", "CLIP encoder freed")

            # ══════════════════════════════════════════════════════════
            #  TIMESTEP SCHEDULE  (single scheduler, reference-correct)
            # ══════════════════════════════════════════════════════════
            num_inference_steps = info.num_inference_steps

            # boundary_ratio controls where the high→low split happens
            # boundary_timestep = boundary_ratio × num_train_timesteps
            # (this is how the reference WanImageToVideoPipeline does it)
            boundary_ratio = cfg.get("boundary_ratio", 0.9)
            if hasattr(info, "boundary_ratio") and info.boundary_ratio is not None:
                boundary_ratio = info.boundary_ratio
            boundary_ratio = max(0.1, min(0.95, float(boundary_ratio)))

            # Single scheduler for the entire loop — preserves UniPC solver state
            main_sched = UniPCMultistepScheduler.from_config(
                sched.config, flow_shift=flow_shift,
            )
            main_sched.set_timesteps(num_inference_steps, device="cpu")
            timesteps = main_sched.timesteps
            all_sigmas = main_sched.sigmas

            # Compute boundary in timestep space (same as reference pipeline)
            boundary_timestep = boundary_ratio * main_sched.config.num_train_timesteps

            # Partition timesteps: high-noise (>= boundary) and low-noise (< boundary)
            high_indices = []
            low_indices = []
            for i, t in enumerate(timesteps):
                if float(t) >= boundary_timestep:
                    high_indices.append(i)
                else:
                    low_indices.append(i)

            print(f"  Schedule: {num_inference_steps} steps, flow_shift={flow_shift}, "
                  f"boundary_ratio={boundary_ratio:.3f}, "
                  f"boundary_timestep={boundary_timestep:.0f}")
            print(f"  High-noise steps: {len(high_indices)}, Low-noise steps: {len(low_indices)}")
            print(f"  Timesteps: {timesteps.tolist()}")
            print(f"  Sigmas: {[f'{s:.4f}' for s in all_sigmas.tolist()]}")
            if self._slog:
                self._slog.log("denoise", f"Timestep schedule: {num_inference_steps} steps, "
                               f"flow_shift={flow_shift}")
                self._slog.log("denoise", f"boundary_ratio={boundary_ratio:.3f}, "
                               f"boundary_timestep={boundary_timestep:.0f}")
                self._slog.log("denoise", f"High-noise steps: {len(high_indices)} "
                               f"({[int(timesteps[i]) for i in high_indices]})")
                self._slog.log("denoise", f"Low-noise steps: {len(low_indices)} "
                               f"({[int(timesteps[i]) for i in low_indices]})")
                self._slog.log("denoise", f"Full sigmas: "
                               f"{[f'{s:.4f}' for s in all_sigmas.tolist()]}")

            guidance_scale = info.guidance_scale
            guidance_scale_2 = getattr(info, "guidance_scale_2", None)
            if guidance_scale_2 is None:
                guidance_scale_2 = guidance_scale
            do_cfg = guidance_scale > 1.0

            self._emit(session_id, "denoise", 20,
                       f"Denoise: {num_inference_steps} steps "
                       f"({len(high_indices)} high + {len(low_indices)} low)")
            print(f"  VRAM before denoise loops: {vram_stats()}")
            print(f"  CFG: gs_high={guidance_scale}, gs_low={guidance_scale_2}, do_cfg={do_cfg}")

            # ══════════════════════════════════════════════════════════
            #  PASS 1: High-noise transformer (load → LoRA → offload → run → free)
            #  Runs timesteps >= boundary_timestep
            # ══════════════════════════════════════════════════════════
            if high_indices:
                self._emit(session_id, "denoise", 22,
                           "Pass 1: Loading high-noise transformer…")
                print(f"  RAM before high-noise load: {ram_stats()}")

                transformer_high = load_single_transformer(cfg, "high")
                print(f"  RAM after high-noise load: {ram_stats()}")
                if self._slog:
                    self._slog.log("denoise", "High-noise transformer loaded")
                    self._slog.vram_event("denoise", "high-noise transformer loaded")

                # Temporary pipe for LoRA loading API
                pipe_pass1 = WanImageToVideoPipeline(
                    tokenizer=tokenizer,
                    text_encoder=None,
                    transformer=transformer_high,
                    vae=None,
                    scheduler=sched,
                    image_processor=None,
                )

                # LoRAs for high-noise ONLY
                self._emit(session_id, "denoise", 24, "Loading high-noise LoRAs…")
                apply_loras_to_transformer(pipe_pass1, cfg, "transformer",
                                           info.lora_scales,
                                           distill_lora_mode=distill_lora_mode)
                print(f"  VRAM after high-noise LoRAs: {vram_stats()}")
                if self._slog:
                    self._slog.vram_event("denoise", "high-noise LoRAs applied")

                # Group offloading
                if use_offload:
                    num_blocks = cfg.get("num_blocks_per_group", 1)
                    print(f"  Setting up {offload_type} offloading for high-noise "
                          f"(num_blocks_per_group={num_blocks})…")
                    offload_kwargs = dict(
                        offload_type=offload_type,
                        offload_device=torch.device("cpu"),
                        onload_device=torch.device("cuda"),
                        use_stream=False,
                    )
                    if offload_type == "block_level":
                        offload_kwargs["num_blocks_per_group"] = num_blocks
                    apply_group_offloading(pipe_pass1.transformer, **offload_kwargs)
                    print(f"  VRAM after offload setup: {vram_stats()}")
                    if self._slog:
                        self._slog.log("denoise", f"High-noise offloading: "
                                       f"{offload_type}, num_blocks={num_blocks}")
                        self._slog.vram_event("denoise", "high-noise offload setup")

                current_model = pipe_pass1.transformer
                print(f"  Pass 1: running {len(high_indices)} high-noise steps "
                      f"(timesteps: {[int(timesteps[i]) for i in high_indices]})")

                # ── Denoise loop (Pass 1: high-noise) ──
                with patched_sdpa(use_chunked_sdpa):
                    for step_idx in high_indices:
                        if self._cancel_flag:
                            raise InterruptedError("Generation cancelled by user")

                        t = timesteps[step_idx]
                        i_global = step_idx  # absolute position in schedule
                        pct = 25 + int(50 * (i_global + 1) / num_inference_steps)
                        gpu_gb = vram_stats().get("allocated_gb", 0)
                        self._emit(session_id, "denoise", pct,
                                   f"Pass 1 step {step_idx+1}/{num_inference_steps} "
                                   f"(t={t:.0f}, GPU: {gpu_gb:.1f} GB)")

                        latent_model_input = torch.cat([latents, condition], dim=1).to(dtype)
                        timestep = t.expand(latents.shape[0])

                        with torch.no_grad():
                            with current_model.cache_context("cond"):
                                noise_pred = current_model(
                                    hidden_states=latent_model_input,
                                    timestep=timestep,
                                    encoder_hidden_states=prompt_embeds,
                                    encoder_hidden_states_image=image_embeds,
                                    return_dict=False,
                                )[0]

                            if do_cfg:
                                with current_model.cache_context("uncond"):
                                    noise_uncond = current_model(
                                        hidden_states=latent_model_input,
                                        timestep=timestep,
                                        encoder_hidden_states=negative_prompt_embeds,
                                        encoder_hidden_states_image=image_embeds,
                                        return_dict=False,
                                    )[0]
                                noise_pred = noise_uncond + guidance_scale * (
                                    noise_pred - noise_uncond)

                        latents = main_sched.step(noise_pred, t, latents,
                                                  return_dict=False)[0]
                        if self._slog:
                            self._slog.log_denoise_step("denoise", 1, step_idx,
                                                        num_inference_steps, float(t))
                            self._slog.vram_denoise_sample("denoise", 1, step_idx,
                                                           num_inference_steps, float(t))

                print(f"  Pass 1 complete. VRAM: {vram_stats()}")
                if self._slog:
                    self._slog.log("denoise", f"Pass 1 complete ({len(high_indices)} steps)")
                    self._slog.vram_event("denoise", "pass 1 complete")

                # ── Free high-noise transformer completely ──
                remove_offloading(pipe_pass1.transformer)
                pipe_pass1.transformer.to("meta")
                pipe_pass1.transformer = None
                del pipe_pass1, current_model, transformer_high
                gc.collect()
                flush_vram()
                print(f"  Freed high-noise. VRAM: {vram_stats()}  RAM: {ram_stats()}")
                if self._slog:
                    self._slog.vram_event("denoise", "high-noise transformer freed")

            # ══════════════════════════════════════════════════════════
            #  PASS 2: Low-noise transformer (load → LoRA → offload → run → free)
            #  Runs timesteps < boundary_timestep
            #  The scheduler object (main_sched) carries over state from Pass 1
            # ══════════════════════════════════════════════════════════
            if low_indices:
                self._emit(session_id, "denoise", 52,
                           "Pass 2: Loading low-noise transformer…")
                print(f"  RAM before low-noise load: {ram_stats()}")

                transformer_low = load_single_transformer(cfg, "low")
                print(f"  RAM after low-noise load: {ram_stats()}")
                if self._slog:
                    self._slog.log("denoise", "Low-noise transformer loaded")
                    self._slog.vram_event("denoise", "low-noise transformer loaded")

                pipe_pass2 = WanImageToVideoPipeline(
                    tokenizer=tokenizer,
                    text_encoder=None,
                    transformer=transformer_low,
                    vae=None,
                    scheduler=sched,
                    image_processor=None,
                )

                # LoRAs for low-noise ONLY
                self._emit(session_id, "denoise", 54, "Loading low-noise LoRAs…")
                apply_loras_to_transformer(pipe_pass2, cfg, "transformer_2",
                                           info.lora_scales,
                                           load_as_primary=True,
                                           distill_lora_mode=distill_lora_mode)
                print(f"  VRAM after low-noise LoRAs: {vram_stats()}")
                if self._slog:
                    self._slog.vram_event("denoise", "low-noise LoRAs applied")

                # Group offloading
                if use_offload:
                    num_blocks = cfg.get("num_blocks_per_group", 1)
                    print(f"  Setting up {offload_type} offloading for low-noise "
                          f"(num_blocks_per_group={num_blocks})…")
                    offload_kwargs = dict(
                        offload_type=offload_type,
                        offload_device=torch.device("cpu"),
                        onload_device=torch.device("cuda"),
                        use_stream=False,
                    )
                    if offload_type == "block_level":
                        offload_kwargs["num_blocks_per_group"] = num_blocks
                    apply_group_offloading(pipe_pass2.transformer, **offload_kwargs)
                    print(f"  VRAM after offload setup: {vram_stats()}")
                    if self._slog:
                        self._slog.log("denoise", f"Low-noise offloading: "
                                       f"{offload_type}, num_blocks={num_blocks}")
                        self._slog.vram_event("denoise", "low-noise offload setup")

                do_cfg_2 = guidance_scale_2 > 1.0
                current_model = pipe_pass2.transformer
                print(f"  Pass 2: running {len(low_indices)} low-noise steps "
                      f"(timesteps: {[int(timesteps[i]) for i in low_indices]})")

                # ── Denoise loop (Pass 2: low-noise) ──
                with patched_sdpa(use_chunked_sdpa):
                    for step_idx in low_indices:
                        if self._cancel_flag:
                            raise InterruptedError("Generation cancelled by user")

                        t = timesteps[step_idx]
                        pct = 25 + int(50 * (step_idx + 1) / num_inference_steps)
                        gpu_gb = vram_stats().get("allocated_gb", 0)
                        self._emit(session_id, "denoise", pct,
                                   f"Pass 2 step {step_idx+1}/{num_inference_steps} "
                                   f"(t={t:.0f}, GPU: {gpu_gb:.1f} GB)")

                        latent_model_input = torch.cat([latents, condition], dim=1).to(dtype)
                        timestep = t.expand(latents.shape[0])

                        with torch.no_grad():
                            with current_model.cache_context("cond"):
                                noise_pred = current_model(
                                    hidden_states=latent_model_input,
                                    timestep=timestep,
                                    encoder_hidden_states=prompt_embeds,
                                    encoder_hidden_states_image=image_embeds,
                                    return_dict=False,
                                )[0]

                            if do_cfg_2:
                                with current_model.cache_context("uncond"):
                                    noise_uncond = current_model(
                                        hidden_states=latent_model_input,
                                        timestep=timestep,
                                        encoder_hidden_states=negative_prompt_embeds,
                                        encoder_hidden_states_image=image_embeds,
                                        return_dict=False,
                                    )[0]
                                noise_pred = noise_uncond + guidance_scale_2 * (
                                    noise_pred - noise_uncond)

                        # Use the SAME scheduler to preserve solver state
                        latents = main_sched.step(noise_pred, t, latents,
                                                 return_dict=False)[0]
                        if self._slog:
                            self._slog.log_denoise_step("denoise", 2, step_idx,
                                                        num_inference_steps, float(t))
                            self._slog.vram_denoise_sample("denoise", 2, step_idx,
                                                           num_inference_steps, float(t))

                print(f"  Pass 2 complete. VRAM: {vram_stats()}")
                if self._slog:
                    self._slog.log("denoise", f"Pass 2 complete ({len(low_indices)} steps)")
                    self._slog.vram_event("denoise", "pass 2 complete")

                # Free low-noise transformer
                remove_offloading(pipe_pass2.transformer)
                pipe_pass2.transformer.to("meta")
                pipe_pass2.transformer = None
                del pipe_pass2, current_model, transformer_low
                gc.collect()
                flush_vram()
            else:
                print(f"  No low-noise steps (all timesteps >= boundary)")
                if self._slog:
                    self._slog.log("denoise", "No low-noise steps needed")

            # ── Save result ──
            self.sm.save_checkpoint(session_id, "latents", latents.cpu())
            if self._slog:
                self._slog.log_checkpoint_size("denoise",
                    self.sm.session_dir(session_id) / "latents.pt")

            # Free everything
            del latents, condition, prompt_embeds, negative_prompt_embeds
            if image_embeds is not None:
                del image_embeds
            gc.collect()
            flush_vram()

            elapsed = time.time() - t0
            self.sm.update_step(session_id, "denoise", StepStatus.DONE, elapsed)
            if self._slog:
                self._slog.vram_event("denoise", "all freed")
                self._slog.step_end("denoise", "done", elapsed)
            self._emit(session_id, "denoise", 100,
                       f"Denoising done in {elapsed:.1f}s "
                       f"({len(high_indices)} high + {len(low_indices)} low "
                       f"= {num_inference_steps} steps)")
            print(f"  === DENOISE DONE === VRAM: {vram_stats()}  RAM: {ram_stats()}",
                  flush=True)
            return True

        except InterruptedError:
            elapsed = time.time() - t0
            self.sm.update_step(session_id, "denoise", StepStatus.FAILED,
                                elapsed, "Cancelled by user")
            if self._slog:
                self._slog.log("denoise", "CANCELLED by user")
                self._slog.step_end("denoise", "CANCELLED", elapsed, "Cancelled by user")
            self._emit(session_id, "denoise", -1, "Cancelled")
            for v in list(locals().values()):
                if isinstance(v, torch.nn.Module):
                    try:
                        v.to("meta")
                    except Exception:
                        pass
            gc.collect()
            flush_vram()
            return False

        except Exception as e:
            elapsed = time.time() - t0
            self.sm.update_step(session_id, "denoise", StepStatus.FAILED,
                                elapsed, str(e))
            if self._slog:
                self._slog.log("denoise", f"EXCEPTION: {e}")
                self._slog.log("denoise", traceback.format_exc())
                self._slog.step_end("denoise", "FAILED", elapsed, str(e))
            self._emit(session_id, "denoise", -1, f"Denoise failed: {e}")
            traceback.print_exc()
            for v in list(locals().values()):
                if isinstance(v, torch.nn.Module):
                    try:
                        v.to("meta")
                    except Exception:
                        pass
            gc.collect()
            flush_vram()
            print(f"  === DENOISE CLEANUP === VRAM: {vram_stats()}", flush=True)
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
        if self._slog:
            self._slog.step_start("vae_decode")
        t0 = time.time()

        try:
            cfg = self.cfg
            force_vae_cpu = cfg.get("force_vae_cpu", False)
            vae_device = "cpu" if force_vae_cpu else "cuda"
            vae = load_vae(cfg)

            if cfg.get("vae_tiling", True):
                vae.enable_tiling()
            if cfg.get("vae_slicing", True):
                vae.enable_slicing()

            vae.to(vae_device)
            if self._slog:
                self._slog.vram_event("vae_decode", f"VAE loaded on {vae_device}")

            self._emit(session_id, "vae_decode", 20, "Loading latents…")
            latents = self.sm.load_checkpoint(session_id, "latents")
            latents = latents.to(vae_device, dtype=torch.float32)
            if self._slog:
                self._slog.log_tensor("vae_decode", "latents", latents)
                self._slog.vram_event("vae_decode", "latents loaded to GPU")

            self._emit(session_id, "vae_decode", 30, "Decoding latents → frames…")

            latents_mean = getattr(vae.config, "latents_mean", None)
            latents_std = getattr(vae.config, "latents_std", None)

            if latents_mean is not None and latents_std is not None:
                latent_mean = torch.tensor(latents_mean).view(1, -1, 1, 1, 1).to(
                    latents.device, latents.dtype)
                latent_std = torch.tensor(latents_std).view(1, -1, 1, 1, 1).to(
                    latents.device, latents.dtype)
                latents = latents * latent_std + latent_mean
            else:
                scaling = getattr(vae.config, "scaling_factor", 1.0)
                if scaling != 1.0:
                    latents = latents / scaling

            with torch.no_grad():
                try:
                    frames_tensor = vae.decode(latents).sample
                except torch.OutOfMemoryError:
                    if vae_device == "cuda":
                        print("  ⚠ OOM during GPU VAE decode; retrying on CPU…",
                              flush=True)
                        flush_vram()
                        vae.to("cpu")
                        latents = latents.to("cpu")
                        frames_tensor = vae.decode(latents).sample
                    else:
                        raise

            frames_tensor = frames_tensor.clamp(-1, 1)
            frames_tensor = ((frames_tensor + 1) / 2 * 255).to(torch.uint8)
            frames_tensor = frames_tensor[0].permute(1, 2, 3, 0).cpu()

            frames_np = frames_tensor.numpy()
            self.sm.save_checkpoint(session_id, "frames",
                                    torch.from_numpy(frames_np))
            if self._slog:
                self._slog.log("vae_decode", f"Decoded {frames_np.shape[0]} frames, shape={frames_np.shape}")
                self._slog.log_checkpoint_size("vae_decode",
                    self.sm.session_dir(session_id) / "frames.pt")

            del vae, latents, frames_tensor
            flush_vram()
            if self._slog:
                self._slog.vram_event("vae_decode", "VAE freed")

            self._emit(session_id, "vae_decode", 90, "Saving preview…")
            self._export_preview(session_id, frames_np, info.fps)

            elapsed = time.time() - t0
            self.sm.update_step(session_id, "vae_decode", StepStatus.DONE, elapsed)
            if self._slog:
                self._slog.step_end("vae_decode", "done", elapsed)
            self._emit(session_id, "vae_decode", 100,
                       f"VAE decode done in {elapsed:.1f}s")
            return True

        except Exception as e:
            elapsed = time.time() - t0
            self.sm.update_step(session_id, "vae_decode", StepStatus.FAILED,
                                elapsed, str(e))
            if self._slog:
                self._slog.log("vae_decode", f"EXCEPTION: {e}")
                self._slog.log("vae_decode", traceback.format_exc())
                self._slog.step_end("vae_decode", "FAILED", elapsed, str(e))
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
        info = self.sm.get_session(session_id)
        if info and "preview.mp4" not in info.checkpoints:
            info.checkpoints.append("preview.mp4")
            self.sm._save_meta(info)

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
        if self._slog:
            self._slog.step_start("export")
        t0 = time.time()

        try:
            import imageio

            frames_tensor = self.sm.load_checkpoint(session_id, "frames")
            frames_np = frames_tensor.numpy()
            if self._slog:
                self._slog.log("export", f"Loaded {frames_np.shape[0]} frames, shape={frames_np.shape}")

            sdir = self.sm.session_dir(session_id)
            video_path = sdir / "output.mp4"

            self._emit(session_id, "export", 30, "Encoding video…")

            writer = imageio.get_writer(
                str(video_path), fps=info.fps, codec="libx264",
                quality=8, pixelformat="yuv420p",
            )
            for frame in frames_np:
                writer.append_data(frame)
            writer.close()

            info = self.sm.get_session(session_id)
            if info and "output.mp4" not in info.checkpoints:
                info.checkpoints.append("output.mp4")
                self.sm._save_meta(info)

            import shutil
            output_dir = Path("output")
            output_dir.mkdir(exist_ok=True)
            final_name = f"video_{info.session_id}.mp4"
            final_path = output_dir / final_name
            shutil.copy2(video_path, final_path)

            elapsed = time.time() - t0
            self.sm.update_step(session_id, "export", StepStatus.DONE, elapsed)
            if self._slog:
                self._slog.log("export", f"Exported {video_path.name} + copied to {final_path}")
                try:
                    sz = os.path.getsize(video_path)
                    self._slog.log("export", f"File size: {sz/1024**2:.1f} MB")
                except OSError:
                    pass
                self._slog.step_end("export", "done", elapsed)
            self._emit(session_id, "export", 100,
                       f"Video exported in {elapsed:.1f}s")
            return True

        except Exception as e:
            elapsed = time.time() - t0
            self.sm.update_step(session_id, "export", StepStatus.FAILED,
                                elapsed, str(e))
            if self._slog:
                self._slog.log("export", f"EXCEPTION: {e}")
                self._slog.step_end("export", "FAILED", elapsed, str(e))
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
            if self._slog:
                self._slog.log("upscale", "Skipped (enable_upscale=False)")
            return True

        video_path = self.sm.get_file_path(session_id, "output.mp4")
        if not video_path:
            self._emit(session_id, "upscale", -1,
                       "Missing video file — run export first")
            return False

        self.sm.update_step(session_id, "upscale", StepStatus.RUNNING)
        self._emit(session_id, "upscale", 0, "Starting upscale…")
        if self._slog:
            self._slog.step_start("upscale")
        t0 = time.time()

        try:
            from upscale import upscale_video

            sdir = self.sm.session_dir(session_id)
            up_path = str(sdir / "output_upscaled.mp4")

            upscale_cfg = dict(self.cfg)
            upscale_cfg["fps"] = info.fps
            upscale_cfg["output_fps"] = getattr(info, "output_fps", info.fps)
            upscale_cfg["target_duration"] = getattr(info, "target_duration", 0)
            # Use session-specific upscale model if set
            if hasattr(info, "upscale_model") and info.upscale_model:
                upscale_cfg["upscale_model"] = info.upscale_model
            if self._slog:
                self._slog.log("upscale", f"Input: {video_path}, Output: {up_path}")
                self._slog.log("upscale", f"fps={info.fps}, output_fps={upscale_cfg['output_fps']}, target_duration={upscale_cfg['target_duration']}")
                self._slog.log("upscale", f"model={upscale_cfg.get('upscale_model', 'default')}")

            def upscale_progress(pct, msg):
                self._emit(session_id, "upscale", pct, msg)
                if self._slog:
                    self._slog.log("upscale", msg)

            upscale_video(str(video_path), up_path, upscale_cfg,
                          progress_fn=upscale_progress)

            info_up = self.sm.get_session(session_id)
            if info_up and "output_upscaled.mp4" not in info_up.checkpoints:
                info_up.checkpoints.append("output_upscaled.mp4")
                self.sm._save_meta(info_up)

            elapsed = time.time() - t0
            self.sm.update_step(session_id, "upscale", StepStatus.DONE, elapsed)
            if self._slog:
                try:
                    sz = os.path.getsize(up_path)
                    self._slog.log("upscale", f"Output size: {sz/1024**2:.1f} MB")
                except OSError:
                    pass
                self._slog.vram_event("upscale", "upscale complete")
                self._slog.step_end("upscale", "done", elapsed)
            self._emit(session_id, "upscale", 100,
                       f"Upscale done in {elapsed:.1f}s")
            flush_vram()
            return True

        except Exception as e:
            elapsed = time.time() - t0
            self.sm.update_step(session_id, "upscale", StepStatus.FAILED,
                                elapsed, str(e))
            if self._slog:
                self._slog.log("upscale", f"EXCEPTION: {e}")
                self._slog.log("upscale", traceback.format_exc())
                self._slog.step_end("upscale", "FAILED", elapsed, str(e))
            self._emit(session_id, "upscale", -1, f"Upscale failed: {e}")
            traceback.print_exc()
            flush_vram()
            return False

    # ─── Run all steps (or from a given step) ───────────────────────
    def run_from_step(self, session_id: str, start_step: str = "encode") -> bool:
        """Run the pipeline starting from a specific step."""
        self._cancel_flag = False
        info = self.sm.get_session(session_id)
        if not info:
            return False

        # Init session logger if not already done (resume case)
        if self._slog is None or start_step != "encode":
            sdir = self.sm.session_dir(session_id)
            is_resume = start_step != "encode"
            self._slog = SessionLogger(sdir, resume=is_resume)
            self._slog.header(info, self.cfg)

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
                if self._slog:
                    energy = self._slog.stop_power_monitor()
                    info = self.sm.get_session(session_id)
                    if info:
                        info.energy_wh = energy.get("total_wh", 0)
                        info.gpu_energy_wh = energy.get("gpu_wh", 0)
                        info.peak_gpu_power_w = energy.get("peak_gpu_w", 0)
                        info.avg_gpu_power_w = energy.get("avg_gpu_w", 0)
                        info.energy_cost_kwh = self.cfg.get("electricity_cost_kwh", 0.12)
                        self.sm._save_meta(info)
                        self._slog.summary(info, time.time() - self._slog._t0)
                    self._slog = None
                return False

            success = step_funcs[step_name](session_id)
            if not success:
                info = self.sm.get_session(session_id)
                if info and info.steps.get(step_name) == StepStatus.FAILED:
                    if self._slog:
                        energy = self._slog.stop_power_monitor()
                        info.energy_wh = energy.get("total_wh", 0)
                        info.gpu_energy_wh = energy.get("gpu_wh", 0)
                        info.peak_gpu_power_w = energy.get("peak_gpu_w", 0)
                        info.avg_gpu_power_w = energy.get("avg_gpu_w", 0)
                        info.energy_cost_kwh = self.cfg.get("electricity_cost_kwh", 0.12)
                        self.sm._save_meta(info)
                        self._slog.summary(info, time.time() - self._slog._t0)
                        self._slog = None
                    return False

        self.sm.update_status(session_id, "done")
        # Write final summary and save energy data
        if self._slog:
            info = self.sm.get_session(session_id)
            if info:
                energy = self._slog.stop_power_monitor()
                # Persist energy data to session metadata
                info.energy_wh = energy.get("total_wh", 0)
                info.gpu_energy_wh = energy.get("gpu_wh", 0)
                info.peak_gpu_power_w = energy.get("peak_gpu_w", 0)
                info.avg_gpu_power_w = energy.get("avg_gpu_w", 0)
                info.energy_cost_kwh = self.cfg.get("electricity_cost_kwh", 0.12)
                self.sm._save_meta(info)
                self._slog.summary(info, time.time() - self._slog._t0)
            self._slog = None
        return True

    def get_resume_step(self, session_id: str) -> str:
        """Determine which step to resume from based on available checkpoints."""
        info = self.sm.get_session(session_id)
        if not info:
            return "encode"

        if self.sm.has_checkpoint(session_id, "frames"):
            return "export"
        if self.sm.has_checkpoint(session_id, "latents"):
            return "vae_decode"
        if self.sm.has_checkpoint(session_id, "embeddings"):
            return "denoise"
        return "encode"
