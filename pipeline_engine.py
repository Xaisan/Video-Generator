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

Memory strategy (denoise step):
  - Transformers are loaded ONE AT A TIME (~10 GB RAM each, not ~20 GB).
  - LoRAs are loaded per-pass (no duplicate adapter warnings).
  - Embeddings are moved to GPU once, not implicitly per step.
  - block_level offloading is used by default (fewer PCIe transfers,
    higher GPU utilisation than leaf_level).
  - VAE is freed before any transformer loads.
"""

import gc
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import yaml
from PIL import Image

from session_manager import SessionManager, StepStatus, STEP_ORDER


# ─── VRAM / RAM helpers ────────────────────────────────────────────
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


# ─── AMD env vars ──────────────────────────────────────────────────
def apply_amd_env(cfg: dict) -> None:
    amd = cfg.get("amd", {})
    os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION",
                          amd.get("hsa_override_gfx_version", "11.0.0"))
    os.environ.setdefault("HIP_VISIBLE_DEVICES", "0")
    alloc_conf = amd.get("pytorch_hip_alloc_conf", "expandable_segments:True")
    os.environ["PYTORCH_ALLOC_CONF"] = alloc_conf
    os.environ["PYTORCH_HIP_ALLOC_CONF"] = alloc_conf
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


# ─── Component loaders (load on demand, free after use) ────────────

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
    """Load VAE in float32. Uses local safetensors if available."""
    from diffusers import AutoencoderKLWan
    model_id = cfg["model_id"]

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


# ─── LoRA loading (per-transformer) ───────────────────────────────
def apply_loras_to_transformer(pipe, cfg: dict, target: str,
                               lora_overrides: list | None = None,
                               load_as_primary: bool = False):
    """Load LoRAs for a SINGLE transformer target.

    target: "transformer" or "transformer_2" — used to FILTER config entries.
    load_as_primary: if True, load into pipe.transformer even when target is
                     "transformer_2".  Needed when the low-noise transformer
                     is mounted as pipe.transformer (because load_lora_weights
                     requires a non-None .transformer to read its config).

    This avoids loading all LoRAs at once (halves RAM pressure) and
    prevents the duplicate peft_config warning.
    """
    import copy
    lora_cfgs = copy.deepcopy(cfg.get("loras", []))
    if not lora_cfgs:
        return

    # Apply scale overrides from session
    if lora_overrides:
        if lora_overrides and isinstance(lora_overrides[0], (int, float)):
            for i, scale_val in enumerate(lora_overrides):
                if i < len(lora_cfgs):
                    lora_cfgs[i]["scale"] = float(scale_val)
        else:
            override_map = {o["adapter_name"]: o["scale"] for o in lora_overrides}
            for lora in lora_cfgs:
                if lora["adapter_name"] in override_map:
                    lora["scale"] = override_map[lora["adapter_name"]]

    # Filter to only LoRAs for this target
    target_loras = [l for l in lora_cfgs if l.get("target", "transformer") == target]
    if not target_loras:
        return

    adapters = []
    scales = {}

    for lora in target_loras:
        path = lora["path"]
        name = lora["adapter_name"]
        scale = lora.get("scale", 1.0)

        if not os.path.isfile(path):
            print(f"  ⚠ LoRA not found, skipping: {path}")
            continue

        print(f"  📦 Loading LoRA: {name} → {target} (scale={scale})")
        try:
            if target == "transformer_2" and not load_as_primary:
                pipe.load_lora_weights(path, adapter_name=name,
                                       load_into_transformer_2=True)
            else:
                pipe.load_lora_weights(path, adapter_name=name)
            adapters.append(name)
            scales[name] = scale
        except Exception as e:
            print(f"  ❌ LoRA {name}: {e}")

    if adapters:
        if load_as_primary:
            model = pipe.transformer
        else:
            model = pipe.transformer if target == "transformer" else pipe.transformer_2
        if model is not None:
            model.set_adapters(adapters, weights=[scales[a] for a in adapters])
            print(f"  ✅ Active adapters on {target}: {adapters}")


def apply_loras(pipe, cfg: dict, lora_overrides: list | None = None):
    """Load LoRAs into the pipeline (legacy wrapper — both targets)."""
    apply_loras_to_transformer(pipe, cfg, "transformer", lora_overrides)
    if hasattr(pipe, "transformer_2") and pipe.transformer_2 is not None:
        apply_loras_to_transformer(pipe, cfg, "transformer_2", lora_overrides)


# ─── Group offloading helpers ──────────────────────────────────────
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


# ═══════════════════════════════════════════════════════════════════
#  Per-session file loggers
# ═══════════════════════════════════════════════════════════════════

class SessionLogger:
    """Writes two human-readable log files per session:

    generation.log — narrative diary of every pipeline event
    vram.log       — timestamped VRAM/RAM allocation timeline
    """

    def __init__(self, session_dir: Path):
        self.session_dir = Path(session_dir)
        self.gen_path = self.session_dir / "generation.log"
        self.vram_path = self.session_dir / "vram.log"
        self._t0: float = time.time()
        self._peak_alloc: float = 0.0
        self._peak_reserved: float = 0.0
        self._last_alloc: float = 0.0
        self._last_reserved: float = 0.0
        self._step_t0: float = 0.0

        # Initialise VRAM log with header
        with open(self.vram_path, "w") as f:
            f.write(f"{'Elapsed':>8s}  {'Step':<14s}  {'Event':<45s}  "
                    f"{'Alloc GB':>9s}  {'Rsrv GB':>9s}  {'Free GB':>9s}  "
                    f"{'Δ Alloc':>9s}  {'Δ Rsrv':>9s}  "
                    f"{'Peak A':>8s}  {'Peak R':>8s}  "
                    f"{'RAM Used':>9s}  {'RAM Avail':>10s}\n")
            f.write("─" * 190 + "\n")

    # ── generation.log helpers ──────────────────────────────────

    def _gwrite(self, text: str):
        with open(self.gen_path, "a") as f:
            f.write(text)

    def header(self, info, cfg: dict):
        """Write the generation parameters header."""
        gpu_name = "N/A"
        gpu_total = 0.0
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_total = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2)

        loras = cfg.get("loras", [])
        lora_overrides = getattr(info, "lora_scales", []) or []

        with open(self.gen_path, "w") as f:
            f.write("═" * 80 + "\n")
            f.write(f"  SESSION: {info.session_id}\n")
            f.write(f"  Started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("═" * 80 + "\n\n")

            f.write("── GPU ──\n")
            f.write(f"  Device:       {gpu_name}\n")
            f.write(f"  Total VRAM:   {gpu_total} GB\n")
            f.write(f"  PyTorch:      {torch.__version__}\n")
            try:
                f.write(f"  CUDA/ROCm:    {torch.version.hip or torch.version.cuda or 'N/A'}\n")
            except Exception:
                pass
            f.write(f"  HSA_GFX:      {os.environ.get('HSA_OVERRIDE_GFX_VERSION', 'N/A')}\n")
            f.write(f"  Alloc conf:   {os.environ.get('PYTORCH_HIP_ALLOC_CONF', 'N/A')}\n\n")

            f.write("── Generation Parameters ──\n")
            f.write(f"  Prompt:           {info.prompt}\n")
            f.write(f"  Negative prompt:  {info.negative_prompt}\n")
            f.write(f"  Resolution:       {info.width}×{info.height} ({info.width*info.height/1e6:.2f} MP)\n")
            f.write(f"  Frames:           {info.num_frames} ({info.duration}s @ {info.fps}fps)\n")
            f.write(f"  Output FPS:       {info.output_fps}\n")
            f.write(f"  Steps (total):    {info.num_inference_steps}\n")
            f.write(f"  CFG High:         {info.guidance_scale}\n")
            f.write(f"  CFG Low:          {info.guidance_scale_2}\n")
            f.write(f"  Flow shift:       {info.flow_shift}\n")
            br = getattr(info, "boundary_ratio", 0.5)
            f.write(f"  Boundary ratio:   {br}\n")
            f.write(f"  Seed:             {info.seed}\n")
            f.write(f"  Upscale:          {info.enable_upscale}\n\n")

            f.write("── Memory Settings ──\n")
            f.write(f"  Offload type:     {cfg.get('offload_type', 'block_level')}\n")
            f.write(f"  Blocks/group:     {cfg.get('num_blocks_per_group', 1)}\n")
            f.write(f"  Group offload:    {cfg.get('enable_group_offload', True)}\n")
            f.write(f"  VAE tiling:       {cfg.get('vae_tiling', True)}\n")
            f.write(f"  VAE slicing:      {cfg.get('vae_slicing', True)}\n")
            f.write(f"  Force VAE CPU:    {cfg.get('force_vae_cpu', False)}\n")
            amd = cfg.get("amd", {})
            f.write(f"  Force FP16:       {amd.get('force_fp16', False)}\n\n")

            f.write("── LoRA Configuration ──\n")
            for i, lora in enumerate(loras):
                scale = lora.get("scale", 1.0)
                if isinstance(lora_overrides, list) and i < len(lora_overrides):
                    if isinstance(lora_overrides[i], (int, float)):
                        scale = lora_overrides[i]
                target = lora.get("target", "transformer")
                badge = "HIGH" if target == "transformer" else "LOW"
                path = lora.get("path", "")
                file_size = "?"
                try:
                    sz = os.path.getsize(path)
                    file_size = f"{sz / 1024**2:.1f} MB"
                except OSError:
                    pass
                f.write(f"  [{i}] {lora['adapter_name']:40s}  {badge:4s}  "
                        f"scale={scale:<6.3f}  {file_size:>10s}  {os.path.basename(path)}\n")
            f.write("\n")

            ram = ram_stats()
            f.write(f"── System RAM at start ──\n")
            f.write(f"  Total:  {ram.get('total_gb', '?')} GB\n")
            f.write(f"  Used:   {ram.get('used_gb', '?')} GB\n")
            f.write(f"  Avail:  {ram.get('available_gb', '?')} GB  ({ram.get('percent', '?')}%)\n\n")

            f.write("═" * 80 + "\n")
            f.write("  PIPELINE EXECUTION\n")
            f.write("═" * 80 + "\n\n")

    def step_start(self, step: str):
        self._step_t0 = time.time()
        elapsed = self._step_t0 - self._t0
        self._gwrite(f"\n┌─ STEP: {step.upper()} ─{'─' * (60 - len(step))}\n")
        self._gwrite(f"│  Started at T+{elapsed:.1f}s  "
                     f"({time.strftime('%H:%M:%S')})\n")
        self.vram_event(step, "step_start")

    def step_end(self, step: str, status: str, elapsed: float, error: str = ""):
        total_elapsed = time.time() - self._t0
        self._gwrite(f"│\n")
        self._gwrite(f"│  Status:   {status}\n")
        self._gwrite(f"│  Duration: {elapsed:.2f}s\n")
        if error:
            self._gwrite(f"│  Error:    {error}\n")
        self._gwrite(f"└─ END {step.upper()} at T+{total_elapsed:.1f}s ─{'─' * (55 - len(step))}\n\n")
        self.vram_event(step, f"step_end ({status})")

    def log(self, step: str, msg: str):
        """Write a timestamped line to generation.log."""
        elapsed = time.time() - self._t0
        self._gwrite(f"│  [{elapsed:7.1f}s] {msg}\n")

    def log_tensor(self, step: str, name: str, tensor):
        """Log a tensor's shape, dtype, and device."""
        if hasattr(tensor, "shape"):
            self.log(step, f"Tensor {name}: shape={list(tensor.shape)}  "
                     f"dtype={tensor.dtype}  device={tensor.device}")

    def log_checkpoint_size(self, step: str, path: str | Path):
        """Log the size of a saved checkpoint file."""
        try:
            sz = os.path.getsize(path)
            if sz > 1024**3:
                self.log(step, f"Saved {os.path.basename(str(path))}: {sz/1024**3:.2f} GB")
            else:
                self.log(step, f"Saved {os.path.basename(str(path))}: {sz/1024**2:.1f} MB")
        except OSError:
            pass

    def log_sigmas(self, step: str, full_sigmas, split_step: int,
                   boundary_sigma: float, boundary_ratio: float):
        """Log the full sigma schedule."""
        total = len(full_sigmas)
        self.log(step, f"Sigma schedule: {total} values, split at step {split_step} "
                 f"(ratio={boundary_ratio:.2f})")
        self.log(step, f"Boundary sigma: {boundary_sigma:.6f}")
        # Log compact sigma list
        sigma_strs = [f"{s:.4f}" for s in full_sigmas]
        self.log(step, f"Full sigmas: [{', '.join(sigma_strs)}]")
        self.log(step, f"Pass 1 ({split_step} steps): "
                 f"[{', '.join(sigma_strs[:split_step])}]")
        self.log(step, f"Pass 2 ({total - split_step} steps): "
                 f"[{', '.join(sigma_strs[split_step:])}]")

    def log_denoise_step(self, step: str, pass_num: int, i: int, total: int,
                         timestep: float):
        """Log a single denoise iteration."""
        v = vram_stats()
        alloc = v.get("allocated_gb", 0)
        self.log(step, f"Pass {pass_num} step {i+1}/{total}  "
                 f"t={timestep:.0f}  VRAM={alloc:.2f} GB")

    def summary(self, info, total_elapsed: float):
        """Write final summary."""
        self._gwrite("\n" + "═" * 80 + "\n")
        self._gwrite("  GENERATION SUMMARY\n")
        self._gwrite("═" * 80 + "\n\n")
        self._gwrite(f"  Final status:     {info.status}\n")
        self._gwrite(f"  Total wall time:  {total_elapsed:.1f}s "
                     f"({total_elapsed/60:.1f} min)\n")
        self._gwrite(f"  Completed at:     {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        self._gwrite("  Per-step breakdown:\n")
        for step_name in STEP_ORDER:
            status = info.steps.get(step_name, "pending")
            t = info.step_times.get(step_name, 0)
            pct = (t / total_elapsed * 100) if total_elapsed > 0 else 0
            self._gwrite(f"    {step_name:<12s}  {str(status):<10s}  "
                         f"{t:7.1f}s  ({pct:4.1f}%)\n")

        self._gwrite(f"\n  VRAM peaks:\n")
        self._gwrite(f"    Allocated: {self._peak_alloc:.2f} GB\n")
        self._gwrite(f"    Reserved:  {self._peak_reserved:.2f} GB\n")

        if info.error_message:
            self._gwrite(f"\n  Error: {info.error_message}\n")

        self._gwrite("\n" + "═" * 80 + "\n")

    # ── vram.log helpers ────────────────────────────────────────

    def vram_event(self, step: str, event: str):
        """Record a VRAM snapshot to vram.log."""
        v = vram_stats()
        r = ram_stats()
        elapsed = time.time() - self._t0
        alloc = v.get("allocated_gb", 0)
        reserved = v.get("reserved_gb", 0)
        total = v.get("total_gb", 0)
        free = round(total - reserved, 2) if total else 0
        d_alloc = round(alloc - self._last_alloc, 3)
        d_reserved = round(reserved - self._last_reserved, 3)

        if alloc > self._peak_alloc:
            self._peak_alloc = alloc
        if reserved > self._peak_reserved:
            self._peak_reserved = reserved

        self._last_alloc = alloc
        self._last_reserved = reserved

        d_alloc_str = f"{'+' if d_alloc >= 0 else ''}{d_alloc:.3f}"
        d_reserved_str = f"{'+' if d_reserved >= 0 else ''}{d_reserved:.3f}"

        ram_used = r.get("used_gb", 0)
        ram_avail = r.get("available_gb", 0)

        with open(self.vram_path, "a") as f:
            f.write(f"{elapsed:7.1f}s  {step:<14s}  {event:<45s}  "
                    f"{alloc:8.3f}   {reserved:8.3f}   {free:8.3f}   "
                    f"{d_alloc_str:>9s}  {d_reserved_str:>9s}  "
                    f"{self._peak_alloc:7.3f}   {self._peak_reserved:7.3f}   "
                    f"{ram_used:8.2f}   {ram_avail:9.2f}\n")

    def vram_denoise_sample(self, step: str, pass_num: int, i: int, total: int,
                            timestep: float):
        """Compact VRAM row for each denoise iteration."""
        self.vram_event(step, f"pass{pass_num} step {i+1}/{total} t={timestep:.0f}")


# ═══════════════════════════════════════════════════════════════════
#  Pipeline Engine
# ═══════════════════════════════════════════════════════════════════

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
            flow_shift = cfg.get("flow_shift", 8.0)

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

            # Load a temporary transformer ONLY to read config values
            # (vae_scale_factor_spatial, z_dim, image_dim).  This loads
            # to CPU/meta and is freed immediately — never touches GPU.
            self._emit(session_id, "denoise", 6, "Reading model config…")
            transformer_cfg_reader = load_single_transformer(cfg, "high")
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                cfg["model_id"], subfolder="tokenizer"
            )

            pipe_vae = WanImageToVideoPipeline(
                tokenizer=tokenizer,
                text_encoder=None,
                transformer=transformer_cfg_reader,
                vae=vae,
                scheduler=sched,
                image_processor=None,
            )

            vae_scale_factor = pipe_vae.vae_scale_factor_spatial
            num_channels_latents = vae.config.z_dim
            image_dim = transformer_cfg_reader.config.image_dim

            # Free config reader immediately — never went to GPU
            pipe_vae.transformer = None
            transformer_cfg_reader.to("meta")
            del transformer_cfg_reader
            gc.collect()
            print(f"  Config read done. VRAM: {vram_stats()}  RAM: {ram_stats()}")
            if self._slog:
                self._slog.log("denoise", f"vae_scale_factor={vae_scale_factor}, z_dim={num_channels_latents}, image_dim={image_dim}")
                self._slog.vram_event("denoise", "config reader freed")

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
            boundary_ratio = cfg.get("boundary_ratio", 0.618)
            if hasattr(info, "boundary_ratio") and info.boundary_ratio is not None:
                boundary_ratio = info.boundary_ratio
            boundary_ratio = max(0.1, min(0.9, float(boundary_ratio)))

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
                                           info.lora_scales)
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
                                           load_as_primary=True)
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
            vae = load_vae(cfg)

            if cfg.get("vae_tiling", True):
                vae.enable_tiling()
            if cfg.get("vae_slicing", True):
                vae.enable_slicing()

            vae.to("cuda")
            if self._slog:
                self._slog.vram_event("vae_decode", "VAE loaded on GPU")

            self._emit(session_id, "vae_decode", 20, "Loading latents…")
            latents = self.sm.load_checkpoint(session_id, "latents")
            latents = latents.to("cuda", dtype=torch.float32)
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
                frames_tensor = vae.decode(latents).sample

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
            if self._slog:
                self._slog.log("upscale", f"Input: {video_path}, Output: {up_path}")
                self._slog.log("upscale", f"fps={info.fps}, output_fps={upscale_cfg['output_fps']}")

            upscale_video(str(video_path), up_path, upscale_cfg)

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
            self._slog = SessionLogger(sdir)
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
                    info = self.sm.get_session(session_id)
                    if info:
                        self._slog.summary(info, time.time() - self._slog._t0)
                    self._slog = None
                return False

            success = step_funcs[step_name](session_id)
            if not success:
                info = self.sm.get_session(session_id)
                if info and info.steps.get(step_name) == StepStatus.FAILED:
                    if self._slog:
                        self._slog.summary(info, time.time() - self._slog._t0)
                        self._slog = None
                    return False

        self.sm.update_status(session_id, "done")
        # Write final summary
        if self._slog:
            info = self.sm.get_session(session_id)
            if info:
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
