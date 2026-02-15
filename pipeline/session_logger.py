"""
pipeline/session_logger.py — Per-session file logger
=====================================================

Writes two human-readable log files per generation session:
  generation.log — narrative diary of every pipeline event
  vram.log       — timestamped VRAM/RAM allocation timeline

Dependencies: vram_utils (vram_stats, ram_stats)
Used by: engine.PipelineEngine (each step logs through this)
"""

import os
import time
from pathlib import Path

import torch

from pipeline.vram_utils import vram_stats, ram_stats
from session_manager import STEP_ORDER


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
            distill = getattr(info, "distill_lora_mode", False)
            f.write(f"  Mode:             {'⚡ DISTILL (fast 4-step)' if distill else '🎨 QUALITY (full CFG)'}\n")
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
                role = lora.get("role", "quality")
                role_badge = "DISTILL" if role == "distill" else "QUALITY"
                skipped = (not distill and role == "distill")
                status = "SKIP" if skipped else "LOAD"
                path = lora.get("path", "")
                file_size = "?"
                try:
                    sz = os.path.getsize(path)
                    file_size = f"{sz / 1024**2:.1f} MB"
                except OSError:
                    pass
                f.write(f"  [{i}] {lora['adapter_name']:40s}  {badge:4s}  "
                        f"{role_badge:7s}  {status:4s}  "
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
