"""
pipeline/session_logger.py — Per-session file logger
=====================================================

Writes two human-readable log files per generation session:
  generation.log — narrative diary of every pipeline event
  vram.log       — timestamped VRAM/RAM allocation timeline

Dependencies: vram_utils (vram_stats, ram_stats)
Used by: engine.PipelineEngine (each step logs through this)
"""

import json
import os
import subprocess
import threading
import time
from pathlib import Path

import torch

from pipeline.vram_utils import vram_stats, ram_stats
from session_manager import STEP_ORDER


# ── Constants for energy estimation ─────────────────────────────────
# PSU efficiency curve (80 Plus Gold approximate):
#   At 20% load ~87%, 50% load ~90%, 100% load ~87%
#   We use a flat 88% as a reasonable average for mixed workloads.
PSU_EFFICIENCY = 0.88
# Fixed system overhead: CPU (~65W avg under mixed load) + RAM/fans/MB (~35W)
SYSTEM_OVERHEAD_W = 100.0


class PowerMonitor:
    """Background thread that samples GPU power draw via rocm-smi.

    Accumulates total energy consumed (in Wh) over the session.
    Also tracks per-step energy and peak power.
    """

    def __init__(self, sample_interval: float = 2.0):
        self._interval = sample_interval
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()

        # Accumulated energy
        self._gpu_energy_j: float = 0.0       # Joules from GPU
        self._total_energy_j: float = 0.0     # Joules total (GPU + system + PSU loss)
        self._samples: int = 0
        self._peak_gpu_w: float = 0.0
        self._last_gpu_w: float = 0.0
        self._last_sample_time: float = 0.0

        # Per-step tracking
        self._current_step: str = ""
        self._step_energy: dict[str, float] = {}  # step -> GPU joules

    def start(self):
        """Start the background sampling thread."""
        self._stop.clear()
        self._last_sample_time = time.time()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def stop(self) -> dict:
        """Stop sampling and return energy summary."""
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None
        return self.get_summary()

    def set_step(self, step: str):
        """Set current pipeline step for per-step tracking."""
        with self._lock:
            self._current_step = step

    def _read_gpu_power(self) -> float | None:
        """Read GPU power from rocm-smi. Returns watts or None."""
        try:
            result = subprocess.run(
                ["rocm-smi", "--showpower", "--json"],
                capture_output=True, text=True, timeout=3,
            )
            if result.returncode == 0:
                data = json.loads(result.stdout)
                card = data.get("card0", {})
                raw = card.get("Average Graphics Package Power (W)")
                if raw is not None:
                    return float(raw)
        except Exception:
            pass
        return None

    def _sample_loop(self):
        """Sampling loop — runs in background thread."""
        while not self._stop.is_set():
            now = time.time()
            dt = now - self._last_sample_time
            self._last_sample_time = now

            gpu_w = self._read_gpu_power()
            if gpu_w is not None:
                with self._lock:
                    # Energy = Power × Time (joules)
                    gpu_j = gpu_w * dt
                    # Total wall power = (GPU + system overhead) / PSU efficiency
                    wall_w = (gpu_w + SYSTEM_OVERHEAD_W) / PSU_EFFICIENCY
                    total_j = wall_w * dt

                    self._gpu_energy_j += gpu_j
                    self._total_energy_j += total_j
                    self._samples += 1
                    self._last_gpu_w = gpu_w
                    if gpu_w > self._peak_gpu_w:
                        self._peak_gpu_w = gpu_w

                    # Per-step accumulation
                    if self._current_step:
                        self._step_energy[self._current_step] = \
                            self._step_energy.get(self._current_step, 0) + gpu_j

            self._stop.wait(self._interval)

    def get_summary(self) -> dict:
        """Return accumulated energy data."""
        with self._lock:
            gpu_wh = self._gpu_energy_j / 3600
            total_wh = self._total_energy_j / 3600
            avg_gpu_w = (self._gpu_energy_j / (self._samples * self._interval)
                         if self._samples > 0 else 0)
            step_wh = {k: v / 3600 for k, v in self._step_energy.items()}
            return {
                "gpu_wh": round(gpu_wh, 3),
                "total_wh": round(total_wh, 3),
                "gpu_kwh": round(gpu_wh / 1000, 6),
                "total_kwh": round(total_wh / 1000, 6),
                "peak_gpu_w": round(self._peak_gpu_w, 1),
                "avg_gpu_w": round(avg_gpu_w, 1),
                "samples": self._samples,
                "step_gpu_wh": {k: round(v, 3) for k, v in step_wh.items()},
                "system_overhead_w": SYSTEM_OVERHEAD_W,
                "psu_efficiency": PSU_EFFICIENCY,
            }


class SessionLogger:
    """Writes two human-readable log files per session:

    generation.log — narrative diary of every pipeline event
    vram.log       — timestamped VRAM/RAM allocation timeline
    """

    def __init__(self, session_dir: Path, resume: bool = False):
        self.session_dir = Path(session_dir)
        self.gen_path = self.session_dir / "generation.log"
        self.vram_path = self.session_dir / "vram.log"
        self._t0: float = time.time()
        self._peak_alloc: float = 0.0
        self._peak_reserved: float = 0.0
        self._last_alloc: float = 0.0
        self._last_reserved: float = 0.0
        self._step_t0: float = 0.0
        self._resume: bool = resume

        # Power monitoring
        self._power_monitor = PowerMonitor(sample_interval=2.0)
        self._power_monitor.start()

        if resume and self.vram_path.exists():
            # Append a resume separator instead of overwriting
            with open(self.vram_path, "a") as f:
                f.write("\n" + "═" * 190 + "\n")
                f.write(f"  RESUMED at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("═" * 190 + "\n")
                f.write(f"{'Elapsed':>8s}  {'Step':<14s}  {'Event':<45s}  "
                        f"{'Alloc GB':>9s}  {'Rsrv GB':>9s}  {'Free GB':>9s}  "
                        f"{'Δ Alloc':>9s}  {'Δ Rsrv':>9s}  "
                        f"{'Peak A':>8s}  {'Peak R':>8s}  "
                        f"{'RAM Used':>9s}  {'RAM Avail':>10s}\n")
                f.write("─" * 190 + "\n")
        else:
            # Fresh start — write header
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
        """Write the generation parameters header.

        On resume, appends a resume banner instead of overwriting.
        """
        gpu_name = "N/A"
        gpu_total = 0.0
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_total = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2)

        loras = cfg.get("loras", [])
        lora_overrides = getattr(info, "lora_scales", []) or []

        if self._resume and self.gen_path.exists():
            # Append a resume section instead of overwriting
            with open(self.gen_path, "a") as f:
                f.write("\n\n" + "═" * 80 + "\n")
                f.write(f"  RESUMED: {info.session_id}\n")
                f.write(f"  Resumed at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("═" * 80 + "\n\n")

                f.write("═" * 80 + "\n")
                f.write("  PIPELINE EXECUTION (continued)\n")
                f.write("═" * 80 + "\n\n")
            return

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
        self._power_monitor.set_step(step)
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

    def stop_power_monitor(self) -> dict:
        """Stop the power monitor and return energy summary."""
        return self._power_monitor.stop()

    def summary(self, info, total_elapsed: float):
        """Write final summary including energy consumption."""
        # Get energy data (monitor may already be stopped by engine)
        energy = self._power_monitor.get_summary()

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
            step_wh = energy.get("step_gpu_wh", {}).get(step_name, 0)
            self._gwrite(f"    {step_name:<12s}  {str(status):<10s}  "
                         f"{t:7.1f}s  ({pct:4.1f}%)  "
                         f"GPU: {step_wh:.2f} Wh\n")

        self._gwrite(f"\n  VRAM peaks:\n")
        self._gwrite(f"    Allocated: {self._peak_alloc:.2f} GB\n")
        self._gwrite(f"    Reserved:  {self._peak_reserved:.2f} GB\n")

        self._gwrite(f"\n  ⚡ Energy consumption:\n")
        self._gwrite(f"    GPU energy:       {energy['gpu_wh']:.2f} Wh\n")
        self._gwrite(f"    Peak GPU power:   {energy['peak_gpu_w']:.0f} W\n")
        self._gwrite(f"    Avg GPU power:    {energy['avg_gpu_w']:.0f} W\n")
        self._gwrite(f"    System overhead:  {energy['system_overhead_w']:.0f} W (constant)\n")
        self._gwrite(f"    PSU efficiency:   {energy['psu_efficiency']*100:.0f}%\n")
        self._gwrite(f"    Est. wall energy: {energy['total_wh']:.2f} Wh "
                     f"({energy['total_kwh']:.4f} kWh)\n")
        cost_kwh = getattr(info, "energy_cost_kwh", 0)
        if cost_kwh > 0:
            cost = energy["total_kwh"] * cost_kwh
            self._gwrite(f"    Est. elec. cost:  ${cost:.4f} "
                         f"(@ ${cost_kwh:.2f}/kWh)\n")
        self._gwrite(f"    Samples taken:    {energy['samples']}\n")

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
