#!/usr/bin/env python3
"""
app.py — Web UI & REST API for Wan 2.2 I2V Video Generator
=========================================================

This is the main entry point for the web application.
Run: python app.py [--host 0.0.0.0] [--port 7860] [--debug]

Responsibilities:
  - Flask web server serving the single-page UI
  - REST API endpoints for generation, session management, uploads
  - Server-Sent Events (SSE) for real-time progress streaming
  - Log capture ring buffer (stdout/stderr -> web UI)
  - Background thread management for generation jobs

Dependencies (project-internal):
  -> session_manager.py  (SessionManager, StepStatus, STEP_ORDER)
  -> pipeline_engine.py  (PipelineEngine, vram_stats, flush_vram, load_config)
  -> amd_tune.py         (apply_amd_optimisations, detect_amd_gpu) -- at startup

Key routes:
  GET  /                              -> index page
  POST /api/generate                  -> start new generation
  POST /api/resume/<session_id>       -> resume from step
  POST /api/cancel                    -> cancel running generation
  GET  /api/sessions                  -> list sessions
  GET  /api/events                    -> SSE stream
  POST /api/upload                    -> upload input image
  GET  /api/vram                      -> VRAM stats
  GET  /api/config                    -> config.yaml as JSON
  GET  /api/logs                      -> log ring buffer

See also: API_REFERENCE.md, ARCHITECTURE.md
"""

import collections
import json
import logging
import os
import queue
import sys
import threading
import time
import uuid
from pathlib import Path

import yaml
from flask import (
    Flask, render_template, request, jsonify, Response,
    send_from_directory, redirect, url_for,
)
from werkzeug.utils import secure_filename

from session_manager import SessionManager, StepStatus, STEP_ORDER
from pipeline_engine import PipelineEngine, vram_stats, flush_vram, load_config
from preset_manager import PresetManager, scan_model_folders
from user_manager import UserManager, AVATAR_OPTIONS

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB uploads

# Custom Jinja2 filter for path basenames
app.jinja_env.filters["basename"] = lambda p: os.path.basename(p) if p else ""

UPLOAD_DIR = Path("input")
UPLOAD_DIR.mkdir(exist_ok=True)

# ─── Global state ──────────────────────────────────────────────────────
sm = SessionManager()
pm = PresetManager()
um = UserManager()
engine: PipelineEngine | None = None
cfg: dict = {}

# SSE event queues per client
sse_clients: dict[str, queue.Queue] = {}
# Active generation thread
active_thread: threading.Thread | None = None
active_session_id: str | None = None
generation_lock = threading.Lock()

# ─── Log capture ring buffer ──────────────────────────────────────
LOG_BUFFER_SIZE = 500
log_buffer: collections.deque = collections.deque(maxlen=LOG_BUFFER_SIZE)
log_lock = threading.Lock()


class LogCapture(logging.Handler):
    """Capture log lines into a ring buffer for the web UI."""
    def emit(self, record):
        try:
            line = self.format(record)
            with log_lock:
                log_buffer.append({
                    "ts": time.time(),
                    "level": record.levelname,
                    "msg": line,
                })
        except Exception:
            pass


# Also capture stdout/stderr prints from pipeline_engine
class StdoutCapture:
    def __init__(self, original):
        self.original = original

    def write(self, text):
        self.original.write(text)
        if text.strip():
            with log_lock:
                log_buffer.append({
                    "ts": time.time(),
                    "level": "INFO",
                    "msg": text.rstrip(),
                })

    def flush(self):
        self.original.flush()

    def isatty(self):
        return False

    def fileno(self):
        return self.original.fileno()

    def writable(self):
        return True


# Install log capture
_log_handler = LogCapture()
_log_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s",
                                            datefmt="%H:%M:%S"))
logging.root.addHandler(_log_handler)
logging.root.setLevel(logging.INFO)
sys.stdout = StdoutCapture(sys.__stdout__)
sys.stderr = StdoutCapture(sys.__stderr__)


def get_engine() -> PipelineEngine:
    global engine
    if engine is None:
        engine = PipelineEngine()
    return engine


def get_cfg() -> dict:
    global cfg
    if not cfg:
        cfg = load_config()
    return cfg


# ─── SSE broadcasting ──────────────────────────────────────────────
def broadcast_event(event_type: str, data: dict):
    """Send an event to all connected SSE clients."""
    msg = f"event: {event_type}\ndata: {json.dumps(data)}\n\n"
    dead = []
    for client_id, q in sse_clients.items():
        try:
            q.put_nowait(msg)
        except queue.Full:
            dead.append(client_id)
    for cid in dead:
        sse_clients.pop(cid, None)


def progress_callback(session_id: str, step: str, pct: float, msg: str):
    """Called by the pipeline engine to broadcast progress."""
    broadcast_event("progress", {
        "session_id": session_id,
        "step": step,
        "percent": pct,
        "message": msg,
        "vram": vram_stats(),
    })


# ─── Routes: Pages ─────────────────────────────────────────────────

@app.route("/")
def index():
    sessions = sm.list_sessions(limit=100)
    config = get_cfg()
    loras = config.get("loras", [])

    # Scan available upscale models
    upscale_dir = Path("models/upscale_models")
    upscale_models = []
    if upscale_dir.is_dir():
        for f in sorted(upscale_dir.iterdir()):
            if f.suffix in (".pth", ".safetensors") and "rife" not in f.name.lower():
                upscale_models.append({
                    "path": str(f),
                    "name": f.stem,
                })

    # Scan available model files for preset editor
    available_models = scan_model_folders()

    # Load presets
    presets = pm.list_presets()

    # Load users
    users = um.list_users()

    return render_template("index.html",
                           sessions=[s.to_dict() for s in sessions],
                           config=config,
                           loras=loras,
                           upscale_models=upscale_models,
                           available_models=available_models,
                           presets=presets,
                           users=users,
                           avatar_options=AVATAR_OPTIONS,
                           step_order=STEP_ORDER)


@app.route("/presets/editor")
def preset_editor():
    """Serve the standalone preset editor page (opens in a new window)."""
    return render_template("preset_editor.html")


@app.route("/presets/report")
def preset_report():
    """Serve a read-only preset report page."""
    presets = pm.list_presets()
    return render_template("preset_report.html", presets=presets)


# ─── Routes: API ───────────────────────────────────────────────────

@app.route("/api/sessions", methods=["GET"])
def api_list_sessions():
    # If user_id param is present (even ""), filter by it.
    # Global (user_id="") shows only sessions with no owner.
    # No param at all → return everything (used internally by stats).
    if "user_id" in request.args:
        user_id = request.args.get("user_id", "")
        sessions = sm.list_sessions_for_user(user_id, limit=100)
    else:
        sessions = sm.list_sessions(limit=100)
    return jsonify([s.to_dict() for s in sessions])


@app.route("/api/sessions/<session_id>", methods=["GET"])
def api_get_session(session_id):
    info = sm.get_session(session_id)
    if not info:
        return jsonify({"error": "Session not found"}), 404
    return jsonify(info.to_dict())


@app.route("/api/sessions/<session_id>", methods=["DELETE"])
def api_delete_session(session_id):
    if sm.delete_session(session_id):
        return jsonify({"ok": True})
    return jsonify({"error": "Session not found"}), 404


@app.route("/api/upload", methods=["POST"])
def api_upload_image():
    """Upload an input image and return its path."""
    if "image" not in request.files:
        return jsonify({"error": "No image file"}), 400
    f = request.files["image"]
    if not f.filename:
        return jsonify({"error": "Empty filename"}), 400

    fname = secure_filename(f.filename)
    # Add unique prefix to avoid collisions
    fname = f"{int(time.time())}_{fname}"
    path = UPLOAD_DIR / fname
    f.save(str(path))
    return jsonify({"path": str(path), "filename": fname})


@app.route("/api/generate", methods=["POST"])
def api_generate():
    """Start a new generation session."""
    global active_thread, active_session_id

    if not generation_lock.acquire(blocking=False):
        return jsonify({"error": "A generation is already running"}), 409

    try:
        data = request.json or {}

        # Merge config defaults with request data
        config = get_cfg()
        params = {
            "prompt": data.get("prompt", config.get("prompt", "")),
            "negative_prompt": data.get("negative_prompt", config.get("negative_prompt", "")),
            "input_image": data.get("input_image", ""),
            "width": int(data.get("width", config.get("width", 832))),
            "height": int(data.get("height", config.get("height", 480))),
            "num_frames": int(data.get("num_frames", config.get("num_frames", 81))),
            "num_inference_steps": int(data.get("num_inference_steps", config.get("num_inference_steps", 20))),
            "guidance_scale": float(data.get("guidance_scale", config.get("guidance_scale", 5.0))),
            "guidance_scale_2": float(data.get("guidance_scale_2", config.get("guidance_scale_2", 5.0))),
            "flow_shift": float(data.get("flow_shift", config.get("flow_shift", 8.0))),
            "seed": int(data.get("seed", config.get("seed", 42))),
            "fps": int(data.get("fps", config.get("fps", 16))),
            "duration": float(data.get("duration", config.get("duration", 5.0))),
            "output_fps": int(data.get("output_fps", config.get("output_fps", 24))),
            "target_duration": float(data.get("target_duration", config.get("target_duration", 0))),
            "enable_upscale": data.get("enable_upscale", False),
            "upscale_model": data.get("upscale_model", config.get("upscale_model", "models/upscale_models/4xRealWebPhoto_v4_dat2.pth")),
            "lora_scales": data.get("lora_scales", []),
            "boundary_ratio": float(data.get("boundary_ratio", config.get("boundary_ratio", 0.9))),
            "distill_lora_mode": bool(data.get("distill_lora_mode", config.get("distill_lora_mode", False))),
            # User assignment
            "user_id": data.get("user_id", ""),
            "user_name": data.get("user_name", ""),
        }

        # Memory / performance overrides — apply to engine cfg
        memory_overrides = {
            "offload_type": data.get("offload_type", config.get("offload_type", "block_level")),
            "num_blocks_per_group": int(data.get("num_blocks_per_group", config.get("num_blocks_per_group", 1))),
            "enable_group_offload": data.get("enable_group_offload", config.get("enable_group_offload", True)),
            "vae_tiling": data.get("vae_tiling", config.get("vae_tiling", True)),
            "vae_slicing": data.get("vae_slicing", config.get("vae_slicing", True)),
            "force_vae_cpu": data.get("force_vae_cpu", config.get("force_vae_cpu", False)),
        }

        # ── Preset model overrides ──
        # If a preset_id is provided, override engine config with preset's model paths and LoRAs
        preset_id = data.get("preset_id", "")
        preset_overrides = {}
        if preset_id:
            preset = pm.get_preset(preset_id)
            if preset:
                params["preset_id"] = preset_id
                params["preset_name"] = preset.get("name", "")
                # Override model paths in engine config
                if preset.get("gguf_transformer_high"):
                    preset_overrides["gguf_transformer_high"] = preset["gguf_transformer_high"]
                if preset.get("gguf_transformer_low"):
                    preset_overrides["gguf_transformer_low"] = preset["gguf_transformer_low"]
                if preset.get("vae_path"):
                    preset_overrides["vae_path"] = preset["vae_path"]
                if preset.get("text_encoder_path"):
                    preset_overrides["text_encoder_path"] = preset["text_encoder_path"]
                # Override LoRA config if preset has LoRAs defined
                if preset.get("loras"):
                    preset_overrides["loras"] = preset["loras"]

        # Validate input image
        if not params["input_image"] or not os.path.isfile(params["input_image"]):
            generation_lock.release()
            return jsonify({"error": "Input image not found"}), 400

        # Snap dimensions to mod 16
        params["width"] = (params["width"] // 16) * 16
        params["height"] = (params["height"] // 16) * 16

        # VRAM estimation guard — warn (but don't block) if settings
        # are likely to OOM on the detected GPU.
        try:
            import torch
            if torch.cuda.is_available():
                total_vram_gb = torch.cuda.get_device_properties(0).total_mem / 1024**3
                # Rough estimate: latent-shaped tensors dominate during denoise.
                # latent size ≈ (W/8)*(H/8)*(F/4)*16 channels * 2 bytes (bf16)
                # With CFG: ~3× that (latent + noise_pred_cond + noise_pred_uncond)
                w, h, f = params["width"], params["height"], params["num_frames"]
                latent_bytes = (w // 8) * (h // 8) * ((f - 1) // 4 + 1) * 16 * 2
                peak_est_gb = (latent_bytes * 3 + 2 * 1024**3) / 1024**3  # +2 GB overhead
                if peak_est_gb > total_vram_gb * 0.85:
                    print(f"  ⚠ VRAM WARNING: {w}×{h}×{f}f estimated peak "
                          f"{peak_est_gb:.1f} GB vs {total_vram_gb:.0f} GB total VRAM")
        except Exception:
            pass  # Non-critical — don't block generation

        # Create session
        session = sm.create_session(params)
        active_session_id = session.session_id

        # Run in background thread
        eng = get_engine()
        eng.set_progress_callback(progress_callback)

        # Apply per-generation memory/performance overrides to engine config
        for k, v in memory_overrides.items():
            eng.cfg[k] = v
        eng.cfg["boundary_ratio"] = params["boundary_ratio"]

        # Apply preset model path overrides to engine config
        for k, v in preset_overrides.items():
            eng.cfg[k] = v

        def run():
            global active_thread, active_session_id
            try:
                eng.run_from_step(session.session_id, "encode")
            finally:
                active_session_id = None
                active_thread = None
                generation_lock.release()
                broadcast_event("generation_complete", {
                    "session_id": session.session_id,
                    "session": sm.get_session(session.session_id).to_dict(),
                })

        active_thread = threading.Thread(target=run, daemon=True)
        active_thread.start()

        return jsonify({
            "session_id": session.session_id,
            "status": "started",
        })

    except Exception as e:
        generation_lock.release()
        return jsonify({"error": str(e)}), 500


@app.route("/api/resume/<session_id>", methods=["POST"])
def api_resume(session_id):
    """Resume a session from a specific step."""
    global active_thread, active_session_id

    info = sm.get_session(session_id)
    if not info:
        return jsonify({"error": "Session not found"}), 404

    if not generation_lock.acquire(blocking=False):
        return jsonify({"error": "A generation is already running"}), 409

    try:
        data = request.json or {}
        start_step = data.get("from_step", "")

        # If no step specified, auto-detect
        eng = get_engine()
        if not start_step:
            start_step = eng.get_resume_step(session_id)

        # Allow updating parameters before resume
        if "prompt" in data:
            info.prompt = data["prompt"]
        if "negative_prompt" in data:
            info.negative_prompt = data["negative_prompt"]
        if "seed" in data:
            info.seed = int(data["seed"])
        if "num_inference_steps" in data:
            info.num_inference_steps = int(data["num_inference_steps"])
        if "guidance_scale" in data:
            info.guidance_scale = float(data["guidance_scale"])
        if "guidance_scale_2" in data:
            info.guidance_scale_2 = float(data["guidance_scale_2"])
        if "lora_scales" in data:
            info.lora_scales = data["lora_scales"]
        if "enable_upscale" in data:
            info.enable_upscale = bool(data["enable_upscale"])
        if "upscale_model" in data:
            info.upscale_model = data["upscale_model"]
        if "output_fps" in data:
            info.output_fps = int(data["output_fps"])
        if "target_duration" in data:
            info.target_duration = float(data["target_duration"])
        if "duration" in data:
            info.duration = float(data["duration"])

        # Reset steps from start_step onwards
        start_idx = STEP_ORDER.index(start_step) if start_step in STEP_ORDER else 0
        for step_name in STEP_ORDER[start_idx:]:
            info.steps[step_name] = StepStatus.PENDING
        info.status = "running"
        info.error_message = ""
        info.updated_at = time.time()
        sm._save_meta(info)

        active_session_id = session_id
        eng.set_progress_callback(progress_callback)

        # Apply preset model overrides if this session has a preset
        if info.preset_id:
            preset = pm.get_preset(info.preset_id)
            if preset:
                if preset.get("gguf_transformer_high"):
                    eng.cfg["gguf_transformer_high"] = preset["gguf_transformer_high"]
                if preset.get("gguf_transformer_low"):
                    eng.cfg["gguf_transformer_low"] = preset["gguf_transformer_low"]
                if preset.get("vae_path"):
                    eng.cfg["vae_path"] = preset["vae_path"]
                if preset.get("text_encoder_path"):
                    eng.cfg["text_encoder_path"] = preset["text_encoder_path"]
                if preset.get("loras"):
                    eng.cfg["loras"] = preset["loras"]

        # Apply per-session config overrides
        eng.cfg["boundary_ratio"] = info.boundary_ratio

        def run():
            global active_thread, active_session_id
            try:
                eng.run_from_step(session_id, start_step)
            finally:
                active_session_id = None
                active_thread = None
                generation_lock.release()
                broadcast_event("generation_complete", {
                    "session_id": session_id,
                    "session": sm.get_session(session_id).to_dict(),
                })

        active_thread = threading.Thread(target=run, daemon=True)
        active_thread.start()

        return jsonify({
            "session_id": session_id,
            "from_step": start_step,
            "status": "resumed",
        })

    except Exception as e:
        generation_lock.release()
        return jsonify({"error": str(e)}), 500


@app.route("/api/cancel", methods=["POST"])
def api_cancel():
    """Cancel the currently running generation."""
    eng = get_engine()
    eng.cancel()
    return jsonify({"ok": True, "message": "Cancel requested"})


@app.route("/api/vram", methods=["GET"])
def api_vram():
    return jsonify(vram_stats())


@app.route("/api/system", methods=["GET"])
def api_system():
    """Return detailed system/hardware stats: GPU power, temp, fan, clocks, CPU, RAM."""
    result = {}

    # ── GPU stats via rocm-smi ──
    try:
        import subprocess
        raw = subprocess.run(
            ["rocm-smi", "--showpower", "--showtemp", "--showfan", "--showuse",
             "--showmeminfo", "vram", "--showclocks", "--json"],
            capture_output=True, text=True, timeout=5,
        )
        if raw.returncode == 0:
            import json as _json
            data = _json.loads(raw.stdout)
            # Use card0 (primary GPU)
            card = data.get("card0", {})
            result["gpu"] = {
                "temp_edge_c":     _safe_float(card.get("Temperature (Sensor edge) (C)")),
                "temp_junction_c": _safe_float(card.get("Temperature (Sensor junction) (C)")),
                "temp_memory_c":   _safe_float(card.get("Temperature (Sensor memory) (C)")),
                "power_w":         _safe_float(card.get("Average Graphics Package Power (W)")),
                "gpu_use_pct":     _safe_float(card.get("GPU use (%)")),
                "fan_rpm":         _safe_float(card.get("Fan RPM")),
                "fan_pct":         _safe_float(card.get("Fan speed (%)")),
                "vram_total_b":    _safe_int(card.get("VRAM Total Memory (B)")),
                "vram_used_b":     _safe_int(card.get("VRAM Total Used Memory (B)")),
                "sclk":            card.get("sclk clock speed:", ""),
                "mclk":            card.get("mclk clock speed:", ""),
                "pcie":            card.get("pcie clock level", ""),
            }
    except Exception:
        result["gpu"] = {}

    # ── CPU / RAM via psutil ──
    try:
        import psutil
        # CPU temperature (AMD k10temp or coretemp)
        cpu_temp = None
        try:
            temps = psutil.sensors_temperatures()
            for chip in ("k10temp", "coretemp", "zenpower"):
                if chip in temps and temps[chip]:
                    # Use Tctl / Package temp (first entry)
                    cpu_temp = temps[chip][0].current
                    break
        except Exception:
            pass

        # CPU power via RAPL (if accessible)
        cpu_power_w = None
        try:
            rapl_path = Path("/sys/class/powercap/intel-rapl:0/energy_uj")
            if rapl_path.exists():
                e1 = int(rapl_path.read_text().strip())
                time.sleep(0.1)
                e2 = int(rapl_path.read_text().strip())
                cpu_power_w = round((e2 - e1) / 100_000, 1)  # µJ over 0.1s → W
        except Exception:
            pass

        result["cpu"] = {
            "model":       _cpu_model(),
            "cores":       psutil.cpu_count(logical=True),
            "usage_pct":   psutil.cpu_percent(interval=0),
            "freq_mhz":    round(psutil.cpu_freq().current, 0) if psutil.cpu_freq() else None,
            "freq_max_mhz": round(psutil.cpu_freq().max, 0) if psutil.cpu_freq() else None,
            "load_avg":    list(os.getloadavg()),
            "temp_c":      cpu_temp,
            "power_w":     cpu_power_w,
        }
        mem = psutil.virtual_memory()
        result["ram"] = {
            "total_gb":     round(mem.total / 1024**3, 1),
            "used_gb":      round(mem.used / 1024**3, 1),
            "available_gb": round(mem.available / 1024**3, 1),
            "percent":      mem.percent,
        }
        swap = psutil.swap_memory()
        result["swap"] = {
            "total_gb":  round(swap.total / 1024**3, 1),
            "used_gb":   round(swap.used / 1024**3, 1),
            "percent":   swap.percent,
        }
    except Exception:
        result["cpu"] = {}
        result["ram"] = {}
        result["swap"] = {}

    # ── Uptime ──
    try:
        import psutil
        import datetime
        boot = datetime.datetime.fromtimestamp(psutil.boot_time())
        uptime = datetime.datetime.now() - boot
        result["uptime_seconds"] = int(uptime.total_seconds())
    except Exception:
        result["uptime_seconds"] = None

    return jsonify(result)


def _safe_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None

def _safe_int(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None

def _cpu_model():
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if "model name" in line:
                    return line.split(":")[1].strip()
    except Exception:
        pass
    return None


@app.route("/api/config", methods=["GET"])
def api_config():
    return jsonify(get_cfg())


@app.route("/api/active", methods=["GET"])
def api_active():
    """Return the currently running session ID (if any)."""
    return jsonify({
        "session_id": active_session_id,
        "running": active_session_id is not None,
    })


@app.route("/api/stats", methods=["GET"])
def api_stats():
    """Return system stats for the dashboard."""
    from amd_tune import detect_amd_gpu

    gpu_info = {}
    try:
        gpu_info = detect_amd_gpu()
    except Exception:
        gpu_info = {"available": False}

    # Session counts by status + energy totals
    sessions = sm.list_sessions(limit=9999)
    counts = {"total": len(sessions), "done": 0, "running": 0, "failed": 0,
              "cancelled": 0, "pending": 0}
    total_energy_wh = 0.0
    total_gpu_energy_wh = 0.0
    peak_gpu_power_all = 0.0
    sessions_with_energy = 0
    for s in sessions:
        st = s.status
        if st in counts:
            counts[st] += 1
        else:
            counts[st] = counts.get(st, 0) + 1
        if getattr(s, 'energy_wh', 0) > 0:
            total_energy_wh += s.energy_wh
            total_gpu_energy_wh += s.gpu_energy_wh
            sessions_with_energy += 1
            if s.peak_gpu_power_w > peak_gpu_power_all:
                peak_gpu_power_all = s.peak_gpu_power_w

    # Disk usage
    sessions_bytes = 0
    output_bytes = 0
    try:
        sessions_dir = Path("sessions")
        if sessions_dir.exists():
            for f in sessions_dir.rglob("*"):
                if f.is_file():
                    sessions_bytes += f.stat().st_size
        output_dir = Path("output")
        if output_dir.exists():
            for f in output_dir.rglob("*"):
                if f.is_file():
                    output_bytes += f.stat().st_size
    except Exception:
        pass

    # Preset count
    preset_count = len(pm.list_presets())

    # User count
    user_count = len(um.list_users())

    # Electricity cost from config
    config = get_cfg()
    cost_kwh = config.get("electricity_cost_kwh", 0.12)
    total_cost = (total_energy_wh / 1000) * cost_kwh

    return jsonify({
        "gpu": gpu_info,
        "sessions": counts,
        "disk": {
            "sessions_bytes": sessions_bytes,
            "output_bytes": output_bytes,
        },
        "preset_count": preset_count,
        "user_count": user_count,
        "energy": {
            "total_wh": round(total_energy_wh, 2),
            "gpu_wh": round(total_gpu_energy_wh, 2),
            "peak_gpu_w": round(peak_gpu_power_all, 1),
            "sessions_tracked": sessions_with_energy,
            "cost_kwh": cost_kwh,
            "total_cost": round(total_cost, 4),
        },
    })


@app.route("/api/logs", methods=["GET"])
def api_logs():
    """Return recent log lines (last N entries)."""
    n = int(request.args.get("n", 100))
    since = float(request.args.get("since", 0))
    with log_lock:
        lines = [l for l in log_buffer if l["ts"] > since]
    return jsonify(lines[-n:])


# ─── Routes: Model Presets ─────────────────────────────────────────

@app.route("/api/models/scan", methods=["GET"])
def api_scan_models():
    """Scan model folders and return available files by type."""
    return jsonify(scan_model_folders())


@app.route("/api/presets", methods=["GET"])
def api_list_presets():
    """List all model presets."""
    return jsonify(pm.list_presets())


@app.route("/api/presets", methods=["POST"])
def api_create_preset():
    """Create a new model preset."""
    data = request.json or {}
    if not data.get("name"):
        return jsonify({"error": "Preset name is required"}), 400
    preset = pm.create_preset(data)
    return jsonify(preset), 201


@app.route("/api/presets/<preset_id>", methods=["GET"])
def api_get_preset(preset_id):
    """Get a single preset by ID."""
    preset = pm.get_preset(preset_id)
    if not preset:
        return jsonify({"error": "Preset not found"}), 404
    return jsonify(preset)


@app.route("/api/presets/<preset_id>", methods=["PUT"])
def api_update_preset(preset_id):
    """Update an existing preset."""
    data = request.json or {}
    preset = pm.update_preset(preset_id, data)
    if not preset:
        return jsonify({"error": "Preset not found"}), 404
    return jsonify(preset)


@app.route("/api/presets/<preset_id>", methods=["DELETE"])
def api_delete_preset(preset_id):
    """Delete a preset."""
    if pm.delete_preset(preset_id):
        return jsonify({"ok": True})
    return jsonify({"error": "Preset not found"}), 404


@app.route("/api/presets/<preset_id>/duplicate", methods=["POST"])
def api_duplicate_preset(preset_id):
    """Duplicate an existing preset."""
    preset = pm.duplicate_preset(preset_id)
    if not preset:
        return jsonify({"error": "Preset not found"}), 404
    return jsonify(preset), 201


@app.route("/api/sessions/<session_id>/clone", methods=["POST"])
def api_clone_session(session_id):
    """Return the generation parameters from a session so the user can
    pre-fill the generate form with them (clone-to-generate)."""
    info = sm.get_session(session_id)
    if not info:
        return jsonify({"error": "Session not found"}), 404
    d = info.to_dict()
    # Return only the params relevant for generating
    clone = {
        "prompt": d.get("prompt", ""),
        "negative_prompt": d.get("negative_prompt", ""),
        "width": d.get("width", 832),
        "height": d.get("height", 480),
        "num_frames": d.get("num_frames", 81),
        "num_inference_steps": d.get("num_inference_steps", 8),
        "guidance_scale": d.get("guidance_scale", 2.0),
        "guidance_scale_2": d.get("guidance_scale_2", 2.0),
        "flow_shift": d.get("flow_shift", 8.0),
        "seed": d.get("seed", 42),
        "fps": d.get("fps", 16),
        "duration": d.get("duration", 5.0),
        "output_fps": d.get("output_fps", 24),
        "target_duration": d.get("target_duration", 0),
        "enable_upscale": d.get("enable_upscale", False),
        "upscale_model": d.get("upscale_model", ""),
        "lora_scales": d.get("lora_scales", []),
        "boundary_ratio": d.get("boundary_ratio", 0.5),
        "distill_lora_mode": d.get("distill_lora_mode", False),
        # input image: absolute path + URL for display
        "input_image": d.get("input_image", ""),
        "input_image_url": f"/sessions/{session_id}/input.png"
                          if (sm.session_dir(session_id) / "input.png").exists() else "",
        # Preset info
        "preset_id": d.get("preset_id", ""),
        "preset_name": d.get("preset_name", ""),
        # User info
        "user_id": d.get("user_id", ""),
        "user_name": d.get("user_name", ""),
    }
    return jsonify(clone)


# ─── Routes: User Profiles ───────────────────────────────────────

@app.route("/api/users", methods=["GET"])
def api_list_users():
    """List all user profiles."""
    return jsonify(um.list_users())


@app.route("/api/users", methods=["POST"])
def api_create_user():
    """Create a new user profile."""
    data = request.json or {}
    if not data.get("name", "").strip():
        return jsonify({"error": "User name is required"}), 400
    user = um.create_user(data)
    return jsonify(user), 201


@app.route("/api/users/<user_id>", methods=["GET"])
def api_get_user(user_id):
    """Get a single user profile."""
    user = um.get_user(user_id)
    if not user:
        return jsonify({"error": "User not found"}), 404
    return jsonify(user)


@app.route("/api/users/<user_id>", methods=["PUT"])
def api_update_user(user_id):
    """Update a user profile."""
    data = request.json or {}
    user = um.update_user(user_id, data)
    if not user:
        return jsonify({"error": "User not found"}), 404
    return jsonify(user)


@app.route("/api/users/<user_id>", methods=["DELETE"])
def api_delete_user(user_id):
    """Delete a user profile. Sessions are preserved (become unowned)."""
    if um.delete_user(user_id):
        return jsonify({"ok": True})
    return jsonify({"error": "User not found"}), 404


@app.route("/api/users/avatars", methods=["GET"])
def api_avatar_options():
    """Return available avatar emoji options."""
    return jsonify(AVATAR_OPTIONS)


# ─── SSE endpoint ──────────────────────────────────────────────────

@app.route("/api/events")
def api_events():
    """Server-Sent Events stream for real-time progress."""
    client_id = str(uuid.uuid4())
    q = queue.Queue(maxsize=200)
    sse_clients[client_id] = q

    def stream():
        try:
            # Send initial ping
            yield f"event: connected\ndata: {json.dumps({'client_id': client_id})}\n\n"
            while True:
                try:
                    msg = q.get(timeout=30)
                    yield msg
                except queue.Empty:
                    # Send keepalive
                    yield ": keepalive\n\n"
        except GeneratorExit:
            pass
        finally:
            sse_clients.pop(client_id, None)

    return Response(stream(), mimetype="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "X-Accel-Buffering": "no",
                        "Connection": "keep-alive",
                    })


# ─── Static file serving for sessions ──────────────────────────────

@app.route("/sessions/<session_id>/<filename>")
def serve_session_file(session_id, filename):
    """Serve files from a session directory (videos, images)."""
    sdir = sm.session_dir(session_id)
    if not sdir.exists():
        return "Not found", 404
    # Only allow safe file types
    allowed = {".mp4", ".png", ".jpg", ".jpeg", ".webp", ".json", ".log"}
    ext = Path(filename).suffix.lower()
    if ext not in allowed:
        return "Forbidden", 403
    return send_from_directory(str(sdir), filename)


@app.route("/input/<filename>")
def serve_input(filename):
    return send_from_directory(str(UPLOAD_DIR), filename)


@app.route("/api/sessions/<session_id>/generation_log", methods=["GET"])
def api_generation_log(session_id):
    """Return the generation.log file content for a session."""
    sdir = sm.session_dir(session_id)
    log_path = sdir / "generation.log"
    if not log_path.exists():
        return jsonify({"content": "(no generation log yet)", "exists": False})
    try:
        content = log_path.read_text(errors="replace")
        return jsonify({"content": content, "exists": True})
    except Exception as e:
        return jsonify({"content": f"Error reading log: {e}", "exists": False})


@app.route("/api/sessions/<session_id>/vram_log", methods=["GET"])
def api_vram_log(session_id):
    """Return the vram.log file content for a session."""
    sdir = sm.session_dir(session_id)
    log_path = sdir / "vram.log"
    if not log_path.exists():
        return jsonify({"content": "(no VRAM log yet)", "exists": False})
    try:
        content = log_path.read_text(errors="replace")
        return jsonify({"content": content, "exists": True})
    except Exception as e:
        return jsonify({"content": f"Error reading log: {e}", "exists": False})


# ─── Main ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    # Apply AMD tuning from config BEFORE any CUDA calls
    startup_cfg = get_cfg()
    from amd_tune import apply_amd_optimisations, detect_amd_gpu
    apply_amd_optimisations(startup_cfg)

    print("=" * 60)
    print("  Wan 2.2 I2V — Web UI")
    print(f"  http://localhost:{args.port}")

    try:
        gpu_info = detect_amd_gpu()
        if gpu_info["available"]:
            print(f"  GPU:   {gpu_info['device_name']}")
            print(f"  VRAM:  {gpu_info['total_vram_gb']} GB")
            print(f"  Arch:  {gpu_info['gfx_arch']}")
            print(f"  HSA:   {os.environ.get('HSA_OVERRIDE_GFX_VERSION', 'not set')}")
            vram_gb = gpu_info['total_vram_gb']
            if vram_gb < 16:
                print(f"  ⚠  Low VRAM ({vram_gb} GB) — ensure force_vae_cpu=true")
                print(f"     Max safe resolution: 832×480 @ 81 frames")
        else:
            print("  ⚠  No GPU detected!")
    except Exception as e:
        print(f"  ⚠  GPU detection failed: {e}")

    print("=" * 60)

    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)
