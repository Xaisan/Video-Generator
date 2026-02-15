#!/usr/bin/env python3
"""
Web UI for Wan 2.2 I2V Video Generator.
Flask + Server-Sent Events for real-time progress.
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

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB uploads

# Custom Jinja2 filter for path basenames
app.jinja_env.filters["basename"] = lambda p: os.path.basename(p) if p else ""

UPLOAD_DIR = Path("input")
UPLOAD_DIR.mkdir(exist_ok=True)

# ─── Global state ──────────────────────────────────────────────────
sm = SessionManager()
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

    return render_template("index.html",
                           sessions=[s.to_dict() for s in sessions],
                           config=config,
                           loras=loras,
                           upscale_models=upscale_models,
                           step_order=STEP_ORDER)


# ─── Routes: API ───────────────────────────────────────────────────

@app.route("/api/sessions", methods=["GET"])
def api_list_sessions():
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


@app.route("/api/logs", methods=["GET"])
def api_logs():
    """Return recent log lines (last N entries)."""
    n = int(request.args.get("n", 100))
    since = float(request.args.get("since", 0))
    with log_lock:
        lines = [l for l in log_buffer if l["ts"] > since]
    return jsonify(lines[-n:])


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
        # input image: absolute path + URL for display
        "input_image": d.get("input_image", ""),
        "input_image_url": f"/sessions/{session_id}/input.png"
                          if (sm.session_dir(session_id) / "input.png").exists() else "",
    }
    return jsonify(clone)


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
    allowed = {".mp4", ".png", ".jpg", ".jpeg", ".webp", ".json"}
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
