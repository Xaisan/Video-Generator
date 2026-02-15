#!/usr/bin/env python3
"""
session_manager.py — Persistent session & checkpoint storage
=============================================================

Manages the lifecycle of generation sessions. Each session is a directory
under sessions/<session_id>/ with JSON metadata and PyTorch checkpoints.

Dependencies (project-internal): none (this is a leaf module)
External: json, os, shutil, time, uuid, torch

Public API:
  Constants:
    SESSIONS_DIR         — Path("sessions")
    STEP_ORDER           — ["encode", "denoise", "vae_decode", "export", "upscale"]

  Enums:
    StepStatus           — pending | running | done | failed | skipped

  Dataclasses:
    SessionInfo          — all session metadata + generation parameters
      .to_dict()         — serialise to JSON-safe dict
      .from_dict()       — deserialise from dict

  Classes:
    SessionManager       — CRUD + checkpoint operations
      .create_session()  — create new session from params dict
      .get_session()     — load session by ID
      .list_sessions()   — list all sessions (newest first)
      .delete_session()  — remove session directory
      .update_step()     — update step status + timing
      .update_status()   — update overall session status
      .save_checkpoint() — save a torch.Tensor as .pt file
      .load_checkpoint() — load a .pt checkpoint
      .has_checkpoint()  — check if checkpoint exists
      .save_file()       — copy a file into session dir
      .get_file_path()   — get path to a session file
      .session_dir()     — get session directory path

Session directory layout:
  sessions/<timestamp>_<uuid8>/
  +-- session.json           # Metadata (params, step statuses, timing)
  +-- input.png              # Copy of input image
  +-- embeddings.pt          # After encode step
  +-- latents.pt             # After denoise step
  +-- frames.pt              # After VAE decode step
  +-- preview.mp4            # Quick preview
  +-- output.mp4             # Final video
  +-- output_upscaled.mp4    # Optional upscaled video
  +-- generation.log         # Pipeline execution diary
  +-- vram.log               # VRAM allocation timeline

Used by: app.py, pipeline_engine.py
"""

import json
import os
import shutil
import time
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any

import torch


SESSIONS_DIR = Path("sessions")


class StepStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"


STEP_ORDER = ["encode", "denoise", "vae_decode", "export", "upscale"]


@dataclass
class SessionInfo:
    session_id: str
    created_at: float
    updated_at: float
    status: str = "created"  # created | running | done | failed | cancelled
    error_message: str = ""

    # Generation parameters (stored so we can resume with them)
    prompt: str = ""
    negative_prompt: str = ""
    input_image: str = ""
    width: int = 832
    height: int = 480
    num_frames: int = 81
    num_inference_steps: int = 20
    guidance_scale: float = 5.0
    guidance_scale_2: float = 5.0
    flow_shift: float = 8.0
    seed: int = 42
    fps: int = 16
    duration: float = 5.0          # Video length in seconds
    output_fps: int = 24           # Output framerate after RIFE interpolation
    target_duration: float = 0     # Target video duration in seconds (0 = preserve original)
    enable_upscale: bool = False
    upscale_model: str = "models/upscale_models/4xRealWebPhoto_v4_dat2.pth"
    boundary_ratio: float = 0.9    # High/low noise boundary split (official model default)
    distill_lora_mode: bool = False # True = fast 4-step distill, False = quality mode

    # Model preset (optional — overrides config.yaml model paths)
    preset_id: str = ""
    preset_name: str = ""

    # User profile (optional — empty string = global/unowned)
    user_id: str = ""
    user_name: str = ""

    # LoRA overrides (list of {adapter_name, scale})
    lora_scales: list = field(default_factory=list)

    # Energy tracking (populated after generation completes)
    energy_wh: float = 0.0             # Estimated total wall energy in Wh
    gpu_energy_wh: float = 0.0         # GPU-only energy in Wh
    peak_gpu_power_w: float = 0.0      # Peak GPU power draw in W
    avg_gpu_power_w: float = 0.0       # Average GPU power draw in W
    energy_cost_kwh: float = 0.0       # Electricity price used ($/kWh)

    # Step statuses
    steps: dict = field(default_factory=lambda: {
        s: StepStatus.PENDING for s in STEP_ORDER
    })

    # Timing info per step
    step_times: dict = field(default_factory=dict)

    # Which checkpoint files exist
    checkpoints: list = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "SessionInfo":
        # Handle enum conversion for steps
        steps = d.get("steps", {})
        for k, v in steps.items():
            if isinstance(v, str):
                steps[k] = StepStatus(v)
        d["steps"] = steps
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class SessionManager:
    def __init__(self, base_dir: str | Path = SESSIONS_DIR):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _session_dir(self, session_id: str) -> Path:
        return self.base_dir / session_id

    def _meta_path(self, session_id: str) -> Path:
        return self._session_dir(session_id) / "session.json"

    # ─── CRUD ───────────────────────────────────────────────────────

    def create_session(self, params: dict) -> SessionInfo:
        """Create a new session with given generation parameters."""
        session_id = f"{int(time.time())}_{uuid.uuid4().hex[:8]}"
        sdir = self._session_dir(session_id)
        sdir.mkdir(parents=True, exist_ok=True)

        now = time.time()
        info = SessionInfo(
            session_id=session_id,
            created_at=now,
            updated_at=now,
            prompt=params.get("prompt", ""),
            negative_prompt=params.get("negative_prompt", ""),
            input_image=params.get("input_image", ""),
            width=int(params.get("width", 832)),
            height=int(params.get("height", 480)),
            num_frames=int(params.get("num_frames", 81)),
            num_inference_steps=int(params.get("num_inference_steps", 8)),
            guidance_scale=float(params.get("guidance_scale", 2.0)),
            guidance_scale_2=float(params.get("guidance_scale_2", 2.0)),
            flow_shift=float(params.get("flow_shift", 8.0)),
            seed=int(params.get("seed", 42)),
            fps=int(params.get("fps", 16)),
            duration=float(params.get("duration", 5.0)),
            output_fps=int(params.get("output_fps", 24)),
            target_duration=float(params.get("target_duration", 0)),
            enable_upscale=bool(params.get("enable_upscale", False)),
            upscale_model=params.get("upscale_model", "models/upscale_models/4xRealWebPhoto_v4_dat2.pth"),
            boundary_ratio=float(params.get("boundary_ratio", 0.9)),
            lora_scales=params.get("lora_scales", []),
            distill_lora_mode=bool(params.get("distill_lora_mode", False)),
            preset_id=params.get("preset_id", ""),
            preset_name=params.get("preset_name", ""),
            user_id=params.get("user_id", ""),
            user_name=params.get("user_name", ""),
        )

        # Copy input image into session
        src_image = params.get("input_image", "")
        if src_image and os.path.isfile(src_image):
            dst = sdir / "input.png"
            shutil.copy2(src_image, dst)
            info.input_image = str(dst)

        self._save_meta(info)
        return info

    def get_session(self, session_id: str) -> SessionInfo | None:
        path = self._meta_path(session_id)
        if not path.exists():
            return None
        with open(path) as f:
            return SessionInfo.from_dict(json.load(f))

    def list_sessions(self, limit: int = 50) -> list[SessionInfo]:
        """Return sessions sorted newest first."""
        sessions = []
        if not self.base_dir.exists():
            return sessions
        for d in sorted(self.base_dir.iterdir(), reverse=True):
            if d.is_dir() and (d / "session.json").exists():
                try:
                    info = self.get_session(d.name)
                    if info:
                        sessions.append(info)
                except Exception:
                    continue
            if len(sessions) >= limit:
                break
        return sessions

    def list_sessions_for_user(self, user_id: str, limit: int = 100) -> list[SessionInfo]:
        """Return sessions filtered by user_id.

        If user_id is empty, returns only 'unowned' sessions (global space).
        If user_id is set, returns only that user's sessions.
        Use list_sessions() for an unfiltered list.
        """
        all_sessions = self.list_sessions(limit=9999)
        filtered = [s for s in all_sessions if s.user_id == user_id]
        return filtered[:limit]

    def delete_session(self, session_id: str) -> bool:
        sdir = self._session_dir(session_id)
        if sdir.exists():
            shutil.rmtree(sdir)
            return True
        return False

    # ─── Step management ────────────────────────────────────────────

    def update_step(self, session_id: str, step: str, status: StepStatus,
                    elapsed: float = 0, error: str = "") -> SessionInfo | None:
        info = self.get_session(session_id)
        if not info:
            return None
        info.steps[step] = status
        info.updated_at = time.time()
        if elapsed > 0:
            info.step_times[step] = round(elapsed, 2)
        if status == StepStatus.FAILED:
            info.status = "failed"
            info.error_message = error
        elif status == StepStatus.RUNNING:
            info.status = "running"
        elif all(v == StepStatus.DONE or v == StepStatus.SKIPPED
                 for v in info.steps.values()):
            info.status = "done"
        self._save_meta(info)
        return info

    def update_status(self, session_id: str, status: str,
                      error: str = "") -> SessionInfo | None:
        info = self.get_session(session_id)
        if not info:
            return None
        info.status = status
        info.error_message = error
        info.updated_at = time.time()
        self._save_meta(info)
        return info

    # ─── Checkpoint files ───────────────────────────────────────────

    def save_checkpoint(self, session_id: str, name: str, tensor: torch.Tensor):
        """Save a PyTorch tensor checkpoint."""
        sdir = self._session_dir(session_id)
        path = sdir / f"{name}.pt"
        torch.save(tensor, path)
        # Update metadata
        info = self.get_session(session_id)
        if info and name not in info.checkpoints:
            info.checkpoints.append(name)
            info.updated_at = time.time()
            self._save_meta(info)

    def load_checkpoint(self, session_id: str, name: str) -> torch.Tensor | None:
        path = self._session_dir(session_id) / f"{name}.pt"
        if path.exists():
            return torch.load(path, map_location="cpu", weights_only=False)
        return None

    def has_checkpoint(self, session_id: str, name: str) -> bool:
        return (self._session_dir(session_id) / f"{name}.pt").exists()

    def save_file(self, session_id: str, filename: str, src_path: str):
        """Copy a file into the session directory."""
        dst = self._session_dir(session_id) / filename
        shutil.copy2(src_path, dst)
        info = self.get_session(session_id)
        if info and filename not in info.checkpoints:
            info.checkpoints.append(filename)
            info.updated_at = time.time()
            self._save_meta(info)

    def get_file_path(self, session_id: str, filename: str) -> Path | None:
        path = self._session_dir(session_id) / filename
        return path if path.exists() else None

    def session_dir(self, session_id: str) -> Path:
        return self._session_dir(session_id)

    # ─── Internal ───────────────────────────────────────────────────

    def _save_meta(self, info: SessionInfo):
        path = self._meta_path(info.session_id)
        with open(path, "w") as f:
            json.dump(info.to_dict(), f, indent=2, default=str)
