#!/usr/bin/env python3
"""
Session Manager — persistent checkpoint storage for the video generation pipeline.

Each session gets a directory under sessions/<session_id>/ containing:
  session.json    — metadata (params, timestamps, status, which checkpoints exist)
  input.png       — copy of the input image
  embeddings.pt   — encoded text + image embeddings (after encode step)
  latents.pt      — denoised latent tensor (after denoise step)
  frames.pt       — decoded video frames tensor (after VAE decode step)
  preview.mp4     — low-quality preview from frames
  output.mp4      — final exported video
  output_upscaled.mp4 — optional upscaled video
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
    num_inference_steps: int = 8
    guidance_scale: float = 2.0
    guidance_scale_2: float = 2.0
    flow_shift: float = 8.0
    seed: int = 42
    fps: int = 16
    duration: float = 5.0          # Video length in seconds
    output_fps: int = 24           # Output framerate after RIFE interpolation
    enable_upscale: bool = False

    # LoRA overrides (list of {adapter_name, scale})
    lora_scales: list = field(default_factory=list)

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
            enable_upscale=bool(params.get("enable_upscale", False)),
            lora_scales=params.get("lora_scales", []),
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
