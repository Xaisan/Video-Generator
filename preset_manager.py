#!/usr/bin/env python3
"""
preset_manager.py — Model preset management
=============================================

Manages named model presets that bundle together:
  - Base GGUF transformers (high-noise + low-noise)
  - VAE model
  - Text encoder
  - LoRA configurations with scales and roles
  - Default generation parameters

Presets are stored in presets.json at the project root.
Each preset can override model paths from config.yaml while
keeping generation parameters as sensible defaults.

Public API:
  PresetManager         — CRUD operations for presets
  scan_model_folders()  — scan models/ for available files

Dependencies: none (leaf module)
"""

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any


PRESETS_FILE = Path("presets.json")

# Supported file extensions per model type
MODEL_EXTENSIONS = {
    "unet": {".gguf"},
    "loras": {".safetensors"},
    "vae": {".safetensors", ".pt", ".pth", ".bin"},
    "text_encoders": {".safetensors", ".pt", ".pth", ".bin"},
    "upscale_models": {".pth", ".safetensors", ".pt"},
}


def scan_model_folders(base: str = "models") -> dict:
    """Scan model folders and return available files grouped by type.

    Returns dict like:
    {
        "unet": [
            {"path": "models/unet/Wan2.2-...-Q8_0.gguf", "name": "Wan2.2-...-Q8_0", "size_mb": 10240},
            ...
        ],
        "loras": [...],
        "vae": [...],
        "text_encoders": [...],
        "upscale_models": [...]
    }

    Each folder supports one level of subfolders (e.g., models/loras/my_set/file.safetensors).
    """
    base_path = Path(base)
    result = {}

    for folder_name, extensions in MODEL_EXTENSIONS.items():
        folder = base_path / folder_name
        files = []
        if folder.is_dir():
            # Scan direct files and one level of subfolders
            for item in sorted(folder.iterdir()):
                if item.is_file() and item.suffix.lower() in extensions:
                    files.append(_file_info(item))
                elif item.is_dir():
                    # One level of subfolders
                    for sub_item in sorted(item.iterdir()):
                        if sub_item.is_file() and sub_item.suffix.lower() in extensions:
                            files.append(_file_info(sub_item))
        result[folder_name] = files

    return result


def _file_info(path: Path) -> dict:
    """Build file info dict for a model file."""
    try:
        size_mb = round(path.stat().st_size / (1024 * 1024), 1)
    except OSError:
        size_mb = 0
    return {
        "path": str(path),
        "name": path.stem,
        "filename": path.name,
        "size_mb": size_mb,
    }


class PresetManager:
    """CRUD operations for model presets stored in presets.json."""

    def __init__(self, path: str | Path = PRESETS_FILE):
        self.path = Path(path)
        self._presets: dict[str, dict] = {}
        self._load()

    def _load(self):
        if self.path.exists():
            try:
                with open(self.path) as f:
                    data = json.load(f)
                self._presets = data.get("presets", {})
            except (json.JSONDecodeError, KeyError):
                self._presets = {}
        else:
            self._presets = {}

    def _save(self):
        with open(self.path, "w") as f:
            json.dump({"presets": self._presets}, f, indent=2)

    def list_presets(self) -> list[dict]:
        """Return all presets as a list, sorted by name."""
        result = []
        for pid, preset in self._presets.items():
            p = dict(preset)
            p["id"] = pid
            result.append(p)
        return sorted(result, key=lambda x: x.get("name", "").lower())

    def get_preset(self, preset_id: str) -> dict | None:
        preset = self._presets.get(preset_id)
        if preset:
            p = dict(preset)
            p["id"] = preset_id
            return p
        return None

    def create_preset(self, data: dict) -> dict:
        """Create a new preset. Returns the created preset with ID."""
        preset_id = f"{int(time.time())}_{uuid.uuid4().hex[:8]}"
        preset = self._validate_preset(data)
        preset["created_at"] = time.time()
        preset["updated_at"] = time.time()
        self._presets[preset_id] = preset
        self._save()
        p = dict(preset)
        p["id"] = preset_id
        return p

    def update_preset(self, preset_id: str, data: dict) -> dict | None:
        """Update an existing preset (merge with existing values)."""
        if preset_id not in self._presets:
            return None
        # Merge: start from existing, overlay incoming data
        existing = dict(self._presets[preset_id])
        # Handle nested 'defaults' merge separately
        if "defaults" in data and "defaults" in existing:
            merged_defaults = dict(existing.get("defaults", {}))
            merged_defaults.update(data["defaults"])
            data = dict(data)
            data["defaults"] = merged_defaults
        merged = dict(existing)
        merged.update(data)
        preset = self._validate_preset(merged)
        preset["created_at"] = existing.get("created_at", time.time())
        preset["updated_at"] = time.time()
        self._presets[preset_id] = preset
        self._save()
        p = dict(preset)
        p["id"] = preset_id
        return p

    def delete_preset(self, preset_id: str) -> bool:
        if preset_id in self._presets:
            del self._presets[preset_id]
            self._save()
            return True
        return False

    def duplicate_preset(self, preset_id: str) -> dict | None:
        """Duplicate an existing preset with a new name."""
        original = self._presets.get(preset_id)
        if not original:
            return None
        data = dict(original)
        data["name"] = data.get("name", "Preset") + " (copy)"
        return self.create_preset(data)

    def _validate_preset(self, data: dict) -> dict:
        """Validate and normalize preset data."""
        preset = {
            "name": str(data.get("name", "Unnamed Preset")),
            "description": str(data.get("description", "")),

            # Model paths
            "gguf_transformer_high": str(data.get("gguf_transformer_high", "")),
            "gguf_transformer_low": str(data.get("gguf_transformer_low", "")),
            "vae_path": str(data.get("vae_path", "")),
            "text_encoder_path": str(data.get("text_encoder_path", "")),

            # LoRA configuration — list of {path, target, adapter_name, scale, role}
            "loras": list(data.get("loras", [])),

            # Default generation parameters (optional — UI fills from these)
            "defaults": {
                "num_inference_steps": int(data.get("defaults", {}).get("num_inference_steps", 20)),
                "guidance_scale": float(data.get("defaults", {}).get("guidance_scale", 5.0)),
                "guidance_scale_2": float(data.get("defaults", {}).get("guidance_scale_2", 5.0)),
                "flow_shift": float(data.get("defaults", {}).get("flow_shift", 8.0)),
                "boundary_ratio": float(data.get("defaults", {}).get("boundary_ratio", 0.9)),
                "distill_lora_mode": bool(data.get("defaults", {}).get("distill_lora_mode", False)),
            },
        }
        return preset
