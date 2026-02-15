"""
pipeline/config.py — Configuration loading and image preparation
================================================================

Public API:
  load_config(path)             — load config.yaml → dict
  prepare_image(path, w, h)     — load + resize image to exact dims
  snap_to_mod(value, mod)       — round down to nearest multiple

Leaf module — no project-internal dependencies.
"""

import yaml
from PIL import Image


def load_config(path: str = "config.yaml") -> dict:
    """Load YAML configuration file."""
    with open(path) as f:
        return yaml.safe_load(f)


def prepare_image(image_path: str, width: int, height: int) -> Image.Image:
    """Load and resize image to exact target dimensions."""
    image = Image.open(image_path).convert("RGB")
    image = image.resize((width, height), Image.LANCZOS)
    return image


def snap_to_mod(value: int, mod: int = 16) -> int:
    """Round down to nearest multiple of mod."""
    return (value // mod) * mod
