"""
Profile persistence – save and load JSON files under ``~/.contextlens``.

Provides a simple ``ModelProfile`` dataclass and helper functions to
store and retrieve profiles as JSON.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class ModelProfile:
    """Container for model architecture and KV‑cache estimation."""

    model_id: str
    num_layers: int
    num_kv_heads: int
    head_dim: int
    dtype: str
    kv_cache_gb_per_1k_tokens: float

    def max_context_at_ram(self, ram_gb: float) -> int:
        """Return the largest context length (tokens) that fits in ``ram_gb``."""
        blocks = int(ram_gb // self.kv_cache_gb_per_1k_tokens)
        return blocks * 1000


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

PROFILE_DIR = Path.home() / ".contextlens"


def _ensure_dir() -> None:
    """Create ``PROFILE_DIR`` if it does not exist."""
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)


def _profile_path(model_id: str) -> Path:
    """Return the JSON file path for a given ``model_id``."""
    safe_name = model_id.replace("/", "_").replace(":", "_")
    return PROFILE_DIR / f"{safe_name}.json"


def save_profile(profile: ModelProfile) -> Path:
    """Write ``profile`` to ``~/.contextlens/<model_id>.json``.

    Returns:
        Path to the saved profile file
    """
    _ensure_dir()
    path = _profile_path(profile.model_id)
    with path.open("w", encoding="utf-8") as f:
        json.dump(asdict(profile), f, indent=2, sort_keys=True)
    return path


def load_profile(model_id: str) -> ModelProfile:
    """Load a saved profile for ``model_id``.

    Raises ``FileNotFoundError`` if the profile does not exist.
    """
    path = _profile_path(model_id)
    if not path.exists():
        raise FileNotFoundError(f"No profile found for '{model_id}'")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return ModelProfile(**data)


def list_profiles() -> List[ModelProfile]:
    """Return a list of all stored ``ModelProfile`` objects."""
    _ensure_dir()
    profiles: List[ModelProfile] = []
    for file in PROFILE_DIR.glob("*.json"):
        with file.open("r", encoding="utf-8") as f:
            data = json.load(f)
        profiles.append(ModelProfile(**data))
    return profiles


def delete_profile(model_id: str) -> None:
    """Delete the profile for ``model_id``.

    Raises ``FileNotFoundError`` if the profile does not exist.
    """
    path = _profile_path(model_id)
    if not path.exists():
        raise FileNotFoundError(f"No profile found for '{model_id}'")
    path.unlink()
