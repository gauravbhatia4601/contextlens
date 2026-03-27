"""
Scanner implementation (Phase 1) – Ollama only.
It contacts the local Ollama API, extracts model architecture details, and returns a
``ModelProfile`` instance used by the CLI.
"""

from __future__ import annotations

import json
from typing import Any, Dict

import requests

from .profiles import ModelProfile

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ollama_show(model_name: str) -> Dict[str, Any]:
    """Call ``POST /api/show`` on the local Ollama server.

    Returns the parsed JSON dictionary.  Any network‑level problem raises a
    ``RuntimeError`` with a concise, user‑friendly message.
    """
    url = "http://localhost:11434/api/show"
    try:
        resp = requests.post(url, json={"name": model_name}, timeout=5)
        resp.raise_for_status()
    except Exception as exc:
        raise RuntimeError(f"Unable to query Ollama at {url}: {exc}") from exc
    return resp.json()


def _parse_ollama_modelinfo(payload: Dict[str, Any], model_name: str) -> ModelProfile:
    """Extract the required fields from Ollama's ``modelinfo`` payload.

    The payload contains a ``model_info`` dict with keys that are prefixed by the
    model family (e.g. ``Llama_num_layers``).  Supported families are listed in
    ``SUPPORTED_FAMILIES``; an unknown family raises ``ValueError``.
    """
    model_info = payload.get("model_info")
    if not isinstance(model_info, dict):
        raise ValueError("Ollama response missing 'model_info' dict")

    # Identify the family – Ollama includes a ``family`` key.
    family = model_info.get("family")
    if not isinstance(family, str):
        raise ValueError("Model family information missing from Ollama response")

    SUPPORTED_FAMILIES = {
        "Llama",
        "Mistral",
        "Phi-3",
        "Gemma",
        "Qwen2",
    }
    if family not in SUPPORTED_FAMILIES:
        raise ValueError(f"Architecture '{family}' is not supported in Phase 1")

    # Keys we need – they follow the pattern ``<Family>_num_layers`` etc.
    prefix = family
    try:
        num_layers = int(model_info[f"{prefix}_num_layers"])
        num_kv_heads = int(model_info[f"{prefix}_num_attention_heads"])
        head_dim = int(model_info[f"{prefix}_head_dim"])
        dtype = str(model_info[f"{prefix}_dtype"])
    except KeyError as exc:
        raise ValueError(f"Missing expected key {exc} in Ollama model info") from exc

    # Rough KV‑cache size per 1k tokens (FP16 = 2 bytes per value).
    # KV cache per token = layers * heads * head_dim * 2 bytes.
    # Convert to GB for a 1 k‑token block.
    kv_bytes_per_1k = num_layers * num_kv_heads * head_dim * 2 * 1000
    kv_gb_per_1k = kv_bytes_per_1k / (1024 ** 3)  # bytes → GiB

    model_id = model_info.get("model_id", model_name)  # fallback to arg if missing
    return ModelProfile(
        model_id=model_id,
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        dtype=dtype,
        kv_cache_gb_per_1k_tokens=kv_gb_per_1k,
    )

# ---------------------------------------------------------------------------
# Public API used by the CLI
# ---------------------------------------------------------------------------

def scan_model(model_name: str, runtime: str = "auto") -> ModelProfile:
    """Return a ``ModelProfile`` for ``model_name``.

    Phase 1 only supports ``ollama`` (or ``auto`` which defaults to Ollama).
    ``runtime`` values other than these raise ``NotImplementedError`` so the CLI
    can report a clear error.
    """
    if runtime not in {"auto", "ollama"}:
        raise NotImplementedError(
            f"Runtime '{runtime}' not implemented in Phase 1 – only Ollama is supported"
        )
    payload = _ollama_show(model_name)
    return _parse_ollama_modelinfo(payload, model_name)

# ---------------------------------------------------------------------------
# End of file
# ---------------------------------------------------------------------------
