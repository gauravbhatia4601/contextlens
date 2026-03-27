"""
Ollama integration for ContextLens.

Patches Ollama Modelfiles to activate ContextLens compression at inference time.
"""

from __future__ import annotations

import subprocess
from typing import Optional, Tuple

import requests

OLLAMA_API = "http://localhost:11434"


def get_modelfile(model_name: str) -> str:
    """Fetch the Modelfile for a given model from Ollama.

    Args:
        model_name: Name of the Ollama model (e.g., "llama3.1:70b")

    Returns:
        The Modelfile content as a string

    Raises:
        RuntimeError: If Ollama is not running or model not found
    """
    url = f"{OLLAMA_API}/api/show"
    try:
        resp = requests.post(url, json={"name": model_name}, timeout=5)
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            "Ollama is not running. Start it with: ollama serve"
        ) from None
    except requests.exceptions.Timeout:
        raise RuntimeError(
            f"Ollama API timed out at {url}. Is Ollama responsive?"
        ) from None
    except Exception as exc:
        raise RuntimeError(f"Failed to fetch Modelfile from Ollama: {exc}") from exc

    return data.get("modelfile", "")


def get_model_info(model_name: str) -> dict:
    """Fetch full model info from Ollama API.

    Args:
        model_name: Name of the Ollama model

    Returns:
        Parsed JSON response with model details

    Raises:
        RuntimeError: If Ollama is not running or model not found
    """
    url = f"{OLLAMA_API}/api/show"
    try:
        resp = requests.post(url, json={"name": model_name}, timeout=5)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            "Ollama is not running. Start it with: ollama serve"
        ) from None
    except Exception as exc:
        raise RuntimeError(f"Failed to fetch model info: {exc}") from exc


def check_model_exists(model_name: str) -> bool:
    """Check if a model exists in Ollama.

    Args:
        model_name: Name of the Ollama model

    Returns:
        True if model exists, False otherwise
    """
    try:
        get_model_info(model_name)
        return True
    except RuntimeError:
        return False


def patch_modelfile(original: str, profile_path: str) -> str:
    """Inject PARAMETER contextlens_profile into Modelfile.

    Args:
        original: Original Modelfile content
        profile_path: Path to the ContextLens profile JSON

    Returns:
        Patched Modelfile content
    """
    injection = f'\nPARAMETER contextlens_profile "{profile_path}"\n'

    lines = original.splitlines()
    for i, line in enumerate(lines):
        if line.strip().upper().startswith("FROM"):
            lines.insert(i + 1, injection)
            break

    return "\n".join(lines)


def apply_to_ollama(model_name: str, profile_path: str) -> None:
    """Apply ContextLens compression to an Ollama model.

    Patches the Modelfile and recreates the model with the new configuration.

    Args:
        model_name: Name of the Ollama model
        profile_path: Path to the ContextLens profile JSON

    Raises:
        RuntimeError: If the model creation fails or Ollama is not running
    """
    if not check_model_exists(model_name):
        raise RuntimeError(
            f"Model '{model_name}' not found. Pull it first: ollama pull {model_name}"
        )

    original = get_modelfile(model_name)
    patched = patch_modelfile(original, profile_path)

    try:
        subprocess.run(
            ["ollama", "create", model_name, "-f", "-"],
            input=patched.encode(),
            capture_output=True,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode() if exc.stderr else "Unknown error"
        raise RuntimeError(f"Failed to create Ollama model: {stderr}") from exc
    except FileNotFoundError:
        raise RuntimeError(
            "Ollama CLI not found. Ensure Ollama is installed and in PATH."
        ) from None


def revert_ollama(model_name: str, original_modelfile: Optional[str] = None) -> None:
    """Remove ContextLens compression from an Ollama model.

    Args:
        model_name: Name of the Ollama model
        original_modelfile: Optional original Modelfile content for restoration

    Raises:
        RuntimeError: If the model creation fails
    """
    if not check_model_exists(model_name):
        raise RuntimeError(f"Model '{model_name}' not found")

    if original_modelfile is not None:
        try:
            subprocess.run(
                ["ollama", "create", model_name, "-f", "-"],
                input=original_modelfile.encode(),
                capture_output=True,
                check=True,
            )
        except subprocess.CalledProcessError as exc:
            stderr = exc.stderr.decode() if exc.stderr else "Unknown error"
            raise RuntimeError(f"Failed to restore original model: {stderr}") from exc
    else:
        current = get_modelfile(model_name)
        lines = [
            line
            for line in current.splitlines()
            if not (line.strip().upper().startswith("PARAMETER")
                    and "contextlens_profile" in line.lower())
        ]
        cleaned = "\n".join(lines)

        try:
            subprocess.run(
                ["ollama", "create", model_name, "-f", "-"],
                input=cleaned.encode(),
                capture_output=True,
                check=True,
            )
        except subprocess.CalledProcessError as exc:
            stderr = exc.stderr.decode() if exc.stderr else "Unknown error"
            raise RuntimeError(f"Failed to revert model: {stderr}") from exc


def backup_modelfile(model_name: str) -> str:
    """Create a backup of the current Modelfile.

    Args:
        model_name: Name of the Ollama model

    Returns:
        The backed up Modelfile content
    """
    return get_modelfile(model_name)
