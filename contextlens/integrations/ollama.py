"""
Ollama integration for ContextLens.

Patches Ollama Modelfiles to activate ContextLens compression at inference time.
Supports both legacy Ollama (< v0.5) and modern blob-based storage (≥ v0.5).
"""

from __future__ import annotations

import subprocess
from typing import Optional, Tuple

import requests
from rich.console import Console

OLLAMA_API = "http://localhost:11434"
console = Console()


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

    Uses Ollama API to create a new derived model with contextlens_profile parameter.
    Works with both legacy and blob-based storage (Ollama v0.5+).

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

    # Create a NEW model name with "-contextlens" suffix
    new_model_name = f"{model_name}-contextlens"
    
    console.print(f"\n[bold blue]Creating compressed model: {new_model_name}[/bold blue]")
    console.print(f"[dim]Profile: {profile_path}[/dim]\n")

    # Build the Modelfile content
    modelfile_content = f"FROM {model_name}\nPARAMETER contextlens_profile \"{profile_path}\"\n"
    
    # Use the API instead of CLI for better compatibility
    url = f"{OLLAMA_API}/api/create"
    try:
        resp = requests.post(
            url,
            json={
                "model": new_model_name,
                "from": model_name,
                "modelfile": modelfile_content,
                "stream": False
            },
            timeout=300
        )
        
        # Check for errors
        if resp.status_code != 200:
            error_data = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {"error": resp.text}
            raise RuntimeError(f"API error: {error_data.get('error', 'Unknown error')}")
        
        console.print(f"\n[bold green]✓ Success![/bold green]")
        console.print(f"\nCompressed model created: [cyan]{new_model_name}[/cyan]")
        console.print(f"\n[dim]To use the compressed model:[/dim]")
        console.print(f"  ollama run {new_model_name}")
        console.print(f"\n[dim]To revert to original:[/dim]")
        console.print(f"  ollama rm {new_model_name}")
        console.print(f"  (original {model_name} is unchanged)\n")
        
    except requests.exceptions.Timeout:
        raise RuntimeError(
            "Ollama API timed out. The model may be too large."
        ) from None
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            "Cannot connect to Ollama. Is it running? Start with: ollama serve"
        ) from None
    except Exception as exc:
        raise RuntimeError(f"Failed to create Ollama model via API: {exc}") from exc


def get_ollama_version() -> str:
    """Get the Ollama version string."""
    try:
        resp = requests.get(f"{OLLAMA_API}/api/version", timeout=5)
        resp.raise_for_status()
        return resp.json().get("version", "unknown")
    except Exception:
        return "unknown"


def revert_ollama(model_name: str, original_modelfile: Optional[str] = None) -> None:
    """Remove ContextLens compression from an Ollama model.

    Args:
        model_name: Name of the Ollama model
        original_modelfile: Optional original Modelfile content for restoration

    Raises:
        RuntimeError: If the model creation fails
    """
    # Check if the contextlens variant exists
    contextlens_variant = f"{model_name}-contextlens"
    
    if check_model_exists(contextlens_variant):
        console.print(f"[bold yellow]Removing compressed variant: {contextlens_variant}[/bold yellow]")
        try:
            subprocess.run(
                ["ollama", "rm", contextlens_variant],
                capture_output=True,
                check=True,
            )
            console.print(f"[green]✓ Removed {contextlens_variant}[/green]")
            console.print(f"[dim]Original model {model_name} is still available.[/dim]")
            return
        except subprocess.CalledProcessError as exc:
            stderr = exc.stderr.decode() if exc.stderr else "Unknown error"
            raise RuntimeError(f"Failed to remove compressed variant: {stderr}") from exc
    
    # If no variant exists, try to clean the original model
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
