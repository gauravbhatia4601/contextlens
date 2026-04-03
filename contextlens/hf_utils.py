"""
HuggingFace utilities for ContextLens.

Provides helper functions for:
- Checking if a model exists in local cache
- Checking HuggingFace authentication status
- Getting HF token from environment or cache
"""

from __future__ import annotations

import os
from typing import Optional


def get_hf_token() -> Optional[str]:
    """Get HuggingFace token from environment or cache.

    Checks in order:
    1. HF_TOKEN environment variable
    2. HUGGING_FACE_HUB_TOKEN environment variable
    3. Cached token from huggingface-cli login

    Returns:
        HF token string if found, None otherwise
    """
    # Check environment variables first
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        return token

    # Try to get cached token from huggingface-cli
    try:
        from huggingface_hub import HfFolder
        token = HfFolder.get_token()
        return token
    except ImportError:
        pass

    return None


def check_model_exists_locally(model_id: str) -> bool:
    """Check if a HuggingFace model exists in local cache.

    Uses try_to_load_from_cache to check for config.json without downloading.

    Args:
        model_id: HuggingFace model ID (e.g., "Qwen/Qwen2-0.5B")

    Returns:
        True if model config exists in cache, False otherwise
    """
    try:
        from huggingface_hub import try_to_load_from_cache

        config_path = try_to_load_from_cache(
            repo_id=model_id,
            filename="config.json",
        )
        return config_path is not None
    except ImportError:
        # If huggingface_hub not installed, assume not available
        return False


def check_hf_auth_status() -> tuple[bool, Optional[str], Optional[str]]:
    """Check HuggingFace authentication status.

    Returns:
        Tuple of (is_authenticated, username, email)
        - is_authenticated: True if logged in or token found
        - username: HF username if authenticated, None otherwise
        - email: HF email if authenticated, None otherwise
    """
    token = get_hf_token()

    if not token:
        return (False, None, None)

    # Try to validate token and get user info
    try:
        from huggingface_hub import whoami
        user_info = whoami(token=token)
        return (
            True,
            user_info.get("name"),
            user_info.get("email")
        )
    except Exception:
        # Token exists but couldn't be validated (might be invalid or network issue)
        return (True, None, None)


def ensure_model_downloaded(model_id: str) -> None:
    """Ensure a model is downloaded locally.

    Raises FileNotFoundError with download instructions if model not in cache.

    Args:
        model_id: HuggingFace model ID

    Raises:
        FileNotFoundError: If model not in local cache
    """
    if not check_model_exists_locally(model_id):
        raise FileNotFoundError(
            f"Model '{model_id}' not found in local cache.\n"
            f"Download it first:\n"
            f"  huggingface-cli download {model_id}\n"
            f"\nOr in Python:\n"
            f"  from huggingface_hub import snapshot_download\n"
            f"  snapshot_download('{model_id}')"
        )


def check_gated_model_access(model_id: str) -> tuple[bool, Optional[str]]:
    """Check if a gated model is accessible.

    Args:
        model_id: HuggingFace model ID

    Returns:
        Tuple of (is_accessible, error_message)
        - is_accessible: True if model can be accessed
        - error_message: Error message if not accessible, None otherwise
    """
    try:
        from huggingface_hub import HfApi
        from huggingface_hub.utils import RepositoryNotFoundError, GatedRepoError

        api = HfApi()
        token = get_hf_token()

        try:
            # Try to get model info - will fail if gated and no access
            api.model_info(model_id, token=token)
            return (True, None)
        except GatedRepoError:
            return (
                False,
                f"Model '{model_id}' is gated. You need to:\n"
                f"  1. Log in: huggingface-cli login\n"
                f"  2. Accept the model license at:\n"
                f"     https://huggingface.co/settings/accepted\n"
                f"  3. Request access if required by the model author"
            )
        except RepositoryNotFoundError:
            return (False, f"Model '{model_id}' not found on HuggingFace")
        except Exception as e:
            return (False, f"Error checking model access: {e}")

    except ImportError:
        return (False, "huggingface_hub not installed")
