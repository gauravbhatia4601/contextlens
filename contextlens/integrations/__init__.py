"""
Integrations package for ContextLens.

Provides HuggingFace Transformers integration with TurboQuant compression.
"""

from .huggingface import (
    patch_model_for_contextlens,
    unpatch_model,
    is_model_patched,
    ContextLensKVCache,
)
from ..hf_utils import (
    get_hf_token,
    check_model_exists_locally,
    check_hf_auth_status,
    ensure_model_downloaded,
    check_gated_model_access,
)

__all__ = [
    # HuggingFace integration
    "patch_model_for_contextlens",
    "unpatch_model",
    "is_model_patched",
    "ContextLensKVCache",
    # HF utilities
    "get_hf_token",
    "check_model_exists_locally",
    "check_hf_auth_status",
    "ensure_model_downloaded",
    "check_gated_model_access",
]
