"""
Scanner implementation for HuggingFace models.

Fetches model architecture from HuggingFace config.json and returns a
ModelProfile instance used by the CLI.
"""

from __future__ import annotations

from typing import Optional
from pathlib import Path

from .profiles import ModelProfile

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_hf_config(model_id: str) -> dict:
    """Load model config from HuggingFace cache (no download).

    Raises FileNotFoundError if model not in cache.
    """
    from huggingface_hub import try_to_load_from_cache

    config_path = try_to_load_from_cache(
        repo_id=model_id,
        filename="config.json",
    )

    if config_path is None:
        raise FileNotFoundError(
            f"Model '{model_id}' not found in local cache.\n"
            f"Download it first:\n"
            f"  huggingface-cli download {model_id}\n"
            f"\nOr in Python:\n"
            f"  from huggingface_hub import snapshot_download\n"
            f"  snapshot_download('{model_id}')"
        )

    from transformers import AutoConfig
    try:
        config = AutoConfig.from_pretrained(
            model_id,
            local_files_only=True,
            trust_remote_code=True,
        )
        return config
    except Exception as e:
        raise FileNotFoundError(
            f"Failed to load config for '{model_id}': {e}"
        )


def _extract_architecture(config, model_id: str) -> ModelProfile:
    """Extract architecture details from HuggingFace config."""

    config_dict = config.to_dict()
    arch_type = getattr(config, 'model_type', 'unknown').lower()

    # Map architecture type to config keys
    key_mappings = {
        'llama': {'layers': 'num_hidden_layers', 'kv_heads': 'num_key_value_heads', 'heads': 'num_attention_heads', 'hidden': 'hidden_size'},
        'mistral': {'layers': 'num_hidden_layers', 'kv_heads': 'num_key_value_heads', 'heads': 'num_attention_heads', 'hidden': 'hidden_size'},
        'qwen2': {'layers': 'num_hidden_layers', 'kv_heads': 'num_key_value_heads', 'heads': 'num_attention_heads', 'hidden': 'hidden_size'},
        'gemma': {'layers': 'num_hidden_layers', 'kv_heads': 'num_key_value_heads', 'heads': 'num_attention_heads', 'hidden': 'hidden_size'},
        'phi': {'layers': 'num_hidden_layers', 'kv_heads': 'num_key_value_heads', 'heads': 'num_attention_heads', 'hidden': 'hidden_size'},
        'phi3': {'layers': 'num_hidden_layers', 'kv_heads': 'num_key_value_heads', 'heads': 'num_attention_heads', 'hidden': 'hidden_size'},
        'gpt2': {'layers': 'n_layer', 'kv_heads': 'n_head', 'heads': 'n_head', 'hidden': 'n_embd'},
        'gpt_neo': {'layers': 'num_layers', 'kv_heads': 'num_heads', 'heads': 'num_heads', 'hidden': 'hidden_size'},
        'falcon': {'layers': 'num_hidden_layers', 'kv_heads': 'num_kv_heads', 'heads': 'num_attention_heads', 'hidden': 'hidden_size'},
    }

    mapping = key_mappings.get(arch_type, key_mappings.get('llama'))

    # Extract values
    num_layers = getattr(config, mapping['layers'], 0)
    num_heads = getattr(config, mapping['heads'], 0)
    hidden_size = getattr(config, mapping['hidden'], 0)

    # KV heads - some models use same heads for QKV
    num_kv_heads = getattr(config, mapping.get('kv_heads', 'num_key_value_heads'), num_heads)
    if num_kv_heads is None:
        num_kv_heads = num_heads

    # Head dimension
    if hasattr(config, 'head_dim') and config.head_dim:
        head_dim = config.head_dim
    elif num_heads > 0:
        head_dim = hidden_size // num_heads
    else:
        head_dim = 64  # Default fallback

    # KV cache size: layers x kv_heads x head_dim x 2 bytes (FP16) x 1000 tokens
    kv_bytes_per_1k = num_layers * num_kv_heads * head_dim * 2 * 1000
    kv_gb_per_1k = kv_bytes_per_1k / (1024 ** 3)

    return ModelProfile(
        model_id=model_id,
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        dtype="float16",
        kv_cache_gb_per_1k_tokens=kv_gb_per_1k,
    )


# ---------------------------------------------------------------------------
# Public API used by the CLI
# ---------------------------------------------------------------------------

def scan_model(model_id: str) -> ModelProfile:
    """Return a ModelProfile for a HuggingFace model.

    Args:
        model_id: HuggingFace model ID (e.g., "Qwen/Qwen2-0.5B")

    Returns:
        ModelProfile with architecture details

    Raises:
        FileNotFoundError: If model not downloaded locally
        ValueError: If config cannot be parsed
    """
    config = _get_hf_config(model_id)
    return _extract_architecture(config, model_id)
