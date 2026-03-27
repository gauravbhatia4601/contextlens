"""
Unit tests for the scanner module (Phase 1/2).
"""

import pytest
from unittest.mock import patch, MagicMock

from contextlens.scanner import scan_model, _ollama_show, _parse_ollama_modelinfo
from contextlens.profiles import ModelProfile


def test_parse_ollama_modelinfo_llama():
    """Test parsing Llama model info from Ollama."""
    payload = {
        "model_info": {
            "model_id": "llama3.1:70b",
            "family": "Llama",
            "Llama_num_layers": 80,
            "Llama_num_attention_heads": 64,
            "Llama_head_dim": 128,
            "Llama_dtype": "float16",
        }
    }

    profile = _parse_ollama_modelinfo(payload, "llama3.1:70b")

    assert profile.model_id == "llama3.1:70b"
    assert profile.num_layers == 80
    assert profile.num_kv_heads == 64
    assert profile.head_dim == 128
    assert profile.dtype == "float16"
    assert profile.kv_cache_gb_per_1k_tokens > 0


def test_parse_ollama_modelinfo_mistral():
    """Test parsing Mistral model info."""
    payload = {
        "model_info": {
            "model_id": "mistral:7b",
            "family": "Mistral",
            "Mistral_num_layers": 32,
            "Mistral_num_attention_heads": 32,
            "Mistral_head_dim": 128,
            "Mistral_dtype": "float16",
        }
    }

    profile = _parse_ollama_modelinfo(payload, "mistral:7b")

    assert profile.model_id == "mistral:7b"
    assert profile.num_layers == 32


def test_parse_ollama_modelinfo_unsupported_family():
    """Test that unsupported architecture raises ValueError."""
    payload = {
        "model_info": {
            "model_id": "unknown:7b",
            "family": "UnknownArch",
            "UnknownArch_num_layers": 32,
            "UnknownArch_num_attention_heads": 32,
            "UnknownArch_head_dim": 128,
            "UnknownArch_dtype": "float16",
        }
    }

    with pytest.raises(ValueError, match="not supported"):
        _parse_ollama_modelinfo(payload, "unknown:7b")


def test_parse_ollama_modelinfo_missing_keys():
    """Test that missing keys raise ValueError."""
    payload = {
        "model_info": {
            "model_id": "incomplete:7b",
            "family": "Llama",
            # Missing required keys
        }
    }

    with pytest.raises(ValueError, match="Missing expected key"):
        _parse_ollama_modelinfo(payload, "incomplete:7b")


def test_scan_model_runtime_not_implemented():
    """Test that unsupported runtime raises NotImplementedError."""
    with pytest.raises(NotImplementedError, match="not implemented"):
        scan_model("test:7b", runtime="vllm")


@patch("contextlens.scanner.requests.post")
def test_ollama_show_success(mock_post):
    """Test successful Ollama API call."""
    mock_response = MagicMock()
    mock_response.json.return_value = {"model_info": {"family": "Llama"}}
    mock_response.raise_for_status.return_value = None
    mock_post.return_value = mock_response

    result = _ollama_show("test:7b")

    assert result == {"model_info": {"family": "Llama"}}
    mock_post.assert_called_once_with(
        "http://localhost:11434/api/show",
        json={"name": "test:7b"},
        timeout=5,
    )


@patch("contextlens.scanner.requests.post")
def test_ollama_show_connection_error(mock_post):
    """Test Ollama API connection error."""
    import requests
    mock_post.side_effect = requests.exceptions.ConnectionError("Connection refused")

    with pytest.raises(RuntimeError, match="Unable to query Ollama"):
        _ollama_show("test:7b")


def test_model_profile_max_context():
    """Test max_context_at_ram calculation."""
    profile = ModelProfile(
        model_id="test:7b",
        num_layers=32,
        num_kv_heads=32,
        head_dim=128,
        dtype="float16",
        kv_cache_gb_per_1k_tokens=0.5,
    )

    # With 0.5 GB per 1k tokens, 16 GB should give 32k tokens
    max_ctx = profile.max_context_at_ram(16.0)
    assert max_ctx == 32000


def test_model_profile_serialization():
    """Test ModelProfile can be serialized to dict."""
    profile = ModelProfile(
        model_id="test:7b",
        num_layers=32,
        num_kv_heads=32,
        head_dim=128,
        dtype="float16",
        kv_cache_gb_per_1k_tokens=0.5,
    )

    data = profile.__dict__

    assert data["model_id"] == "test:7b"
    assert data["num_layers"] == 32
