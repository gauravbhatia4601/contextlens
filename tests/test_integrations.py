"""
Unit tests for integration modules (Phase 4).
"""

import pytest
from unittest.mock import patch, MagicMock

from contextlens.integrations.ollama import (
    get_modelfile,
    patch_modelfile,
    check_model_exists,
)
from contextlens.integrations.llamacpp import (
    generate_flags,
    write_config_file,
    check_llama_cpp_installed,
)
from contextlens.integrations.huggingface import (
    patch_model_for_contextlens,
    unpatch_model,
    is_model_patched,
)
from contextlens.profiles import ModelProfile


# -----------------------------------------------------------------------------
# Ollama Integration Tests
# -----------------------------------------------------------------------------

@patch("contextlens.integrations.ollama.requests.post")
def test_get_modelfile_success(mock_post):
    """Test successful Modelfile retrieval."""
    mock_response = MagicMock()
    mock_response.json.return_value = {"modelfile": "FROM llama3.1:70b"}
    mock_response.raise_for_status.return_value = None
    mock_post.return_value = mock_response

    result = get_modelfile("llama3.1:70b")

    assert result == "FROM llama3.1:70b"
    mock_post.assert_called_once()


@patch("contextlens.integrations.ollama.requests.post")
def test_get_modelfile_ollama_not_running(mock_post):
    """Test error when Ollama is not running."""
    import requests
    mock_post.side_effect = requests.exceptions.ConnectionError()

    with pytest.raises(RuntimeError, match="Ollama is not running"):
        get_modelfile("llama3.1:70b")


def test_patch_modelfile():
    """Test Modelfile patching."""
    original = """FROM llama3.1:70b
SYSTEM You are a helpful assistant.
"""
    result = patch_modelfile(original, "/path/to/profile.json")

    assert 'PARAMETER contextlens_profile "/path/to/profile.json"' in result
    assert "FROM llama3.1:70b" in result


def test_patch_modelfile_injection_position():
    """Test that injection happens right after FROM line."""
    original = """FROM model
SYSTEM test
PARAMETER temp 0.5
"""
    result = patch_modelfile(original, "/profile.json")
    lines = result.split("\n")
    
    from_idx = next(i for i, line in enumerate(lines) if line.startswith("FROM"))
    param_idx = next(i for i, line in enumerate(lines) if "contextlens_profile" in line)
    
    assert param_idx == from_idx + 1


# -----------------------------------------------------------------------------
# llama.cpp Integration Tests
# -----------------------------------------------------------------------------

def test_generate_flags():
    """Test flag generation for llama.cpp."""
    profile = ModelProfile(
        model_id="test:7b",
        num_layers=32,
        num_kv_heads=32,
        head_dim=128,
        dtype="float16",
        kv_cache_gb_per_1k_tokens=0.5,
    )

    flags = generate_flags(profile, reuse_cache=256)

    assert "--cache-type-k q3_K" in flags
    assert "--cache-type-v q3_K" in flags
    assert "--cache-reuse 256" in flags


def test_generate_flags_no_reuse():
    """Test flag generation without cache reuse."""
    profile = ModelProfile(
        model_id="test:7b",
        num_layers=32,
        num_kv_heads=32,
        head_dim=128,
        dtype="float16",
        kv_cache_gb_per_1k_tokens=0.5,
    )

    flags = generate_flags(profile, reuse_cache=0)

    assert "--cache-type-k q3_K" in flags
    assert "--cache-reuse" not in flags


@patch("contextlens.integrations.llamacpp.Path")
def test_write_config_file(mock_path):
    """Test config file writing."""
    mock_file = MagicMock()
    mock_path.return_value.expanduser.return_value = mock_file
    mock_file.parent = MagicMock()

    profile = ModelProfile(
        model_id="test:7b",
        num_layers=32,
        num_kv_heads=32,
        head_dim=128,
        dtype="float16",
        kv_cache_gb_per_1k_tokens=0.5,
    )

    write_config_file(profile, "/tmp/test.conf")

    mock_file.write_text.assert_called_once()
    content = mock_file.write_text.call_args[0][0]
    assert "test:7b" in content
    assert "--cache-type-k q3_K" in content


def test_check_llama_cpp_installed():
    """Test llama-cpp-python installation check."""
    result = check_llama_cpp_installed()
    assert isinstance(result, bool)


# -----------------------------------------------------------------------------
# HuggingFace Integration Tests
# -----------------------------------------------------------------------------

def test_is_model_patched_unpatched():
    """Test that unpatched model returns False."""
    mock_model = MagicMock()
    mock_model.model.layers = []
    
    result = is_model_patched(mock_model)
    assert result is False


def test_unpatch_model_no_layers():
    """Test unpatching model without detectable layers."""
    mock_model = MagicMock(spec=[])
    
    result = unpatch_model(mock_model)
    assert result is mock_model


# -----------------------------------------------------------------------------
# Integration Test Helpers
# -----------------------------------------------------------------------------

def test_model_profile_for_integration():
    """Test ModelProfile creation for integration tests."""
    profile = ModelProfile(
        model_id="llama3.1:8b",
        num_layers=32,
        num_kv_heads=8,
        head_dim=128,
        dtype="float16",
        kv_cache_gb_per_1k_tokens=0.1,
    )

    assert profile.model_id == "llama3.1:8b"
    assert profile.num_layers == 32
    assert profile.max_context_at_ram(16.0) > 0
