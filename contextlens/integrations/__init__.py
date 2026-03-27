"""
Integrations package for ContextLens.

Provides runtime-specific integration modules for:
- Ollama
- llama.cpp  
- HuggingFace Transformers
"""

from .ollama import (
    get_modelfile,
    patch_modelfile,
    apply_to_ollama,
    revert_ollama,
    get_model_info,
    check_model_exists,
    backup_modelfile,
)
from .llamacpp import (
    generate_flags,
    write_config_file,
    generate_llama_cpp_python_hook,
    get_llama_cpp_version,
    check_llama_cpp_installed,
)
from .huggingface import (
    patch_model_for_contextlens,
    unpatch_model,
    is_model_patched,
    ContextLensKVCache,
)

__all__ = [
    # Ollama
    "get_modelfile",
    "patch_modelfile",
    "apply_to_ollama",
    "revert_ollama",
    "get_model_info",
    "check_model_exists",
    "backup_modelfile",
    # llama.cpp
    "generate_flags",
    "write_config_file",
    "generate_llama_cpp_python_hook",
    "get_llama_cpp_version",
    "check_llama_cpp_installed",
    # HuggingFace
    "patch_model_for_contextlens",
    "unpatch_model",
    "is_model_patched",
    "ContextLensKVCache",
]
