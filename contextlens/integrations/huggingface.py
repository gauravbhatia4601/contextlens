"""
HuggingFace Transformers integration for ContextLens.

Monkey-patches the attention forward() in each layer to intercept
KV tensors and apply TurboQuant compression/decompression inline.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import torch

from contextlens.compressor import TurboQuantCompressor, CompressedKCache, CompressedVCache


class ContextLensKVCache:
    """Wrapper for compressed KV cache that decompresses on access.
    
    This class allows seamless integration with HuggingFace's cache system
    by providing a drop-in replacement for the standard KV cache tensors.
    """
    
    def __init__(
        self,
        ck: CompressedKCache,
        cv: CompressedVCache,
        compressor: TurboQuantCompressor,
    ):
        self.ck = ck
        self.cv = cv
        self.compressor = compressor
        self._k_cache: Optional[torch.Tensor] = None
        self._v_cache: Optional[torch.Tensor] = None
    
    def get_keys(self) -> torch.Tensor:
        """Decompress and return the key cache."""
        if self._k_cache is None:
            self._k_cache = self.compressor.decompress_k_cache(self.ck)
        return self._k_cache
    
    def get_values(self) -> torch.Tensor:
        """Decompress and return the value cache."""
        if self._v_cache is None:
            self._v_cache = self.compressor.decompress_v_cache(self.cv)
        return self._v_cache
    
    @property
    def key_shape(self) -> tuple:
        """Return the original key cache shape."""
        return self.ck.original_shape
    
    @property
    def value_shape(self) -> tuple:
        """Return the original value cache shape."""
        return self.cv.original_shape


def make_compressed_forward(
    original_forward: Callable,
    compressor: TurboQuantCompressor,
    layer_idx: int,
) -> Callable:
    """Create a wrapped forward function that compresses KV cache.

    Args:
        original_forward: Original attention forward method
        compressor: TurboQuantCompressor instance
        layer_idx: Layer index for seeding

    Returns:
        Wrapped forward function
    """
    def compressed_forward(*args, **kwargs):
        """Forward pass with KV cache compression."""
        # Extract past_key_values if present
        past_key_values = kwargs.get("past_key_values", None)
        
        # Call original forward
        outputs = original_forward(*args, **kwargs)
        
        # If we have new key/value states, compress them
        if hasattr(outputs, "past_key_value") and outputs.past_key_value is not None:
            key, value = outputs.past_key_value
            
            if isinstance(key, torch.Tensor) and isinstance(value, torch.Tensor):
                # Compress the new KV states
                ck = compressor.compress_k_cache(key, layer_idx)
                cv = compressor.compress_v_cache(value, layer_idx)
                
                # Wrap in our compressed cache class
                compressed_cache = ContextLensKVCache(ck, cv, compressor)
                outputs.past_key_value = compressed_cache
        
        return outputs

    return compressed_forward


def patch_model_for_contextlens(model: Any) -> Any:
    """Monkey-patch a HuggingFace model to use ContextLens compression.

    Args:
        model: A HuggingFace AutoModelForCausalLM instance

    Returns:
        The same model instance, patched in-place
    """
    compressor = TurboQuantCompressor(bits=3)

    # Detect model architecture and get the layers
    layers = _get_model_layers(model)
    
    if layers is None:
        raise ValueError(
            f"Could not detect model layers for {type(model).__name__}. "
            "This architecture is not yet supported."
        )

    # Patch each layer's attention mechanism
    for layer_idx, layer in enumerate(layers):
        _patch_layer_attention(layer, compressor, layer_idx)

    return model


def _get_model_layers(model: Any) -> Optional[list]:
    """Extract the layer list from various model architectures.

    Args:
        model: HuggingFace model

    Returns:
        List of transformer layers or None if not found
    """
    # Common patterns for different architectures
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers  # Llama, Mistral, etc.
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h  # GPT-2, etc.
    elif hasattr(model, "layers"):
        return model.layers  # Some other architectures
    elif hasattr(model, "encoder") and hasattr(model.encoder, "layer"):
        return model.encoder.layer  # Encoder-only models
    return None


def _patch_layer_attention(
    layer: Any,
    compressor: TurboQuantCompressor,
    layer_idx: int,
) -> None:
    """Patch the attention module in a single layer.

    Args:
        layer: Transformer layer
        compressor: TurboQuantCompressor instance
        layer_idx: Layer index
    """
    # Find the attention module
    self_attn = None
    
    if hasattr(layer, "self_attn"):
        self_attn = layer.self_attn  # Llama, Mistral
    elif hasattr(layer, "self_attention"):
        self_attn = layer.self_attention  # Some other architectures
    elif hasattr(layer, "attn"):
        self_attn = layer.attn  # GPT-style
    
    if self_attn is None:
        return
    
    # Store original forward if not already stored
    if not hasattr(self_attn, "_contextlens_original_forward"):
        self_attn._contextlens_original_forward = self_attn.forward
        
        # Wrap with compression
        self_attn.forward = make_compressed_forward(
            self_attn._contextlens_original_forward,
            compressor,
            layer_idx,
        )


def unpatch_model(model: Any) -> Any:
    """Remove ContextLens patches from a model.

    Args:
        model: A previously patched model

    Returns:
        The model with original forward methods restored
    """
    layers = _get_model_layers(model)
    
    if layers is None:
        return model

    for layer in layers:
        self_attn = None
        
        if hasattr(layer, "self_attn"):
            self_attn = layer.self_attn
        elif hasattr(layer, "self_attention"):
            self_attn = layer.self_attention
        elif hasattr(layer, "attn"):
            self_attn = layer.attn
        
        if self_attn is None:
            continue
        
        # Restore original forward if it was saved
        if hasattr(self_attn, "_contextlens_original_forward"):
            self_attn.forward = self_attn._contextlens_original_forward
            delattr(self_attn, "_contextlens_original_forward")

    return model


def is_model_patched(model: Any) -> bool:
    """Check if a model has been patched with ContextLens.

    Args:
        model: HuggingFace model to check

    Returns:
        True if patched, False otherwise
    """
    layers = _get_model_layers(model)
    
    if layers is None:
        return False
    
    for layer in layers:
        self_attn = None
        
        if hasattr(layer, "self_attn"):
            self_attn = layer.self_attn
        elif hasattr(layer, "self_attention"):
            self_attn = layer.self_attention
        elif hasattr(layer, "attn"):
            self_attn = layer.attn
        
        if self_attn is not None and hasattr(self_attn, "_contextlens_original_forward"):
            return True
    
    return False
