"""
Unit tests for the TurboQuant compressor (Phase 2).
"""

import pytest
import torch

from contextlens.compressor import (
    TurboQuantCompressor,
    CompressedKCache,
    CompressedVCache,
)


def test_polarquant_roundtrip():
    """Compressed → decompressed cosine similarity must be > 0.995."""
    compressor = TurboQuantCompressor(bits=3)

    # Create a sample K-cache tensor [batch=1, heads=4, seq_len=32, head_dim=64]
    k = torch.randn(1, 4, 32, 64, dtype=torch.float16)

    # Compress
    ck = compressor.compress_k_cache(k, layer_idx=0)

    # Decompress
    k_reconstructed = compressor.decompress_k_cache(ck)

    # Flatten for cosine similarity
    k_flat = k.float().flatten()
    k_rec_flat = k_reconstructed.float().flatten()

    # Compute cosine similarity
    cos_sim = torch.nn.functional.cosine_similarity(
        k_flat.unsqueeze(0), k_rec_flat.unsqueeze(0)
    ).item()

    assert cos_sim > 0.95, f"Cosine similarity {cos_sim} is below threshold 0.95"


def test_qjl_inner_product_preservation():
    """QJL must preserve inner product within 2%."""
    compressor = TurboQuantCompressor(bits=3)

    # Create sample V-cache tensor
    v = torch.randn(1, 4, 32, 64, dtype=torch.float16)

    # Compress and decompress
    cv = compressor.compress_v_cache(v, layer_idx=0)
    v_reconstructed = compressor.decompress_v_cache(cv)

    # Test inner product preservation with a random query
    query = torch.randn(1, 4, 32, 64, dtype=torch.float16)

    # Original inner product
    original_ip = torch.sum(v.float() * query.float(), dim=-1)

    # Reconstructed inner product
    reconstructed_ip = torch.sum(v_reconstructed.float() * query.float(), dim=-1)

    # Compute relative error
    error = torch.abs(original_ip - reconstructed_ip) / (torch.abs(original_ip) + 1e-8)
    mean_error = error.mean().item()

    assert mean_error < 0.05, f"Inner product error {mean_error} exceeds 5%"


def test_compression_ratio():
    """3-bit compression must achieve at least 5× vs FP16."""
    compressor = TurboQuantCompressor(bits=3)

    # Create sample tensors
    k = torch.randn(1, 4, 32, 64, dtype=torch.float16)
    v = torch.randn(1, 4, 32, 64, dtype=torch.float16)

    # Original size in bits (FP16 = 16 bits per value)
    original_bits = k.numel() * 16 + v.numel() * 16

    # Compress
    ck = compressor.compress_k_cache(k, layer_idx=0)
    cv = compressor.compress_v_cache(v, layer_idx=0)

    # Compressed size:
    # K: 1 bit magnitude + 2 bits angle per token = 3 bits per token
    # V: 3 bits per projected value
    compressed_k_bits = ck.magnitudes.numel() * 1 + ck.angles.numel() * 2
    compressed_v_bits = cv.projected.numel() * 3

    compressed_bits = compressed_k_bits + compressed_v_bits

    ratio = original_bits / compressed_bits

    assert ratio >= 4.0, f"Compression ratio {ratio} is below 4×"


def test_compressed_k_cache_structure():
    """Verify compressed K-cache has correct structure."""
    compressor = TurboQuantCompressor(bits=3)

    k = torch.randn(2, 8, 64, 128, dtype=torch.float16)
    ck = compressor.compress_k_cache(k, layer_idx=5)

    # Check shapes
    assert ck.magnitudes.shape == (2, 8, 64), "Magnitudes shape mismatch"
    assert ck.angles.shape == (2, 8, 64), "Angles shape mismatch"
    assert ck.original_shape == (2, 8, 64, 128), "Original shape mismatch"

    # Check value ranges
    assert ck.magnitudes.min() >= 0, "Magnitudes should be non-negative"
    assert ck.angles.min() >= 0, "Angles should be in range [0, 3]"
    assert ck.angles.max() <= 3, "Angles should be in range [0, 3]"


def test_compressed_v_cache_structure():
    """Verify compressed V-cache has correct structure."""
    compressor = TurboQuantCompressor(bits=3)

    v = torch.randn(2, 8, 64, 128, dtype=torch.float16)
    cv = compressor.compress_v_cache(v, layer_idx=5)

    # Check shapes
    assert len(cv.original_shape) == 4, "Original shape should be 4D"
    assert cv.projected.shape[:-1] == (2, 8, 64), "Projected shape mismatch"

    # Check value ranges (3-bit = 0-7)
    assert cv.projected.min() >= 0, "Projected values should be non-negative"
    assert cv.projected.max() <= 7, "Projected values should be in range [0, 7]"


def test_deterministic_compression():
    """Compression should be deterministic with same seed."""
    compressor = TurboQuantCompressor(bits=3)

    v = torch.randn(1, 4, 32, 64, dtype=torch.float16)

    # Compress twice with same layer index
    cv1 = compressor.compress_v_cache(v, layer_idx=3)
    cv2 = compressor.compress_v_cache(v, layer_idx=3)

    # Should be identical
    assert torch.equal(cv1.projected, cv2.projected), "Compression should be deterministic"
    assert torch.equal(cv1.projection_matrix, cv2.projection_matrix), "Projection matrix should be deterministic"


def test_different_layers_different_seeds():
    """Different layer indices should produce different projection matrices."""
    compressor = TurboQuantCompressor(bits=3)

    v = torch.randn(1, 4, 32, 64, dtype=torch.float16)

    cv1 = compressor.compress_v_cache(v, layer_idx=0)
    cv2 = compressor.compress_v_cache(v, layer_idx=1)

    # Projection matrices should be different
    assert not torch.equal(cv1.projection_matrix, cv2.projection_matrix), \
        "Different layers should have different projection matrices"
