#!/usr/bin/env python3
"""
TurboQuant Compression Profiler

Measures actual memory savings from TurboQuant compression
compared to uncompressed KV cache.
"""

import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from contextlens.compressor import TurboQuantCompressor, CompressedKCache, CompressedVCache


def get_tensor_memory(tensor: torch.Tensor) -> int:
    """Get memory usage of a tensor in bytes."""
    return tensor.numel() * tensor.element_size()


def format_bytes(bytes_val: int) -> str:
    """Format bytes to human readable string."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_val < 1024:
            return f"{bytes_val:.2f} {unit}"
        bytes_val /= 1024
    return f"{bytes_val:.2f} TB"


def profile_compression(
    batch_size: int = 1,
    seq_len: int = 512,
    num_kv_heads: int = 8,
    head_dim: int = 64,
    num_layers: int = 16,
):
    """Profile TurboQuant compression for given model dimensions."""

    print("=" * 70)
    print("TurboQuant Compression Profiler")
    print("=" * 70)
    print(f"\nModel Configuration:")
    print(f"  Batch Size: {batch_size}")
    print(f"  Sequence Length: {seq_len} tokens")
    print(f"  KV Heads: {num_kv_heads}")
    print(f"  Head Dimension: {head_dim}")
    print(f"  Layers: {num_layers}")
    print()

    # Create sample KV tensors (simulating a model's KV cache)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    # Initialize compressor
    compressor = TurboQuantCompressor(bits=3.5)

    total_original_k = 0
    total_compressed_k = 0
    total_original_v = 0
    total_compressed_v = 0

    print("-" * 70)
    print("Per-Layer Compression Analysis")
    print("-" * 70)
    print(f"{'Layer':<8} {'K Original':<14} {'K Compressed':<14} {'K Ratio':<10} {'V Original':<14} {'V Compressed':<14} {'V Ratio':<10}")
    print("-" * 90)

    for layer_idx in range(num_layers):
        # Create K and V cache tensors
        # Shape: (batch, seq_len, num_kv_heads, head_dim)
        k_cache = torch.randn(
            batch_size, seq_len, num_kv_heads, head_dim,
            dtype=torch.float32, device=device
        )
        v_cache = torch.randn(
            batch_size, seq_len, num_kv_heads, head_dim,
            dtype=torch.float32, device=device
        )

        # Measure original sizes
        k_original_bytes = get_tensor_memory(k_cache)
        v_original_bytes = get_tensor_memory(v_cache)

        # Compress
        start = time.time()
        compressed_k = compressor.compress_k_cache(k_cache, layer_idx)
        compressed_v = compressor.compress_v_cache(v_cache, layer_idx)
        compress_time = time.time() - start

        # Measure compressed sizes
        k_compressed_bytes = compressor.get_compressed_size(compressed_k)
        v_compressed_bytes = compressor.get_compressed_size(compressed_v)

        # Accumulate totals
        total_original_k += k_original_bytes
        total_original_v += v_original_bytes
        total_compressed_k += k_compressed_bytes
        total_compressed_v += v_compressed_bytes

        k_ratio = k_compressed_bytes / k_original_bytes if k_original_bytes > 0 else 0
        v_ratio = v_compressed_bytes / v_original_bytes if v_original_bytes > 0 else 0

        print(f"{layer_idx:<8} {format_bytes(k_original_bytes):<14} {format_bytes(k_compressed_bytes):<14} {k_ratio:.2f}x      {format_bytes(v_original_bytes):<14} {format_bytes(v_compressed_bytes):<14} {v_ratio:.2f}x")

    print("-" * 90)

    # Calculate totals
    total_original = total_original_k + total_original_v
    total_compressed = total_compressed_k + total_compressed_v
    overall_ratio = total_original / total_compressed if total_compressed > 0 else 0

    k_overall_ratio = total_original_k / total_compressed_k if total_compressed_k > 0 else 0
    v_overall_ratio = total_original_v / total_compressed_v if total_compressed_v > 0 else 0

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n{'Metric':<35} {'Value':<20}")
    print("-" * 55)
    print(f"{'K-Cache Original':<35} {format_bytes(total_original_k)}")
    print(f"{'K-Cache Compressed':<35} {format_bytes(total_compressed_k)}")
    print(f"{'K-Cache Compression Ratio':<35} {k_overall_ratio:.2f}x")
    print()
    print(f"{'V-Cache Original':<35} {format_bytes(total_original_v)}")
    print(f"{'V-Cache Compressed':<35} {format_bytes(total_compressed_v)}")
    print(f"{'V-Cache Compression Ratio':<35} {v_overall_ratio:.2f}x")
    print()
    print(f"{'TOTAL Original KV Cache':<35} {format_bytes(total_original)}")
    print(f"{'TOTAL Compressed KV Cache':<35} {format_bytes(total_compressed)}")
    print(f"{'TOTAL Compression Ratio':<35} {overall_ratio:.2f}x")
    print()
    print(f"{'Memory Saved':<35} {format_bytes(total_original - total_compressed)}")
    print(f"{'Memory Reduction':<35} {(1 - total_compressed/total_original)*100:.1f}%")
    print("=" * 70)

    # Scale to different context lengths
    print("\n" + "=" * 70)
    print("KV Cache Memory at Different Context Lengths")
    print("=" * 70)
    print(f"{'Context Length':<18} {'Original KV':<18} {'Compressed KV':<18} {'Saved':<18}")
    print("-" * 72)

    for ctx_len in [256, 512, 1024, 2048, 4096, 8192, 16384, 32768]:
        scale_factor = ctx_len / seq_len
        orig_scaled = total_original * scale_factor
        comp_scaled = total_compressed * scale_factor
        saved_scaled = orig_scaled - comp_scaled
        print(f"{ctx_len:<18} {format_bytes(int(orig_scaled)):<18} {format_bytes(int(comp_scaled)):<18} {format_bytes(int(saved_scaled))}")

    print("=" * 70)

    return {
        'original_bytes': total_original,
        'compressed_bytes': total_compressed,
        'compression_ratio': overall_ratio,
        'k_ratio': k_overall_ratio,
        'v_ratio': v_overall_ratio,
    }


if __name__ == "__main__":
    # Profile with qwen2:0.5b configuration
    print("\n*** Profiling for Qwen2-0.5B ***\n")
    profile_compression(
        batch_size=1,
        seq_len=512,
        num_kv_heads=2,      # Qwen2-0.5B has 2 KV heads
        head_dim=64,         # Head dimension
        num_layers=24,       # 24 layers
    )

    print("\n\n*** Profiling for Llama-3.2-1B ***\n")
    profile_compression(
        batch_size=1,
        seq_len=512,
        num_kv_heads=8,      # Llama-3.2-1B has 8 KV heads
        head_dim=64,
        num_layers=16,       # 16 layers
    )
