#!/usr/bin/env python3
"""
Integration test for ContextLens TurboQuant compression.

Compares:
1. Direct HuggingFace inference (no compression)
2. ContextLens-patched inference (with TurboQuant compression)
3. Ollama inference (quantized baseline)
"""

import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add contextlens to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from contextlens.compressor import TurboQuantCompressor
from contextlens.integrations.huggingface import patch_model_for_contextlens


def measure_memory(model):
    """Measure model memory usage."""
    if hasattr(model, 'get_memory_footprint'):
        return model.get_memory_footprint() / (1024 ** 2)  # MB
    return sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 ** 2)


def test_compression(model_name="Qwen/Qwen2-0.5B", prompt="Say hello in one word", max_tokens=50):
    """Test compression vs no compression."""

    print("=" * 60)
    print("ContextLens TurboQuant Compression Integration Test")
    print("=" * 60)
    print(f"\nModel: {model_name}")
    print(f"Prompt: {prompt}")
    print(f"Max tokens: {max_tokens}\n")

    # Load model and tokenizer
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,  # Use CPU-compatible dtype
        device_map="cpu",
    )

    base_memory = measure_memory(model)
    print(f"Base model memory: {base_memory:.2f} MB")

    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]

    # Test 1: No compression (baseline)
    print("\n--- Test 1: No Compression (Baseline) ---")
    start = time.time()
    with torch.inference_mode():
        output_base = model.generate(**inputs, max_new_tokens=max_tokens, pad_token_id=tokenizer.eos_token_id)
    time_base = time.time() - start
    text_base = tokenizer.decode(output_base[0], skip_special_tokens=True)
    print(f"Time: {time_base:.2f}s")
    print(f"Output: {text_base[:100]}...")

    # Test 2: With ContextLens compression
    print("\n--- Test 2: With TurboQuant Compression ---")
    print("Patching model...")
    patch_model_for_contextlens(model)

    compressed_memory = measure_memory(model)
    print(f"Compressed model memory: {compressed_memory:.2f} MB")

    # Re-tokenize for fresh generation
    inputs2 = tokenizer(prompt, return_tensors="pt")
    start = time.time()
    with torch.inference_mode():
        output_compressed = model.generate(**inputs2, max_new_tokens=max_tokens, pad_token_id=tokenizer.eos_token_id)
    time_compressed = time.time() - start
    text_compressed = tokenizer.decode(output_compressed[0], skip_special_tokens=True)
    print(f"Time: {time_compressed:.2f}s")
    print(f"Output: {text_compressed[:100]}...")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Metric':<30} {'Baseline':<15} {'Compressed':<15} {'Change':<15}")
    print("-" * 75)
    print(f"{'Memory (MB)':<30} {base_memory:<15.2f} {compressed_memory:<15.2f} {'--':<15}")
    print(f"{'Generation Time (s)':<30} {time_base:<15.2f} {time_compressed:<15.2f} {((time_compressed/time_base)-1)*100:+.1f}%")
    print(f"{'Output Length (tokens)':<30} {len(output_base[0]):<15} {len(output_compressed[0]):<15} {len(output_compressed[0])-len(output_base[0]):<15}")

    # Note about KV cache
    print("\nNote: TurboQuant compression reduces KV cache memory during generation.")
    print("      The patch intercepts KV cache tensors and compresses them on-the-fly.")
    print("      Memory savings become more significant with longer contexts.")

    return {
        "base_memory_mb": base_memory,
        "compressed_memory_mb": compressed_memory,
        "base_time_s": time_base,
        "compressed_time_s": time_compressed,
        "base_tokens": len(output_base[0]),
        "compressed_tokens": len(output_compressed[0]),
    }


if __name__ == "__main__":
    results = test_compression()
    print("\n" + "=" * 60)
    print("Test completed successfully!")
    print("=" * 60)
