# ContextLens Production Roadmap

**Current Status:** ~20% production-ready  
**Estimated Effort:** 8-12 weeks of focused engineering

---

## Executive Summary

ContextLens has solid foundations: clean CLI, good documentation, PyPI publication, and a valid problem statement (KV cache memory consumption). However, after testing and reviewing the original TurboQuant paper (arXiv:2504.19874), the gaps are more significant than initially assessed:

1. **Compression is not hooked into actual inference** — tensors are fully decompressed before attention
2. **Runtime integrations don't work** — Ollama ignores the `PARAMETER contextlens_profile` (not a recognized parameter)
3. **Algorithm deviates from TurboQuant paper** — missing random rotation, incorrect QJL usage, no residual quantization
4. **No compressed attention implementation** — the paper's key insight is computing attention WITHOUT decompression

---

## Key Insights from TurboQuant Paper

The Google Research TurboQuant paper describes a different algorithm than what ContextLens implements:

| Component | TurboQuant Paper | ContextLens Implementation | Gap |
|-----------|-----------------|---------------------------|-----|
| **Preprocessing** | Random rotation to simplify geometry | ❌ Missing | High |
| **K-cache (PolarQuant)** | Recursive polar transforms on rotated vectors | Simple angle/magnitude on first 2 dims | High |
| **V-cache (QJL)** | QJL on **residual** after MSE quantization | QJL on full vector | High |
| **Attention** | Special estimator computes Q·K^T without decompression | Full decompression before attention | Critical |
| **Bits claimed** | 3.5 bits/channel with "absolute quality neutrality" | 3 bits, claims <1% accuracy loss | Unverified |

**Critical Finding:** The paper's main contribution is an **unbiased inner product estimator** that works directly on compressed representations. ContextLens decompresses everything first, which defeats the entire purpose.

---

## Critical Issues (Must Fix)

### 1. Implement TurboQuant Algorithm Correctly

**File:** `contextlens/compressor.py`

**Problem:** Current implementation deviates significantly from the TurboQuant paper (arXiv:2504.19874).

#### 1.1 Missing: Random Rotation Preprocessing

The paper states: *"TurboQuant achieves this by randomly rotating input vectors, inducing a concentrated Beta distribution on coordinates."*

```python
# MISSING: Random rotation before quantization
def random_rotate(x: torch.Tensor, seed: int) -> torch.Tensor:
    """Apply random orthogonal rotation to simplify vector geometry."""
    torch.manual_seed(seed)
    d = x.shape[-1]
    # Generate random orthogonal matrix via QR decomposition
    random_matrix = torch.randn(d, d, dtype=x.dtype, device=x.device)
    Q, R = torch.linalg.qr(random_matrix)
    return torch.matmul(x, Q)
```

**Why it matters:** Without rotation, the coordinate distribution is not uniform, making quantization less effective.

#### 1.2 Missing: Residual QJL Structure

The paper uses a **two-stage** approach:
1. Apply MSE-optimized quantizer (e.g., 2.5 bits)
2. Apply 1-bit QJL to the **residual** (error), not the full vector

```python
# Current (wrong) - QJL on full vector
cv = compressor.compress_v_cache(v, layer_idx=0)

# Paper approach (correct) - QJL on residual
v_quant = mse_quantize(v, bits=2.5)  # Primary quantization
v_residual = v - mse_dequantize(v_quant)  # Compute error
v_qjl = qjl_compress(v_residual, bits=1)  # QJL on residual only
```

#### 1.3 Missing: Unbiased Inner Product Estimator

The paper's key contribution (Section 3.2):

```python
# From paper: QJL enables unbiased inner product estimation
# QJL map: Q_qjl(x) := sign(S · x) where S is sketch matrix
# Inverse: Q_qjl^(-1)(z) := sqrt(π/2)/d · S^T · z

# For attention: compute Q · K^T without full K reconstruction
def compressed_attention_score(
    query: torch.Tensor,      # [B, H, S_q, D]
    ck: CompressedKCache,     # Compressed K
) -> torch.Tensor:            # [B, H, S_q, S_k]
    """
    Compute attention scores Q · K^T using compressed K.
    
    Paper approach:
    - K is stored as (magnitudes, angles) from PolarQuant
    - Use polar coordinate properties to compute dot product
    - Q · K ≈ |Q| · |K| · cos(θ_Q - θ_K) for 2D case
    - For higher dimensions, use recursive polar decomposition
    """
    # TODO: Implement without materializing full K tensor
    pass
```

**Acceptance Criteria:**
- [ ] Random rotation added before quantization
- [ ] Residual QJL structure implemented (MSE + 1-bit QJL)
- [ ] Inner product estimator works without full decompression
- [ ] Unit tests verify unbiased estimation: E[⟨y, Q^(-1)(Q(x))⟩] = ⟨y, x⟩

---

### 2. Fix Runtime Integrations

**Verified Issue (2026-04-02):** I tested the integration end-to-end. The `llama3.2:1b-contextlens` model was created, but:
- Both original and "compressed" models produce identical outputs
- The Modelfiles are identical (no `PARAMETER` stored)
- Ollama's API silently ignores unknown parameters

#### 2.1 Ollama Integration

**File:** `contextlens/integrations/ollama.py`

**Problem:** The `PARAMETER contextlens_profile` written to Modelfile (line 138) is **ignored by Ollama**. Ollama doesn't recognize this parameter.

```python
# Current (line 138) - doesn't work
modelfile_content = f"FROM {model_name}\nPARAMETER contextlens_profile \"{profile_path}\"\n"
```

**Options:**

| Approach | Effort | Pros | Cons |
|----------|--------|------|------|
| **A. Fork ollama/llama.cpp** | 4 weeks | Full control, immediate integration | Maintenance burden, diverges from upstream |
| **B. Proxy server** | 2 weeks | No upstream changes needed | Adds latency, infrastructure complexity |
| **C. Upstream contribution** | 6+ weeks | Official support, community benefit | Uncertain timeline, review process |

**Recommended:** Start with **Option B (proxy)** for quick validation, pursue **Option C (upstream)** in parallel.

**Proxy Server Sketch:**
```python
# contextlens/server.py (new file)
from flask import Flask, request, stream_with_context
import requests

app = Flask(__name__)
OLLAMA_BASE = "http://localhost:11434"

@app.route("/api/generate", methods=["POST"])
def generate():
    data = request.json
    model = data.get("model")
    
    # Check if model has ContextLens profile
    profile = load_profile_if_exists(model)
    if profile:
        # Apply compression to KV cache during generation
        return stream_with_context(compressed_generate(data, profile))
    
    # Forward to Ollama unchanged
    return requests.post(f"{OLLAMA_BASE}/api/generate", json=data)
```

**Acceptance Criteria:**
- [ ] Compressed model variant actually uses compression during inference
- [ ] Memory savings are measurable via `nvidia-smi` or similar
- [ ] No manual intervention required to use compressed models

---

#### 2.2 HuggingFace Integration

**File:** `contextlens/integrations/huggingface.py` (missing/minimal)

**Problem:** The `patch_model_for_contextlens()` function is imported but implementation is not visible.

**Required Implementation:**
```python
# contextlens/integrations/huggingface.py
from typing import Any
import torch
from transformers import PreTrainedModel

from ..compressor import TurboQuantCompressor, CompressedKCache, CompressedVCache


class CompressedKVCache:
    """Manages compressed KV cache during generation."""
    
    def __init__(self, compressor: TurboQuantCompressor):
        self.compressor = compressor
        self.key_cache: list[CompressedKCache] = []
        self.value_cache: list[CompressedVCache] = []
    
    def update(self, layer_idx: int, key: torch.Tensor, value: torch.Tensor):
        """Compress and store KV for current layer."""
        ck = self.compressor.compress_k_cache(key, layer_idx)
        cv = self.compressor.compress_v_cache(value, layer_idx)
        
        if layer_idx < len(self.key_cache):
            self.key_cache[layer_idx] = ck
            self.value_cache[layer_idx] = cv
        else:
            self.key_cache.append(ck)
            self.value_cache.append(cv)
    
    def get(self, layer_idx: int):
        """Retrieve compressed KV for layer."""
        return self.key_cache[layer_idx], self.value_cache[layer_idx]


def patch_model_for_contextlens(model: PreTrainedModel) -> PreTrainedModel:
    """
    Patch model's attention mechanism to use compressed KV cache.
    
    This replaces the model's _attn_implementation with a compressed variant.
    """
    compressor = TurboQuantCompressor(bits=3)
    cache = CompressedKVCache(compressor)
    
    # Hook into model's forward pass
    # This is model-family specific (Llama, Mistral, etc.)
    
    # TODO: Implement model-specific patching
    # - LlamaAttention.forward
    # - MistralAttention.forward
    # - etc.
    
    return model
```

**Acceptance Criteria:**
- [ ] Works with Llama 3.x family
- [ ] Works with Mistral/Mixtral
- [ ] Compatible with transformers >= 4.40.0
- [ ] Graceful fallback if compression fails

---

### 3. Fix Compression Algorithm Bugs

#### 3.1 Magnitude Quantization (No-Op Bug)

**File:** `contextlens/compressor.py`  
**Line:** 97-99

```python
# BUG: This is always True, so magnitudes are always 1.0
magnitudes_quantized = (mean_mag >= 0).float()  # Always true, placeholder

# FIX: Store actual magnitude information
# Option A: Per-block scale
magnitude_scale = magnitude.mean(dim=-2, keepdim=True)  # [B, H, 1, 1]
magnitudes_normalized = magnitude / (magnitude_scale + eps)
magnitudes_quantized = torch.round(magnitudes_normalized * 3).clamp(0, 3)

# Return both quantized magnitudes AND scale for reconstruction
```

**Impact:** Current code loses all magnitude information, causing significant reconstruction error.

---

#### 3.2 V-Cache Dequantization (Incomplete)

**File:** `contextlens/compressor.py`  
**Line:** 221-229

```python
# Current code admits it's incomplete:
# "This is a simplification - production code would store per-token stats"

# FIX: Store and use min/max per token
v_min = projected.min(dim=-1, keepdim=True).values
v_max = projected.max(dim=-1, keepdim=True).values
v_range = v_max - v_min + 1e-8

# Store these in CompressedVCache
@dataclass
class CompressedVCache:
    projected: torch.Tensor
    projection_matrix: torch.Tensor
    original_shape: Tuple[int, int, int, int]
    v_min: torch.Tensor  # NEW: for dequantization
    v_max: torch.Tensor  # NEW: for dequantization
```

---

#### 3.3 K-Cache Reconstruction (Dimension Loss)

**File:** `contextlens/compressor.py`  
**Line:** 190-199

```python
# Current: Only reconstructs first 2 dimensions
direction = torch.zeros(batch, heads, seq_len, head_dim, ...)
direction[..., 0] = direction_x  # cos(angle)
direction[..., 1] = direction_y  # sin(angle)
# All other dimensions are ZERO - information loss!

# FIX: Use spherical coordinates for higher dimensions
# Or: Store additional angle components for head_dim > 2
```

**Impact:** For typical head_dim=128, 126/128 dimensions are zeros after reconstruction.

---

### 4. Add Per-Token/Per-Block Scale Storage

**File:** `contextlens/compressor.py`

**Problem:** Quantization loses scale information that varies across tokens and heads.

**Required:**
```python
@dataclass
class CompressedKCache:
    magnitudes: torch.Tensor      # 1-bit magnitude encoding
    angles: torch.Tensor          # 2-bit angle encoding
    magnitude_scale: torch.Tensor # NEW: per-block scale [B, H, S, 1]
    original_shape: Tuple[int, ...]

@dataclass  
class CompressedVCache:
    projected: torch.Tensor
    projection_matrix: torch.Tensor
    v_min: torch.Tensor           # NEW: per-token min
    v_max: torch.Tensor           # NEW: per-token max
    original_shape: Tuple[int, ...]
```

**Memory Overhead:** ~10-15% increase in compressed size, but essential for accuracy.

---

## High-Priority Features

### 5. Memory Profiling & Verification

**New File:** `contextlens/memory_profiler.py`

```python
"""
Memory profiler to verify actual savings during inference.
"""
import torch
import pynvml  # For GPU memory

class MemoryProfiler:
    def __init__(self):
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    
    def get_gpu_memory(self) -> int:
        """Get current GPU memory usage in bytes."""
        info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
        return info.used
    
    def profile_generation(
        self,
        model,
        prompt: str,
        max_tokens: int = 100,
    ) -> dict:
        """Profile memory during text generation."""
        memory_before = self.get_gpu_memory()
        
        # Generate
        outputs = model.generate(...)
        
        memory_peak = self.get_gpu_memory()
        memory_after = self.get_gpu_memory()
        
        return {
            "memory_before": memory_before,
            "memory_peak": memory_peak,
            "memory_after": memory_after,
            "memory_delta": memory_peak - memory_before,
        }
```

**Acceptance Criteria:**
- [ ] CLI command: `llm-contextlens bench-memory llama3.1:70b`
- [ ] Reports memory savings with/without compression
- [ ] Validates "5.3×" claim or adjusts documentation

---

### 6. CUDA Kernels for Compressed Attention

**New File:** `contextlens/kernels/compressed_attn.cu`

**Why:** Python-loop decompression is slow. Custom CUDA kernels can operate on compressed tensors directly.

```cuda
// Compressed PolarQuant attention kernel
__global__ void polar_attention_kernel(
    const float* query,           // [batch, heads, seq, dim]
    const uint8_t* k_magnitudes,  // [batch, heads, seq] - 1-bit
    const uint8_t* k_angles,      // [batch, heads, seq] - 2-bit
    const float* k_scale,         // [batch, heads, seq, 1]
    float* attention_output,      // [batch, heads, seq, dim]
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim
) {
    // TODO: Implement compressed Q·K^T without full K reconstruction
}
```

**Effort:** 2-4 weeks for someone with CUDA experience.

---

### 7. Configuration System

**New File:** `contextlens/config.py`

```python
"""
Configuration management for ContextLens.
"""
from dataclasses import dataclass
from pathlib import Path
import yaml

@dataclass
class ContextLensConfig:
    bits: int = 3
    method: str = "turboquant"
    skip_benchmark: bool = False
    force_apply: bool = False
    benchmark_dataset: str = "mmlu"
    benchmark_questions: int = 500
    
    # Runtime settings
    use_cuda_kernels: bool = False
    debug_mode: bool = False
    
    @classmethod
    def load(cls, path: Path = None) -> "ContextLensConfig":
        if path is None:
            path = Path.home() / ".contextlens" / "config.yaml"
        if path.exists():
            with path.open() as f:
                data = yaml.safe_load(f)
            return cls(**data)
        return cls()
    
    def save(self, path: Path = None):
        if path is None:
            path = Path.home() / ".contextlens" / "config.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            yaml.dump(self.__dict__, f)
```

---

### 8. Logging System

**File:** `contextlens/utils.py` (extend)

```python
"""
Logging utilities for ContextLens.
"""
import logging
from pathlib import Path

def setup_logging(debug: bool = False) -> logging.Logger:
    """Configure logging for ContextLens."""
    log_path = Path.home() / ".contextlens" / "contextlens.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    level = logging.DEBUG if debug else logging.INFO
    
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(),  # Console
        ],
    )
    
    return logging.getLogger("contextlens")

logger = setup_logging()
```

**Replace all `console.print()` with proper logging:**
```python
# Before
console.print(f"[bold green]Scanning model:[/bold green] {model}")

# After
logger.info(f"Scanning model: {model}")
```

---

## Testing Gaps

### 9. Add Integration Tests

**New File:** `tests/test_inference_integration.py`

```python
"""
Integration tests for compressed inference.
"""
import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from contextlens.compressor import TurboQuantCompressor
from contextlens.integrations.huggingface import patch_model_for_contextlens


@pytest.mark.slow  # Mark for CI skip
@pytest.mark.parametrize("model_name", [
    "Qwen/Qwen2.5-0.5B-Instruct",  # Small, fast
    "meta-llama/Llama-3.2-1B-Instruct",  # If accessible
])
def test_compressed_inference_matches_uncompressed(model_name):
    """Verify compressed model produces similar output to baseline."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    prompt = "The capital of France is"
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Baseline
    baseline_output = model.generate(**inputs, max_new_tokens=20)
    baseline_text = tokenizer.decode(baseline_output[0], skip_special_tokens=True)
    
    # Compressed
    compressed_model = patch_model_for_contextlens(model)
    compressed_output = compressed_model.generate(**inputs, max_new_tokens=20)
    compressed_text = tokenizer.decode(compressed_output[0], skip_special_tokens=True)
    
    # Outputs should be similar (not necessarily identical due to quantization)
    assert baseline_text[:50] == compressed_text[:50], \
        f"Outputs diverge: {baseline_text} vs {compressed_text}"


@pytest.mark.gpu
def test_actual_memory_saved_during_inference():
    """Measure GPU memory with/without compression."""
    # TODO: Use pynvml or torch.cuda.memory_allocated()
    pass
```

---

**New File:** `tests/test_end_to_end.py`

```python
"""
End-to-end tests with Ollama.
"""
import pytest
import requests
import subprocess
import time

from contextlens.cli import app
from typer.testing import CliRunner

runner = CliRunner()


@pytest.fixture
def ollama_running():
    """Ensure Ollama is running for tests."""
    # Start Ollama if not running
    try:
        requests.get("http://localhost:11434/api/version", timeout=5)
        yield
    except requests.exceptions.ConnectionError:
        # Start Ollama
        proc = subprocess.Popen(["ollama", "serve"])
        time.sleep(3)  # Wait for startup
        yield
        proc.terminate()


@pytest.mark.slow
def test_full_pipeline_ollama(ollama_running):
    """Test: scan -> apply -> integrate -> generate."""
    model = "llama3.2:3b"
    
    # 1. Scan
    result = runner.invoke(app, ["scan", model])
    assert result.exit_code == 0
    
    # 2. Apply compression
    result = runner.invoke(app, ["apply", model, "--skip-benchmark"])
    assert result.exit_code == 0
    
    # 3. Integrate
    result = runner.invoke(app, ["integrate", "ollama", "--model", model])
    assert result.exit_code == 0
    
    # 4. Generate with compressed model
    compressed_model = f"{model}-contextlens"
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": compressed_model,
            "prompt": "Hello, world!",
            "stream": False,
        },
    )
    assert response.status_code == 200
    assert "response" in response.json()
```

---

### 10. Add Memory Benchmark Tests

**New File:** `tests/test_memory_savings.py`

```python
"""
Verify memory savings claims.
"""
import torch
import pynvml

from contextlens.compressor import TurboQuantCompressor


def test_compression_ratio_matches_claims():
    """Verify 5.3× compression ratio claim."""
    compressor = TurboQuantCompressor(bits=3)
    
    # Simulate KV cache for 70B model at 32k context
    # 80 layers, 64 KV heads, 128 head dim, 32k seq_len
    k = torch.randn(1, 64, 32768, 128, dtype=torch.float16)
    v = torch.randn(1, 64, 32768, 128, dtype=torch.float16)
    
    # Original size (FP16 = 16 bits per value)
    original_bytes = (k.numel() + v.numel()) * 2  # 2 bytes per FP16 value
    
    # Compress
    ck = compressor.compress_k_cache(k, layer_idx=0)
    cv = compressor.compress_v_cache(v, layer_idx=0)
    
    # Compressed size (with scale storage overhead)
    compressed_bytes = (
        ck.magnitudes.numel() * 1 +  # 1-bit magnitudes
        ck.angles.numel() * 2 +      # 2-bit angles
        ck.magnitude_scale.numel() * 2 +  # FP16 scale
        cv.projected.numel() * 3 +   # 3-bit projected values
        cv.v_min.numel() * 2 +       # FP16 min
        cv.v_max.numel() * 2         # FP16 max
    ) / 8  # Convert bits to bytes
    
    ratio = original_bytes / compressed_bytes
    
    # Should achieve at least 4× (5.3× is ideal but 4× is acceptable)
    assert ratio >= 4.0, f"Compression ratio {ratio} below 4× threshold"
```

---

## Documentation Updates

### 11. Update Claims to Match Reality

**File:** `README.md`

**Current (misleading):**
```markdown
**Compress your local LLM KV cache with 5.3× memory reduction and zero accuracy loss.**
```

**Should be:**
```markdown
**Compress your local LLM KV cache with 5.3× storage reduction and <1% accuracy loss.**

> Note: Runtime memory savings require supported integrations. See [Runtime Support](#runtime-support) below.
```

---

### 12. Add Limitations Section

**File:** `README.md` (add before Installation)

```markdown
## Limitations

### Runtime Support

| Runtime | Storage Savings | Inference Savings | Status |
|---------|-----------------|-------------------|--------|
| Ollama | ✅ Yes | ⚠️ Proxy required | Beta |
| llama.cpp | ✅ Yes | ⚠️ Manual config | Beta |
| HuggingFace | ✅ Yes | ✅ Full support | Alpha |

### Algorithm Limitations

- **PolarQuant reconstruction** is approximate for `head_dim > 2`
- **QJL projection** adds ~5-10% computational overhead
- **Per-token scale storage** increases compressed size by ~15%

### Model Support

- Tested: Llama 3.x, Mistral, Qwen 2.5
- Untested: Gemma, Phi-3, Yi, StableLM
- Unsupported: MoE architectures (Mixtral) without modification
```

---

### 13. Add Architecture Decision Record (ADR)

**New File:** `docs/adr/001-compression-strategy.md`

```markdown
# ADR 001: TurboQuant Compression Strategy

## Status
Accepted

## Context
KV cache consumes significant memory during LLM inference, scaling linearly with context length.

## Decision
Use PolarQuant for K-cache and QJL for V-cache, achieving 3-bit representation.

## Rationale
- K-cache: Polar coordinates allow efficient angle/magnitude separation
- V-cache: Johnson-Lindenstrauss projection preserves inner products

## Consequences
- Requires custom attention kernels for full benefit
- Reconstruction is lossy (acceptable: <1% accuracy impact)
- Per-token scale storage adds overhead but improves accuracy

## Alternatives Considered
1. **Standard quantization (INT8/INT4)** - Simpler but lower compression ratio
2. **Low-rank decomposition** - Higher compression but more reconstruction error
3. **No compression** - Baseline, doesn't solve the problem
```

---

## Code Quality Fixes

### 14. Minor Issues

| File | Line | Issue | Fix |
|------|------|-------|-----|
| `cli.py` | 15, 18 | Duplicate `import typer` | Remove line 18 |
| `compressor.py` | 58-62 | Hard errors for non-3-bit | Support 2-bit, 4-bit or remove params |
| `scanner.py` | 76 | Hardcoded `dtype = "float16"` | Extract actual dtype from model |
| `pyproject.toml` | 13 | Generic email | Use real maintainer contact |
| `cli.py` | 101 | Duplicate `import typer` | Already imported at line 15 |

---

### 15. Security Hardening

**File:** `contextlens/cli.py` and all integrations

```python
# Add input validation
def validate_model_name(model: str) -> str:
    """Sanitize model name to prevent injection."""
    # Allow: alphanumeric, colon, slash, dot, underscore, hyphen
    import re
    if not re.match(r'^[a-zA-Z0-9:/._-]+$', model):
        raise ValueError(f"Invalid model name: {model}")
    return model


def validate_profile_path(path: str) -> Path:
    """Prevent path traversal attacks."""
    from pathlib import Path
    profile_path = Path(path).resolve()
    
    # Must be under ~/.contextlens
    allowed_base = Path.home() / ".contextlens"
    if not str(profile_path).startswith(str(allowed_base)):
        raise ValueError(f"Profile path must be under {allowed_base}")
    
    return profile_path
```

---

## Priority Roadmap

### Phase 1: Critical Fixes (Week 1-2)

| Task | Owner | Status |
|------|-------|--------|
| Fix magnitude quantization bug | | ⬜ TODO |
| Fix V-cache dequantization | | ⬜ TODO |
| Fix K-cache reconstruction | | ⬜ TODO |
| Add per-token scale storage | | ⬜ TODO |

### Phase 2: Runtime Integration (Week 3-5)

| Task | Owner | Status |
|------|-------|--------|
| HuggingFace patch_model implementation | | ⬜ TODO |
| Ollama proxy server (MVP) | | ⬜ TODO |
| Memory profiler | | ⬜ TODO |

### Phase 3: Testing & Validation (Week 6-8)

| Task | Owner | Status |
|------|-------|--------|
| Integration tests | | ⬜ TODO |
| Memory savings verification | | ⬜ TODO |
| End-to-end Ollama tests | | ⬜ TODO |

### Phase 4: Optimization (Week 9-12)

| Task | Owner | Status |
|------|-------|--------|
| CUDA kernels for compressed attention | | ⬜ TODO |
| Configuration system | | ⬜ TODO |
| Logging system | | ⬜ TODO |
| Documentation updates | | ⬜ TODO |

---

## Success Criteria

ContextLens is production-ready when:

1. ✅ **Memory savings verified:** `llm-contextlens bench-memory` shows >4× KV cache reduction during inference
2. ✅ **Accuracy validated:** All supported models show <1% accuracy degradation on MMLU/HellaSwag
3. ✅ **Runtime integrations work:** Ollama, llama.cpp, and HuggingFace all apply compression during generation
4. ✅ **Tests pass:** >90% code coverage, all integration tests passing
5. ✅ **Documentation accurate:** Claims match measured results, limitations clearly documented
6. ✅ **Security reviewed:** No obvious injection vulnerabilities, input validation in place

---

## Appendix: Current vs. Target Architecture

### Current Architecture (Incomplete)

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│    CLI      │────▶│   Scanner    │────▶│  Compressor │
│  (working)  │     │  (working)   │     │  (partial)  │
└─────────────┘     └──────────────┘     └─────────────┘
                                                │
                                                ▼
                                       ┌──────────────┐
                                       │ Integrations │
                                       │   (broken)   │
                                       └──────────────┘
```

### Target Architecture (Production)

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│    CLI      │────▶│   Scanner    │────▶│  Compressor │
│  (working)  │     │  (working)   │     │   (fixed)   │
└─────────────┘     └──────────────┘     └─────────────┘
                             │                  │
                             ▼                  ▼
                    ┌──────────────┐    ┌──────────────┐
                    │   Profiler   │    │ Integrations │
                    │   (new)      │    │   (fixed)    │
                    └──────────────┘    └──────────────┘
                                                │
                    ┌───────────────────────────┼───────────────────────────┐
                    ▼                           ▼                           ▼
            ┌──────────────┐          ┌──────────────┐          ┌──────────────┐
            │   Ollama     │          │  llama.cpp   │          │ HuggingFace  │
            │   (proxy)    │          │   (patch)    │          │   (patch)    │
            └──────────────┘          └──────────────┘          └──────────────┘
```

---

*Last updated: 2026-04-02*
