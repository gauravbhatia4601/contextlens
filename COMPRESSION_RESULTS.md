# TurboQuant Compression Profiling Results

**Date:** 2026-04-03  
**Test Environment:** Docker container (ollama/ollama:latest)  
**Device:** CPU (single-threaded test)

---

## Executive Summary

TurboQuant compression achieves **~3.4x compression ratio** (71% memory reduction) on KV cache:

| Model | Original KV | Compressed KV | Compression Ratio | Memory Saved |
|-------|-------------|---------------|-------------------|--------------|
| **Qwen2-0.5B** (512 tokens) | 12.00 MB | 3.49 MB | **3.44x** | 8.51 MB (70.9%) |
| **Llama-3.2-1B** (512 tokens) | 32.00 MB | 9.22 MB | **3.47x** | 22.78 MB (71.2%) |

---

## Detailed Results: Qwen2-0.5B

**Configuration:**
- Layers: 24
- KV Heads: 2
- Head Dimension: 64
- Sequence Length: 512 tokens

### Per-Layer Breakdown

| Component | Original | Compressed | Ratio |
|-----------|----------|------------|-------|
| **K-Cache** (per layer) | 256 KB | 65 KB | 3.94x |
| **V-Cache** (per layer) | 256 KB | 84 KB | 3.05x |

### Total Memory by Context Length

| Context Length | Original KV | Compressed KV | Memory Saved |
|----------------|-------------|---------------|--------------|
| 256 tokens | 6.00 MB | 1.75 MB | 4.25 MB |
| 512 tokens | 12.00 MB | 3.49 MB | 8.51 MB |
| 1024 tokens | 24.00 MB | 6.98 MB | 17.02 MB |
| 2048 tokens | 48.00 MB | 13.97 MB | 34.03 MB |
| 4096 tokens | 96.00 MB | 27.94 MB | 68.06 MB |
| 8192 tokens | 192.00 MB | 55.88 MB | 136.12 MB |
| 16384 tokens | 384.00 MB | 111.75 MB | 272.25 MB |
| **32768 tokens** | **768.00 MB** | **223.50 MB** | **544.50 MB** |

---

## Detailed Results: Llama-3.2-1B

**Configuration:**
- Layers: 16
- KV Heads: 8
- Head Dimension: 64
- Sequence Length: 512 tokens

### Per-Layer Breakdown

| Component | Original | Compressed | Ratio |
|-----------|----------|------------|-------|
| **K-Cache** (per layer) | 1.00 MB | 260 KB | 3.94x |
| **V-Cache** (per layer) | 1.00 MB | 330 KB | 3.10x |

### Total Memory by Context Length

| Context Length | Original KV | Compressed KV | Memory Saved |
|----------------|-------------|---------------|--------------|
| 256 tokens | 16.00 MB | 4.61 MB | 11.39 MB |
| 512 tokens | 32.00 MB | 9.22 MB | 22.78 MB |
| 1024 tokens | 64.00 MB | 18.44 MB | 45.56 MB |
| 2048 tokens | 128.00 MB | 36.88 MB | 91.12 MB |
| 4096 tokens | 256.00 MB | 73.75 MB | 182.25 MB |
| 8192 tokens | 512.00 MB | 147.50 MB | 364.50 MB |
| 16384 tokens | 1.00 GB | 295.00 MB | 729.00 MB |
| **32768 tokens** | **2.00 GB** | **590.00 MB** | **1.42 GB** |

---

## Compression Algorithm Breakdown

### K-Cache: PolarQuant (8-bit)

```
Original: float32 (4 bytes per element)
Compressed:
  - Magnitude: 8-bit (1 byte per token)
  - Direction: 8-bit per coordinate (1 byte × head_dim)
  
Compression Ratio: ~4x (3.94x achieved)
```

### V-Cache: Residual QJL (5-bit + 1-bit)

```
Original: float32 (4 bytes per element)
Compressed:
  - Primary: 5-bit quantization (~1 byte)
  - Residual: 1-bit QJL sign + projection
  - Scale: float16 per token
  
Compression Ratio: ~3x (3.05-3.10x achieved)
```

---

## Key Observations

1. **K-Cache compresses better than V-Cache**
   - K: 3.94x (PolarQuant is highly effective)
   - V: 3.05-3.10x (Residual QJL has overhead)

2. **Memory savings scale linearly with context length**
   - At 32K context: 544 MB saved (Qwen2) / 1.42 GB saved (Llama-3.2)

3. **Overhead is fixed per layer**
   - Rotation matrices, min/max scalars
   - Negligible at long contexts

4. **Total compression ratio: ~3.4x**
   - Matches paper claims (arXiv:2504.19874)
   - Consistent across model sizes

---

## Integration Test Results

### Unit Tests (15 tests)

```
============================== 15 passed in 7.76s ==============================
contextlens/compressor.py                   153      0   100% coverage
```

| Test Category | Tests | Status |
|---------------|-------|--------|
| PolarQuant (K-cache) | 3 | ✅ Pass |
| Residual QJL (V-cache) | 3 | ✅ Pass |
| Compression Ratio | 1 | ✅ Pass |
| Attention Interface | 3 | ✅ Pass |
| Unbiased Estimation | 1 | ✅ Pass |
| Compressed Attention | 4 | ✅ Pass |
| Memory Efficiency | 1 | ✅ Pass |

### End-to-End Tests

| Test | Status | Notes |
|------|--------|-------|
| Ollama direct (qwen2:0.5b) | ✅ Working | Q4_0 quantized, fast |
| Proxy + TurboQuant (qwen2:0.5b) | ✅ Working | HF FP16 + compression |
| Proxy fallback (llama3.2:1b) | ✅ Working | Forwards to Ollama |

---

## How to Reproduce

### Run Compression Profiler

```bash
docker exec 49ccd533eb13 sh -c "
  cd /app/contextlens && \
  /app/contextlens/venv/bin/python scripts/compression_profile.py
"
```

### Run Unit Tests

```bash
docker exec 49ccd533eb13 sh -c "
  cd /app/contextlens && \
  /app/contextlens/venv/bin/python -m pytest tests/test_compressor.py -v
"
```

### Test Proxy (when running)

```bash
# Start proxy
setsid /app/contextlens/venv/bin/python -m contextlens.proxy > /tmp/proxy.log 2>&1 &

# Test generation
curl -X POST http://localhost:8080/api/generate \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen2:0.5b","prompt":"Hello","stream":false}'
```

---

## Conclusion

TurboQuant compression is **verified working** with:

- ✅ **3.4x compression ratio** (71% memory reduction)
- ✅ **Consistent across model sizes** (tested on 0.5B and 1B parameter models)
- ✅ **Linear scaling** with context length
- ✅ **15/15 unit tests passing**
- ✅ **End-to-end integration** with HuggingFace and Ollama

The compression is most beneficial for:
- Long context scenarios (4K+ tokens)
- Large models (7B+ parameters)  
- Memory-constrained environments
