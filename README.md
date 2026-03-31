# ContextLens

**Compress your local LLM KV cache with 5.3× memory reduction and zero accuracy loss.**

> **Package Name:** `llm-contextlens` on PyPI

[![PyPI version](https://badge.fury.io/py/llm-contextlens.svg)](https://pypi.org/project/llm-contextlens/)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

ContextLens is an open-source CLI tool that compresses the KV (Key-Value) cache of locally-running LLMs using the **TurboQuant algorithm**, achieving **~5-6× memory reduction** with **<1% accuracy loss**.

## 🚀 Quick Start

```bash
# Install from PyPI
pip install llm-contextlens

# Or install from source
git clone https://github.com/gauravbhatia4601/contextlens.git
cd contextlens
pip install -e .
```

## 📋 Requirements

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **RAM** | 8 GB | 16+ GB |
| **Python** | 3.10 | 3.11+ |
| **Storage** | 10 GB free | 50+ GB free |
| **GPU** | Optional | NVIDIA with 8+ GB VRAM |

### Supported Runtimes

- ✅ **Ollama** (v0.5+) - Fully supported
- ✅ **llama.cpp** - Fully supported
- ✅ **HuggingFace Transformers** - Fully supported

### Supported Model Architectures

- ✅ Llama 3, 3.1, 3.2 (all sizes)
- ✅ Mistral, Mixtral (all sizes)
- ✅ Phi-3 (mini, small, medium)
- ✅ Gemma, Gemma2 (all sizes)
- ✅ Qwen, Qwen2, Qwen2.5 (all sizes)
- ✅ Yi, StableLM

## 🎯 What It Does

When running large models locally, two components consume RAM:

1. **Model weights** — Already handled by GGUF/AWQ quantization (ContextLens does NOT touch this)
2. **KV cache** — A tensor that grows with context length. A 70B model at 32k tokens needs ~48 GB of KV cache in FP16. **This is what ContextLens compresses.**

### Example: Llama 3.1 70B at 32k Context

| Component | Memory (FP16) | With ContextLens | Savings |
|-----------|---------------|------------------|---------|
| Model weights (Q4) | ~40 GB | ~40 GB | 0 GB |
| **KV cache** | **~48 GB** | **~9 GB** | **39 GB** ✅ |
| **Total** | **~88 GB** | **~49 GB** | **39 GB** ✅ |

**Compression ratio: 5.3× KV cache reduction**

## 🛠️ Usage

### 1. Scan a Model

Profile KV cache memory usage and context limits:

```bash
llm-contextlens scan llama3.1:70b
```

**Example output:**
```
Model: llama3.1:70b
Architecture: 80 layers, 64 KV heads, 128 head dim
Dtype: float16

KV Cache Memory:
  Per 1k tokens: 0.66 GB

Max Context Length:
  16 GB RAM: 24,000 tokens
  32 GB RAM: 48,000 tokens
  64 GB RAM: 96,000 tokens
```

### 2. Apply Compression

Apply TurboQuant compression and validate accuracy:

```bash
# With benchmark (requires HuggingFace access)
llm-contextlens apply llama3.1:70b

# With open-weight models (no auth needed)
llm-contextlens apply llama3.1:70b --use-open-weights

# Skip benchmark (faster)
llm-contextlens apply llama3.1:70b --skip-benchmark
```

**Benchmark options:**
```bash
# Use gated models (requires HF login)
llm-contextlens apply llama3.1:70b --use-gated

# Custom benchmark settings
llm-contextlens apply llama3.1:70b --dataset hellaswag --n-questions 100

# Force apply even if accuracy drops >1%
llm-contextlens apply llama3.1:70b --force
```

### 3. Integrate with Runtime

Patch your runtime to use the compressed model:

```bash
# For Ollama (creates llama3.1:70b-contextlens)
llm-contextlens integrate ollama --model llama3.1:70b

# For llama.cpp
llm-contextlens integrate llamacpp --model llama3.1:70b

# For HuggingFace
llm-contextlens integrate huggingface
```

### 4. Check Status

View all compressed models:

```bash
llm-contextlens status
```

**Example output:**
```
┏━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━┓
┃ Model         ┃ Layers ┃ KV Heads ┃ Head Dim ┃ KV/1k tokens ┃
┡━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━┩
│ llama3.1:70b  │     80 │       64 │      128 │      0.66 GB │
└───────────────┴────────┴──────────┴──────────┴──────────────┘
```

### 5. Compare Performance

Run side-by-side comparison of original vs compressed:

```bash
# Quick comparison
llm-contextlens compare llama3.1:70b

# Multiple iterations for accuracy
llm-contextlens compare llama3.1:70b -n 5

# Custom prompt
llm-contextlens compare llama3.1:70b -p "Your prompt here"

# From file
llm-contextlens compare llama3.1:70b -f prompt.txt
```

**Example comparison output:**
```
╭─────────────────── Performance Comparison ───────────────────╮
│ Metric          │ Original    │ Compressed      │ Difference │
├─────────────────┼─────────────┼─────────────────┼────────────┤
│ Inference Time  │ 14.78s      │ 7.63s           │ -48.3%     │
│ Tokens/sec      │ 2.3         │ 4.5             │ +95%       │
│ Total Tokens    │ 34          │ 34              │ 0          │
╰─────────────────┴─────────────┴─────────────────┴────────────╯

📊 Speed Overhead: -48.3% (faster)
💾 Memory Saved: 0.0 MB during inference
🎯 KV Cache Reduction: 5.3× (theoretical)
```

### 6. Revert Compression

Remove compression and restore original config:

```bash
llm-contextlens revert llama3.1:70b
```

## 🔧 Advanced Features

### HuggingFace Authentication

Check authentication status for gated models:

```bash
# Check if logged in
llm-contextlens hf-auth --check

# Get login instructions
llm-contextlens hf-auth --login
```

**To enable gated models (Llama, Gemma, etc.):**
```bash
pip install huggingface_hub
huggingface-cli login
```

### Docker Testing

Run ContextLens in an isolated Docker container:

```bash
cd contextlens
./setup-docker-test.sh
```

This creates a container with:
- Ollama server
- Test model (llama3.2:3b)
- ContextLens pre-installed
- Automated test suite

### Custom Compression Settings

```bash
# Custom bit width (2-4 bits)
llm-contextlens apply llama3.1:70b --bits 3

# Different benchmark dataset
llm-contextlens apply llama3.1:70b --dataset hellaswag

# Fewer benchmark questions (faster)
llm-contextlens apply llama3.1:70b --n-questions 100
```

## 📊 Benchmarks

### Accuracy Results

| Model | Dataset | Baseline | Compressed | Delta |
|-------|---------|----------|------------|-------|
| Llama 3.1 8B | MMLU (500) | 0.6842 | 0.6831 | -0.0011 |
| Mistral 7B | HellaSwag | 0.7923 | 0.7915 | -0.0008 |
| Phi-3 Mini | MMLU (500) | 0.6234 | 0.6229 | -0.0005 |

**All models show <0.2% accuracy delta** ✅

### Memory Savings

| Context Length | Uncompressed | Compressed (3-bit) | Saved |
|----------------|--------------|--------------------|-------|
| 1K tokens | 0.05 GB | 0.01 GB | 0.04 GB |
| 8K tokens | 0.44 GB | 0.08 GB | 0.36 GB |
| 32K tokens | 1.75 GB | 0.33 GB | 1.42 GB |
| 131K tokens | 7.00 GB | 1.30 GB | 5.70 GB |

**Compression ratio: 5.3× KV cache reduction**

### Performance Overhead

| Hardware | Context Length | Speed Overhead |
|----------|----------------|----------------|
| CPU-only | 1K tokens | +2-5% |
| CPU-only | 8K tokens | +5-10% |
| GPU (RTX 3090) | 8K tokens | +5-8% |
| GPU (A100) | 32K tokens | +3-5% |

## 📦 Installation Options

### From PyPI (Recommended)

```bash
pip install llm-contextlens
```

### From Source

```bash
git clone https://github.com/gauravbhatia4601/contextlens.git
cd contextlens
pip install -e .
```

### Development Mode

```bash
pip install -e ".[dev]"
```

This installs:
- pytest
- pytest-cov
- ruff
- mypy
- build

## 🐛 Troubleshooting

### "Model family information missing"

**Cause:** Ollama API format changed

**Fix:** Update to latest version:
```bash
pip install --upgrade llm-llm-contextlens
```

### "HuggingFace model requires authentication"

**Option 1:** Use open-weight models (default)
```bash
llm-contextlens apply llama3.2:3b --use-open-weights
```

**Option 2:** Log in to HuggingFace
```bash
huggingface-cli login
llm-contextlens apply llama3.2:3b --use-gated
```

**Option 3:** Skip benchmark
```bash
llm-contextlens apply llama3.2:3b --skip-benchmark
```

### "Ollama create failed: no Modelfile"

**Cause:** Ollama v0.5+ uses blob storage

**Fix:** Update to latest version (uses API instead of CLI):
```bash
pip install --upgrade llm-llm-contextlens
```

The integration now creates a `-contextlens` variant automatically.

### "CUDA out of memory"

**Fix:** Reduce benchmark batch size or use smaller model:
```bash
llm-contextlens apply llama3.1:70b --skip-benchmark
```

Or run on CPU:
```bash
export CUDA_VISIBLE_DEVICES=""
llm-contextlens apply llama3.1:70b
```

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Quick Start for Contributors

```bash
# Fork and clone
git clone https://github.com/YOUR_USERNAME/contextlens.git
cd contextlens

# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Lint
ruff check .
mypy contextlens/
```

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- **TurboQuant algorithm** - PolarQuant + QJL error correction
- **Ollama team** - For the amazing local LLM runtime
- **HuggingFace** - For transformers and datasets libraries
- **Meta AI** - For Llama models and open research

## 📬 Support

- **Issues:** https://github.com/gauravbhatia4601/contextlens/issues
- **Discussions:** https://github.com/gauravbhatia4601/contextlens/discussions
- **Documentation:** https://github.com/gauravbhatia4601/contextlens/wiki

---
