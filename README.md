# ContextLens

**Compress your local LLM KV cache with 3.4× memory reduction and minimal accuracy loss.**

> **Package Name:** `llm-contextlens` on PyPI

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

ContextLens is an open-source CLI tool that compresses the KV (Key-Value) cache of locally-running LLMs using the **TurboQuant algorithm** (arXiv:2504.19874), achieving **~3.4× memory reduction** with minimal accuracy loss.

## 🚀 Quick Start

### Installation

**Using pipx (Recommended for CLI tools)**
```bash
pipx install llm-contextlens
```

**Using pip with virtual environment**
```bash
python3 -m venv ~/llm-contextlens-venv
source ~/llm-contextlens-venv/bin/activate
pip install llm-contextlens
```

**From source**
```bash
git clone https://github.com/gauravbhatia4601/contextlens.git
cd contextlens
python3 -m venv venv
source venv/bin/activate
pip install -e .
```

### Verify Installation

```bash
llm-contextlens --help
```

### Upgrade

**If installed with pip:**
```bash
pip install --upgrade llm-contextlens
```

**If installed with pipx:**
```bash
pipx upgrade llm-contextlens
```

## 📋 Requirements

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **RAM** | 8 GB | 16+ GB |
| **Python** | 3.10 | 3.11+ |
| **Storage** | 10 GB free | 50+ GB free |
| **GPU** | Optional | NVIDIA with 8+ GB VRAM |

### Supported Model Architectures

- ✅ Llama 3, 3.1, 3.2 (all sizes)
- ✅ Mistral, Mixtral (all sizes)
- ✅ Phi, Phi-2, Phi-3 (all sizes)
- ✅ Gemma, Gemma2 (all sizes)
- ✅ Qwen, Qwen2, Qwen2.5 (all sizes)
- ✅ Falcon, Yi, StableLM

## 🎯 What It Does

When running large models locally, two components consume RAM:

1. **Model weights** — Already handled by quantization (ContextLens does NOT touch this)
2. **KV cache** — A tensor that grows with context length. A 70B model at 32k tokens needs ~48 GB of KV cache in FP16. **This is what ContextLens compresses.**

### Example: Llama 3.1 70B at 32k Context

| Component | Memory (FP16) | With ContextLens | Savings |
|-----------|---------------|------------------|---------|
| Model weights (Q4) | ~40 GB | ~40 GB | 0 GB |
| **KV cache** | **~48 GB** | **~14 GB** | **34 GB** ✅ |
| **Total** | **~88 GB** | **~54 GB** | **34 GB** ✅ |

**Compression ratio: ~3.4× KV cache reduction**

## 🛠️ Usage

### 1. Download a Model

ContextLens works with HuggingFace models that are downloaded locally:

```bash
huggingface-cli download Qwen/Qwen2-0.5B
```

For gated models (e.g., Llama, Gemma), first log in:

```bash
huggingface-cli login
```

### 2. Scan a Model

Profile KV cache memory usage and context limits:

```bash
llm-contextlens scan Qwen/Qwen2-0.5B
```

Output:
```
Model: Qwen/Qwen2-0.5B
Architecture: 24 layers, 2 KV heads, 64 head dim
Dtype: float16

KV Cache Memory:
  Per 1k tokens: 0.05 GB

Max Context Length:
  16 GB RAM: 327,680 tokens
  32 GB RAM: 655,360 tokens
  64 GB RAM: 1,310,720 tokens
```

### 3. Apply Compression

Apply TurboQuant compression to the model:

```bash
llm-contextlens apply Qwen/Qwen2-0.5B
```

This will:
1. Scan the model architecture
2. Run an accuracy benchmark (optional, can be skipped)
3. Save a compression profile to `~/.contextlens/`

### 4. Use Compression in Your Code

Load the model with compression enabled:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from contextlens.integrations.huggingface import patch_model_for_contextlens

# Load model
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2-0.5B",
    torch_dtype="auto",
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B")

# Apply compression
model = patch_model_for_contextlens(model)

# Generate text - KV cache will be compressed automatically
inputs = tokenizer("Hello, how are you?", return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## 📊 Compression Results

Based on profiling with the TurboQuant algorithm:

| Model | Original KV | Compressed KV | Compression Ratio | Memory Saved |
|-------|-------------|---------------|-------------------|--------------|
| **Qwen2-0.5B** (512 tokens) | 12.00 MB | 3.49 MB | **3.44x** | 8.51 MB (70.9%) |
| **Llama-3.2-1B** (512 tokens) | 32.00 MB | 9.22 MB | **3.47x** | 22.78 MB (71.2%) |

### Memory by Context Length

| Context Length | Qwen2-0.5B Original | Qwen2-0.5B Compressed | Saved |
|----------------|---------------------|----------------------|-------|
| 4K tokens | 48 MB | 14 MB | 34 MB |
| 16K tokens | 192 MB | 56 MB | 136 MB |
| 32K tokens | 384 MB | 112 MB | 272 MB |
| 128K tokens | 1.5 GB | 448 MB | 1.1 GB |

## 🔧 CLI Commands

| Command | Description |
|---------|-------------|
| `llm-contextlens scan <model>` | Profile KV cache memory usage |
| `llm-contextlens apply <model>` | Apply TurboQuant compression |
| `llm-contextlens status` | List all compressed models |
| `llm-contextlens integrate <model>` | Show integration instructions |
| `llm-contextlens revert <model>` | Remove compression profile |
| `llm-contextlens hf-auth --check` | Check HuggingFace auth status |
| `llm-contextlens hf-auth --login` | Show login instructions |

## 🔐 HuggingFace Authentication

For gated models (Llama, Gemma, etc.), you need to authenticate:

**Option 1: Login via CLI**
```bash
huggingface-cli login
```

**Option 2: Set environment variable**
```bash
export HF_TOKEN=your_token_here
```

**Option 3: Use open-weight alternatives**
Models like Qwen are freely available without authentication.

## 🏗️ Architecture

ContextLens implements the **TurboQuant** algorithm from arXiv:2504.19874:

1. **K-Cache: PolarQuant (8-bit)** - Decomposes keys into magnitude and direction, quantizing each separately
2. **V-Cache: Residual QJL (5-bit + 1-bit)** - Two-stage quantization with Johnson-Lindenstrauss projection on the residual

The compression is applied transparently during inference - your code does not need to change.

## 📦 API Server (Optional)

ContextLens includes an optional OpenAI-compatible API server:

```bash
# Start the server
python -m contextlens.proxy serve --port 8080

# Use with curl
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2-0.5B",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a PR on GitHub.

## 📄 License

MIT License - see LICENSE file for details.

## 📚 References

- TurboQuant Paper: https://arxiv.org/abs/2504.19874
- HuggingFace Transformers: https://huggingface.co/docs/transformers
- HuggingFace Hub: https://huggingface.co/docs/hub
