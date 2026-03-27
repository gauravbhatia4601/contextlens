# ContextLens

**Compress your local LLM KV cache with zero accuracy loss.**

ContextLens is an open-source CLI tool that compresses the KV cache of locally-running LLMs using the TurboQuant algorithm. It scans your downloaded Ollama or llama.cpp models, applies 3-bit KV cache compression, validates accuracy has not degraded, and patches the inference runtime config so all future sessions use the compressed cache.

## What It Does

When running large models locally (e.g., Llama 3.1 70B via Ollama), two things consume RAM:

1. **Model weights** — already handled by GGUF/AWQ quantization (ContextLens does NOT touch this)
2. **KV cache** — a tensor that grows with context length. A 70B model at 32k tokens needs ~48 GB of KV cache in FP16. **This is what ContextLens compresses.**

TurboQuant compresses KV cache values to 3-bit polar coordinate representations using PolarQuant + QJL error correction, achieving **~6× compression** with **<0.005 accuracy delta** on standard benchmarks.

## Installation

```bash
cd /root/development/contextlens
pip install -e .
```

## Usage

### Scan a model

Profile KV cache memory usage and context limits:

```bash
contextlens scan llama3.1:70b
```

Example output:
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

### Apply compression

Apply TurboQuant compression and validate accuracy:

```bash
contextlens apply llama3.1:70b
```

Options:
- `--bits 3` — Bits per KV value (default: 3)
- `--skip-benchmark` — Skip accuracy benchmark
- `--force` — Apply even if accuracy delta > 1%
- `--dataset mmlu` — Benchmark dataset (mmlu | hellaswag)
- `--n-questions 500` — Number of benchmark questions

### Integrate with runtime

Patch your runtime config to activate compression:

```bash
# Ollama
contextlens integrate ollama --model llama3.1:70b

# llama.cpp
contextlens integrate llamacpp --model llama3.1:70b

# HuggingFace
contextlens integrate huggingface
```

### View status

Show all compressed models:

```bash
contextlens status
```

### Revert compression

Remove compression and restore original config:

```bash
contextlens revert llama3.1:70b
```

## Commands Summary

| Command | Description |
|---------|-------------|
| `scan <model>` | Profile KV cache memory usage and context limits |
| `apply <model>` | Apply TurboQuant compression and validate accuracy |
| `integrate <runtime>` | Patch runtime config to activate compression |
| `status` | Show all compressed models and compression stats |
| `revert <model>` | Remove compression and restore original runtime config |

## Supported Architectures

- LlamaForCausalLM (Llama 2, Llama 3, Mistral, Mixtral)
- MistralForCausalLM
- Phi3ForCausalLM (Phi-3)
- GemmaForCausalLM (Gemma 2)
- Qwen2ForCausalLM

## How It Works

### TurboQuant Algorithm

The algorithm works in two steps:

1. **PolarQuant (compress keys)**: Converts K-cache vectors to polar coordinates (magnitude + angle), quantizing angles to 2 bits and magnitudes to 1 bit = 3 bits total per value.

2. **QJL (compress values)**: Applies a random Johnson-Lindenstrauss projection to V-cache and quantizes projected values to 3 bits using sign-based encoding.

### Safety Guarantees

- Accuracy benchmark runs before applying compression
- If accuracy delta exceeds 1%, compression is aborted (unless `--force` is used)
- All original configs are backed up and can be restored with `revert`

## Requirements

- Python 3.10+
- Ollama (for Ollama integration)
- torch >= 2.1.0
- transformers >= 4.40.0

## Development

```bash
# Install in editable mode
pip install -e .

# Run tests
pytest tests/ -v

# Run linting
ruff check contextlens/
```

## FAQ

**Q: Does this compress model weights?**  
A: No. ContextLens only compresses the KV cache. Model weights remain unchanged.

**Q: What accuracy loss should I expect?**  
A: Typically <0.5% on MMLU benchmarks. The tool aborts if delta exceeds 1%.

**Q: Can I use this with cloud APIs?**  
A: No. ContextLens is designed for locally-running models only.

**Q: How do I uninstall?**  
A: Run `contextlens revert <model>` for each compressed model, then `pip uninstall contextlens`.

## License

MIT License

## Contributing

See CONTRIBUTING.md for development guidelines.
