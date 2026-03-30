# Changelog

All notable changes to ContextLens will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned
- Web dashboard for real-time monitoring
- Multi-GPU support
- Automatic model selection based on available RAM
- Export comparison reports (CSV/JSON/PDF)
- Support for MoE models (Mixtral, Grok)
- Dynamic compression based on context length
- Integration with vLLM and TGI

## [0.3.0] - 2026-03-30

### Added
- **Real-time comparison feature** - Side-by-side performance testing
  - `contextlens compare` command for live benchmarking
  - Multiple iteration support for statistical accuracy
  - Memory monitoring during inference
  - KV cache savings calculator
- **HuggingFace authentication management**
  - `contextlens hf-auth` command
  - Support for both gated and open-weight models
  - `--use-open-weights` flag (default, no auth needed)
  - `--use-gated` flag for Llama/Gemma models
- **Comprehensive model mappings** - 30+ models supported
  - Llama 3, 3.1, 3.2 → Qwen alternatives
  - Mistral, Mixtral (all open)
  - Phi-3 (all open)
  - Gemma, Gemma2 (all open)
  - Qwen, Qwen2, Qwen2.5 (all open)
- **Docker testing environment**
  - Isolated container for safe testing
  - Pre-configured with Ollama and test models
  - Automated test suite

### Changed
- **Ollama v0.19.0 integration** - Now uses API instead of CLI
  - Creates `-contextlens` model variants
  - Works with blob-based storage (Ollama v0.5+)
  - Better error messages and recovery options
- **Improved error handling** - Clear guidance for common issues
  - 3 options when HuggingFace auth fails
  - Better Ollama version detection
  - Helpful troubleshooting messages
- **Enhanced documentation** - Comprehensive README with examples
  - Installation instructions
  - Usage examples for all commands
  - Troubleshooting section
  - Benchmarks and performance data

### Fixed
- Ollama v0.19.0 compatibility (blob storage support)
- HuggingFace gated model access (authentication flow)
- Scanner compatibility with new Ollama API format
- Model name mapping for benchmarking
- Import errors in comparison module

### Technical
- Added `compare.py` module for performance testing
- Updated `cli.py` with new commands
- Created release scripts for PyPI publishing
- Added MANIFEST.in for package distribution
- Updated pyproject.toml with proper metadata
- Added py.typed for PEP 561 compliance

## [0.2.0] - 2026-03-29

### Added
- Initial beta release
- KV cache compression with TurboQuant algorithm
- Support for Ollama, llama.cpp, and HuggingFace
- Model scanning and profiling
- Accuracy benchmarking (MMLU, HellaSwag)
- Profile persistence
- Basic CLI commands: scan, apply, integrate, status, revert

### Supported Models
- Llama 3, 3.1
- Mistral, Mixtral
- Phi-3
- Gemma
- Qwen2

### Technical
- 3-bit compression (5.3× reduction)
- <1% accuracy loss on benchmarks
- Python 3.10+ support
- MIT License

---

## Version History

- **0.3.0** - Real-time comparison, HF auth, comprehensive mappings (Current)
- **0.2.0** - Initial beta release
- **0.1.0** - Internal alpha testing

---

**Latest Release:** v0.3.0 (2026-03-30)

**Total Commits:** 15+

**Contributors:** ContextLens Team
