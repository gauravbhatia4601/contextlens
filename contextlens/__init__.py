"""
ContextLens - KV cache compression for local LLMs.

Compresses the KV cache of locally-running LLMs using the TurboQuant algorithm
(PolarQuant + QJL error correction) achieving ~6x compression with <0.005
accuracy delta on standard benchmarks.
"""

__version__ = "0.2.0"

__all__ = [
    "cli",
    "scanner",
    "compressor",
    "benchmarks",
    "profiles",
    "utils",
    "integrations",
]
