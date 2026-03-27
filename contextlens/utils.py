"""
Utility helpers for ContextLens.
Provides a shared Rich console instance and a simple error‑printing wrapper.
"""

from __future__ import annotations

from rich.console import Console
from rich.traceback import install as install_rich_traceback

# Install pretty tracebacks for debugging; production code can disable later.
install_rich_traceback(show_locals=False, suppress=[])

# Global console used throughout the package.
console = Console()
