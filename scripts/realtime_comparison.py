#!/usr/bin/env python3
"""
Real-time comparison demo for ContextLens.

Run this to see actual memory and performance differences.
"""

import sys
sys.path.insert(0, '/app/contextlens')

from contextlens.compare import compare_models, calculate_kv_cache, display_comparison, run_inference
from rich.console import Console
from rich.panel import Panel

console = Console()

def main():
    console.print(Panel.fit(
        "[bold cyan]ContextLens Real-Time Comparison[/bold cyan]\n\n"
        "This will show ACTUAL measurable differences between\n"
        "original and compressed models.",
        title="🔬 Live Demo",
        border_style="cyan"
    ))
    
    # Test configuration
    original = "llama3.2:3b"
    compressed = "llama3.2:3b-contextlens"
    
    # Simple prompt for quick test
    prompt = "Explain quantum computing in 3 sentences."
    
    console.print(f"\n[bold]Test Setup:[/bold]")
    console.print(f"  Original:  {original}")
    console.print(f"  Compressed: {compressed}")
    console.print(f"  Prompt: {len(prompt)} chars")
    console.print()
    
    # Run comparison
    compare_models(
        original_model=original,
        compressed_model=compressed,
        prompt=prompt,
        container_name="contextlens-test",
        timeout=120
    )
    
    console.print("\n[bold green]✅ Comparison complete![/bold green]")
    console.print("\n[dim]Note: Memory savings are most visible with long contexts (8k+ tokens)[/dim]")

if __name__ == "__main__":
    main()
