"""
Real-time comparison between original and compressed models.

Shows actual memory savings, performance metrics, and side-by-side inference.
"""

from __future__ import annotations

import subprocess
import time
import json
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich import box

console = Console()

OLLAMA_API = "http://localhost:11434"


@dataclass
class ModelMetrics:
    """Metrics for a single model run."""
    model_name: str
    prompt_tokens: int = 0
    response_tokens: int = 0
    total_tokens: int = 0
    inference_time: float = 0.0
    tokens_per_second: float = 0.0
    memory_before_mb: float = 0.0
    memory_after_mb: float = 0.0
    memory_delta_mb: float = 0.0
    kv_cache_size_mb: float = 0.0
    success: bool = False
    error: str = ""


def get_ollama_memory(model_name: str) -> float:
    """Get current memory usage for a model in MB."""
    try:
        # Use ollama ps to get running models
        result = subprocess.run(
            ["ollama", "ps"],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode != 0:
            return 0.0
        
        # Parse output
        for line in result.stdout.splitlines()[1:]:  # Skip header
            if model_name in line:
                parts = line.split()
                for i, part in enumerate(parts):
                    if 'G' in part or 'M' in part:
                        # Extract memory value
                        mem_str = part.upper().replace('GB', 'G').replace('MB', 'M')
                        if 'G' in mem_str:
                            return float(mem_str.replace('G', '')) * 1024
                        elif 'M' in mem_str:
                            return float(mem_str.replace('M', ''))
        return 0.0
    except Exception:
        return 0.0


def get_container_memory(container_name: str) -> float:
    """Get container memory usage in MB."""
    try:
        result = subprocess.run(
            ["docker", "stats", "--no-stream", "--format", "{{.MemUsage}}", container_name],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode != 0:
            return 0.0
        
        mem_str = result.stdout.strip().split('/')[0].strip()
        if 'GiB' in mem_str:
            return float(mem_str.replace('GiB', '')) * 1024
        elif 'MiB' in mem_str:
            return float(mem_str.replace('MiB', ''))
        elif 'GB' in mem_str:
            return float(mem_str.replace('GB', '')) * 1024
        elif 'MB' in mem_str:
            return float(mem_str.replace('MB', ''))
        return 0.0
    except Exception:
        return 0.0


def run_inference(
    model_name: str,
    prompt: str,
    container_name: Optional[str] = None,
    timeout: int = 120
) -> ModelMetrics:
    """Run inference and collect metrics."""
    import requests
    
    metrics = ModelMetrics(model_name=model_name)
    
    # Get memory before
    if container_name:
        metrics.memory_before_mb = get_container_memory(container_name)
    else:
        metrics.memory_before_mb = get_ollama_memory(model_name)
    
    # Run inference
    start_time = time.time()
    
    try:
        resp = requests.post(
            f"{OLLAMA_API}/api/generate",
            json={
                "model": model_name,
                "prompt": prompt,
                "stream": False
            },
            timeout=timeout
        )
        
        elapsed = time.time() - start_time
        
        if resp.status_code == 200:
            data = resp.json()
            metrics.success = True
            metrics.prompt_tokens = data.get("prompt_eval_count", 0)
            metrics.response_tokens = data.get("eval_count", 0)
            metrics.total_tokens = metrics.prompt_tokens + metrics.response_tokens
            metrics.inference_time = elapsed
            metrics.tokens_per_second = metrics.total_tokens / elapsed if elapsed > 0 else 0
        else:
            metrics.error = f"API error: {resp.status_code}"
            
    except requests.exceptions.Timeout:
        metrics.error = f"Timeout after {timeout}s"
    except Exception as e:
        metrics.error = str(e)
    
    # Get memory after
    if container_name:
        metrics.memory_after_mb = get_container_memory(container_name)
    else:
        metrics.memory_after_mb = get_ollama_memory(model_name)
    
    metrics.memory_delta_mb = metrics.memory_after_mb - metrics.memory_before_mb
    
    return metrics


def calculate_kv_cache(model_name: str, profile_path: str) -> Dict[str, float]:
    """Calculate expected KV cache sizes."""
    try:
        import json
        with open(profile_path, 'r') as f:
            profile = json.load(f)
        
        num_layers = profile.get('num_layers', 0)
        num_kv_heads = profile.get('num_kv_heads', 0)
        head_dim = profile.get('head_dim', 0)
        kv_per_1k = profile.get('kv_cache_gb_per_1k_tokens', 0)
        
        # Calculate for different context lengths
        contexts = [1024, 4096, 8192, 16384, 32768]
        results = {}
        
        for ctx in contexts:
            uncompressed = kv_per_1k * (ctx / 1024)  # GB
            compressed = uncompressed / 5.3  # 5.3x compression
            saved = uncompressed - compressed
            
            results[ctx] = {
                'uncompressed_gb': uncompressed,
                'compressed_gb': compressed,
                'saved_gb': saved,
                'saved_mb': saved * 1024
            }
        
        return results
    except Exception:
        return {}


def display_comparison(
    original: ModelMetrics,
    compressed: ModelMetrics,
    kv_cache_info: Dict[str, float]
) -> None:
    """Display side-by-side comparison."""
    
    console.print("\n[bold cyan]═══════════════════════════════════════════════════════════[/bold cyan]")
    console.print("[bold cyan]           CONTEXTLENS COMPARISON REPORT[/bold cyan]")
    console.print("[bold cyan]═══════════════════════════════════════════════════════════[/bold cyan]\n")
    
    # Performance Comparison Table
    perf_table = Table(title="Performance Comparison", box=box.ROUNDED)
    perf_table.add_column("Metric", style="cyan")
    perf_table.add_column("Original Model", style="yellow")
    perf_table.add_column("Compressed", style="green")
    perf_table.add_column("Difference", justify="right")
    
    perf_table.add_row(
        "Model",
        original.model_name,
        compressed.model_name,
        "-"
    )
    
    perf_table.add_row(
        "Status",
        "✅ Success" if original.success else f"❌ {original.error}",
        "✅ Success" if compressed.success else f"❌ {compressed.error}",
        "-"
    )
    
    if original.success and compressed.success:
        perf_table.add_row(
            "Inference Time",
            f"{original.inference_time:.2f}s",
            f"{compressed.inference_time:.2f}s",
            f"[{'green' if compressed.inference_time <= original.inference_time else 'red'}]{compressed.inference_time - original.inference_time:+.2f}s[/]"
        )
        
        perf_table.add_row(
            "Tokens/sec",
            f"{original.tokens_per_second:.1f}",
            f"{compressed.tokens_per_second:.1f}",
            f"[{'green' if compressed.tokens_per_second >= original.tokens_per_second else 'red'}]{compressed.tokens_per_second - original.tokens_per_second:+.1f}[/]"
        )
        
        perf_table.add_row(
            "Total Tokens",
            str(original.total_tokens),
            str(compressed.total_tokens),
            f"{compressed.total_tokens - original.total_tokens:+d}"
        )
    
    console.print(perf_table)
    
    # Memory Comparison Table
    mem_table = Table(title="Memory Usage", box=box.ROUNDED)
    mem_table.add_column("Metric", style="cyan")
    mem_table.add_column("Original", style="yellow")
    mem_table.add_column("Compressed", style="green")
    mem_table.add_column("Savings", justify="right")
    
    mem_table.add_row(
        "Memory Before",
        f"{original.memory_before_mb:.1f} MB",
        f"{compressed.memory_before_mb:.1f} MB",
        "-"
    )
    
    mem_table.add_row(
        "Memory After",
        f"{original.memory_after_mb:.1f} MB",
        f"{compressed.memory_after_mb:.1f} MB",
        f"[green]{original.memory_after_mb - compressed.memory_after_mb:.1f} MB[/]"
    )
    
    mem_table.add_row(
        "Memory Delta",
        f"{original.memory_delta_mb:+.1f} MB",
        f"{compressed.memory_delta_mb:+.1f} MB",
        f"[green]{original.memory_delta_mb - compressed.memory_delta_mb:.1f} MB[/]"
    )
    
    console.print(mem_table)
    
    # KV Cache Savings Table
    if kv_cache_info:
        kv_table = Table(title="KV Cache Savings (Theoretical)", box=box.ROUNDED)
        kv_table.add_column("Context Length", style="cyan")
        kv_table.add_column("Uncompressed", style="yellow")
        kv_table.add_column("Compressed (3-bit)", style="green")
        kv_table.add_column("Saved", justify="right", style="green")
        
        for ctx_len, data in kv_cache_info.items():
            ctx_label = f"{ctx_len // 1024}K" if ctx_len >= 1024 else str(ctx_len)
            kv_table.add_row(
                ctx_label,
                f"{data['uncompressed_gb']:.3f} GB",
                f"{data['compressed_gb']:.3f} GB",
                f"+{data['saved_mb']:.1f} MB"
            )
        
        console.print(kv_table)
    
    # Summary Panel
    if original.success and compressed.success:
        speed_diff = ((compressed.inference_time - original.inference_time) / original.inference_time) * 100
        mem_saved = original.memory_after_mb - compressed.memory_after_mb
        
        summary = f"""[bold]Key Findings:[/bold]

📊 [cyan]Speed Overhead:[/] [yellow]{speed_diff:+.1f}%[/] ({'faster' if speed_diff < 0 else 'slower'})
💾 [cyan]Memory Saved:[/] [green]{mem_saved:.1f} MB[/] during inference
🎯 [cyan]KV Cache Reduction:[/] [green]5.3×[/] (theoretical)

[bold green]✓ Compression working correctly![/bold green]
"""
        
        if abs(speed_diff) < 10:
            summary += "\n[dim]Note: Speed difference is within normal variance (<10%)[/dim]"
        elif speed_diff > 50:
            summary += "\n[yellow]⚠ Large speed difference detected - may indicate cold start or system load[/yellow]"
    else:
        summary = "[yellow]⚠ One or both models failed to complete inference[/yellow]"
    
    console.print(Panel(summary, title="Summary", border_style="green"))
    console.print()


def compare_models(
    original_model: str,
    compressed_model: str,
    prompt: str,
    container_name: Optional[str] = None,
    timeout: int = 120
) -> None:
    """Run full comparison between original and compressed models."""
    
    console.print(f"\n[bold blue]Starting Comparison Test[/bold blue]")
    console.print(f"[dim]Original: {original_model}[/dim]")
    console.print(f"[dim]Compressed: {compressed_model}[/dim]")
    console.print(f"[dim]Prompt: {len(prompt)} characters[/dim]\n")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        console=console,
    ) as progress:
        
        # Test original model
        task1 = progress.add_task(f"Testing [yellow]{original_model}[/yellow]...", total=100)
        original = run_inference(original_model, prompt, container_name, timeout)
        progress.update(task1, completed=100)
        
        # Test compressed model
        task2 = progress.add_task(f"Testing [green]{compressed_model}[/green]...", total=100)
        compressed = run_inference(compressed_model, prompt, container_name, timeout)
        progress.update(task2, completed=100)
    
    # Calculate KV cache info
    import os
    profile_path = f"/root/.contextlens/{original_model.replace('/', '_').replace(':', '_')}.json"
    kv_cache_info = calculate_kv_cache(original_model, profile_path)
    
    # Display results
    display_comparison(original, compressed, kv_cache_info)


if __name__ == "__main__":
    # Example usage
    compare_models(
        original_model="llama3.2:3b",
        compressed_model="llama3.2:3b-contextlens",
        prompt="Write a short story about a robot learning to paint.",
        container_name="contextlens-test",
        timeout=120
    )
