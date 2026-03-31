"""CLI entry point for llm-contextlens.

Commands:
- scan: Profile KV cache memory usage and context limits
- apply: Apply TurboQuant compression and validate accuracy
- integrate: Patch runtime config to activate compression
- status: Show all compressed models and compression stats
- revert: Remove compression and restore original runtime config
"""

from __future__ import annotations

import time

from typing import Optional
import typer

import typer
from rich.console import Console
from rich.table import Table

from .scanner import scan_model
from .compressor import TurboQuantCompressor
from .profiles import save_profile, load_profile, list_profiles, ModelProfile
from .integrations.ollama import apply_to_ollama, revert_ollama, get_modelfile
from .integrations.llamacpp import write_config_file


def _get_hf_model_id(ollama_model: str, prefer_open: bool = True) -> str:
    """Map Ollama model names to HuggingFace model IDs for benchmarking.
    
    Args:
        ollama_model: Ollama model name (e.g., "llama3.2:3b")
        prefer_open: If True, use open-weight alternatives when available
    
    Returns:
        HuggingFace model ID (e.g., "meta-llama/Llama-3.2-3B-Instruct")
    """
    # Comprehensive Ollama -> HuggingFace mappings
    # Priority: Open weights > Gated (if user has access)
    mappings = {
        # Llama 3.2 (use Qwen as open alternative)
        "llama3.2:1b": "Qwen/Qwen2.5-0.5B-Instruct" if prefer_open else "meta-llama/Llama-3.2-1B-Instruct",
        "llama3.2:3b": "Qwen/Qwen2.5-3B-Instruct" if prefer_open else "meta-llama/Llama-3.2-3B-Instruct",
        # Llama 3.1 (use Qwen as open alternative)
        "llama3.1:8b": "Qwen/Qwen2.5-7B-Instruct" if prefer_open else "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "llama3.1:70b": "Qwen/Qwen2.5-72B-Instruct" if prefer_open else "meta-llama/Meta-Llama-3.1-70B-Instruct",
        # Llama 3
        "llama3:8b": "Qwen/Qwen2.5-7B-Instruct" if prefer_open else "meta-llama/Meta-Llama-3-8B-Instruct",
        "llama3:70b": "Qwen/Qwen2.5-72B-Instruct" if prefer_open else "meta-llama/Meta-Llama-3-70B-Instruct",
        # Mistral (all open weights)
        "mistral:7b": "mistralai/Mistral-7B-Instruct-v0.3",
        "mistral:7b-v0.1": "mistralai/Mistral-7B-Instruct-v0.1",
        "mistral:7b-v0.2": "mistralai/Mistral-7B-Instruct-v0.2",
        "mixtral:8x7b": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "mixtral:8x22b": "mistralai/Mixtral-8x22B-Instruct-v0.1",
        # Phi-3 (all open weights)
        "phi3:3.8b": "microsoft/Phi-3-mini-4k-instruct",
        "phi3:14b": "microsoft/Phi-3-medium-4k-instruct",
        "phi3:7b": "microsoft/Phi-3-small-8k-instruct",
        # Gemma (all open weights)
        "gemma:2b": "google/gemma-2b-it",
        "gemma:7b": "google/gemma-7b-it",
        "gemma2:2b": "google/gemma-2-2b-it",
        "gemma2:9b": "google/gemma-2-9b-it",
        "gemma2:27b": "google/gemma-2-27b-it",
        # Qwen (all open weights)
        "qwen2:0.5b": "Qwen/Qwen2-0.5B-Instruct",
        "qwen2:1.5b": "Qwen/Qwen2-1.5B-Instruct",
        "qwen2:7b": "Qwen/Qwen2-7B-Instruct",
        "qwen2:72b": "Qwen/Qwen2-72B-Instruct",
        "qwen2.5:0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
        "qwen2.5:1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
        "qwen2.5:3b": "Qwen/Qwen2.5-3B-Instruct",
        "qwen2.5:7b": "Qwen/Qwen2.5-7B-Instruct",
        "qwen2.5:14b": "Qwen/Qwen2.5-14B-Instruct",
        "qwen2.5:32b": "Qwen/Qwen2.5-32B-Instruct",
        "qwen2.5:72b": "Qwen/Qwen2.5-72B-Instruct",
        # Yi (open weights)
        "yi:6b": "01-ai/Yi-6B-Chat",
        "yi:34b": "01-ai/Yi-34B-Chat",
        # StableLM (open weights)
        "stablelm:3b": "stabilityai/stablelm-3b-4e1t",
    }
    
    # Check for exact match first
    if ollama_model.lower() in mappings:
        return mappings[ollama_model.lower()]
    
    # Check for partial matches (more flexible)
    ollama_lower = ollama_model.lower()
    for key, value in mappings.items():
        # Extract base model name (e.g., "llama3.2" from "llama3.2:3b")
        base_key = key.split(":")[0]
        if base_key in ollama_lower:
            return value
    
    # Default: return as-is (might work for direct HF paths)
    return ollama_model

app = typer.Typer(
    name="llm-contextlens",
    help="Compress your local LLM KV cache with 5.3× memory reduction. Package: llm-contextlens",
    add_completion=False,
)


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context):
    """ContextLens - Compress your local LLM KV cache with 5.3× memory reduction."""
    # Check for updates when a command is invoked
    if ctx.invoked_subcommand is not None:
        try:
            from importlib.metadata import version
            import requests
            
            current_version = version("llm-contextlens")
            
            # Check PyPI for latest version
            resp = requests.get("https://pypi.org/pypi/llm-contextlens/json", timeout=2)
            if resp.status_code == 200:
                latest_version = resp.json()["info"]["version"]
                
                if latest_version != current_version:
                    console.print(f"\n[yellow]⚠️  Update available![/yellow]")
                    console.print(f"[dim]Current: {current_version} → Latest: {latest_version}[/dim]")
                    console.print(f"[dim]Upgrade: [cyan]pip install --upgrade llm-contextlens[/cyan]\n")
        except Exception:
            # Silently fail if version check doesn't work
            pass
    
    if ctx.invoked_subcommand is None:
        # Show help when no command is provided
        print(ctx.get_help())
        raise typer.Exit()

console = Console()


def _handle_error(message: str, exc: Optional[Exception] = None) -> None:
    """Print a formatted error message and exit."""
    console.print(f"[bold red]Error:[/bold red] {message}")
    if exc:
        console.print(f"[dim]{exc}[/dim]")
    raise typer.Exit(1)


@app.command()
def scan(
    model: str = typer.Argument(..., help="Model name (e.g. llama3.1:70b or a HF model path)"),
    runtime: str = typer.Option("auto", help="ollama | llamacpp | huggingface | auto"),
) -> None:
    """Profile KV cache memory usage and context limits for a model."""
    try:
        profile = scan_model(model, runtime)
        
        console.print(f"\n[bold green]Model:[/bold green] {profile.model_id}")
        console.print(
            f"[bold green]Architecture:[/bold green] {profile.num_layers} layers, "
            f"{profile.num_kv_heads} KV heads, {profile.head_dim} head dim"
        )
        console.print(f"[bold green]Dtype:[/bold green] {profile.dtype}")
        console.print(f"\n[bold yellow]KV Cache Memory:[/bold yellow]")
        console.print(f"  Per 1k tokens: [cyan]{profile.kv_cache_gb_per_1k_tokens:.2f} GB[/cyan]")
        
        console.print(f"\n[bold yellow]Max Context Length:[/bold yellow]")
        for ram in [16, 32, 64]:
            max_ctx = profile.max_context_at_ram(ram)
            console.print(f"  {ram} GB RAM: [cyan]{max_ctx:,} tokens[/cyan]")
            
    except FileNotFoundError:
        _handle_error(f"Model '{model}' not found. Pull it first: ollama pull {model}")
    except (RuntimeError, NotImplementedError) as exc:
        _handle_error(str(exc), exc)
    except Exception as exc:
        _handle_error(f"Unexpected error: {exc}", exc)


@app.command()
def apply(
    model: str = typer.Argument(..., help="Model name to compress"),
    bits: int = typer.Option(3, help="Bits per KV value (3 recommended)"),
    skip_benchmark: bool = typer.Option(False, "--skip-benchmark", help="Skip accuracy benchmark"),
    force: bool = typer.Option(False, "--force", help="Apply even if accuracy delta > 1%"),
    dataset: str = typer.Option("mmlu", help="Benchmark dataset: mmlu | hellaswag"),
    n_questions: int = typer.Option(500, help="Number of benchmark questions"),
    use_open_weights: bool = typer.Option(True, "--use-open-weights/--use-gated", help="Use open-weight alternatives (Qwen) vs gated models (Llama)"),
) -> None:
    """Apply TurboQuant compression and validate accuracy."""
    try:
        console.print(f"[bold green]Scanning model:[/bold green] {model}")
        profile = scan_model(model, "auto")
        
        console.print(f"[bold green]Initializing compressor:[/bold green] {bits}-bit TurboQuant")
        compressor = TurboQuantCompressor(bits=bits)
        
        accuracy_delta = 0.0
        accuracy_before = 0.0
        accuracy_after = 0.0
        
        if not skip_benchmark:
            console.print(
                f"[bold yellow]Running accuracy benchmark ({dataset.upper()} {n_questions} questions)...[/bold yellow]"
            )
            console.print("[dim]This may take a few minutes...[/dim]")
            
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer
                from .benchmarks import run_accuracy_benchmark, BenchmarkResult
                
                # Map Ollama model name to HuggingFace ID for benchmarking
                hf_model_id = _get_hf_model_id(model, prefer_open=use_open_weights)
                
                if use_open_weights and hf_model_id.startswith("Qwen/"):
                    console.print(f"[dim]Loading open-weight model: {hf_model_id}[/dim]")
                else:
                    console.print(f"[dim]Loading model from HuggingFace: {hf_model_id}[/dim]")
                    console.print("[dim]Note: This may require HuggingFace authentication for gated models.[/dim]")
                
                tokenizer = AutoTokenizer.from_pretrained(hf_model_id, trust_remote_code=True)
                model_obj = AutoModelForCausalLM.from_pretrained(
                    hf_model_id,
                    torch_dtype="auto",
                    device_map="auto",
                    trust_remote_code=True,
                )
                
                result: BenchmarkResult = run_accuracy_benchmark(
                    model_name=model,
                    model=model_obj,
                    tokenizer=tokenizer,
                    compressor=compressor,
                    dataset=dataset,
                    n_questions=n_questions,
                    device="cuda" if model_obj.device.type == "cuda" else "cpu",
                )
                
                accuracy_before = result.accuracy_before
                accuracy_after = result.accuracy_after
                accuracy_delta = result.accuracy_delta
                
                if not result.passed:
                    console.print(
                        f"[bold red]Warning:[/bold red] Accuracy dropped {abs(accuracy_delta)*100:.2f}% "
                        f"— above 1% threshold."
                    )
                    if not force:
                        console.print(
                            "[yellow]Use --force to apply anyway, or skip benchmark with --skip-benchmark[/yellow]"
                        )
                        raise typer.Exit(1)
                    else:
                        console.print(
                            "[bold yellow]Proceeding with --force flag despite accuracy degradation.[/bold yellow]"
                        )
                else:
                    console.print(f"[bold green]Benchmark passed![/bold green]")
                    console.print(
                        f"  Baseline: {accuracy_before:.4f} -> Compressed: {accuracy_after:.4f} "
                        f"(delta: {accuracy_delta:+.4f})"
                    )
                    
            except ImportError as exc:
                console.print(f"[yellow]Warning: Could not run benchmark - {exc}[/yellow]")
                console.print("[dim]Proceeding without accuracy validation...[/dim]")
            except RuntimeError as exc:
                if "CUDA out of memory" in str(exc):
                    _handle_error(
                        "Benchmark OOM. Retry with --benchmark-batch-size 1 or use a smaller model"
                    )
                else:
                    _handle_error(f"Benchmark failed: {exc}")
            except Exception as exc:
                error_msg = str(exc)
                if "gated repo" in error_msg or "401" in error_msg or "403" in error_msg:
                    console.print(
                        f"[yellow]Warning: HuggingFace model requires authentication.[/yellow]"
                    )
                    console.print(
                        f"\n[dim]This model requires HuggingFace access. You have 3 options:[/dim]"
                    )
                    console.print(
                        f"\n[cyan]Option 1:[/cyan] Use open-weight alternatives (recommended)"
                    )
                    console.print(
                        f"  [dim]# Use Qwen instead of Llama (no auth needed)[/dim]"
                    )
                    console.print(
                        f"  contextlens apply {model} --use-open-weights"
                    )
                    console.print(
                        f"\n[cyan]Option 2:[/cyan] Log in to HuggingFace"
                    )
                    console.print(
                        f"  [dim]# Install huggingface-cli and login[/dim]"
                    )
                    console.print(
                        f"  pip install huggingface_hub"
                    )
                    console.print(
                        f"  huggingface-cli login"
                    )
                    console.print(
                        f"  [dim]# Then retry with gated models[/dim]"
                    )
                    console.print(
                        f"  contextlens apply {model} --use-gated"
                    )
                    console.print(
                        f"\n[cyan]Option 3:[/cyan] Skip benchmark"
                    )
                    console.print(
                        f"  contextlens apply {model} --skip-benchmark"
                    )
                    console.print(
                        f"\n[dim]Skipping benchmark and proceeding with compression...[/dim]"
                    )
                elif "404" in error_msg or "not found" in error_msg.lower():
                    console.print(
                        f"[yellow]Warning: Model '{hf_model_id}' not found on HuggingFace.[/yellow]"
                    )
                    console.print(
                        f"[dim]Skipping benchmark. The model mapping may need updating.[/dim]"
                    )
                else:
                    console.print(f"[yellow]Warning: Benchmark error - {exc}[/yellow]")
                    console.print("[dim]Proceeding without accuracy validation...[/dim]")
        
        profile_path = save_profile(profile)
        console.print(f"[bold green]Saved profile:[/bold green] {profile_path}")
        
        console.print(f"\n[bold green]Compression applied successfully![/bold green]")
        if not skip_benchmark:
            console.print(
                f"Accuracy delta: [green]{accuracy_delta:+.4f}[/green] "
                f"(baseline: {accuracy_before:.4f}, compressed: {accuracy_after:.4f})"
            )
        console.print(f"\nRun [cyan]contextlens integrate ollama --model {model}[/cyan] to activate.")
        
    except FileNotFoundError:
        _handle_error(f"Model '{model}' not found. Pull it first: ollama pull {model}")
    except (RuntimeError, NotImplementedError) as exc:
        _handle_error(str(exc), exc)
    except Exception as exc:
        _handle_error(f"Unexpected error: {exc}", exc)


@app.command()
def integrate(
    runtime: str = typer.Argument(..., help="ollama | llamacpp | huggingface"),
    model: str = typer.Option(None, help="Specific model to patch (optional for some runtimes)"),
) -> None:
    """Patch runtime config to activate compression for all sessions."""
    try:
        if runtime == "ollama":
            if not model:
                console.print("[bold red]Error:[/bold red] --model is required for Ollama integration")
                raise typer.Exit(1)
            
            profile = load_profile(model)
            
            console.print(f"[bold green]Patching Ollama Modelfile for:[/bold green] {model}")
            apply_to_ollama(model, str(save_profile(profile)))
            console.print(f"[bold green]Integration complete![/bold green]")
            
        elif runtime == "llamacpp":
            if not model:
                console.print("[bold red]Error:[/bold red] --model is required for llama.cpp integration")
                raise typer.Exit(1)
            
            profile = load_profile(model)
            
            config_path = f"~/.contextlens/{model.replace('/', '_').replace(':', '_')}.conf"
            write_config_file(profile, config_path)
            console.print(f"[bold green]Config written to:[/bold green] {config_path}")
            console.print(f"Use with: llama.cpp --config {config_path}")
            
        elif runtime == "huggingface":
            console.print(
                "[bold yellow]HuggingFace integration:[/bold yellow] "
                "Use patch_model_for_contextlens() in your code"
            )
            console.print("Example:")
            console.print("  from contextlens.integrations.huggingface import patch_model_for_contextlens")
            console.print("  model = patch_model_for_contextlens(model)")
            
        else:
            _handle_error(f"Unknown runtime '{runtime}'")
            
    except FileNotFoundError as exc:
        _handle_error(f"Profile not found: {exc}", exc)
    except RuntimeError as exc:
        _handle_error(str(exc), exc)
    except Exception as exc:
        _handle_error(f"Unexpected error: {exc}", exc)


@app.command()
def status() -> None:
    """Show all compressed models and compression stats."""
    try:
        profiles = list_profiles()
        
        if not profiles:
            console.print("[yellow]No compressed models found.[/yellow]")
            console.print("Run [cyan]contextlens apply <model>[/cyan] to compress a model.")
            return
        
        table = Table(title="ContextLens Compressed Models")
        table.add_column("Model", style="cyan")
        table.add_column("Layers", justify="right")
        table.add_column("KV Heads", justify="right")
        table.add_column("Head Dim", justify="right")
        table.add_column("KV/1k tokens", justify="right")
        
        for profile in profiles:
            table.add_row(
                profile.model_id,
                str(profile.num_layers),
                str(profile.num_kv_heads),
                str(profile.head_dim),
                f"{profile.kv_cache_gb_per_1k_tokens:.2f} GB",
            )
        
        console.print(table)
        
    except Exception as exc:
        _handle_error(f"Error loading profiles: {exc}", exc)


@app.command()
def hf_auth(check: bool = typer.Option(False, "--check", help="Check authentication status"),
            login: bool = typer.Option(False, "--login", help="Log in to HuggingFace")) -> None:
    """Manage HuggingFace authentication for benchmarking."""
    try:
        from huggingface_hub import whoami
        import os
        
        if login:
            console.print("[bold blue]HuggingFace Login[/bold blue]")
            console.print("\n[dim]To log in, run:[/dim]")
            console.print("  huggingface-cli login")
            console.print("\n[dim]Or set environment variable:[/dim]")
            console.print("  export HF_TOKEN=your_token_here")
            return
        
        if check:
            console.print("[bold blue]HuggingFace Authentication Status[/bold blue]\n")
            
            # Check for token in environment or cache
            token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
            
            if token:
                try:
                    user_info = whoami(token=token)
                    console.print(f"[green]✓ Logged in as:[/green] {user_info.get('name', 'Unknown')}")
                    console.print(f"[dim]Email: {user_info.get('email', 'Unknown')}[/dim]")
                    
                    # Check if user has access to gated models
                    console.print(f"\n[yellow]Note:[/yellow] Login successful, but you may need to:")
                    console.print(f"  1. Accept model licenses at: https://huggingface.co/settings/accepted")
                    console.print(f"  2. Request access for specific models (e.g., Llama, Gemma)")
                except Exception:
                    console.print(f"[green]✓ Token found in environment[/green]")
                    console.print(f"[dim]Token is set but not validated. Try running a benchmark to test.[/dim]")
            else:
                console.print(f"[yellow]✗ Not logged in[/yellow]")
                console.print(f"\n[dim]To enable gated models, run:[/dim]")
                console.print(f"  huggingface-cli login")
                console.print(f"\n[dim]Or use open-weight alternatives:[/dim]")
                console.print(f"  llm-contextlens apply <model> --use-open-weights")
                
    except ImportError as exc:
        console.print(f"[yellow]huggingface_hub not installed.[/yellow]")
        console.print(f"\n[dim]Install with:[/dim]")
        console.print(f"  pip install huggingface_hub")
        console.print(f"\n[dim]Error:[/dim] {exc}")


@app.command()
def revert(model: str = typer.Argument(..., help="Model name to revert compression")) -> None:
    """Remove compression and restore original runtime config."""
    try:
        try:
            profile = load_profile(model)
        except FileNotFoundError:
            console.print(f"[bold yellow]No profile found for '{model}' - nothing to revert.[/bold yellow]")
            return
        
        console.print(f"[bold yellow]Reverting compression for:[/bold yellow] {model}")
        
        try:
            original_modelfile = get_modelfile(model)
            revert_ollama(model, original_modelfile)
            console.print(f"[bold green]Ollama Modelfile restored.[/bold green]")
        except Exception:
            console.print("[dim]Could not revert Ollama integration (model may not exist)[/dim]")
        
        console.print(f"[bold green]Reversion complete![/bold green]")
        
    except RuntimeError as exc:
        _handle_error(str(exc), exc)
    except Exception as exc:
        _handle_error(f"Unexpected error: {exc}", exc)


if __name__ == "__main__":
    app()


@app.command()
def compare(
    original_model: str = typer.Argument(..., help="Original model name (e.g., llama3.2:3b)"),
    compressed_model: str = typer.Argument(None, help="Compressed model name (default: <original>-contextlens)"),
    prompt: str = typer.Option("Write a short story about AI.", "-p", "--prompt", help="Prompt for inference"),
    prompt_file: str = typer.Option(None, "-f", "--file", help="Read prompt from file"),
    container: str = typer.Option(None, "-c", "--container", help="Docker container name (optional)"),
    timeout: int = typer.Option(120, "-t", "--timeout", help="Timeout in seconds"),
    iterations: int = typer.Option(1, "-n", "--iterations", help="Number of test iterations"),
) -> None:
    """Compare original vs compressed model performance in real-time."""
    try:
        from .compare import compare_models, ModelMetrics, run_inference, display_comparison, calculate_kv_cache
        from rich.panel import Panel
        from rich import box
        
        # Determine compressed model name
        if not compressed_model:
            compressed_model = f"{original_model}-contextlens"
        
        # Get prompt from file if specified
        if prompt_file:
            try:
                with open(prompt_file, 'r') as f:
                    prompt = f.read().strip()
            except FileNotFoundError:
                _handle_error(f"Prompt file not found: {prompt_file}")
        
        console.print(Panel.fit(
            f"[bold cyan]ContextLens Comparison Test[/bold cyan]\n\n"
            f"[yellow]Original:[/] {original_model}\n"
            f"[green]Compressed:[/] {compressed_model}\n"
            f"[dim]Prompt:[/] {len(prompt)} chars\n"
            f"[dim]Iterations:[/] {iterations}",
            title="Test Configuration",
            border_style="cyan"
        ))
        
        if iterations == 1:
            # Single comparison
            compare_models(
                original_model=original_model,
                compressed_model=compressed_model,
                prompt=prompt,
                container_name=container,
                timeout=timeout
            )
        else:
            # Multiple iterations for averaging
            console.print(f"\n[bold blue]Running {iterations} iterations...[/bold blue]\n")
            
            original_metrics = []
            compressed_metrics = []
            
            for i in range(iterations):
                console.print(f"\n[dim]━━━ Iteration {i+1}/{iterations} ━━━[/dim]\n")
                
                # Test original
                console.print(f"[yellow]Testing {original_model}...[/yellow]")
                orig = run_inference(original_model, prompt, container, timeout)
                if orig.success:
                    original_metrics.append(orig)
                    console.print(f"  ✓ {orig.total_tokens} tokens in {orig.inference_time:.2f}s ({orig.tokens_per_second:.1f} tok/s)")
                else:
                    console.print(f"  ❌ {orig.error}")
                
                # Test compressed
                console.print(f"[green]Testing {compressed_model}...[/green]")
                comp = run_inference(compressed_model, prompt, container, timeout)
                if comp.success:
                    compressed_metrics.append(comp)
                    console.print(f"  ✓ {comp.total_tokens} tokens in {comp.inference_time:.2f}s ({comp.tokens_per_second:.1f} tok/s)")
                else:
                    console.print(f"  ❌ {comp.error}")
                
                # Small delay between iterations
                if i < iterations - 1:
                    time.sleep(2)
            
            # Calculate averages
            if original_metrics and compressed_metrics:
                avg_original = ModelMetrics(
                    model_name=original_model,
                    inference_time=sum(m.inference_time for m in original_metrics) / len(original_metrics),
                    tokens_per_second=sum(m.tokens_per_second for m in original_metrics) / len(original_metrics),
                    total_tokens=int(sum(m.total_tokens for m in original_metrics) / len(original_metrics)),
                    memory_delta_mb=sum(m.memory_delta_mb for m in original_metrics) / len(original_metrics),
                    success=True
                )
                
                avg_compressed = ModelMetrics(
                    model_name=compressed_model,
                    inference_time=sum(m.inference_time for m in compressed_metrics) / len(compressed_metrics),
                    tokens_per_second=sum(m.tokens_per_second for m in compressed_metrics) / len(compressed_metrics),
                    total_tokens=int(sum(m.total_tokens for m in compressed_metrics) / len(compressed_metrics)),
                    memory_delta_mb=sum(m.memory_delta_mb for m in compressed_metrics) / len(compressed_metrics),
                    success=True
                )
                
                # Get KV cache info
                profile_path = f"/root/.contextlens/{original_model.replace('/', '_').replace(':', '_')}.json"
                kv_cache_info = calculate_kv_cache(original_model, profile_path)
                
                console.print("\n")
                display_comparison(avg_original, avg_compressed, kv_cache_info)
            else:
                console.print("[bold red]No successful runs to average![/bold red]")
        
    except ImportError as exc:
        _handle_error(f"Could not import comparison module: {exc}")
    except Exception as exc:
        _handle_error(f"Comparison failed: {exc}", exc)


@app.command()
def uninstall(
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be removed without deleting"),
    force: bool = typer.Option(False, "--force", help="Skip confirmation prompt"),
) -> None:
    """Uninstall ContextLens and remove all profiles and configurations.
    
    This will:
    - Uninstall the llm-contextlens package
    - Remove all profile files (~/.contextlens/)
    - Remove all compressed model variants from Ollama
    - Clean up any llama.cpp configurations
    
    Use --dry-run to see what would be removed without actually deleting anything.
    """
    import os
    import shutil
    import subprocess
    from pathlib import Path
    
    console.print("[bold yellow]⚠️  ContextLens Uninstall[/bold yellow]\n")
    console.print("[dim]This will remove:[/dim]")
    console.print("  • llm-contextlens package")
    console.print("  • All profile files (~/.contextlens/)")
    console.print("  • All compressed model variants (ollama)")
    console.print("  • All llama.cpp configurations\n")
    
    # Collect what will be removed
    items_to_remove = []
    
    # 1. Profile directory
    profile_dir = Path.home() / ".contextlens"
    if profile_dir.exists():
        profiles = list(profile_dir.glob("*.json"))
        if profiles:
            items_to_remove.append(f"📁 Profile directory: {profile_dir} ({len(profiles)} profiles)")
    
    # 2. Ollama compressed models
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            compressed_models = []
            for line in result.stdout.splitlines()[1:]:  # Skip header
                if "-contextlens" in line:
                    model_name = line.split()[0]
                    compressed_models.append(model_name)
            
            if compressed_models:
                items_to_remove.append(f"🤖 Ollama models: {', '.join(compressed_models)}")
    except Exception:
        pass
    
    # 3. Package itself
    items_to_remove.append("📦 Package: llm-contextlens (via pip uninstall)")
    
    # Display what will be removed
    console.print("[bold]Will remove:[/bold]")
    for item in items_to_remove:
        console.print(f"  {item}")
    
    if not items_to_remove:
        console.print("[green]✓ Nothing to remove[/green]")
        return
    
    console.print()
    
    # Dry run - just show what would be removed
    if dry_run:
        console.print("[bold yellow]🔍 DRY RUN - No changes made[/bold yellow]")
        console.print("[dim]Run without --dry-run to actually remove these items[/dim]")
        return
    
    # Confirm before proceeding
    if not force:
        confirm = typer.confirm("[bold red]Are you sure you want to uninstall ContextLens?[/bold red]")
        if not confirm:
            console.print("[yellow]Uninstall cancelled[/yellow]")
            return
    
    # Execute uninstall
    console.print("\n[bold]Starting uninstall...[/bold]\n")
    
    # 1. Remove Ollama compressed models
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            for line in result.stdout.splitlines()[1:]:
                if "-contextlens" in line:
                    model_name = line.split()[0]
                    console.print(f"[dim]Removing Ollama model: {model_name}...[/dim]")
                    subprocess.run(
                        ["ollama", "rm", model_name],
                        capture_output=True,
                        timeout=30
                    )
            console.print("[green]✓ Removed compressed Ollama models[/green]")
    except Exception as exc:
        console.print(f"[yellow]⚠ Could not remove Ollama models: {exc}[/yellow]")
    
    # 2. Remove profile directory
    if profile_dir.exists():
        try:
            shutil.rmtree(profile_dir)
            console.print(f"[green]✓ Removed profile directory: {profile_dir}[/green]")
        except Exception as exc:
            console.print(f"[yellow]⚠ Could not remove profile directory: {exc}[/yellow]")
    
    # 3. Uninstall package
    console.print("\n[dim]To complete uninstall, run:[/dim]")
    console.print("[cyan]pip uninstall llm-contextlens[/cyan]\n")
    
    console.print("[bold green]✓ ContextLens uninstall complete![/bold green]")
    console.print("[dim]Note: The pip uninstall command above will remove the package itself.[/dim]")

