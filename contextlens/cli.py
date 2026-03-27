"""
CLI entry point for ContextLens.

Commands:
- scan: Profile KV cache memory usage and context limits
- apply: Apply TurboQuant compression and validate accuracy
- integrate: Patch runtime config to activate compression
- status: Show all compressed models and compression stats
- revert: Remove compression and restore original runtime config
"""

from __future__ import annotations

from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from .scanner import scan_model
from .compressor import TurboQuantCompressor
from .profiles import save_profile, load_profile, list_profiles, ModelProfile
from .integrations.ollama import apply_to_ollama, revert_ollama, get_modelfile
from .integrations.llamacpp import write_config_file

app = typer.Typer(
    name="contextlens",
    help="Compress your local LLM KV cache with zero accuracy loss.",
    add_completion=False,
)

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
                
                console.print("[dim]Loading model...[/dim]")
                tokenizer = AutoTokenizer.from_pretrained(model)
                model_obj = AutoModelForCausalLM.from_pretrained(
                    model,
                    torch_dtype="auto",
                    device_map="auto",
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
