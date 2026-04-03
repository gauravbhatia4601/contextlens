"""CLI entry point for llm-contextlens.

Commands:
- scan: Profile KV cache memory usage and context limits
- apply: Apply TurboQuant compression and validate accuracy
- status: Show all compressed models and compression stats
- hf-auth: Manage HuggingFace authentication
"""

from __future__ import annotations

import signal
import sys
import time

from typing import Optional
import typer

import typer
from rich.console import Console
from rich.table import Table

from .scanner import scan_model
from .compressor import TurboQuantCompressor
from .profiles import save_profile, load_profile, list_profiles
from .integrations.huggingface import patch_model_for_contextlens

app = typer.Typer(
    name="llm-contextlens",
    help="Compress your local LLM KV cache with 5.3× memory reduction. Package: llm-contextlens",
    add_completion=False,
)


def _handle_interrupt(signum, frame):
    """Handle Ctrl+C gracefully without showing traceback."""
    console.print("\n[yellow]Operation cancelled by user.[/yellow]")
    sys.exit(0)


signal.signal(signal.SIGINT, _handle_interrupt)


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    version: bool = typer.Option(False, "--version", "-v", help="Show version and exit"),
):
    """ContextLens - Compress your local LLM KV cache with 5.3× memory reduction."""
    if version:
        from importlib.metadata import version as get_version
        console.print(f"llm-contextlens {get_version('llm-contextlens')}")
        raise typer.Exit(0)
    
    # Check for updates when a command is invoked
    if ctx.invoked_subcommand is not None:
        try:
            from importlib.metadata import version as get_version
            import requests
            
            current_version = get_version("llm-contextlens")
            
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
    model: str = typer.Argument(..., help="HuggingFace model ID (e.g. Qwen/Qwen2-0.5B)"),
) -> None:
    """Profile KV cache memory usage and context limits for a model."""
    try:
        profile = scan_model(model)

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

    except FileNotFoundError as exc:
        _handle_error(
            f"Model '{model}' not found in local cache.\n"
            f"Download it first:\n"
            f"  huggingface-cli download {model}",
            exc
        )
    except (RuntimeError, NotImplementedError) as exc:
        _handle_error(str(exc), exc)
    except Exception as exc:
        _handle_error(f"Unexpected error: {exc}", exc)


@app.command()
def apply(
    model: str = typer.Argument(..., help="HuggingFace model ID to compress (e.g. Qwen/Qwen2-0.5B)"),
    bits: int = typer.Option(3, help="Bits per KV value (3 recommended)"),
    skip_benchmark: bool = typer.Option(False, "--skip-benchmark", help="Skip accuracy benchmark"),
    force: bool = typer.Option(False, "--force", help="Apply even if accuracy delta > 1%"),
    dataset: str = typer.Option("mmlu", help="Benchmark dataset: mmlu | hellaswag"),
    n_questions: int = typer.Option(500, help="Number of benchmark questions"),
) -> None:
    """Apply TurboQuant compression and validate accuracy."""
    try:
        console.print(f"[bold green]Scanning model:[/bold green] {model}")
        profile = scan_model(model)

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

                console.print(f"[dim]Loading model from HuggingFace: {model}[/dim]")
                console.print("[dim]Note: This may require HuggingFace authentication for gated models.[/dim]")

                tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
                model_obj = AutoModelForCausalLM.from_pretrained(
                    model,
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
                        f"\n[dim]This model requires HuggingFace access. You have 2 options:[/dim]"
                    )
                    console.print(
                        f"\n[cyan]Option 1:[/cyan] Log in to HuggingFace"
                    )
                    console.print(
                        f"  huggingface-cli login"
                    )
                    console.print(
                        f"  [dim]# Then retry the apply command[/dim]"
                    )
                    console.print(
                        f"\n[cyan]Option 2:[/cyan] Skip benchmark"
                    )
                    console.print(
                        f"  contextlens apply {model} --skip-benchmark"
                    )
                    console.print(
                        f"\n[dim]Skipping benchmark and proceeding with compression...[/dim]"
                    )
                elif "404" in error_msg or "not found" in error_msg.lower():
                    console.print(
                        f"[yellow]Warning: Model '{model}' not found on HuggingFace.[/yellow]"
                    )
                    console.print(
                        f"[dim]Skipping benchmark.[/dim]"
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
        console.print(f"\n[dim]Profile saved. Use patch_model_for_contextlens() in your code to activate.[/dim]")

    except FileNotFoundError as exc:
        _handle_error(
            f"Model '{model}' not found in local cache.\n"
            f"Download it first:\n"
            f"  huggingface-cli download {model}",
            exc
        )
    except (RuntimeError, NotImplementedError) as exc:
        _handle_error(str(exc), exc)
    except Exception as exc:
        _handle_error(f"Unexpected error: {exc}", exc)


@app.command()
def integrate(
    model: str = typer.Argument(..., help="Model to activate compression for"),
) -> None:
    """Show how to activate compression in your HuggingFace code."""
    try:
        profile = load_profile(model)

        console.print(f"[bold green]Model profile loaded:[/bold green] {model}")
        console.print(f"\n[bold yellow]HuggingFace integration:[/bold yellow]")
        console.print("Use patch_model_for_contextlens() in your code:")
        console.print("")
        console.print("  [cyan]from contextlens.integrations.huggingface import patch_model_for_contextlens[/cyan]")
        console.print("  [cyan]model = patch_model_for_contextlens(model)[/cyan]")
        console.print("")
        console.print("[dim]Compression is applied when loading the model in your Python code.[/dim]")

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
    """Manage HuggingFace authentication for gated models."""
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
                    console.print(f"[dim]Token is set but not validated. Try loading a gated model to test.[/dim]")
            else:
                console.print(f"[yellow]✗ Not logged in[/yellow]")
                console.print(f"\n[dim]To enable gated models, run:[/dim]")
                console.print(f"  huggingface-cli login")

    except ImportError as exc:
        console.print(f"[yellow]huggingface_hub not installed.[/yellow]")
        console.print(f"\n[dim]Install with:[/dim]")
        console.print(f"  pip install huggingface_hub")
        console.print(f"\n[dim]Error:[/dim] {exc}")


@app.command()
def revert(model: str = typer.Argument(..., help="Model name to remove profile")) -> None:
    """Remove a model's compression profile."""
    try:
        from .profiles import delete_profile

        try:
            delete_profile(model)
            console.print(f"[bold green]Profile removed for:[/bold green] {model}")
        except FileNotFoundError:
            console.print(f"[bold yellow]No profile found for '{model}' - nothing to revert.[/bold yellow]")

    except RuntimeError as exc:
        _handle_error(str(exc), exc)
    except Exception as exc:
        _handle_error(f"Unexpected error: {exc}", exc)


if __name__ == "__main__":
    app()