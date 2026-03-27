"""
Accuracy benchmarking for ContextLens (Phase 3).

Provides MMLU and HellaSwag benchmark runners with Rich progress bars
to validate that compression does not degrade model accuracy beyond the 1% threshold.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from rich.console import Console
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn

console = Console()


@dataclass
class BenchmarkResult:
    """Container for benchmark results.

    Attributes:
        model_id: Model identifier
        dataset: Name of the benchmark dataset (e.g., "mmlu", "hellaswag")
        n_questions: Number of questions evaluated
        accuracy_before: Accuracy score before compression (0.0-1.0)
        accuracy_after: Accuracy score after compression (0.0-1.0)
        accuracy_delta: Difference (after - before), negative means degradation
        passed: Whether the accuracy delta is within the 1% threshold
    """
    model_id: str
    dataset: str
    n_questions: int
    accuracy_before: float
    accuracy_after: float
    accuracy_delta: float
    passed: bool


class MMLUBenchmark:
    """MMLU benchmark runner using a 500-question subset.

    Args:
        n_questions: Number of questions to sample (default 500)
        seed: Random seed for reproducibility (default 42)
    """

    def __init__(self, n_questions: int = 500, seed: int = 42):
        self.n_questions = n_questions
        self.seed = seed
        self._dataset: Optional[List[Dict[str, Any]]] = None

    def load_dataset(self) -> List[Dict[str, Any]]:
        """Load and sample the MMLU dataset.

        Returns:
            List of question dictionaries with 'question', 'choices', and 'answer'
        """
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "datasets library is required for benchmarking. "
                "Install with: pip install datasets"
            )

        if self._dataset is not None:
            return self._dataset

        console.print("[dim]Loading MMLU dataset...[/dim]")
        dataset = load_dataset("cais/mmlu", "all", split="test")

        torch.manual_seed(self.seed)
        indices = torch.randperm(len(dataset))[: self.n_questions].tolist()
        self._dataset = [dataset[i] for i in indices]
        return self._dataset

    def run(
        self,
        model: Any,
        tokenizer: Any,
        device: str = "cuda",
        show_progress: bool = True,
    ) -> float:
        """Run the MMLU benchmark and return accuracy.

        Args:
            model: The model to evaluate (transformers.AutoModelForCausalLM)
            tokenizer: The tokenizer (transformers.AutoTokenizer)
            device: Device to run on ("cuda" or "cpu")
            show_progress: Whether to display progress bar

        Returns:
            Accuracy score (0.0-1.0)
        """
        dataset = self.load_dataset()
        correct = 0
        total = len(dataset)

        model.eval()
        model.to(device)

        if show_progress:
            with Progress(
                TextColumn("[bold blue]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TimeRemainingColumn(),
                console=console,
            ) as progress:
                task = progress.add_task("Running MMLU benchmark", total=total)

                for item in dataset:
                    answer = item["answer"]
                    choices = item["choices"]
                    question = item["question"]

                    prompt = self._format_question(question, choices)
                    prediction = self._get_prediction(model, tokenizer, prompt, choices, device)

                    if prediction == answer:
                        correct += 1

                    progress.update(task, advance=1)
        else:
            for item in dataset:
                answer = item["answer"]
                choices = item["choices"]
                question = item["question"]

                prompt = self._format_question(question, choices)
                prediction = self._get_prediction(model, tokenizer, prompt, choices, device)

                if prediction == answer:
                    correct += 1

        return correct / total if total > 0 else 0.0

    def _format_question(self, question: str, choices: List[str]) -> str:
        """Format a question with its choices into a prompt."""
        lines = [question]
        for i, choice in enumerate(choices):
            lines.append(f"{chr(ord('A') + i)}. {choice}")
        lines.append("Answer:")
        return "\n".join(lines)

    def _get_prediction(
        self,
        model: Any,
        tokenizer: Any,
        prompt: str,
        choices: List[str],
        device: str,
    ) -> int:
        """Get the model's predicted answer index."""
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated = tokenizer.decode(
            outputs[0, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        ).strip()

        generated_upper = generated.upper()
        for i, choice in enumerate(choices):
            if chr(ord("A") + i) in generated_upper or choice.upper() in generated_upper:
                return i

        if generated_upper and "A" <= generated_upper[0] <= "Z":
            idx = ord(generated_upper[0]) - ord("A")
            if 0 <= idx < len(choices):
                return idx

        return -1


class HellaSwagBenchmark:
    """HellaSwag benchmark runner (faster alternative to MMLU).

    Args:
        n_questions: Number of questions to sample (default 500)
        seed: Random seed for reproducibility (default 42)
    """

    def __init__(self, n_questions: int = 500, seed: int = 42):
        self.n_questions = n_questions
        self.seed = seed
        self._dataset: Optional[List[Dict[str, Any]]] = None

    def load_dataset(self) -> List[Dict[str, Any]]:
        """Load and sample the HellaSwag dataset."""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "datasets library is required for benchmarking. "
                "Install with: pip install datasets"
            )

        if self._dataset is not None:
            return self._dataset

        console.print("[dim]Loading HellaSwag dataset...[/dim]")
        dataset = load_dataset("Rowan/hellaswag", split="validation")

        torch.manual_seed(self.seed)
        indices = torch.randperm(len(dataset))[: self.n_questions].tolist()
        self._dataset = [dataset[i] for i in indices]
        return self._dataset

    def run(
        self,
        model: Any,
        tokenizer: Any,
        device: str = "cuda",
        show_progress: bool = True,
    ) -> float:
        """Run HellaSwag benchmark and return accuracy.

        Args:
            model: The model to evaluate
            tokenizer: The tokenizer
            device: Device to run on
            show_progress: Whether to display progress bar

        Returns:
            Accuracy score (0.0-1.0)
        """
        dataset = self.load_dataset()
        correct = 0
        total = len(dataset)

        model.eval()
        model.to(device)

        if show_progress:
            with Progress(
                TextColumn("[bold blue]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TimeRemainingColumn(),
                console=console,
            ) as progress:
                task = progress.add_task("Running HellaSwag benchmark", total=total)

                for item in dataset:
                    ctx = item["ctx"]
                    endings = item["endings"]
                    label = int(item["label"])

                    prediction = self._get_prediction(model, tokenizer, ctx, endings, device)

                    if prediction == label:
                        correct += 1

                    progress.update(task, advance=1)
        else:
            for item in dataset:
                ctx = item["ctx"]
                endings = item["endings"]
                label = int(item["label"])

                prediction = self._get_prediction(model, tokenizer, ctx, endings, device)

                if prediction == label:
                    correct += 1

        return correct / total if total > 0 else 0.0

    def _get_prediction(
        self,
        model: Any,
        tokenizer: Any,
        context: str,
        endings: List[str],
        device: str,
    ) -> int:
        """Get the model's predicted ending index."""
        scores = []
        for ending in endings:
            full_text = context + " " + ending
            inputs = tokenizer(full_text, return_tensors="pt").to(device)

            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits

            scores.append(logits[0, -1, :].max().item())

        return int(torch.argmax(torch.tensor(scores)).item())


def run_accuracy_benchmark(
    model_name: str,
    model: Any,
    tokenizer: Any,
    compressor: Optional[Any] = None,
    dataset: str = "mmlu",
    n_questions: int = 500,
    seed: int = 42,
    device: str = "cuda",
) -> BenchmarkResult:
    """Run accuracy benchmark before and after compression.

    Args:
        model_name: Model identifier
        model: The model to evaluate
        tokenizer: The tokenizer
        compressor: TurboQuantCompressor instance (if None, only baseline is run)
        dataset: "mmlu" or "hellaswag"
        n_questions: Number of questions
        seed: Random seed
        device: Device to run on

    Returns:
        BenchmarkResult with accuracy metrics and delta
    """
    if dataset == "mmlu":
        benchmark = MMLUBenchmark(n_questions=n_questions, seed=seed)
    elif dataset == "hellaswag":
        benchmark = HellaSwagBenchmark(n_questions=n_questions, seed=seed)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    console.print("[bold green]Running baseline benchmark (uncompressed)...[/bold green]")
    accuracy_before = benchmark.run(model, tokenizer, device, show_progress=True)
    console.print(f"[green]Baseline accuracy:[/green] {accuracy_before:.4f}")

    if compressor is None:
        return BenchmarkResult(
            model_id=model_name,
            dataset=dataset,
            n_questions=n_questions,
            accuracy_before=accuracy_before,
            accuracy_after=accuracy_before,
            accuracy_delta=0.0,
            passed=True,
        )

    console.print("[bold yellow]Running benchmark with compression...[/bold yellow]")
    # Apply compression to the model
    from .integrations.huggingface import patch_model_for_contextlens
    patched_model = patch_model_for_contextlens(model)

    accuracy_after = benchmark.run(patched_model, tokenizer, device, show_progress=True)
    console.print(f"[yellow]Compressed accuracy:[/yellow] {accuracy_after:.4f}")

    accuracy_delta = accuracy_after - accuracy_before
    passed = abs(accuracy_delta) <= 0.01

    console.print(f"[bold]Accuracy delta:[/bold] {accuracy_delta:+.4f} ({accuracy_delta*100:+.2f}%)")

    return BenchmarkResult(
        model_id=model_name,
        dataset=dataset,
        n_questions=n_questions,
        accuracy_before=accuracy_before,
        accuracy_after=accuracy_after,
        accuracy_delta=accuracy_delta,
        passed=passed,
    )
