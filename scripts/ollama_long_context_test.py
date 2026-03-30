#!/usr/bin/env python3
"""
Long-context comparison for ContextLens vs baseline Ollama models.

Uses Ollama's prompt_eval_count (not character heuristics) to ensure the
prompt actually reaches at least --min-prompt-tokens (default 8000).

Default corpus: /tmp/large-document.txt (non-repetitive long text you supply;
e.g. docker cp large-document.txt <container>:/tmp/large-document.txt).

Usage:
  docker cp scripts/ollama_long_context_test.py <container>:/tmp/
  docker cp /path/to/large-document.txt <container>:/tmp/large-document.txt
  docker exec <container> python3 /tmp/ollama_long_context_test.py

Or after image rebuild:
  python3 /app/contextlens/scripts/ollama_long_context_test.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


def build_corpus(target_chars: int, needle: str, needle_after_chars: int) -> str:
    """Synthetic fallback: varied technical lines plus a quarantined needle."""
    parts: list[str] = []
    acc = 0
    i = 0
    needle_inserted = False
    while acc < target_chars:
        if not needle_inserted and acc >= needle_after_chars:
            parts.append(
                "\n--- QUARANTINED SECTION ---\n"
                f"The designated test passphrase is exactly: {needle}\n"
                "--- END QUARANTINED SECTION ---\n\n"
            )
            needle_inserted = True
            acc = sum(len(p) for p in parts)
        line = (
            f"[{i}] Shard S-{i % 220} queue_depth={i % 600} consumer=CG-{i % 45} "
            f"threshold_k={(i % 85) + 12} owner=OU-{i % 28} region=R-{i % 9}.\n"
        )
        parts.append(line)
        acc += len(line)
        i += 1
    return "".join(parts)


def ollama_generate(
    base_url: str,
    model: str,
    prompt: str,
    *,
    num_ctx: int,
    num_predict: int,
    timeout: int,
) -> dict:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_ctx": num_ctx,
            "num_predict": num_predict,
        },
    }
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/generate",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def probe_prompt_for_document(corpus: str) -> str:
    return (
        "You will read a long document. The full text follows.\n\n"
        f"{corpus}\n\n"
        "After reading the document, reply with exactly one word: OK\n"
    )


def answer_prompt_for_document(corpus: str, task: str) -> str:
    return (
        "You are given a long document. Answer using only that document.\n\n"
        f"{corpus}\n\n"
        f"Task: {task}\n"
    )


def count_prompt_tokens_probe(
    base_url: str,
    model: str,
    corpus: str,
    *,
    num_ctx: int,
    timeout: int,
    use_synthetic_needle: bool,
) -> int:
    if use_synthetic_needle:
        prompt = (
            "You are given internal reference logs. A short question follows the logs.\n\n"
            f"{corpus}\n\n"
            "Question: Reply with exactly one word: OK\n"
        )
    else:
        prompt = probe_prompt_for_document(corpus)
    out = ollama_generate(
        base_url,
        model,
        prompt,
        num_ctx=num_ctx,
        num_predict=2,
        timeout=timeout,
    )
    return int(out.get("prompt_eval_count") or 0)


def run_answer(
    base_url: str,
    model: str,
    corpus: str,
    *,
    num_ctx: int,
    num_predict: int,
    timeout: int,
    use_synthetic_needle: bool,
    needle: str,
    task: str,
) -> tuple[dict, float]:
    if use_synthetic_needle:
        prompt = (
            "You are given internal reference logs. Answer using only the passage "
            "marked QUARANTINED SECTION.\n\n"
            f"{corpus}\n\n"
            "Question: What is the exact test passphrase from the QUARANTINED SECTION? "
            "Reply with only the passphrase string, no quotes or extra words.\n"
        )
    else:
        prompt = answer_prompt_for_document(corpus, task)
    t0 = time.perf_counter()
    out = ollama_generate(
        base_url,
        model,
        prompt,
        num_ctx=num_ctx,
        num_predict=num_predict,
        timeout=timeout,
    )
    elapsed = time.perf_counter() - t0
    return out, elapsed


def grow_corpus_to_min_tokens(
    base_url: str,
    probe_model: str,
    min_tokens: int,
    *,
    num_ctx: int,
    timeout: int,
    start_chars: int,
    max_chars: int,
    needle: str,
) -> tuple[str, int]:
    step = 8000
    n = start_chars
    last_count = 0
    while n <= max_chars:
        corpus = build_corpus(n, needle=needle, needle_after_chars=max(n // 3, 5000))
        last_count = count_prompt_tokens_probe(
            base_url,
            probe_model,
            corpus,
            num_ctx=num_ctx,
            timeout=timeout,
            use_synthetic_needle=True,
        )
        print(f"  corpus_chars={len(corpus):_} prompt_eval_count={last_count}", flush=True)
        if last_count >= min_tokens:
            return corpus, last_count
        n += step
    raise SystemExit(
        f"Could not reach {min_tokens} prompt tokens (last prompt_eval_count={last_count}, "
        f"chars={n}). Raise --max-chars or --num-ctx."
    )


def load_document(path: Path) -> str:
    if not path.is_file():
        raise SystemExit(
            f"Document not found: {path}\n"
            f"Copy it into the container, e.g.:\n"
            f"  docker cp /host/path/large-document.txt <container>:{path}\n"
        )
    text = path.read_text(encoding="utf-8", errors="replace")
    if not text.strip():
        raise SystemExit(f"Document is empty: {path}")
    return text


def main() -> None:
    p = argparse.ArgumentParser(description="Compare Ollama models at >=8k-token context.")
    p.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    p.add_argument("--baseline-model", default="llama3.2:3b")
    p.add_argument("--compressed-model", default="llama3.2:3b-contextlens")
    p.add_argument("--min-prompt-tokens", type=int, default=8000)
    p.add_argument("--num-ctx", type=int, default=32768)
    p.add_argument("--num-predict", type=int, default=256)
    p.add_argument("--timeout", type=int, default=900)
    p.add_argument(
        "--document",
        type=Path,
        default=Path("/tmp/large-document.txt"),
        help="Path to long non-repetitive text (default: /tmp/large-document.txt).",
    )
    p.add_argument(
        "--synthetic",
        action="store_true",
        help="Generate synthetic corpus and grow until min tokens (ignore --document).",
    )
    p.add_argument(
        "--start-chars",
        type=int,
        default=65_000,
        help="With --synthetic: initial corpus size before growing.",
    )
    p.add_argument("--max-chars", type=int, default=400_000)
    p.add_argument("--needle", default="CQ-TEST-PHRASE-Z9")
    p.add_argument(
        "--task",
        default=(
            "Write exactly 5 bullet points summarizing the main themes or concrete facts "
            "in the document. Each bullet must reflect specific content from the text."
        ),
        help="Instruction appended after the document (ignored with --synthetic needle task).",
    )
    p.add_argument("--probe-only", action="store_true", help="Only verify token count; no full model comparison.")
    args = p.parse_args()

    print(
        f"Target: prompt_eval_count >= {args.min_prompt_tokens} (num_ctx={args.num_ctx})\n"
        f"Probe model: {args.baseline_model}\n",
        flush=True,
    )

    if args.synthetic:
        print("Mode: synthetic corpus (grow until min tokens)\n", flush=True)
        corpus, tok = grow_corpus_to_min_tokens(
            args.ollama_url,
            args.baseline_model,
            args.min_prompt_tokens,
            num_ctx=args.num_ctx,
            timeout=args.timeout,
            start_chars=args.start_chars,
            max_chars=args.max_chars,
            needle=args.needle,
        )
        use_synthetic = True
    else:
        print(f"Mode: file document {args.document}\n", flush=True)
        corpus = load_document(args.document)
        print(f"  file_bytes={len(corpus.encode('utf-8')):_} chars={len(corpus):_}", flush=True)
        tok = count_prompt_tokens_probe(
            args.ollama_url,
            args.baseline_model,
            corpus,
            num_ctx=args.num_ctx,
            timeout=args.timeout,
            use_synthetic_needle=False,
        )
        print(f"  prompt_eval_count={tok}", flush=True)
        if tok < args.min_prompt_tokens:
            raise SystemExit(
                f"prompt_eval_count {tok} < {args.min_prompt_tokens}. "
                "Use a longer document, raise --num-ctx if context was truncated, or lower "
                "--min-prompt-tokens."
            )
        use_synthetic = False

    print(f"\n✓ Ready: prompt_eval_count={tok}\n", flush=True)

    if args.probe_only:
        return

    for label, model in (
        ("Baseline", args.baseline_model),
        ("ContextLens", args.compressed_model),
    ):
        print(f"=== {label}: {model} ===", flush=True)
        try:
            out, elapsed = run_answer(
                args.ollama_url,
                model,
                corpus,
                num_ctx=args.num_ctx,
                num_predict=args.num_predict,
                timeout=args.timeout,
                use_synthetic_needle=use_synthetic,
                needle=args.needle,
                task=args.task,
            )
        except urllib.error.HTTPError as e:
            print(f"HTTP error: {e.read().decode()[:500]}", file=sys.stderr)
            raise
        pec = out.get("prompt_eval_count")
        gen = out.get("eval_count")
        text = (out.get("response") or "").strip()
        print(f"  wall_s={elapsed:.1f} prompt_eval_count={pec} eval_count={gen}", flush=True)
        print(f"  response: {text[:800]}{'…' if len(text) > 800 else ''}", flush=True)
        if use_synthetic:
            ok = args.needle.lower() in text.lower() or args.needle in text
            print(f"  needle_found={ok}\n", flush=True)
        else:
            print("", flush=True)


if __name__ == "__main__":
    main()
