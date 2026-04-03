"""
FastAPI proxy server for ContextLens compressed inference.

Provides an OpenAI-compatible API for HuggingFace models with TurboQuant compression.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
import uvicorn

from contextlens.compressor import TurboQuantCompressor
from contextlens.integrations.huggingface import patch_model_for_contextlens
from contextlens.profiles import load_profile, PROFILE_DIR
from contextlens.hf_utils import get_hf_token, check_model_exists_locally


# ============================================================================
# Configuration
# ============================================================================

PROFILES_DIR = Path(os.environ.get("CONTEXTLENS_PROFILES_DIR", PROFILE_DIR))
DEFAULT_PORT = int(os.environ.get("CONTEXTLENS_PORT", "8080"))


# ============================================================================
# Request/Response Models (OpenAI-compatible API)
# ============================================================================

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    stream: bool = False
    temperature: float = 0.8
    max_tokens: int = 256
    top_p: float = 1.0


class ChatChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: str = "stop"


class ChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatChoice]
    usage: Optional[Dict[str, int]] = None


class ChatStreamChunk(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[Dict[str, Any]]


# ============================================================================
# Model Registry
# ============================================================================

class ModelRegistry:
    """Manages loaded models and their compression state."""

    def __init__(self):
        self._models: Dict[str, Any] = {}
        self._compressors: Dict[str, TurboQuantCompressor] = {}
        self._profiles: Dict[str, dict] = {}

    def has_profile(self, model_name: str) -> bool:
        """Check if a model has a ContextLens profile."""
        profile_path = PROFILES_DIR / f"{model_name.replace('/', '_').replace(':', '_')}.json"
        return profile_path.exists()

    def load_model(self, model_name: str, use_compression: bool = True) -> Any:
        """Load a model with optional compression."""
        if model_name in self._models:
            return self._models[model_name]

        from transformers import AutoModelForCausalLM, AutoTokenizer

        # Check if model exists locally
        if not check_model_exists_locally(model_name):
            raise ValueError(
                f"Model '{model_name}' not found in local cache.\n"
                f"Download it first: huggingface-cli download {model_name}"
            )

        # Get HF token for gated models
        hf_token = get_hf_token()

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            token=hf_token,
        )

        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            token=hf_token,
        )

        # Apply compression if requested and profile exists
        if use_compression and self.has_profile(model_name):
            profile = self.load_profile(model_name)
            self._profiles[model_name] = profile
            model = patch_model_for_contextlens(model)
            self._compressors[model_name] = TurboQuantCompressor(bits=3.5)

        self._models[model_name] = {
            "model": model,
            "tokenizer": tokenizer,
        }

        return self._models[model_name]

    def load_profile(self, model_name: str) -> dict:
        """Load a model profile."""
        profile_path = PROFILES_DIR / f"{model_name.replace('/', '_').replace(':', '_')}.json"
        if not profile_path.exists():
            raise FileNotFoundError(f"Profile not found: {profile_path}")

        with open(profile_path, "r") as f:
            return json.load(f)

    def get_compressor(self, model_name: str) -> Optional[TurboQuantCompressor]:
        """Get the compressor for a model."""
        return self._compressors.get(model_name)

    def is_compressed(self, model_name: str) -> bool:
        """Check if a model is loaded with compression."""
        return model_name in self._compressors


# ============================================================================
# FastAPI App
# ============================================================================

app = FastAPI(title="ContextLens API", version="0.1.0")
registry = ModelRegistry()


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.get("/v1/models")
async def list_models():
    """List available models."""
    # List models that have profiles
    try:
        from contextlens.profiles import list_profiles
        profiles = list_profiles()
        models = []
        for profile in profiles:
            models.append({
                "id": profile.model_id,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "contextlens",
            })
        return {"object": "list", "data": models}
    except Exception:
        return {"object": "list", "data": []}


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    """Chat completion endpoint."""
    model_name = request.model

    # Load model
    try:
        model_data = registry.load_model(model_name, use_compression=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

    model = model_data["model"]
    tokenizer = model_data["tokenizer"]

    # Build prompt from messages
    prompt = _messages_to_prompt(request.messages, tokenizer)

    if request.stream:
        return StreamingResponse(
            _chat_stream(model, tokenizer, request, prompt),
            media_type="text/event-stream",
        )
    else:
        return await _chat_non_stream(model, tokenizer, request, prompt)


# ============================================================================
# Helper Functions
# ============================================================================

def _messages_to_prompt(messages: List[ChatMessage], tokenizer: Any) -> str:
    """Convert chat messages to prompt."""
    # Use tokenizer's chat template if available
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            [{"role": m.role, "content": m.content} for m in messages],
            tokenize=False,
            add_generation_prompt=True,
        )

    # Fallback: simple concatenation
    prompt_parts = []
    for msg in messages:
        if msg.role == "system":
            prompt_parts.append(f"System: {msg.content}")
        elif msg.role == "user":
            prompt_parts.append(f"User: {msg.content}")
        elif msg.role == "assistant":
            prompt_parts.append(f"Assistant: {msg.content}")

    prompt_parts.append("Assistant:")
    return "\n".join(prompt_parts)


def _chat_stream(
    model: Any,
    tokenizer: Any,
    request: ChatRequest,
    prompt: str,
) -> Generator[str, None, None]:
    """Generate chat response with streaming."""
    start_time = time.time()

    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_length = inputs["input_ids"].shape[1]

    # Generate
    max_new_tokens = request.max_tokens
    temperature = request.temperature

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_ids = outputs[0][input_length:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    # Stream token by token
    response_id = f"chatcmpl-{int(time.time())}"
    created = int(time.time())

    # Send empty first chunk (as OpenAI does)
    first_chunk = {
        "id": response_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": request.model,
        "choices": [{
            "index": 0,
            "delta": {"role": "assistant"},
            "finish_reason": None
        }]
    }
    yield f"data: {json.dumps(first_chunk)}\n\n"

    # Stream content
    for char in generated_text:
        chunk = {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": request.model,
            "choices": [{
                "index": 0,
                "delta": {"content": char},
                "finish_reason": None
            }]
        }
        yield f"data: {json.dumps(chunk)}\n\n"

    # Final chunk
    final_chunk = {
        "id": response_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": request.model,
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": "stop"
        }]
    }
    yield f"data: {json.dumps(final_chunk)}\n\n"
    yield "data: [DONE]\n\n"


async def _chat_non_stream(
    model: Any,
    tokenizer: Any,
    request: ChatRequest,
    prompt: str,
) -> JSONResponse:
    """Generate chat response without streaming."""
    start_time = time.time()

    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_length = inputs["input_ids"].shape[1]

    # Generate
    max_new_tokens = request.max_tokens
    temperature = request.temperature

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_ids = outputs[0][input_length:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    total_time = time.time() - start_time

    response = ChatResponse(
        id=f"chatcmpl-{int(time.time())}",
        created=int(time.time()),
        model=request.model,
        choices=[
            ChatChoice(
                index=0,
                message=ChatMessage(role="assistant", content=generated_text),
                finish_reason="stop"
            )
        ],
        usage={
            "prompt_tokens": input_length,
            "completion_tokens": len(generated_ids),
            "total_tokens": input_length + len(generated_ids),
        }
    )

    return JSONResponse(content=response.dict())


# ============================================================================
# CLI
# ============================================================================

def run_proxy(host: str = "0.0.0.0", port: int = DEFAULT_PORT):
    """Run the ContextLens API server."""
    print(f"Starting ContextLens API on {host}:{port}")
    print(f"Profiles directory: {PROFILES_DIR}")
    print()
    print("Endpoints:")
    print("  GET  /health           - Health check")
    print("  GET  /v1/models        - List available models")
    print("  POST /v1/chat/completions - Chat completion")
    print()

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import sys

    # Simple CLI argument parsing
    host = "0.0.0.0"
    port = DEFAULT_PORT

    if len(sys.argv) > 1 and sys.argv[1] == "serve":
        args = sys.argv[2:]
        i = 0
        while i < len(args):
            arg = args[i]
            if arg.startswith("--host="):
                host = arg.split("=", 1)[1]
            elif arg == "--host" and i + 1 < len(args):
                host = args[i + 1]
                i += 1
            elif arg.startswith("--port="):
                port = int(arg.split("=", 1)[1])
            elif arg == "--port" and i + 1 < len(args):
                port = int(args[i + 1])
                i += 1
            elif arg == "--help":
                print("Usage: python -m contextlens.proxy serve [--host=HOST] [--port=PORT]")
                print()
                print("Options:")
                print("  --host    Host to bind to (default: 0.0.0.0)")
                print("  --port    Port to bind to (default: 8080)")
                sys.exit(0)
            i += 1

    run_proxy(host, port)
