# ContextLens Proxy Server

FastAPI proxy server that provides Ollama-compatible API endpoints with TurboQuant compressed inference.

## Architecture

```
User Request → ContextLens Proxy → [Has Profile?]
                                      ├─ Yes → Load model via HuggingFace + TurboQuant compression
                                      └─ No  → Forward to Ollama
```

## Quick Start

### 1. Build and Run with Docker Compose

```bash
# Start the proxy server
docker-compose -f docker-compose.proxy.yml up -d

# Check logs
docker-compose -f docker-compose.proxy.yml logs -f
```

### 2. Manual Docker Build

```bash
# Build the image
docker build -f Dockerfile.proxy -t contextlens-proxy:latest .

# Run the container
docker run -d \
  -p 8080:8080 \
  -v ~/.contextlens:/root/.contextlens:ro \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e OLLAMA_HOST=http://host.docker.internal:11434 \
  --add-host=host.docker.internal:host-gateway \
  contextlens-proxy:latest
```

### 3. Development Mode

```bash
# Install proxy dependencies
pip install -e ".[proxy]"

# Run the proxy server
python -m contextlens.proxy

# Or with custom settings
CONTEXTLENS_PORT=9000 OLLAMA_HOST=http://localhost:11434 python -m contextlens.proxy
```

## API Endpoints

### Health Check

```bash
curl http://localhost:8080/health
# {"status": "healthy"}
```

### List Models

```bash
curl http://localhost:8080/api/tags
```

### Generate Text

```bash
# Streaming
curl -X POST http://localhost:8080/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.2:1b",
    "prompt": "Hello, world!",
    "stream": true
  }'

# Non-streaming
curl -X POST http://localhost:8080/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.2:1b",
    "prompt": "Hello, world!",
    "stream": false,
    "options": {"num_predict": 100}
  }'
```

### Chat Completion

```bash
curl -X POST http://localhost:8080/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.2:1b",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Hello!"}
    ],
    "stream": false
  }'
```

## Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `CONTEXTLENS_PORT` | `8080` | Port to listen on |
| `CONTEXTLENS_PROFILES_DIR` | `/root/.contextlens` | Directory containing model profiles |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama API URL for non-compressed models |

## How It Works

1. **Profile Detection**: When a request arrives, the proxy checks if a profile exists for the requested model in `~/.contextlens/`.

2. **Compressed Inference**: If a profile exists and the model has a HuggingFace mapping:
   - Model weights are loaded via HuggingFace transformers
   - TurboQuant compression is applied during inference
   - Response is streamed in Ollama-compatible format

3. **Fallback to Ollama**: If no profile exists OR the model requires authentication:
   - Request is forwarded to Ollama unchanged
   - Response is returned as-is

## Model Name Mapping

The proxy maps Ollama model names to HuggingFace repo IDs. Non-gated models included by default:

- `tinyllama:1b` → `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- `qwen2:0.5b` → `Qwen/Qwen2-0.5B`
- `qwen2:1.5b` → `Qwen/Qwen2-1.5B`
- `phi2:2.7b` → `microsoft/phi-2`

For gated models (Llama 3, Mistral, etc.), set your HuggingFace token:

```bash
export HF_TOKEN=hf_xxx
# Or the proxy will automatically fall back to Ollama
```

## Testing

```bash
# Run the test script
./scripts/test-proxy.sh

# Or with custom URL
PROXY_URL=http://localhost:9000 ./scripts/test-proxy.sh
```

## Memory Savings

With TurboQuant compression:
- **K-cache**: 8-bit PolarQuant (~2x compression)
- **V-cache**: 5-bit primary + 1-bit QJL residual (~3x compression)
- **Overall**: ~1.5-2x KV cache reduction

## Limitations

- Image inputs are not yet supported
- Only text generation models are supported
- GPU memory must be available for model loading
