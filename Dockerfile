from ollama/ollama:latest

# Install Python and build tools
run apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
workdir /app

# Create virtual environment
run python3 -m venv /app/venv
env PATH="/app/venv/bin:$PATH"

# Copy ContextLens project
copy . /app/contextlens

# Install ContextLens in virtual environment
run cd /app/contextlens && pip install -e .

# Copy startup script
copy entrypoint.sh /app/entrypoint.sh
run chmod +x /app/entrypoint.sh

# Expose Ollama port
expose 11434

# Override entrypoint and set command
entrypoint []
cmd ["/app/entrypoint.sh"]
