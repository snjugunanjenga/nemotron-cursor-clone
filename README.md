# nemotron-cursor-clone

A Cursor-like AI coding assistant powered by Nvidia Nemotron.

This repository scaffolds a vLLM-based inference backend and a Gradio frontend with simple RAG support.

Quickstart (local, dev)

1. Copy environment variables:
   cp .env.example .env

2. Start the backend (stub or proxy):
   # Run the minimal FastAPI wrapper (stubbed) for local dev
   python -m backend.server

3. Start the Gradio frontend (connects to the backend):
   python app.py

Docker (backend)

Build:
  docker build -t nemotron-cursor .
Run (GPU-enabled):
  docker run --gpus all -p 8000:8000 -e VLLM_API_URL=http://localhost:8000 nemotron-cursor

MCP servers

See mcp-servers/ for docker-compose + Playwright smoke test runner (created in-session).

Contributing

- Do not commit model checkpoints or large binary artifacts. Use MODEL_DIR or MODEL_NAME environment variables and document download locations.
- Update .github/copilot-instructions.md after adding build/test commands so Copilot sessions can find them automatically.
