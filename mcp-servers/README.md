mcp-servers: skeleton configs for common MCP servers

Provided files:
- docker-compose.yml: services for Triton, TorchServe, FastAPI (sample), and Playwright runner.
- fastapi/: minimal FastAPI inference wrapper and Dockerfile.
- playwright/: Playwright tests and package.json to run smoke tests against fastapi service.
- torchserve/: README describing model-store usage.

Quickstart

1. Build and start the FastAPI and Playwright services (no GPU required):
   docker-compose up --build fastapi playwright-runner

2. To start all services (GPU required for Triton):
   docker-compose up --build

Notes
- Triton official images are published on NVIDIA NGC and may require login or GPU access; replace the image with a suitable private build if necessary.
- This setup is intentionally minimal: add real model repositories, .mar files for TorchServe, and CI workflows as needed.
