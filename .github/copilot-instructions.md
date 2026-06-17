# Copilot instructions for this repository

Purpose
- This repo is a Cursor-like AI coding assistant powered by Nvidia Nemotron (README only). Current tree: README.md, LICENSE.

What was detected
- No package manifest (package.json / pyproject.toml / Cargo.toml) or CI workflows were found.
- No tests, linters, or source files present.

Build, test, and lint (detected / actionable)
- No build/test/lint commands detected in this repository. Update this file when you add a manifest or CI.
- Helpful examples to document when you add a language/runtime (include exact scripts):
  - Node.js (package.json):
    - Full test suite: `npm test` or `npm run test`
    - Single test by name: `npm test -- -t "<test name or pattern>"`
    - Lint: `npm run lint`
  - Python (pyproject.toml / pytest):
    - Full test suite: `pytest`
    - Single test: `pytest path/to/test_file.py::test_name -q`
    - Lint: `flake8` or `ruff` (document exact command)
  - Add the exact commands/scripts used in your manifest so Copilot can surface them.

High-level architecture (what Copilot should assume / look for)
- Purpose: a Nemotron-powered assistant. Expect three main areas once present: model code (training / checkpoints), inference/serving code (API or CLI), and a UI/frontend (optional).
- Where to start in future sessions:
  1. Look for package manifests (package.json, pyproject.toml, Cargo.toml) or a Makefile to find build/test commands.
  2. Check .github/workflows for CI steps and test matrix.
  3. Search for directories named `src`, `inference`, `model`, `api`, `web`, `notebooks`, and `tests`.
  4. Look for configuration or secrets references (e.g., `.env`, config/*.yaml) that name model checkpoint locations.

Key conventions for this repository (establish and document when adding code)
- Do not commit model checkpoints or large binary artifacts. Document download locations and expected path in a config file (e.g., ENV var MODEL_DIR or config/model_path).
- Prefer clear entrypoints: `src/main.py` or `src/server.js` (document the actual entrypoint in README or package manifest).
- Tests: place unit tests under `tests/` and integration tests under `tests/integration/`. Use explicit markers (pytest: `-m integration`) so CI can filter GPU-heavy tests.
- CI: keep GPU-specific steps gated behind labeled workflows or matrix entries so ordinary PRs don't require GPU hardware.

AI assistant / Copilot tips specific to this repo
- Start by reading README.md, then search for manifests and CI. If none exist, ask the maintainer where the canonical entrypoints and checkpoints live before making edits.
- If adding new files, update this instructions file with concrete build/test/lint commands and any environment variables required for model paths or credentials.
- If you need to run or test model inference locally, prefer small smoke tests that don't require full checkpoints; document those commands in the repo.

Other AI assistant configs to check and incorporate (if added)
- CLAUDE.md, AGENTS.md, .cursorrules, .cursor/, .windsurfrules, CONVENTIONS.md, AIDER_CONVENTIONS.md, .clinerules — copy relevant rules into this file when they appear so Copilot sessions have a single source of truth.

Maintainers: please update this file whenever you add language-specific manifests, CI workflows, or change entrypoints. Copilot uses this file to locate build/test commands and repo conventions quickly.
