# Repository Guidelines

## Project Structure & Module Organization

This is a Python 3.12+ agent system for component anomaly QA, tracing, and evaluation. Core orchestration lives in `agent/`, shared schemas and LLM access in `schemas.py` and `llm_client.py`, and tool implementations in `tools/`. Evaluation logic is in `evaluator/`; runnable workflows are in `scripts/`. Configs are under `configs/`, with presets in `configs/experiments/`. Tests live in `tests/`. Generated artifacts belong in `outputs/`, `logs/`, `.tmp/`, and downloaded models in `models/`.

The `2025/` directory and `2025_dataset.jsonl` are task data. Do not use raw Markdown/HTML/JSON inside `2025/` as retrieval knowledge unless explicitly allowed; use `knowledge_base/` for curated local retrieval sources.

## Build, Test, and Development Commands

- `uv sync`: install dependencies from `pyproject.toml` and `uv.lock`.
- `uv run python scripts/validate_data.py`: validate dataset records and image paths.
- `uv run python scripts/run_infer.py --sample-id 1326045 --experiment test_one --no-resume`: run one inference sample.
- `uv run python scripts/run_eval.py --predictions outputs/test_one/predictions.jsonl --experiment test_one`: evaluate predictions.
- `uv run python scripts/run_experiment.py --experiment agent_limit5 --limit 5`: run a small experiment.
- `uv run python -m pytest tests`: run the test suite.

## Coding Style & Naming Conventions

Use 4-space indentation, `from __future__ import annotations` where useful, and standard Python naming: `snake_case` for functions, variables, and modules; `PascalCase` for classes. Keep code within existing boundaries: agent planning/execution in `agent/`, external capabilities in `tools/`, and metrics in `evaluator/`. Prefer schema objects from `schemas.py` over ad hoc dictionaries for cross-module data.

## Testing Guidelines

Tests use `unittest` assertions and are runnable through pytest. Name files `tests/test_<feature>.py`, classes `<Feature>Tests`, and methods `test_<expected_behavior>`. Add focused tests for planner/tool behavior, config parsing, evaluation metrics, and data processing changes. Avoid network-dependent tests unless isolated or skipped by default.

## Commit & Pull Request Guidelines

Recent commits use concise Conventional-style prefixes such as `feat:` and `refactor:`, often followed by a short Chinese summary. Keep commit messages imperative and scoped, for example `feat: add search timeout diagnostics`.

Pull requests should include purpose, key behavior changes, commands run, linked issues or experiment IDs, and relevant output paths such as `outputs/<experiment>/eval_results.jsonl`. Include screenshots only for UI-facing tools like `scripts/trace_viewer.py`.

## Security & Configuration Tips

Keep API keys out of git. Use environment variables such as `DASHSCOPE_API_KEY`, or copy `configs/local.example.yaml` to ignored `configs/local.yaml`. Do not commit generated caches, model files, logs, or experiment outputs unless they are intentional documentation artifacts.
