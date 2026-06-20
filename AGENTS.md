# Menipy Codex Guide

Use [`docs/CODEBASE_MAP.md`](docs/CODEBASE_MAP.md) as the canonical navigation
guide before making changes. It maps execution paths, subsystem ownership,
plugins, tests, documentation, and maintenance tooling.

## Canonical locations

- `src/menipy/`: application code installed as the `menipy` package.
- `plugins/`: dynamically discovered analysis and detector extensions.
- `tests/`: unit, integration, CLI, and offscreen Qt coverage.
- `docs/guides/` and `docs/contracts/`: developer guidance and result contracts.
- `pyproject.toml`: package entry points and tool configuration.
- `.github/workflows/`: CI, lint, resource-build, and pre-commit behavior.

Treat `archive/` as historical context, not current architecture. Treat
`build/`, `dist/`, `out/`, `plot/`, caches, coverage files, and generated graph
artifacts as derived output. Do not infer current ownership from them. Root-level
experimental scripts and old HTML mockups are non-canonical unless a task names
them explicitly.

## Working rules

1. Start from the task-to-file routes in `docs/CODEBASE_MAP.md`.
2. Read the nearest tests before changing behavior.
3. For pipeline output changes, read the matching file in `docs/contracts/`.
4. Keep GUI work on PySide6 and preserve the controller/service/view boundaries.
5. Run Python commands through the repository virtual environment.

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m mypy src/menipy/models --config-file=pyproject.toml
```

Set `QT_QPA_PLATFORM=offscreen` when running GUI tests in a headless environment.

Update `docs/CODEBASE_MAP.md` whenever an entry point, subsystem boundary,
pipeline or plugin flow, canonical documentation location, or CI/test route
changes. Keep this file concise and leave detailed navigation in the map.
