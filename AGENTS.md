# Menipy Codex & Coding Agent Guide

Use [`docs/CODEBASE_MAP.md`](docs/CODEBASE_MAP.md) as the canonical navigation
guide before making changes. It maps execution paths, subsystem ownership,
plugins, tests, documentation, and maintenance tooling. For a complete deep-dive
into agent workflows, see [`docs/guides/llm_coding_agent_guide.md`](docs/guides/llm_coding_agent_guide.md).

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

1. **Start from the task-to-file routes in [`docs/CODEBASE_MAP.md`](docs/CODEBASE_MAP.md).**
2. **Read the nearest tests before changing behavior.**
3. **Respect strict `Context` validation**: `src/menipy/models/context.py` sets `extra="forbid"`. Do not dynamically assign undeclared attributes to `ctx`; declare them on the model.
4. **For pipeline output changes, read the matching file in `docs/contracts/`.**
5. **Keep GUI work on PySide6 and preserve controller/service/view boundaries.** Never run blocking computation on the UI thread.
6. **Use `uv` for all Python, lint, and test execution:**

```powershell
# Set offscreen platform for Qt GUI tests
$env:QT_QPA_PLATFORM="offscreen"

# Run tests
uv run --extra test pytest

# Run linter
uv run --extra dev ruff check .

# Run type checker
uv run --extra dev mypy src/menipy/models --config-file=pyproject.toml

# Run GUI or CLI
uv run menipy
uv run adsa --help
```

Update `docs/CODEBASE_MAP.md` whenever an entry point, subsystem boundary,
pipeline or plugin flow, canonical documentation location, or CI/test route
changes. Keep this file concise and leave detailed navigation in the map.
