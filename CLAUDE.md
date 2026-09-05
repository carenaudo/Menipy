# CLAUDE.md - Menipy Agent & Contributor Guide

Menipy is an open-source Python toolkit for droplet shape analysis, tensiometry, and contact angle measurements (PySide6 GUI + headless CLI).

---

## 1. Project Management & Command Workflows

**Always use `uv` for environment management, script execution, and testing.**

### Environment Setup
```powershell
# Synchronize environment with all dev and test dependencies
uv sync --extra dev --extra test
```

### Running Tests
```powershell
# Run full test suite with offscreen Qt platform
$env:QT_QPA_PLATFORM="offscreen"
uv run --extra test pytest

# Run a specific test file
uv run --extra test pytest tests/test_sessile_geometry.py

# Run tests matching a keyword
uv run --extra test pytest -k "test_adsa"
```
*(On Linux/macOS, use `QT_QPA_PLATFORM=offscreen uv run --extra test pytest`)*

### Linting & Type Checking
```powershell
# Run Ruff linter
uv run --extra dev ruff check .

# Fix auto-fixable Ruff issues
uv run --extra dev ruff check --fix .

# Type checking with mypy
uv run --extra dev mypy src/menipy/models --config-file=pyproject.toml
```

### Running the Application
```powershell
# Launch PySide6 GUI
uv run menipy
# or
uv run python -m menipy

# Launch headless CLI / ADSA
uv run adsa --help
# or
uv run menipy-cli --help
```

---

## 2. Canonical Codebase Navigation

Before modifying code, consult **`docs/CODEBASE_MAP.md`**. It defines canonical file ownership and task routing.

| Subsystem | Key Locations | Responsibility |
|---|---|---|
| **GUI** | `src/menipy/gui/` | PySide6 views, controllers, services, dialogs. Entry: `app.py`. |
| **CLI** | `src/menipy/cli/` | Headless execution, batch processing, SOP workflows. Entry: `cli/__init__.py`. |
| **Pipelines** | `src/menipy/pipelines/` | Pipeline lifecycle (`base.py`), discovery (`discover.py`), runner (`runner.py`), and modes (`sessile`, `pendant`, `sessile_dynamic`, etc.). |
| **Context & State** | `src/menipy/models/` | Shared pipeline context (`context.py`), config models (`config.py`), results (`results.py`). |
| **Common Algorithms**| `src/menipy/common/` | Image utilities, edge detection, auto-calibration, material DB, units (`pint`). |
| **Math & Physics** | `src/menipy/math/` | Young-Laplace ODE integration, surface tension solvers. |
| **Plugins** | `plugins/` | Runtime-discovered detector and processor extensions. |
| **Contracts** | `docs/contracts/` | Canonical output schemas and result contracts for each pipeline mode. |
| **Guides** | `docs/guides/` | Detailed architecture and domain guides. |

---

## 3. Critical Architectural Rules & Pitfalls for LLMs

### ⚠️ Rule 1: `Context` Model is Strictly Validated (`extra="forbid"`)
- `src/menipy/models/context.py` uses `model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)`.
- **Trap**: You **cannot** attach arbitrary new attributes to `ctx` (e.g. `ctx.my_custom_var = 123`). This will raise a `ValidationError`.
- **Solution**: If a pipeline stage requires a new persistent state field, it **must** be explicitly declared on the `Context` class in `src/menipy/models/context.py`.

### ⚠️ Rule 2: GUI Separation of Concerns
- Menipy strictly follows **View -> Controller -> Service** architecture.
- **Views** (`src/menipy/gui/views/`): Pure Qt widgets, UI rendering, signal emission. Never run heavy business logic here.
- **Controllers** (`src/menipy/gui/controllers/`): Wire events, manipulate view models, invoke services.
- **Services** (`src/menipy/gui/services/`): Headless execution, background worker delegation via `pipeline_runner.py` / `QThreadPool`.
- Never execute blocking pipeline operations directly in the GUI thread.

### ⚠️ Rule 3: Headless Qt Tests Require `offscreen`
- Whenever running tests that touch GUI components, controllers, or Qt widgets, set the environment variable:
  `QT_QPA_PLATFORM=offscreen` (PowerShell: `$env:QT_QPA_PLATFORM="offscreen"`).

### ⚠️ Rule 4: Plugin Design & Registration
- Plugins in `plugins/` must be registered via `src/menipy/common/registry.py` (e.g. exposing dictionaries like `EDGE_DETECTORS`, `SOLVERS`, or implementing `register(registries)`).
- Detector plugins must return a single-frame `DetectionResult` and **must not** maintain hidden mutable temporal state.

### ⚠️ Rule 5: Docstring & Coding Standards
- All public functions, classes, and modules must have **NumPy-style docstrings**.
- Code formatting follows Black (88-character line limit) and Ruff.
- Imports must use absolute package paths (e.g., `from menipy.models.context import Context`).
- Treat `archive/` as historical reference only; do not import from or modify files in `archive/`. Treat `build/`, `dist/`, and `.tmp` as derived output.
