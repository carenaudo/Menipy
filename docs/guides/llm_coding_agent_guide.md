# Menipy Developer Guide for LLM Models & Coding Agents

This guide is the canonical operational manual for AI coding agents (Claude Code, Cursor, Gemini, Copilot, Antigravity, CODEX) and human contributors working on Menipy. It codifies the architecture, non-negotiable constraints, execution workflows, and testing practices.

---

## 1. Quick Reference: The 7 Golden Rules

1. **Always Use `uv`**: All python execution, dependency management, and tool invocations must use `uv run` and `uv sync`. Never use raw `pip` or untracked python binaries.
2. **Strict `Context` Schema (`extra="forbid"`)**: `src/menipy/models/context.py` rejects undeclared attributes. Never dynamically assign ad-hoc attributes to `ctx`; declare them on the Pydantic model first.
3. **Headless Qt Requires Offscreen**: When running any tests touching PySide6, set `QT_QPA_PLATFORM=offscreen`.
4. **Follow `docs/CODEBASE_MAP.md`**: Treat `docs/CODEBASE_MAP.md` as the canonical navigation router before touching unfamiliar files.
5. **Preserve GUI Architecture (View -> Controller -> Service)**: Never embed pipeline execution, solvers, or heavy image math in Qt views or controllers. Delegate to services and background workers (`pipeline_runner.py`).
6. **Honor Result Contracts**: Any changes to pipeline metric outputs (`ctx.results`, `ctx.final_output`) must adhere to the schemas in `docs/contracts/`.
7. **Do Not Touch Derived or Archived Folders**: `archive/` contains historical reference material; do not import from or modify it. `build/`, `dist/`, `out/`, and `.tmp/` are transient artifacts.

---

## 2. Command Cheat-Sheet (Using `uv`)

### Environment & Dependencies
```powershell
# Sync full virtual environment (dev tools, pytest, mypy, ruff)
uv sync --extra dev --extra test

# Add a runtime dependency
uv add <package>

# Add a development/testing dependency
uv add --dev <package>
```

### Running Tests
```powershell
# Run all tests with offscreen Qt platform
$env:QT_QPA_PLATFORM="offscreen"
uv run --extra test pytest

# Run a specific test module (fast turnaround)
uv run --extra test pytest tests/test_sessile_geometry.py -q

# Run tests matching a keyword or subsystem
uv run --extra test pytest -k "sessile and not dynamic" -q

# Run with test coverage
uv run --extra test pytest --cov=src/menipy --cov-report=term-missing
```

### Code Quality & Static Analysis
```powershell
# Lint codebase with Ruff
uv run --extra dev ruff check .

# Automatically apply Ruff safe fixes
uv run --extra dev ruff check --fix .

# Format code with Black / Ruff
uv run --extra dev ruff format .

# Type checking on models and critical modules
uv run --extra dev mypy src/menipy/models --config-file=pyproject.toml
```

### Launching Application
```powershell
# Launch PySide6 GUI
uv run menipy
# Equivalent fallback:
uv run python -m menipy

# Launch headless CLI / ADSA
uv run adsa --help
uv run menipy-cli --help
```

---

## 3. Core Architecture & Subsystem Boundaries

### 3.1. Pipeline Lifecycle (`src/menipy/pipelines/`)
All analysis modes (`sessile`, `pendant`, `sessile_dynamic`, `captive_bubble`, `capillary_rise`, `oscillating`) inherit from `PipelineBase` (`src/menipy/pipelines/base.py`).

The canonical 12-stage lifecycle is executed sequentially:
```
1. acquisition         -> Load image or video frames (ctx.image, ctx.frames)
2. preprocessing       -> Filter, crop, normalize (ctx.preprocessed, ctx.roi)
3. feature_detection   -> Detect substrate, needle, apex, contact points
4. contour_extraction  -> Extract raw drop contour (ctx.detected_contour)
5. contour_refinement  -> Substrate clipping, smoothing, outlier removal
6. calibration         -> Compute px_per_mm scale from needle or manual input
7. geometric_features  -> Circle fit, ellipse fit, apex radius (r0), drop height
8. physics             -> Fluid/drop densities, capillary constant (c = Δρ*g/γ)
9. profile_fitting     -> Young-Laplace ODE integration, ADSA optimization
10. compute_metrics    -> Calculate contact angles, surface tension, volume
11. overlay            -> Render visual annotation layers (ctx.overlay, ctx.preview)
12. validation         -> Sanity checks, error bounds, status warnings (ctx.qa)
```

Pipelines are discovered dynamically by `src/menipy/pipelines/discover.py` from subdirectories under `src/menipy/pipelines/` that define a `PipelineBase` subclass in `stages.py` or `__init__.py`.

### 3.2. Shared State: `Context` Model (`src/menipy/models/context.py`)
`Context` is the shared state object passed between all pipeline stages.

```python
class Context(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)
    ...
```

* **Why `extra="forbid"`?** It prevents typos and silent state corruption across stages.
* **Adding New State**: If your new algorithm or stage produces intermediate data needed by downstream stages:
  1. Open `src/menipy/models/context.py`.
  2. Declare the new field with a type hint and default (e.g., `my_metric: float | None = None`).
  3. Verify with `uv run --extra dev mypy src/menipy/models --config-file=pyproject.toml`.

### 3.3. GUI Architecture (`src/menipy/gui/`)
The GUI uses PySide6 with strict layer separation:

```
                  ┌─────────────────────────────────────────┐
                  │                 Views                   │
                  │   src/menipy/gui/views/ (main_window,   │
                  │      preview_panel, results_panel)      │
                  └────────────────────┬────────────────────┘
                                       │ Qt Signals & Slots
                                       ▼
                  ┌─────────────────────────────────────────┐
                  │              Controllers                │
                  │   src/menipy/gui/controllers/ (main,    │
                  │    pipeline_controller, setup_ctrl)     │
                  └────────────────────┬────────────────────┘
                                       │ Delegates background jobs
                                       ▼
                  ┌─────────────────────────────────────────┐
                  │               Services                  │
                  │   src/menipy/gui/services/ (runner,     │
                  │     material_catalog, plugin_service)   │
                  └────────────────────┬────────────────────┘
                                       │ Headless execution
                                       ▼
                  ┌─────────────────────────────────────────┐
                  │              Pipelines                  │
                  │      src/menipy/pipelines/<mode>/       │
                  └─────────────────────────────────────────┘
```

* **Threading Rule**: Long-running operations (pipeline execution, calibration, video batch processing) **must** run in background threads via `src/menipy/gui/services/pipeline_runner.py` (which uses Qt's `QThreadPool`). Never block the Qt event loop!
* **Step Test Panel**: Accessible in the GUI (**View -> Focus -> Science -> Test**). It sandboxes stage parameters for interactive parameter tuning.

### 3.4. Plugin Subsystem (`plugins/` and `src/menipy/common/`)
* Discovered automatically by scanning `plugins/` and tracked in `menipy_plugins.sqlite`.
* Registered in `src/menipy/common/registry.py`.
* Can register via module-level dictionaries (e.g., `EDGE_DETECTORS = {"my_detector": func}`) or via a `register(registries)` function.
* **Statelessness Rule**: Plugins must be deterministic and return a single-frame `DetectionResult`. Do not store mutable global or instance state across frames.

---

## 4. Task-to-File Routing Matrix

When assigned a task, start at the listed files and run the corresponding tests:

| Task | Primary Implementation Files | Key Tests to Run |
|---|---|---|
| **Sessile drop contact angles & geometry** | `src/menipy/pipelines/sessile/` <br>`src/menipy/common/sessile_detection.py` | `tests/test_sessile_geometry.py`<br>`tests/test_sessile_contact_angles.py` |
| **Pendant drop & Young-Laplace fitting** | `src/menipy/pipelines/pendant/` <br>`src/menipy/math/young_laplace.py` | `tests/test_pendant_pipeline.py`<br>`tests/test_adsa_geometry_phase_b.py` |
| **Dynamic sessile / video analysis** | `src/menipy/pipelines/sessile_dynamic/` <br>`src/menipy/common/temporal_sessile.py` | `tests/test_phase_d_dynamic_sessile.py` |
| **Auto-calibration & feature detection** | `src/menipy/common/auto_calibrator.py` <br>`src/menipy/common/detection_helpers.py` | `tests/test_auto_calibrator.py`<br>`tests/test_detection_plugins.py` |
| **GUI startup, layout & main window** | `src/menipy/gui/app.py` <br>`src/menipy/gui/views/main_window.py` | `tests/test_gui_startup_preview.py`<br>`tests/test_guided_ui_simplification.py` |
| **GUI pipeline orchestration & runner** | `src/menipy/gui/controllers/pipeline_controller.py` <br>`src/menipy/gui/services/pipeline_runner.py` | `tests/test_smoke_controller_flows.py`<br>`tests/test_pipeline_runner.py` |
| **Preview overlays & display** | `src/menipy/gui/views/preview_panel.py` <br>`src/menipy/gui/controllers/overlay_manager.py` | `tests/test_preview_overlay_layers.py` |
| **CLI commands, batch & exports** | `src/menipy/cli/` <br>`src/menipy/pipelines/runner.py` | `tests/test_cli.py` |
| **Plugin creation & registration** | `plugins/<plugin_name>.py` <br>`src/menipy/common/registry.py` | `tests/test_plugin_consolidation.py` |
| **Adding/modifying pipeline stage state** | `src/menipy/models/context.py` <br>`src/menipy/models/config.py` | `tests/test_models.py` |

---

## 5. Common AI Agent Anti-Patterns & Pitfalls

### ❌ Anti-Pattern 1: Adding Ad-Hoc Attributes to `Context`
```python
# BAD: Will raise pydantic_core.ValidationError at runtime!
ctx.my_custom_value = 42.0

# GOOD:
# 1. Add `my_custom_value: float | None = None` to `src/menipy/models/context.py`
# 2. Assign:
ctx.my_custom_value = 42.0
```

### ❌ Anti-Pattern 2: Running GUI Tests Without Offscreen Platform
```powershell
# BAD: May crash with display connection error or hang in headless CI
uv run pytest tests/test_gui.py

# GOOD:
$env:QT_QPA_PLATFORM="offscreen"
uv run --extra test pytest tests/test_gui.py
```

### ❌ Anti-Pattern 3: Writing Blocking Logic in GUI Views
```python
# BAD: Freezes the UI event loop
def on_analyze_clicked(self):
    results = run_heavy_adsa_pipeline(self.image)
    self.update_results(results)

# GOOD: Delegate via controller and service to background worker
def on_analyze_clicked(self):
    self.pipeline_controller.start_analysis_async()
```

### ❌ Anti-Pattern 4: Importing from `archive/` or Modifying `archive/`
Files under `archive/` represent past cleanups and deprecated modules. Never import from `archive/`. Look for the active equivalent under `src/menipy/` or `plugins/`.

### ❌ Anti-Pattern 5: Using Raw Python or Pip
```powershell
# BAD: Bypasses project-managed environment
python -m pytest
pip install foo

# GOOD:
uv run --extra test pytest
uv add foo
```

---

## 6. Conformance Test Suites

Menipy uses deterministic, fixture-manifest-backed conformance tests:
* **ADSA Detector Conformance**: `tests/test_adsa_detector_conformance.py` (validates against `tests/data/adsa_detector_manifest.json` across 48 synthetic cases).
* **ADSA Geometry Conformance**: `tests/test_adsa_geometry_phase_b.py` (validates 60 geometric cases against `tests/data/adsa_geometry_manifest.json`).
* **Temporal Sessile Conformance**: `tests/test_phase_d_dynamic_sessile.py` (validates 48 dynamic sequences without large binary files).

Always run the relevant conformance suite when modifying low-level edge detection, baseline clamping, or ODE optimization routines.
