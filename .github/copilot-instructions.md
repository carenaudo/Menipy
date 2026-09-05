# GitHub Copilot Instructions for Menipy

Menipy is an open-source Python toolkit for droplet and meniscus shape analysis from images (PySide6 GUI + headless CLI).

## Key Development Rules

1. **Python Environment & Commands**:
   - Use `uv` for all environment tasks: `uv run --extra test pytest`, `uv run --extra dev ruff check .`, `uv run menipy`.
   - In PowerShell, run GUI tests with `$env:QT_QPA_PLATFORM="offscreen"`.

2. **Strict Context State Model**:
   - `src/menipy/models/context.py` defines the shared pipeline state `Context` with `ConfigDict(extra="forbid")`.
   - Never set arbitrary new fields on `ctx` without declaring them in `Context` first.

3. **Subsystem Architecture**:
   - **GUI**: PySide6 widgets live in `src/menipy/gui/views/`, logic in `src/menipy/gui/controllers/`, background jobs in `src/menipy/gui/services/`.
   - **Pipelines**: Analysis pipelines live in `src/menipy/pipelines/<mode>/stages.py` and extend `PipelineBase`.
   - **Plugins**: Discoverable extensions live in `plugins/` and register with `src/menipy/common/registry.py`.
   - **Canonical Map**: Refer to `docs/CODEBASE_MAP.md` for ownership and task routes.

4. **Code Quality**:
   - Write NumPy-style docstrings for all new modules, classes, and public functions.
   - Code formatting must adhere to Black line length (88) and pass `ruff check .`.
   - Do not reference or import from `archive/` or `build/`.
