# Menipy: AI-Driven Droplet Shape Analysis Toolkit

This `GEMINI.md` file provides an overview of the Menipy project, specifically tailored for an AI agent like Gemini, to facilitate understanding and interaction with the codebase.

## 1. Project Overview

Menipy is a Python-based scientific toolkit designed for analyzing droplet shapes from images (contact angles, surface tension, and axisymmetric drop shape analysis - ADSA).

**Key Features:**
*   **Image Processing:** Classical CV algorithms and optional ONNX models (MobileSAM) for edge detection, segmentation, and feature extraction.
*   **PySide6 GUI:** Interactive graphical interface with live preview, auto-calibration wizard, and step-testing sandboxes.
*   **Physical Property Estimation:** Young-Laplace ODE profile fitting, Bond number optimization, and contact angle calculation.
*   **Headless CLI:** Batch image processing and temporal video sequence analysis.

## 2. AI-Driven Development Philosophy

The development of Menipy leverages AI agents and specialized sub-agents. Gemini's role is to operate within this architecture with precision and safety.

*   **Canonical Navigation:** Always consult [`docs/CODEBASE_MAP.md`](docs/CODEBASE_MAP.md) before making changes.
*   **Comprehensive Guide:** Refer to [`docs/guides/llm_coding_agent_guide.md`](docs/guides/llm_coding_agent_guide.md) for detailed agent rules.
*   **Result Contracts:** When changing analysis outputs, consult [`docs/contracts/`](docs/contracts/).
*   **Technical Specifications:** Detailed scientific descriptions, equations, and algorithms reside in [`docs/guides/`](docs/guides/) (e.g. `physics_models.md`, `drop_analysis.md`, `image_processing.md`).

## 3. Repository Structure

*   **`src/menipy/`**: Application package source.
    *   `gui/`: PySide6 application (views, controllers, services, dialogs).
    *   `cli/`: Command-line entry points and batch workflows.
    *   `pipelines/`: Modular analysis pipelines (`sessile`, `pendant`, `sessile_dynamic`, `captive_bubble`, etc.).
    *   `models/`: Pydantic data schemas (`context.py`, `config.py`, `results.py`).
    *   `common/`: Auto-calibration, geometry, edge detection, unit management, and plugin DB.
    *   `math/`: Young-Laplace numerical solvers and ODE integration.
*   **`plugins/`**: Custom runtime-discovered extensions (edge detectors, preprocessors, solvers).
*   **`docs/`**: Canonical documentation, contracts, research benchmarks, and guides.
*   **`tests/`**: Comprehensive unit, regression, integration, and UI test suite.
*   **`pyproject.toml`**: Package metadata, tools configuration (Ruff, Black, Mypy, Pytest).

## 4. Key Architectural Rules for Gemini

### 4.1. Strict `Context` Schema (`extra="forbid"`)
`src/menipy/models/context.py` enforces `model_config = ConfigDict(extra="forbid")`.
*   **Do NOT** dynamically attach arbitrary new fields to `ctx` without declaring them in `Context`.
*   If a stage needs new state, add the typed field definition directly to `Context` in `src/menipy/models/context.py`.

### 4.2. GUI Separation of Concerns
*   Keep GUI work strictly in PySide6.
*   Maintain the **View -> Controller -> Service** boundaries.
*   Never run blocking operations or pipelines on the main GUI thread; use `src/menipy/gui/services/pipeline_runner.py`.

### 4.3. Pipeline & Plugin Protocol
*   Pipelines inherit from `PipelineBase` (`src/menipy/pipelines/base.py`) and are discovered dynamically from `src/menipy/pipelines/<mode>/`.
*   Plugins live in `plugins/` and register with `src/menipy/common/registry.py`. They must remain stateless per frame.

## 5. Getting Started & Commands

**Always use `uv` for environment management, dependency resolution, and running scripts.**

1.  **Environment Sync:**
    ```powershell
    uv sync --extra dev --extra test
    ```

2.  **Running Tests:**
    Always set `QT_QPA_PLATFORM=offscreen` to run GUI and headless tests without display errors:
    ```powershell
    $env:QT_QPA_PLATFORM="offscreen"
    uv run --extra test pytest
    ```
    (Or specify a test file: `uv run --extra test pytest tests/test_sessile_geometry.py`)

3.  **Code Quality & Linting:**
    ```powershell
    uv run --extra dev ruff check .
    uv run --extra dev mypy src/menipy/models --config-file=pyproject.toml
    ```

4.  **Launching the Application:**
    *   Start the PySide6 GUI:
        ```powershell
        uv run menipy
        # or: uv run python -m menipy
        ```
    *   Run headless CLI:
        ```powershell
        uv run adsa --help
        ```

## 6. Interaction Guidelines

*   **Consult `docs/CODEBASE_MAP.md` First:** Locate the affected subsystems and recommended tests before editing.
*   **Test-Driven Execution:** Run the nearest tests to verify changes before and after modifying code.
*   **No Assumptions:** Always confirm file contents using `view_file` or `grep_search` rather than guessing internal APIs.
*   **Preserve Existing Contracts:** Never break existing output keys in `ctx.results` without updating the matching contract in `docs/contracts/`.