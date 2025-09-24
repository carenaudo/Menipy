# Menipy: AI-Driven Droplet Shape Analysis Toolkit

This `GEMINI.md` file provides an overview of the Menipy project, specifically tailored for an AI agent like Gemini, to facilitate understanding and interaction with the codebase.

## 1. Project Overview

Menipy is a Python-based toolkit designed for analyzing droplet shapes from images. Its primary goal is to achieve development with minimal human involvement, leveraging an AI agent named CODEX to orchestrate various specialized sub-agents.

**Key Features:**
*   **Image Processing:** Algorithms for segmentation, edge detection, and feature extraction.
*   **PySide6 GUI:** A graphical user interface for interactive analysis.
*   **Physical Property Estimation:** Calculation of surface tension, contact angles, and other droplet metrics.

## 2. AI-Driven Development Philosophy

The development of Menipy is driven by an AI agent and its sub-agents. Gemini's role is to understand and contribute within this AI-orchestrated framework.

*   **CODEX:** The primary AI orchestrator.
*   **Sub-Agents:** Specialized agents (e.g., Documentation, Scaffold, Environment, Processing, Modeling, GUI, Batch, CI & Packaging) whose roles are defined in `AGENTS.md`.
*   **Development Plan:** High-level plans and desired features are outlined in `PLAN.md`.
*   **Reference Material:** Detailed technical specifications, equations, and workflow descriptions are provided in the `doc/` directory.

## 3. Repository Structure (Relevant for AI Agent)

Understanding the repository structure is crucial for effective navigation and contribution.

*   **`AGENTS.md`**: Defines the roles and responsibilities of the various AI sub-agents involved in Menipy's development.
*   **`PLAN.md`**: Contains the high-level development plan, including desired directory layout, technology stack, and feature set.
*   **`doc/`**: A directory containing supporting Markdown files (e.g., `physics_models.md`, `numerical_methods.md`, `image_processing.md`, `gui_design.md`, `drop_analysis.md`) that provide detailed technical context for the AI agents.
*   **`src/menipy/pipelines/`**: This directory houses the modular analysis pipelines (e.g., `pendant`, `sessile`). New pipelines are added as subdirectories here.
*   **`plugins/`**: This directory is for custom plugins that extend Menipy's functionality (e.g., image filters, solvers).
*   **`requirements.txt`**: Lists all Python dependencies required for the project.
*   **`tests/`**: Contains unit and integration tests for various components of the application.
*   **`.venv/`**: The virtual environment directory, which should be used for all Python-related commands.

## 4. Key Architectural Concepts

### 4.1. Pipelines

Menipy's analysis capabilities are built on a flexible, stage-based pipeline architecture.
*   **`PipelineBase`**: The base class (`src/menipy/pipelines/base.py`) defining a series of stages (Acquisition, Preprocessing, Edge Detection, Geometry, Scaling, Physics, Solver, Optimization, Outputs, Overlay, Validation).
*   **`Context` Object**: A central data container passed between pipeline stages, enabling loose coupling.
*   **Discovery**: Pipelines are automatically discovered from subdirectories within `src/menipy/pipelines/`.

### 4.2. Plugins

The plugin system allows for extending functionality with new algorithms and processing stages.
*   **Discovery**: Plugins are discovered by scanning designated directories (e.g., `plugins/`) and managed by `PluginDB` (`src/menipy/common/plugin_db.py`).
*   **Registration**: Plugins register their functionality with a central `registry` (`src/menipy/common/registry.py`).
*   **Types (Kinds)**: Plugins are categorized by "kind" (e.g., `acquisition`, `edge_detection`, `solver`).

## 5. Getting Started (for Gemini)

To interact with and contribute to the Menipy project, follow these guidelines:

1.  **Environment Setup:**
    *   Ensure a virtual environment exists at `.venv/`.
    *   Install all dependencies using the virtual environment's pip:
        ```bash
        .venv\Scripts\python.exe -m pip install -r requirements.txt
        ```

2.  **Running Tests:**
    *   Execute tests using the virtual environment's pytest:
        ```bash
        .venv\Scripts\python.exe -m pytest
        ```
        (Or specify a particular test file: `.venv\Scripts\python.exe -m pytest tests/test_your_file.py`)

3.  **Launching the Application:**
    *   Start the Menipy GUI:
        ```bash
        .venv\Scripts\python.exe -m src
        ```

## 6. Interaction Guidelines for Gemini

*   **Consult Documentation:** Always refer to `AGENTS.md`, `PLAN.md`, and the `doc/` directory for context, requirements, and technical details before making significant changes.
*   **Adhere to Conventions:** Mimic existing code style, structure, and architectural patterns.
*   **Test-Driven Approach:** When implementing new features or fixing bugs, prioritize writing or updating tests to ensure correctness and prevent regressions.
*   **Virtual Environment:** Always use the Python executable within the `.venv/` directory for all Python-related commands.
*   **Explain Critical Commands:** Before executing commands that modify the file system or codebase, provide a brief explanation.
*   **No Assumptions:** Do not make assumptions about file contents; use `read_file` or `read_many_files` to confirm.