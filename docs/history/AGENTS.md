# CODEX Agents Configuration

This file defines the specialized agents (sub-agents) that CODEX will instantiate and coordinate to implement the droplet shape analysis application. Each agent is responsible for a distinct layer of the architecture, with clear input/output interfaces and validation steps.

---

## 1. Documentation Agent

**Role:** Ingest and validate all reference material in `/doc/`.

**Responsibilities:**
- Read and parse markdown files in `/doc/` (physics_models.md, numerical_methods.md, image_processing.md, gui_design.md).
- Extract equations, algorithm parameters, and workflow descriptions.
- Populate an internal knowledge base for downstream agents.

**Inputs:** `/doc/*.md` files
**Outputs:** Parsed algorithm metadata, list of required functions and constants.

---

## 2. Scaffold Agent

**Role:** Create initial project structure and configuration files.

**Responsibilities:**
- Generate directory layout as specified in `PLAN.md`.
- Create `requirements.txt` with placeholders for each dependency.
- Write `setup.py` stub with package metadata and entry point.
- Initialize a Git repository (if applicable).

**Inputs:** `PLAN.md` project structure section
**Outputs:** Empty directories and stub files in `/src`, `/tests`, `/data`, and root.

---

## 3. Environment Agent

**Role:** Set up and verify the Python environment.

**Responsibilities:**
- Create a Python 3.9+ virtual environment.
- Install dependencies from `requirements.txt`.
- Validate installed versions match pins.

**Inputs:** `requirements.txt`
**Outputs:** Active venv with all packages installed.

---

## 4. Processing Agent

**Role:** Implement image loading and preprocessing modules.

**Responsibilities:**
- Write `processing/reader.py` for generic image I/O (OpenCV, scikit-image wrappers).
- Write `processing/segmentation.py` implementing Otsu, adaptive thresholding, morphological filters.
- Provide unit tests for each function.

**Inputs:** Parsed segmentation algorithms from Documentation Agent
**Outputs:** `processing` module with functions and tests under `tests/test_processing.py`

---

## 5. Modeling Agent

**Role:** Develop geometric and physical shape-fitting models.

**Responsibilities:**
- Implement circle-fit, ellipse-fit, tangent polynomial in `models/geometry.py`.
- Implement Young–Laplace ODE solver and ADSA optimization in `models/physics.py`.
- Expose property calculators in `models/properties.py` (surface tension, contact angle, volume).
- Write corresponding tests in `tests/test_models.py`.

**Inputs:** Parsed equations from physics_models.md and numerical_methods.md
**Outputs:** `models` package and test suite

---

## 6. GUI Agent

**Role:** Build interactive interface using PySide6.

**Responsibilities:**
- Create `gui/main_window.py` with a two-panel layout (QGraphicsView for image + control widgets).
- Implement dialogs for file open, calibration, model selector, and batch parameters.
- Connect processing and modeling functions to GUI actions.
- Ensure overlays of contours and fits render correctly.
- Add “Save annotated image” functionality.

**Inputs:** UI design descriptions from gui_design.md, Processing & Modeling modules
**Outputs:** Working GUI application code and `tests/test_gui.py`

---

## 7. Batch Agent

**Role:** Automate bulk processing and result aggregation.

**Responsibilities:**
- Implement `batch.py` to iterate over an image directory.
- Apply default segmentation and modeling for each image.
- Aggregate results into a pandas DataFrame and export CSV.
- Provide CLI entry points and usage documentation.

**Inputs:** Processing & Modeling modules
**Outputs:** `batch.py` and sample run in documentation

---

## 8. CI & Packaging Agent

**Role:** Finalize packaging and continuous integration.

**Responsibilities:**
- Complete `setup.py` with metadata, dependencies, and entry point.
- Write `GitHub Actions` workflow to install deps, run pytest, and measure coverage.
- Ensure wheel and source distributions build successfully.

**Inputs:** Project codebase, tests
**Outputs:** `.github/workflows/ci.yml`, built artifacts

---

## 9. CODEXLOG Agent

**Role:** Maintain a persistent activity log for CODEX tasks.

**Responsibilities:**
- Append a new entry to `CODEXLOG.md` after each CODEX task.
- Summarize the task description and CODEX's response.
- Keep the log concise and in chronological order.

**Inputs:** Task descriptions and summaries from other agents.
**Outputs:** Updated `CODEXLOG.md` with a new entry.

---
*End of AGENTS.md*
