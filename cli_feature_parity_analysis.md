# GUI vs. CLI Feature Parity & Consolidation (Completed)

This document summarizes the completed work to eliminate command-line entry point conflicts and close the feature gaps between the PySide6 GUI and the CLI command-runner in the **Menipy** droplet shape analysis toolkit.

---

## 1. Consolidated CLI Architecture

To resolve entry collisions, the legacy conflicting script `src/menipy/cli.py` has been completely deleted.
The CLI is now centralized under `src/menipy/cli/` using a modern, unified package design:
*   **`src/menipy/cli/__main__.py`**: Standard package entry point (`python -m menipy.cli`).
*   **`src/menipy/cli/__init__.py`**: Contains the full argument parser, SQLite database connections, SOP parser, file directory walking logic, and visual overlay pipelines.
*   **`pyproject.toml` console scripts**: The `adsa` and `menipy-cli` hooks have been registered to point directly to `menipy.cli:main`, guaranteeing consistent runtime behavior.

---

## 2. Completed Feature Parity Matrix

All identified functional gaps between the graphical user interface and the terminal command runner have been successfully closed:

| Feature / Capability | GUI Support | CLI Support | Status / Resolution |
| :--- | :--- | :--- | :--- |
| **Batch Processing** | **YES** <br>Processes full folders sequentially. | **YES** (Closed!) | Added `--input-dir` / `-i` and `--glob` / `-g` options. Walks folders recursively, processes matching images sequentially, generates visuals, and exports a unified `results.csv`. |
| **Auto-Calibration** | **YES** <br>Runs `AutoCalibrator` dynamically on frames. | **YES** (Closed!) | Added `--auto-calibrate` / `-a` toggle. If ROI or needle bounding coordinates are omitted, the CLI dynamically executes baseline `run_auto_calibration` on the first frame to determine boundaries on-the-fly. |
| **Material/Needle DBs** | **YES** <br>SQLite DB dialogs for fluids/needles. | **YES** (Closed!) | Connected to SQLite database `menipy_materials.sqlite`. Added `--material` / `--fluid-name` and `--needle-name` parameters to query fluid densities (`rho1`, `rho2`) and needle outer diameters (e.g. `22G` gauge matches `0.72` mm) by name. |
| **SOP Config Loading** | **YES** <br>Loads preprocessor/edge state profiles. | **YES** (Closed!) | Added `--sop` / `-s` to load standard JSON SOP configuration templates (injecting Canny thresholds, CLAHE settings, blur settings, etc.) with command-line overrides taking precedence. |
| **Visuals & CSV Export** | **YES** <br>Visual overlays and run tables. | **YES** (Closed!) | Batch mode exports a structured run table `results.csv` including file names, pipeline status, QA checks, and physical measurements. Output directory is fully customizable via `--output-dir` / `-o`. Collision-proof visuals are exported as `<basename>_overlay.png` and `<basename>_preview.png`. |
| **Headless Cleanliness** | **YES** (Active Qt main loop). | **YES** (Closed!) | The consolidated CLI imports core algorithms and databases completely isolated from PySide6 view classes, ensuring flawless, headless execution on CI/CD servers, remote shells, or cloud run environments without requiring a display server. |

---

## 3. Integration Test Coverage

We created the comprehensive CLI integration test suite **`tests/test_cli.py`**, verifying:
1.  **Coordinate Parsing Errors**: Asserts proper parsing validation for malformed ROI sizes, non-numeric strings, and coincident line endpoints.
2.  **Materials SQLite DB Queries**: Asserts accurate physical property lookups for densities and needle outer diameters using temporary in-memory database clones.
3.  **SOP Loading Validation**: Asserts proper standard operating procedure parameter ingestion.
4.  **Single-Image Pipeline**: Executes sessile droplet analysis using Auto-Calibration fallback and validates that visual preview/overlay images and a structured `results.json` are written correctly.
5.  **Batch Processing Directory Walker**: Iterates over multiple images, checks for collision-preventing file naming, and validates that a detailed consolidated `results.csv` is correctly created and filled with droplet measurements.
6.  **Argument Overrides**: Verifies that custom preprocessors and Sobel/Canny edge overrides take priority over defaults.
