# Codebase Analysis Summary

This document summarizes the analysis of the Menipy codebase, focusing on documentation, file structure, and implementation status.

## 1. Project Status Overview

The Menipy project is currently in a **transition phase**.
- **Current State**: Most analysis pipelines (`pendant`, `sessile`, `capillary_rise`, `oscillating`) are implemented in a "simplified" manner, where the core logic is concentrated in `stages.py` files.
- **Goal**: The detailed plan files (`*_plan_pipeline.md`) outline a roadmap to modularize this logic into dedicated files (`geometry.py`, `physics.py`, `solver.py`, etc.), which currently exist mostly as empty placeholders or stubs in the respective directories.

## 2. Outdated and Legacy Files

The following files appear to be outdated, legacy, or candidates for cleanup:

*   **`src/menipy/pipelines/sessile/zold_drawing_alt.py`**: Explicitly named "old".
*   **`src/menipy/pipelines/sessile/zold_geometry_alt.py`**: Explicitly named "old".
*   **`src/menipy/ui/main_window.py`**: Described in `PYTHONFILES.md` as "previously used, now kept for reference". The active main window is `src/menipy/gui/mainwindow.py`.
*   **`PYTHONFILES.md` & `IMPORTED.md`**: These appear to be auto-generated documentation. They may be outdated if the generation script hasn't been run recently.
*   **`scripts/generate_legacy_map.py`**: Suggests a past migration effort.

## 3. Documentation and Plans

The project has a comprehensive set of documentation files. While there is some overlap, they generally serve different purposes:

*   **High-Level**:
    *   `PLAN.md`: The main roadmap.
    *   `TODO.md`: Specific task tracking.
    *   `GEMINI.md`: Context for AI agents.

*   **Detailed Pipeline Plans**:
    *   `pendant_plan_pipeline.md`
    *   `sessile_plan_pipeline.md`
    *   `capillary_rise_plan_pipeline.md`
    *   `oscillating_plan_pipeline.md`
    *   `captive_bubble_plan_pipeline.md`
    *   *Note*: These files are **not duplicates**. They contain specific, detailed technical specifications for each pipeline's refactoring and feature set. They accurately reflect the current "simplified" vs. "planned modular" state.

*   **Specific Features/Reference**:
    *   `needle_detection_plan.md`: Specific feature spec.
    *   `plan_DRy.md`: Reference for library usage (scikit-image, scipy, etc.).
    *   `pint_integration.md`: Specific integration plan.

## 4. Implementation Discrepancies

*   **Pipeline Modules**: As noted in the pipeline plans, many modules within `src/menipy/pipelines/<type>/` (e.g., `solver.py`, `physics.py`) are currently empty or minimal, with logic "inlined" in `stages.py`. This is a known state documented in the plans, not an accidental discrepancy.
*   **Sessile Pipeline**: The `sessile` pipeline has some "alt" files (`geometry_alt.py`, `drawing_alt.py`) and "zold" files, indicating an incomplete cleanup or experimentation phase.

## 5. Completed Actions

1.  **Cleanup**: Deleted `src/menipy/pipelines/sessile/zold_*.py` files.
2.  **Legacy UI**: Confirmed `src/menipy/ui/main_window.py` does not exist (likely already removed).
3.  **Documentation Update**: Regenerated `PYTHONFILES.md` and `IMPORTED.md` using a new script `scripts/generate_docs.py`.
4.  **Plan Consolidation**: Updated `PLAN.md` to link all `*_plan_pipeline.md` files.
