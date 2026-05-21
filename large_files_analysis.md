# Large Files & Refactoring Strategy Report

This document presents a rigorous analysis of the source files in the **Menipy** droplet shape analysis toolkit that exceed recommended code size thresholds. Monolithic classes and bloated modules are cataloged below, along with precise architectural refactoring and splitting strategies.

> [!NOTE]
> **Major GUI Reorganization Completed**: All active GUI files have been successfully relocated into logical packages (`views/`, `controllers/`, `dialogs/`, `helpers/`), and all obsolete panel views and legacy workflows have been completely deleted.

---

## 1. Code Size Threshold Definitions

*   <span style="color:red">**CRITICAL (>30,000 Bytes)**</span>: High urgency. Monolithic files exhibiting "God Class" symptoms, mixed concerns, or excessive event-handler boilerplate. Mandatory split candidates.
*   <span style="color:orange">**WARNING (15,000 – 30,000 Bytes)**</span>: Medium urgency. Complex controllers, complex dialog layouts, or heavily populated utility layers. Splitting or encapsulation is highly recommended.
*   <span style="color:blue">**NOTE (10,000 – 15,000 Bytes)**</span>: Low urgency. Stable mathematical collections, generated Qt layouts, or simple panels. Candidates for ongoing code size monitoring.

---

## 2. Comprehensive Code Size Inventory (Sorted Descending)

| Rank | File Path | Size (Bytes) | Est. Lines | Severity | Reorganization / Active Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | [views/main_window.py](file:///Users/charly/Coding/Menipy/src/menipy/gui/views/main_window.py) | 50,093 | 1,266 | <span style="color:red">**CRITICAL**</span> | **ACTIVE (Relocated to views/)** — Main UI workspace layout construction and action bindings. |
| 2 | [controllers/setup_panel_controller.py](file:///Users/charly/Coding/Menipy/src/menipy/gui/controllers/setup_panel_controller.py) | 40,878 | 1,026 | <span style="color:red">**CRITICAL**</span> | **ACTIVE** — Setup controller driving acquisition, calibration, and SOP state. |
| 3 | [views/results_panel.py](file:///Users/charly/Coding/Menipy/src/menipy/gui/views/results_panel.py) | 39,651 | 1,027 | <span style="color:red">**CRITICAL**</span> | **ACTIVE (Relocated to views/)** — Results history table and exporter module. |
| 4 | `src/menipy/gui/views/adsa_main_window.py` | 37,920 | 1,031 | <span style="color:gray">**DELETED**</span> | **CLEANED UP** — Obsolete experiment-selector main window. |
| 5 | `src/menipy/gui/views/sessile_drop_window.py` | 34,888 | 911 | <span style="color:gray">**DELETED**</span> | **CLEANED UP** — Obsolete experiment-selector sessile view. |
| 6 | `src/menipy/gui/views/pendant_drop_window.py` | 34,714 | 918 | <span style="color:gray">**DELETED**</span> | **CLEANED UP** — Obsolete experiment-selector pendant view. |
| 7 | [controllers/pipeline_controller.py](file:///Users/charly/Coding/Menipy/src/menipy/gui/controllers/pipeline_controller.py) | 33,186 | 847 | <span style="color:red">**CRITICAL**</span> | **ACTIVE** — Dynamic stage pipeline execution coordinator. |
| 8 | [views/image_view.py](file:///Users/charly/Coding/Menipy/src/menipy/gui/views/image_view.py) | 33,147 | 904 | <span style="color:red">**CRITICAL**</span> | **ACTIVE** — Interactive custom QGraphicsView widget. |
| 9 | [dialogs/calibration_wizard_dialog.py](file:///Users/charly/Coding/Menipy/src/menipy/gui/dialogs/calibration_wizard_dialog.py) | 32,347 | 812 | <span style="color:red">**CRITICAL**</span> | **ACTIVE** — Multi-step calibration state machine wizard. |
| 10 | [auto_calibrator.py](file:///Users/charly/Coding/Menipy/src/menipy/common/auto_calibrator.py) | 31,760 | 794 | <span style="color:red">**CRITICAL**</span> | **ACTIVE** — Backend calibration contour detection engine. |
| 11 | [views/preview_panel.py](file:///Users/charly/Coding/Menipy/src/menipy/gui/views/preview_panel.py) | 28,689 | 750 | <span style="color:orange">**WARNING**</span> | **ACTIVE (Relocated to views/)** — Stream viewer and layout manager hooks. |
| 12 | [dialogs/preprocessing_config_dialog.py](file:///Users/charly/Coding/Menipy/src/menipy/gui/dialogs/preprocessing_config_dialog.py) | 27,742 | 710 | <span style="color:orange">**WARNING**</span> | **ACTIVE** — Filter setup dialog tabs. |
| 13 | `src/menipy/gui/panels/setup_panel.py` | 27,116 | 680 | <span style="color:gray">**DELETED**</span> | **CLEANED UP** — Obsolete panels setup class (replaced by setup_panel_controller). |
| 14 | [dialogs/edge_detection_config_dialog.py](file:///Users/charly/Coding/Menipy/src/menipy/gui/dialogs/edge_detection_config_dialog.py) | 26,054 | 640 | <span style="color:orange">**WARNING**</span> | **ACTIVE** — Edge parameters settings. |
| 15 | [dialogs/analysis_settings_dialog.py](file:///Users/charly/Coding/Menipy/src/menipy/gui/dialogs/analysis_settings_dialog.py) | 24,812 | 580 | <span style="color:orange">**WARNING**</span> | **ACTIVE** — Custom pipeline parameters widget. |
| 16 | [preprocessing_helpers.py](file:///Users/charly/Coding/Menipy/src/menipy/common/preprocessing_helpers.py) | 24,331 | 550 | <span style="color:orange">**WARNING**</span> | **ACTIVE** — CV image helpers. |
| 17 | [views/setup_panel.ui](file:///Users/charly/Coding/Menipy/src/menipy/gui/views/setup_panel.ui) | 23,848 | N/A | <span style="color:orange">**WARNING**</span> | **ACTIVE** — Side rail XML layout. |
| 18 | [controllers/main_controller.py](file:///Users/charly/Coding/Menipy/src/menipy/gui/controllers/main_controller.py) | 23,523 | 602 | <span style="color:orange">**WARNING**</span> | **ACTIVE (Relocated to controllers/)** — Main application coordinator. |
| 19 | [views/ui_main_window.py](file:///Users/charly/Coding/Menipy/src/menipy/gui/views/ui_main_window.py) | 22,687 | 520 | <span style="color:orange">**WARNING**</span> | **ACTIVE** — Auto-generated Qt layout code. |
| 20 | [geometry.py](file:///Users/charly/Coding/Menipy/src/menipy/common/geometry.py) | 22,372 | 480 | <span style="color:orange">**WARNING**</span> | **ACTIVE** — Math and normal vector solvers. |
| 21 | [dialogs/overlay_config_dialog.py](file:///Users/charly/Coding/Menipy/src/menipy/gui/dialogs/overlay_config_dialog.py) | 21,720 | 510 | <span style="color:orange">**WARNING**</span> | **ACTIVE** — Overlay visibility toggles. |
| 22 | [helpers/theme.py](file:///Users/charly/Coding/Menipy/src/menipy/gui/helpers/theme.py) | 18,463 | 420 | <span style="color:orange">**WARNING**</span> | **ACTIVE (Relocated to helpers/)** — Styling and HSL colors sheet. |
| 23 | [dialogs/material_dialog.py](file:///Users/charly/Coding/Menipy/src/menipy/gui/dialogs/material_dialog.py) | 17,304 | 400 | <span style="color:orange">**WARNING**</span> | **ACTIVE** — SQLite density database grid manager. |
| 24 | [controllers/dialog_coordinator.py](file:///Users/charly/Coding/Menipy/src/menipy/gui/controllers/dialog_coordinator.py) | 16,840 | 410 | <span style="color:orange">**WARNING**</span> | **ACTIVE** — Maps dialog activations to state trees. |
| 25 | [dialogs/detector_test_dialog.py](file:///Users/charly/Coding/Menipy/src/menipy/gui/dialogs/detector_test_dialog.py) | 15,044 | 380 | <span style="color:orange">**WARNING**</span> | **ACTIVE** — Detector verification grid view. |

---

## 3. Dedicated Refactoring & Splitting Strategies for the Remaining Active Critical Files

The following actionable blueprints outline how to decompose the remaining active critical files (excluding decommissioned/obsolete ADSA experiment-selector views) into single-responsibility, highly testable components:

### 1. `src/menipy/gui/main_window.py` (50,089 Bytes)
*   **Problem**: Serves as a monolithic "God Class" coordinating overall layout construction, menu item actions, dynamic stylesheets, sidebar assembly, dialog triggers, and cleanup routines.
*   **Strategy**:
    *   Extract dynamic menu action wiring and keyboard shortcuts to a helper: `src/menipy/gui/helpers/window_actions.py`.
    *   Delegate layout presets and toggle panels completely to `layout_manager.py` instead of performing geometry logic inside `main_window.py`.
    *   Move the complex workflow-bar state mapping (`_sync_workflow_source_mode_buttons`, etc.) to a specialized sub-widget class.

### 2. `src/menipy/gui/controllers/setup_panel_controller.py` (40,870 Bytes)
*   **Problem**: Violates the Single Responsibility Principle by driving camera configuration, batch walking loops, directory listings, dynamic coordinate draw modes, manual calibrations, and SOP updates.
*   **Strategy**: Split into three specialized controllers under `src/menipy/gui/controllers/`:
    1.  `CameraSetupController`: Manages cameras, frames, FPS spinboxes, and device scanning.
    2.  `BatchSetupController`: Coordinates folder path selection, walking directories, file extension filtering, list lists, and batch UI states.
    3.  `CalibrationSetupController`: Manages mouse drawing overlays (ROI, baseline) and triggers coordinates resolution.

### 3. `src/menipy/gui/views/results_panel.py` (39,643 Bytes)
*   **Problem**: Combines UI grid rendering (table row highlights, fonts, priority column lists) with calculations (SI to CGS metric conversions, standard deviations) and export file writing operations (CSV, images).
*   **Strategy**:
    *   Move SI-to-CGS and localized string formatting directly to a shared units system: `src/menipy/common/units.py`.
    *   Extract file exports (rendering overlays to disk and writing CSV files) into `src/menipy/gui/services/results_exporter.py`.
    *   Move run list records coordination into a lightweight coordinator: `ResultsHistoryCoordinator`.

### 4. Obsolete ADSA Experiment-Selector Views (No Refactor Planned)
*   **Impacted Files**:
    *   `src/menipy/gui/views/adsa_main_window.py` (37,920 Bytes)
    *   `src/menipy/gui/views/sessile_drop_window.py` (34,888 Bytes)
    *   `src/menipy/gui/views/pendant_drop_window.py` (34,714 Bytes)
*   **Decision**: **Decommissioned and Marked for Deletion**. No code refactoring or splitting is required. These files are listed in the code inventory strictly for sizing visibility, but are marked for complete removal under the GUI cleanup sweep.

### 5. `src/menipy/gui/controllers/pipeline_controller.py` (33,186 Bytes)
*   **Problem**: Mixes backend solver execution logic (module discovery, dynamic import hooks) with graphical side-effects (launching error boxes, updating window status bars, updating results tables).
*   **Strategy**:
    *   Move dynamic module discovery and python loader hooks to a clean service: `src/menipy/gui/services/pipeline_loader.py`.
    *   Abstract results formatting away from the controller into a clean mapper function.

### 6. `src/menipy/gui/views/image_view.py` (33,147 Bytes)
*   **Problem**: Combines high-frequency QGraphicsView mouse events (panning, zoom, tracking coordinates) with custom drawing algorithms for ROI rectangles, needle coordinates, and baseline paths.
*   **Strategy**:
    *   Extract the layered graphics items (ROI, needle, baseline, contour) and coordinate conversions into a clean manager class: `src/menipy/gui/helpers/drawing_layer_manager.py`.
    *   Keep `image_view.py` focused purely on rendering, zooming, panning, and viewport bounds calculations.

### 7. `src/menipy/gui/dialogs/calibration_wizard_dialog.py` (32,347 Bytes)
*   **Problem**: Integrates multiple custom WizardPages inside a single python class with extensive page transition, pixel-ratio computation, and validation checks.
*   **Strategy**:
    *   Extract the custom layouts and inputs of each step (e.g. Needle Calibration page, grid scale calibration, ROI setting) into individual, self-contained `QWizardPage` widget classes located under a new folder: `src/menipy/gui/dialogs/calibration_wizard/`.

### 8. `src/menipy/common/auto_calibrator.py` (31,760 Bytes)
*   **Problem**: Monolithic image processing file containing multiple complex math and computer vision calculations (Hough transforms for needle shafts, edge filtering, ROI bounding coordinates, baseline fit formulas).
*   **Strategy**:
    *   Modularize the CV operations into separate, single-responsibility solvers in a new module: `src/menipy/common/auto_calibration/`:
        -   `NeedleSolver`: Detects the needle bounds and tip coordinate.
        -   `SubstrateSolver`: Fits the horizontal baseline.
        -   `RoiSolver`: Computes the optimized droplet bounding box.
    *   Leave `auto_calibrator.py` as a clean coordinator that delegates to these solvers and returns the resolved coordinates.
