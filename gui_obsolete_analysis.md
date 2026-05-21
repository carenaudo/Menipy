# Obsolete GUI Files Audit (Completed)

This document details the completed analysis and cleanup of obsolete, redundant, placeholder, or decommissioned files within the PySide6 user interface implementation of the **Menipy** droplet shape analysis toolkit.

Following architectural review, the **new ADSA experiment-selector workflow** was officially designated as **OBSOLETE**. The classic workspace entry point (`app.py` -> `main_window.py`) remains the single active production GUI path, and all files associated with the experiment-selector workflow have been successfully decommissioned and removed.

---

## 1. Decommissioned & Deleted Obsolete Files

The following files have been completely removed from the active codebase.

### A. Legacy Stubs & Backups (Successfully Deleted)
*   **`src/menipy/gui/mainwindow.py`** (5,928 bytes) — Legacy main window shim (superseded by `main_window.py`).
*   **`src/menipy/gui/panels/discover.py`** (50 bytes) — Empty panel discovery stub (superseded by backend pipeline auto-discovery).
*   **`src/menipy/gui/viewmodels/results_vm.py`** (38 bytes) — Empty placeholder viewmodel.
*   **`src/menipy/gui/overlay.py`** (4,042 bytes) — Legacy overlay painter (superseded by `overlay_manager.py` and backend `overlay.py`).
*   **`src/menipy/gui/views/ui_main_window.py.bak`** (8,295 bytes) — Outdated backup file.

### B. ADSA Experiment-Selector Workflow Files (Successfully Deleted)
*   **`src/menipy/gui/adsa_app.py`** (1,887 bytes) — Obsolete selector application wrapper.
*   **`src/menipy/gui/views/adsa_main_window.py`** (37,920 bytes) — Unused selector view window.
*   **`src/menipy/gui/views/experiment_selector.py`** (11,402 bytes) — Unused grid selection UI.
*   **`src/menipy/gui/views/sessile_drop_window.py`** (34,888 bytes) — Unused three-pane sessile drop window.
*   **`src/menipy/gui/views/pendant_drop_window.py`** (34,714 bytes) — Unused pendant drop window.
*   **`src/menipy/gui/views/tilted_sessile_window.py`** (13,420 bytes) — Unused tilted sessile drop window.
*   **`src/menipy/gui/views/base_experiment_window.py`** (13,559 bytes) — Obsolete template class parent.
*   **`src/menipy/gui/widgets/experiment_card.py`** (7,911 bytes) — Unused experiment click cards.
*   **`src/menipy/gui/widgets/pendant_results_widget.py`** (11,162 bytes) — Unused pendant statistics panel.
*   **`src/menipy/gui/widgets/tilted_sessile_results_widget.py`** (10,533 bytes) — Unused tilted sessile results panel.

### C. Obsolete Test Suites & CLI Scripts (Successfully Deleted)
*   **`tests/test_project_loading.py`** — Legacy GUI test asserting obsolete layouts.
*   **`src/menipy/cli.py`** — Legacy conflicting command-line script (superseded by the consolidated `src/menipy/cli/` package).

### D. Obsolete Panels & Manual Wizard (Successfully Deleted in Phase 2)
*   **`src/menipy/gui/dialogs/calibration_wizard.py`** (14,463 bytes) — Legacy manual calibration wizard dialog (superseded by `calibration_wizard_dialog.py`).
*   **`src/menipy/gui/panels/setup_panel.py`** (27,116 bytes) — Legacy widget class (superseded by dynamic `SetupPanelController` and `setup_panel.ui`).
*   **`src/menipy/gui/panels/image_source_panel.py`** (11,431 bytes) — Legacy panel from the obsolete ADSA selector workspace.
*   **`src/menipy/gui/panels/tilt_stage_panel.py`** (9,452 bytes) — Legacy panel from the obsolete ADSA selector workspace.
*   **`src/menipy/gui/panels/needle_calibration_panel.py`** (9,126 bytes) — Legacy panel from the obsolete ADSA selector workspace.
*   **`src/menipy/gui/panels/calibration_panel.py`** (6,909 bytes) — Legacy panel from the obsolete ADSA selector workspace.
*   **`src/menipy/gui/panels/action_panel.py`** (6,551 bytes) — Legacy panel from the obsolete ADSA selector workspace.
*   **`src/menipy/gui/panels/parameters_panel.py`** (11,176 bytes) — Legacy panel from the obsolete ADSA selector workspace.
*   **`src/menipy/gui/panels/__init__.py`** (637 bytes) — Unused package initializer.

---

## 2. Active Panel Views Relocation & Reorganization (Phase 2)

To completely eliminate the redundant `panels/` layer and unify all visual elements, the two active panel view files were relocated:
*   **`src/menipy/gui/panels/preview_panel.py`** -> **`src/menipy/gui/views/preview_panel.py`**
*   **`src/menipy/gui/panels/results_panel.py`** -> **`src/menipy/gui/views/results_panel.py`**

Following this relocation, the empty directory **`src/menipy/gui/panels/`** and its pycache artifacts were completely deleted.

---

## 3. Integrity Verification & Impact

1.  **Zero Import Regressions**: Active controller and panel views (`main_window.py`, `camera_manager.py`, `image_marking.py`, etc.) were refactored to import relocated panels from `menipy.gui.views` rather than `menipy.gui.panels`.
2.  **Classic Workspace Intact**: The classic split-pane workspace entry flow (`app.py` -> `main_window.py`) remains 100% active, fully tested, and unaffected by the cleanup.
3.  **Automated Verification**: Run complete test suite (`.venv/bin/pytest`) to confirm zero regressions:
    *   **Result**: All 279 unit and GUI integration tests pass cleanly with 100% success.

