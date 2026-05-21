# Code Cleanup Tracker

This file tracks duplicated and unused code in the Menipy project, and provides a plan for safe removal.

Analysis date: May 19, 2026

Scope used for this pass:
- Runtime entry chain: src/__main__.py -> src/menipy/gui/app.py -> src/menipy/gui/main_window.py
- Import-following static scan: python scripts/find_unreferenced.py
- Duplicate heuristics: normalized text similarity pass over plugins/ and src/menipy/common/

---

## Duplicated Code

Status legend:
- Candidate: suspected duplicate, not yet approved for refactor
- Confirmed: reviewed and approved for consolidation

1) Candidate - Detection logic duplicated between detector and preprocessor plugins
- plugins/detect_drop.py
- plugins/preproc_detect_drop.py
- Why flagged: very similar sessile pipeline stages (CLAHE, adaptive threshold, contour filtering)
- Risk: medium (different APIs: function-style detector vs ctx mutating preprocessor)
- Validation before refactor: run tests/test_preproc_plugins.py and tests/test_detection_plugins.py

2) Candidate - Needle logic duplicated between detector and preprocessor plugins
- plugins/detect_needle.py
- plugins/preproc_detect_needle.py
- Why flagged: overlap in sessile thresholding and pendant shaft/contact calculations
- Risk: medium-high (pendant contact-point behavior is sensitive)
- Validation before refactor: run tests/test_preproc_plugins.py and tests/test_detection_plugins.py

3) Candidate - Edge plugin boilerplate duplication
- plugins/circle_edge.py
- plugins/sine_edge.py
- Why flagged: near-identical registration, optional cv2 path, and fallback structure
- Risk: low-medium
- Validation before refactor: run tests/test_edge_detectors_plugin.py

4) Candidate - Generic plugin skeleton overlap
- plugins/scaler_identity.py
- plugins/validator_basic.py
- Why flagged: plugin boilerplate patterns are repeated
- Risk: low
- Validation before refactor: run tests/test_plugin_consolidation.py

## GUI Detection Baseline (Needle + Drop)

Decision:
- Baseline for needle/drop detection is the GUI auto-calibration path, not preprocessor variants.
- Authoritative flow: main window Auto-Calibrate -> main controller -> calibration wizard -> AutoCalibrator.

Reference flow files:
- src/menipy/gui/main_window.py (Auto-Calibrate button wiring)
- src/menipy/gui/main_controller.py (on_auto_calibrate_requested)
- src/menipy/gui/dialogs/calibration_wizard_dialog.py (run_detection and re-detection with manual substrate/needle)
- src/menipy/common/auto_calibrator.py (baseline algorithms)

Baseline behavior to preserve:
- Sessile needle:
	- CLAHE + adaptive threshold + morphology
	- First contour touching top border (y < 5)
- Sessile drop:
	- Adaptive threshold segmentation and substrate mask
	- Needle-proximity rejection (minimum vertical gap from needle)
	- Rectangularity rejection (ROI-like contours)
	- Prefer substrate-touching contours, fallback to floating/largest
	- Contact points clamped to substrate line
- Pendant drop:
	- Otsu threshold + morphology
	- Largest centered contour with min area fraction
- Pendant needle:
	- Shaft-based contact detection from top region
	- Contact deviation tolerance and needle rect from contact points

Current drift observed against baseline:
- plugins/detect_drop.py (sessile): missing explicit needle-proximity filter present in AutoCalibrator
- plugins/detect_drop.py (sessile): substrate_touch_tolerance default differs (10 vs 15 in AutoCalibrator)
- plugins/detect_drop.py (sessile): contact-point extraction is simpler than AutoCalibrator near-substrate logic
- plugins/detect_needle.py: largely aligned with AutoCalibrator for sessile and pendant paths

Alignment tasks (baseline-first):
1. Refactor detect_drop_sessile to delegate shared selection logic to baseline helper extracted from AutoCalibrator.
2. Keep plugin public API unchanged; only internal contour-selection logic changes.
3. Add regression tests that compare plugin output against AutoCalibrator on the same fixture images.
4. Only then refactor preproc_detect_drop/preproc_detect_needle toward shared helpers.

Progress on baseline alignment:
- Completed: plugins/detect_drop.py aligned closer to AutoCalibrator sessile baseline
	- Added needle-proximity filter (minimum gap + alignment guard)
	- Updated substrate_touch_tolerance default to 15 (matching baseline)
	- Improved contact-point extraction using near-substrate selection and baseline clamping logic
	- Preserved public API compatibility (optional needle_rect also accepted via kwargs)
- Completed: plugins/preproc_detect_drop.py now delegates to detect_drop plugin (baseline-aligned path)
	- Removed duplicated sessile/pendant contour-selection implementation from preprocessor
	- Preserved context outputs (detected_contour, contact_points, apex_point)
- Completed: parity tests added for baseline consistency
	- tests/test_detection_plugins.py includes AutoCalibrator parity checks for sessile drop and pendant needle detector
- Completed: plugins/preproc_detect_needle.py now delegates to detect_needle plugin (baseline-aligned path)
	- Removed duplicated sessile/pendant needle extraction logic from preprocessor
	- Preserved context outputs (needle_rect, contact_points)
- Completed: sessile needle parity check against AutoCalibrator baseline
	- tests/test_detection_plugins.py validates geometry compatibility with GUI baseline
- Validation after change:
	- pytest -q tests/test_preproc_plugins.py tests/test_detection_plugins.py tests/test_sessile_auto_detection.py tests/test_pipeline_runner.py tests/test_gui_startup_preview.py
	- Result: 46 passed, 1 skipped

## Unused Code

Important: this section is static-analysis output and may include false positives where dynamic imports are used.

### A) High-confidence candidates (legacy/empty, no direct import hits, or officially decommissioned)

- src/menipy/common/zold_detection.py (archived and removed from active tree)
- src/menipy/gui/zold_drawing_alt.py (archived and removed from active tree)
- src/menipy/gui/mainwindow.py (compatibility shim; no current imports found)
- **ADSA Experiment-Selector Workflow** (officially decommissioned and marked for removal):
  - src/menipy/gui/adsa_app.py (selector entry point)
  - src/menipy/gui/views/adsa_main_window.py (obsolete multi-stage main view)
  - src/menipy/gui/views/experiment_selector.py (unused grid selector container)
  - src/menipy/gui/views/sessile_drop_window.py (unused three-pane sessile view)
  - src/menipy/gui/views/pendant_drop_window.py (unused pendant view)
  - src/menipy/gui/views/tilted_sessile_window.py (unused tilted sessile view)
  - src/menipy/gui/views/base_experiment_window.py (obsolete window base class)
  - src/menipy/gui/widgets/experiment_card.py (unused click cards widget)
  - src/menipy/gui/widgets/pendant_results_widget.py (unused results panel)
  - src/menipy/gui/widgets/tilted_sessile_results_widget.py (unused results panel)

Archive location for removed low-risk files:
- archive/2026-05-cleanup/src/menipy/common/zold_detection.py
- archive/2026-05-cleanup/src/menipy/gui/zold_drawing_alt.py
- archive/2026-05-cleanup/src/menipy/math/rayleigh_lamb.py

Post-removal validation:
- pytest -q tests/test_preproc_plugins.py tests/test_detection_plugins.py tests/test_sessile_auto_detection.py tests/test_pipeline_runner.py tests/test_gui_startup_preview.py tests/test_import_health.py tests/test_imports.py
- Result: 48 passed, 1 skipped

Additional triage completed:
- Placeholder module archived/removed: src/menipy/math/rayleigh_lamb.py
- Validation result after removal: 48 passed, 1 skipped

### B) Unreferenced module candidates from scripts/find_unreferenced.py

- src/menipy/common/acquisition.py
- src/menipy/common/optimization.py
- src/menipy/common/outputs.py
- src/menipy/common/overlay.py
- src/menipy/common/physics.py
- src/menipy/common/scaling.py
- src/menipy/gui/dialogs/analysis_settings/__init__.py
- src/menipy/gui/dialogs/analysis_settings/captive_bubble_settings.py
- src/menipy/gui/dialogs/analysis_settings/pendant_settings.py
- src/menipy/gui/dialogs/analysis_settings/sessile_settings.py
- src/menipy/gui/dialogs/calibration_wizard.py
- src/menipy/gui/overlay.py
- src/menipy/gui/panels/discover.py
- src/menipy/gui/panels/setup_panel.py
- src/menipy/gui/resources/menipy_icons_rc.py
- src/menipy/gui/theme.py
- src/menipy/gui/viewmodels/results_vm.py
- src/menipy/gui/views/step_item_widget.py
- src/menipy/math/rayleigh_lamb.py (archived and removed from active tree)
- src/menipy/models/typing.py
- src/menipy/models/unit_types.py
- src/menipy/pipelines/pendant/metrics.py
- src/menipy/pipelines/pendant/outputs.py
- src/menipy/pipelines/pendant/physics.py
- src/menipy/pipelines/pendant/scaling.py
- src/menipy/pipelines/pendant/validation.py
- src/menipy/viz/plots.py

### C) Do-not-delete-until-verified list

- Any module loaded through plugin discovery/registry at runtime
- Any module referenced by UI resources or generated code paths
- Any module used only by optional workflows or skipped tests

---

## Cleanup Plan

### Phase 0 - Baseline and guardrails

1. Create a baseline test run and save results.
2. Archive-first policy: before deleting, move candidate files to archive/2026-05-cleanup/ when practical.
3. Work in very small commits by family (one family per commit).

Baseline captured:
- Command: pytest -q tests/test_detection_plugins.py tests/test_preproc_plugins.py tests/test_edge_detectors_plugin.py tests/test_plugin_consolidation.py tests/test_gui_startup_preview.py
- Result: 56 passed in 3.27s

### Phase 1 - Remove low-risk dead code

1. Start with high-confidence legacy/empty files:
	- src/menipy/common/zold_detection.py
	- src/menipy/gui/zold_drawing_alt.py
2. Run targeted tests after each removal.
3. If green, proceed to src/menipy/gui/mainwindow.py only after confirming no runtime or test import requirements.

### Phase 2 - Validate static candidates before deletion

For each file in "Unreferenced module candidates":
1. Search import/usage across src, tests, scripts, plugins.
2. If no usage, move to archive/2026-05-cleanup/ first.
3. Run targeted tests for impacted area.
4. After one full successful cycle, delete archived file in a separate commit.

### Phase 3 - Consolidate duplicate logic

1. Extract shared helpers from AutoCalibrator first (baseline source of truth).
2. Keep external APIs stable; only internal delegation should change.
3. Refactor one pair at a time:
	- detect_drop/preproc_detect_drop
	- detect_needle/preproc_detect_needle
	- circle_edge/sine_edge
4. Run targeted tests, then full test subset.
5. Add parity checks: plugin detector outputs must stay consistent with AutoCalibrator baseline on test fixtures.

### Phase 4 - Full verification and documentation

1. Run full test suite (or project default CI subset).
2. Smoke-test GUI startup and one sessile + one pendant workflow.
3. Update this tracker: mark each candidate as kept, archived, deleted, or refactored.
4. Add notes in CHANGELOG.md for any public behavior change.

---

### Checklist

- [x] Initial static analysis completed
- [x] Initial candidate list documented
- [x] Baseline tests captured before first deletion
- [x] GUI auto-calibration baseline documented for needle/drop
- [x] Low-risk dead code archived/removed
- [x] Unreferenced candidate modules triaged one by one
- [x] First duplicate pair refactored with tests (detect_drop + preproc_detect_drop)
- [x] Second duplicate pair refactored with tests (detect_needle + preproc_detect_needle)
- [x] Full verification complete
- [x] Tracker finalized (kept vs removed decisions)

---

_Last updated: May 21, 2026_
