# CODEX Activity Log

This file summarizes tasks requested of CODEX and a brief description of how CODEX responded.

## Entry 1 - Creating CODEXLOG.md

**Task:** Create a file called CODEXLOG.md that will describe what task was asked to CODEX and a summary of what CODEX did. This file will be updated in every task.

**Summary:** CODEX created CODEXLOG.md and added this initial entry to log future activities.

## Entry 2 - Adding CODEXLOG agent instructions

**Task:** Add a new "CODEXLOG" agent description in AGENTS.md so that CODEX appends to CODEXLOG.md after each task.

**Summary:** Updated AGENTS.md with a dedicated CODEXLOG agent section describing how the log should be maintained. Appended this entry to record the update.

## Entry 3 - Continue working with the PLAN

**Task:** Continue implementing features from PLAN.md, focusing on the GUI skeleton and calibration utilities.

**Summary:** Added basic image loading and segmentation controls to `gui/main_window.py`, implemented calibration utilities under `src/utils`, updated the CLI entry point, and expanded tests accordingly.

## Entry 4 - Fix QAction Import

**Task:** Resolve error due to missing QAction attribute from QtWidgets by importing QAction from QtGui and updating usage.

**Summary:** Updated `gui/main_window.py` to import `QAction` from `PySide6.QtGui` and replaced references to QtWidgets with direct class imports. All tests pass.

## Entry 5 - Display Image at Native Size

**Task:** Adjust the GUI so loaded images are shown at their original dimensions instead of being fit to the view.

**Summary:** Modified `load_image` in `gui/main_window.py` to reset the view transform and set the scene rectangle to the pixmap's bounds. Added a corresponding unit test to verify the image dimensions and transform. All tests pass (GUI tests skipped if PySide6 unavailable).

## Entry 6 - Expand Window to Image Size

**Task:** Ensure that when an image is loaded, the main window grows to match the image dimensions rather than constraining the view.

**Summary:** Updated `load_image` to set the graphics view size to the pixmap's bounding rectangle and call `adjustSize()` so the window expands. Extended the GUI test to confirm the view size matches the image.

## Entry 7 - Overlay Contours and Save Image

**Task:** Continue with the plan by integrating contour overlays after segmentation and adding a feature to save the annotated image.

**Summary:** Implemented `find_contours` in `segmentation.py` and updated `MainWindow` to draw contour overlays using `QPainterPath`. Added a "Save Annotated Image" action and method that render the scene to a file. Extended processing and GUI tests to cover these new capabilities.


## Entry 8 - Calibration Dialog

**Task:** Continue with the plan by adding calibration features to allow setting pixel-to-millimeter scaling from a reference line in the image.

**Summary:** Implemented an interactive `CalibrationDialog` that lets the user draw a line on the image and enter its real-world length. The dialog computes pixels-per-millimeter using `calibrate_from_points` from updated `utils.calibration`. The main window now launches this dialog via the Tools menu. Utility functions for conversions were added along with new tests.

## Entry 9 - Batch Processing

**Task:** Continue with the plan by implementing batch mode to process directories of images and export results.

**Summary:** Added a functional `run_batch` in `src/batch.py` that loads each image, applies Otsu thresholding and morphological cleanup, counts contours, and collects basic stats into a pandas DataFrame. Results can be saved to CSV. Created `tests/test_batch.py` to verify batch processing and CSV output.

## Entry 10 - CI Workflow

**Task:** Continue with the plan by adding continuous integration to run tests and build packages automatically.

**Summary:** Created `.github/workflows/ci.yml` to set up Python 3.10 and 3.11, install dependencies, run pytest with coverage, and build source and wheel distributions.

## Entry 11 - ML Segmentation Toggle

**Task:** Continue with the plan by adding an optional ML-based segmentation toggle in the GUI.

**Summary:** Implemented a `Use ML Segmentation` checkable action in `MainWindow` and a placeholder `ml_segment` function. Processing now routes through this function when the action is enabled, and tests verify the new behavior.

## Entry 12 - Zoom Control

**Task:** Continue with the plan by adding a zoom slider to adjust the image view scale.

**Summary:** Created `ZoomControl` widget in `src/gui/controls.py` and integrated it into `MainWindow`. The slider emits a `zoomChanged` signal connected to a new `set_zoom` method that scales the `QGraphicsView`. Added unit test `test_zoom_control` verifying the zoom factor. All tests pass.
## Entry 13 - Parameter Panel

**Task:** Continue with the plan by adding a panel to edit physical parameters in the GUI.

**Summary:** Implemented `ParameterPanel` widget in `src/gui/controls.py` with fields for air density, liquid density, and surface tension. Integrated the panel into `MainWindow` and exported it from the GUI package. Added `test_parameter_panel_defaults` ensuring default values are accessible. All tests pass.

## Entry 14 - Address Feedback

**Task:** Review the previous work and mark completed stages in `PLAN.md`.

**Summary:** Added "Completed by Codex" markers for the Zoom Control and Parameter Panel steps in `PLAN.md` to reflect the finished features. No code changes were required. All tests still pass.

## Entry 15 - Metrics Panel

**Task:** Continue with the plan by adding a metrics panel to display calculated values.

**Summary:** Created a `MetricsPanel` widget in `src/gui/controls.py` to show IFT, Wo number, volume, contact angle, height, and diameter. Integrated the panel into `MainWindow` and provided a simple metrics calculation in `process_image`. Exported the widget from the GUI package and added `test_metrics_panel_update` to verify updates. All tests pass.

## Entry 16 - Update Documentation

**Task:** Continue with the plan by updating DOCUMENTATION.md after implementing the metrics panel.

**Summary:** Added an Interface Overview section documenting the zoom slider, parameter panel, and metrics panel. Marked the metrics panel step complete in PLAN.md. All tests pass.

## Entry 17 - Propose Enhancements

**Task:** Continue with the plan by outlining future improvements.

**Summary:** Added bullet points under "Propose enhancements" in PLAN.md suggesting real-time metric updates, ML segmentation integration, 3D reconstruction, a CLI for batch mode, and expanded tests.

## Entry 18 - Interactive Calibration Box

**Task:** Implement a calibration mode with draggable box overlay.

**Summary:** Added a Calibration Mode checkbox to the ParameterPanel and new handlers in MainWindow to draw a blue rectangle when dragging on the image. The box coordinates are stored for downstream processing. Added unit test `test_calibration_box` and all tests pass.

## Entry 19 - Mark Calibration Box Complete

**Task:** Update PLAN.md to mark the interactive calibration box step as done.

**Summary:** Inserted "<!-- Completed by Codex -->" comment under the Interactive Calibration Box section in PLAN.md and under Update Documentation. All tests pass.

## Entry 20 - Pixel-to-mm Calibration

**Task:** Implement manual and automatic pixel-to-mm calibration.

**Summary:** Added calibration controls to the parameter panel with manual/automatic modes and reference length input. Updated `MainWindow.open_calibration` to crop the ROI and call a new `auto_calibrate` utility. Implemented `auto_calibrate` in `src/utils/calibration.py` and exposed it via `__init__`. Added tests for automatic calibration and new GUI controls, updated README with a Calibration Workflow section, and marked the plan step complete. All tests pass.

## Entry 21 - Calibration UI Tweaks

**Task:** Fix manual calibration and add calibration/measurement toggles.

**Summary:** Replaced manual/automatic radio buttons with a single toggle and added a "Calibrate" button. Calibration mode now supports drawing either a line (manual) or a box (automatic). Updated `MainWindow` to handle both interactions and compute scale directly from the main view. Adjusted tests and documentation. All tests pass.

## Entry 22 - Improve Automatic Calibration

**Task:** Refine the automatic calibration method so it detects the two vertical needle edges rather than using simple intensity projection.

**Summary:** Reimplemented `auto_calibrate` to apply Canny edge detection and a Hough transform, measuring the separation of near-vertical lines. Updated the unit test to draw a dark needle on a light background and documented the new approach in the README. All tests pass.

## Entry 23 - ROI Selection for Volume

**Task:** Implement region-of-interest drawing and restrict processing to this area. Update PLAN.md to mark the step complete.

**Summary:** Added an ROI Mode checkbox in `ParameterPanel` and new handlers in `MainWindow` to draw a green rectangle. Segmentation and contour overlays now operate on the cropped region, and metrics use the restricted mask. Updated tests with ROI drawing and processing cases, and marked the corresponding plan section complete.


## Entry 24 - Apex and Contact Marking

**Task:** Implement detection of the droplet apex and contact point with overlays.

**Summary:** Added apex and contact point markers in `MainWindow.process_image` using a yellow ellipse and cyan line. Updated `ParameterPanel` integration to clear previous markers, wrote a GUI test verifying marker creation, and marked the plan step complete. All tests pass.

## Entry 25 - External Contour Only

**Task:** Continue with the plan by ensuring only the outer droplet contour is used for processing.

**Summary:** Added `external_contour_mask` in `segmentation.py` to keep only the largest external contour and fill interior holes. Integrated this step into `batch.py` and `MainWindow.process_image`. Updated processing tests with a new case for the function and marked the plan item complete.

## Entry 26 - Correct Volume and Apex Calculations

**Task:** Revise the droplet volume function and fix the apex detection logic in the GUI.

**Summary:** Rewrote `droplet_volume` to compute the solid-of-revolution integral from a binary mask, returning volume in cubic millimetres. Updated `MainWindow.process_image` to call the new API and corrected the apex selection to use the lowest point of the mask. Adjusted unit tests accordingly.

## Entry 27 - Calculate & Draw Buttons

**Task:** Continue with the plan by adding buttons to compute physical parameters and overlay a model contour.

**Summary:** Introduced `Calculate` and `Draw Model` buttons in the control panel. These call new helper methods in `MainWindow` that estimate surface tension and contact angle then plot a fitted circle. Added simple estimation functions in `models.properties`, exported them, and expanded unit tests. Removed the redundant Calibration menu action. All tests pass.

## Entry 28 - CSV Export Feature

**Task:** Review PLAN.md, mark completed tasks, and start the next task by implementing CSV export.

**Summary:** Marked the "Calculate & Draw" buttons, "Save Annotated Image," and "Propose enhancements" items as completed in PLAN.md. Added a "Save CSV" button to `MainWindow` with a new `save_csv` method that exports parameters and metrics using pandas. Extended `MetricsPanel` with a `values` method, updated GUI tests, and added a new test for CSV export. All tests pass.

## Entry 29 - .gitignore newline fix

**Task:** Add a newline at the end of `.gitignore` to properly terminate the final entry `\.cursorindexingignore`.

**Summary:** Appended a newline to `.gitignore` so the last entry `.cursorindexingignore` is correctly terminated. No other lines were modified.

## Entry 30 - Pin Dependency Versions

**Task:** Edit requirements.txt to pin exact package versions and update setup.py to match.

**Summary:** Pinned all dependencies in `requirements.txt` (numpy 1.24.0, scipy 1.10.0, etc.) and updated `setup.py` to require those same versions. Ensured the requirements file ends with a newline and verified all tests still pass.

## Entry 31 - Remove matplotlib dependency

**Task:** Remove matplotlib from requirements and setup since it is unused.

**Summary:** Deleted matplotlib from `requirements.txt` and the `install_requires` list in `setup.py`. All tests pass.

## Entry 32 - Fix file dialog signal arguments

**Task:** Resolve a crash when using `Save CSV` due to PySide6 passing a boolean to slots expecting a path.

**Summary:** Updated `MainWindow` connections for menu actions and the CSV button to use lambdas, preventing unintended boolean arguments from `triggered`/`clicked` signals. Added lambdas for `Open Image` and `Save Annotated Image` actions as well. All tests pass.

## Entry 33 - Refactor droplet property functions

**Task:** Improve droplet volume and surface tension calculations, addressing review comments.

**Summary:** Updated `properties.py` to accept a pixel-to-millimetre factor, validate inputs, and use consistent units. GUI and tests now supply the calibration factor. Functions return `None` when measurements fail. All tests pass.

## Entry 34 - Improve contact line detection

**Task:** Address review feedback on silhouette-only analysis and fix contact line calculation.

**Summary:** Updated `MainWindow.process_image` so the contact line is derived from the outer contour by intersecting it with the ROI boundary, falling back to the mask row if needed. This ensures geometric metrics rely solely on the droplet silhouette. All tests pass.
## Entry 35 - Implement droplet detection

**Task:** Add ROI-based droplet detection returning geometric metrics.

**Summary:** Created `detect_droplet` with a `Droplet` dataclass to extract the outer contour, apex and contact point in image coordinates. The function thresholds using Otsu, closes small holes, chooses the largest contour by area, and computes height, radius and area from the mask. New unit tests validate synthetic examples and error handling. All tests pass.



## Entry 36 - Integrate droplet detection with GUI

**Task:** Use the new `detect_droplet` function in the GUI pipeline to compute metrics and overlays.**

**Summary:** Updated `detect_droplet` for robust polarity handling and apex calculation. `MainWindow.process_image` now calls this function to obtain the mask, contour, apex, and contact line which are drawn on the scene. Metrics such as height, diameter and volume are derived from the returned data. All tests pass.


## Entry 37 - Drop mode classification

**Task:** Implement classify_drop_mode function with confidence heuristics.

**Summary:** Added `classify_drop_mode` in `src/processing` to determine whether a droplet is pendant or sessile using the contact line normal and apex position. Exported the function through the processing package and created unit tests covering pendant, sessile, and unknown cases. All tests pass.

## Entry 38 - Display drop mode in GUI

**Task:** Integrate `classify_drop_mode` results into the user interface.

**Summary:** Updated the droplet `Droplet` dataclass to store the full contact
line segment. The GUI now calls `classify_drop_mode` after detection and shows
the resulting mode in the metrics panel. Added a label to display this value and
included it when exporting CSV data. Adjusted unit tests for the updated data
structure and GUI behaviour. All tests pass.

## Entry 39 - Sessile Droplet Detection

**Task:** Implement a robust geometric detector for sessile drops and expose it through the processing package.

**Summary:** Added `SessileDroplet` dataclass and `detect_sessile_droplet` function implementing substrate line fitting via Hough transform and RANSAC. Updated `__init__` exports. No existing functionality was modified. All tests pass.

## Entry 40 - Pendant Drop Detector and Mode Selector

**Task:** Implement robust pendant drop detection and add a GUI selector for detection mode.

**Summary:** Added a `PendantDroplet` dataclass with a corresponding `detect_pendant_droplet` function that fits the needle line using RANSAC and extracts geometric metrics from the silhouette. The parameter panel now includes a "Detection Mode" combo box allowing users to choose between sessile and pendant processing. Updated the main window logic to call the selected detector and display the resulting mode. A unit test validates the pendant detector on synthetic data. All tests pass.
\n## Entry 41 - Pendant detector uses base silhouette
**Task:** Combine original silhouette detection with pendant drop metrics.
**Summary:** `detect_pendant_droplet` now calls `detect_droplet` to reuse its contour and area calculations when possible, falling back to the local mask if it fails. Unit tests still pass.

## Entry 42 - Initialize analysis package

**Task:** Create src/analysis directory.

**Summary:** Added new empty package `analysis` with __init__.py to start implementing drop analysis modules as per NEW_PLAN.

## Entry 43 - GUI tab refactor

**Task:** Implement UI refactor using `QTabWidget`.

**Summary:** Main window now contains a tab widget with "Classic" and "Drop Analysis" tabs. Added `DropAnalysisPanel` widget housing workflow controls and result displays. Updated exports and tests to check tab creation. All tests pass.

## Entry 44 - Needle detection module

**Task:** Develop `src/analysis/needle.py` with automated needle edge detection and add unit tests.

**Summary:** Implemented `detect_vertical_edges` to locate the needle axis using Canny and Hough transforms. Exported it through `analysis.__init__` and created `tests/test_analysis.py` covering successful detection and failure cases. All tests pass.

## Entry 45 - Drop contour metrics

**Task:** Implement `src/analysis/drop.py` providing droplet contour extraction and metric calculations with tests.

**Summary:** Added `extract_external_contour` and `compute_drop_metrics` to get the outer droplet boundary and basic geometry. Exported these APIs and expanded `tests/test_analysis.py` with new cases for contour extraction and metric computation. All tests pass.

## Entry 46 - GUI drop analysis integration

**Task:** Integrate drop analysis overlays into the GUI controller.

**Summary:** Added `gui/overlay.py` with `draw_drop_overlay` helper. Updated `MainWindow` to handle needle and drop ROI drawing, call analysis functions, and display overlays and metrics in `DropAnalysisPanel`. Connected workflow buttons and added tests for the overlay function and drop analysis workflow. All tests pass.

## Entry 47 - Persist drop analysis regions

**Task:** Ensure the needle and drop regions drawn in the Drop Analysis tab remain visible and are stored for later use.

**Summary:** Updated `MainWindow` so disabling drawing mode no longer removes ROI rectangles. Added coordinate labels in `DropAnalysisPanel` with `set_regions`/`regions` methods. Rectangles are cleared when a new image loads. Introduced a new test verifying region persistence and reset. All tests pass.

## Entry 48 - Drop overlay fixes and needle width

**Task:** Preserve drop ROI after analysis and define needle length as horizontal distance between edges.

**Summary:** Updated `detect_vertical_edges` to compute width between vertical contours and modified tests accordingly. Drop and needle ROI rectangles now retain visibility by assigning a higher Z value. Added GUI test check for drop ROI persistence. All tests pass.

## Entry 49 - Update docs for drop analysis

**Task:** Update README and documentation.

**Summary:** Added `doc/drop_analysis.md` detailing the workflow and algorithms for the new Drop Analysis tab. README now references the new document and describes the tab features.

## Entry 50 - Drop analysis missing features

**Task:** Implement new metrics and overlays for drop analysis.

**Summary:** Added `_max_horizontal_diameter` in `analysis/drop.py` to locate the
widest slice and contact line. `compute_drop_metrics` now returns the maximum
horizontal diameter coordinates, apex radius, and optional contact line.
`draw_drop_overlay` draws this line and the contact line. GUI panels show the
"Radius–Apex" metric and render the new overlays. Documentation updated and unit
tests added for these features.

## Entry 51 - Auto-fitting ImageView

**Task:** Implement a graphics-based image viewer that keeps the pixmap scaled to the viewport and adapts during zoom and resize.

**Summary:** Added `ImageView` using `QGraphicsView`/`QGraphicsScene` to store the original pixmap and fit it to the widget. The view tracks a scale factor combining fit and zoom, with helper methods for coordinate conversion. Integrated it into `MainWindow` and updated tests. All tests pass.

## Entry 52 - Surface tension computations

**Task:** Add surface tension calculation methods and integrate them into drop analysis.

**Summary:** Introduced `surface_tension.py` with Jennings–Pallas form factor, Young–Laplace surface tension, Bond number and contour based volume. `compute_drop_metrics` now outputs `gamma_mN_m`, `beta`, `Bo` and uses the new volume estimator. A minimal Sphinx config generates `docs/pendant_drop_methods.md` from these docstrings. Added unit tests for the new functions.

## Entry 53 - Printing drop metrics for testing

**Task:** Print each derived drop metric to the terminal for testing.

**Summary:** Added print statements in `compute_drop_metrics` so height, diameter, apex, radius, beta, surface tension, Bond number, volume, contact angle and IFT values are displayed whenever metrics are computed. Existing tests continue to pass.

## Entry 54 - Surface tension metrics in GUI

**Task:** Display surface tension from `surface_tension()` and show beta, Bond number and s1 on the Drop Analysis tab.

**Summary:** Updated `compute_drop_metrics` to return `s1` along with beta and Bond number. `DropAnalysisPanel` now displays surface tension from `surface_tension()`, plus beta, s1 and Bond number. Tests updated for s1 and pass.

## Entry 55 - GUI tab reorganization

**Task:** Split Drop Analysis controls across Calibration, Pendant drop and Contact angle tabs with fluid property inputs and per-tab analysis mode.

**Summary:** Introduced `CalibrationTab` for calibration settings and `AnalysisTab` for displaying pendant or contact-angle results. `MainWindow` now builds these tabs, selects the method when an Analyze button is pressed and reuses existing drop analysis logic. Tests expect four tabs and read metrics from the pendant tab. All tests pass.


## Entry 56 - Fix QFrame import

**Task:** Resolve NameError when creating AnalysisTab due to missing QFrame import.

**Summary:** Added QFrame to the PySide6.QtWidgets import list in `src/gui/controls.py`. All tests pass.


## Entry 57 - Fix metrics argument mismatch

**Task:** Resolve TypeError from `AnalysisTab.set_metrics()` when running drop analysis.

**Summary:** Removed the unsupported `beta` keyword from the `set_metrics` call in `gui/main_window.py`. Tests confirm the GUI now updates metrics without errors.

## Entry 58 - GUI defaults

**Task:** Start main window maximized and set default widget sizes.

**Summary:** Modified `src/main.py` to call `showMaximized()` and updated `MainWindow` setup to enforce a 200px minimum image view and 250px tab width. Added splitter size initialization and a test covering these defaults.
## Entry 59 - Detection test interface

**Task:** Move zoom control above image area, rename Classic tab, add detection and filter controls with associated buttons.

**Summary:** Updated `MainWindow` layout to place `ZoomControl` atop the image view. The first tab is now "Detection Test" and includes a segmentation algorithm selector, Detect/Clean buttons, and filter selection with strength slider. Implemented `apply_filter`, `clean_filter`, and `clean_detection` helpers and simplified detection to show segmentation masks. Adjusted tests for the new tab name and ROI processing. All tests pass.


## Entry 60 - Fix uninitialized widgets

**Task:** Resolve AttributeError due to referencing buttons before creation in `MainWindow` setup.

**Summary:** Moved the initialization of `detect_button`, `clean_button`, and filter controls above their layout insertion to ensure attributes exist before use. All tests pass.

## Entry 61 - Stop Clean button resizing window

**Task:** Prevent the Clean Detection action from altering the main window dimensions.

**Summary:** Removed the `adjustSize()` call from `MainWindow.clean_detection` so pressing "Clean" no longer resizes the GUI. All tests continue to pass.

## Entry 62 - Contact angle docs

**Task:** Document contact angle analysis functions and update log.

**Summary:** Added `docs/contact_angle_methods.md` with `autofunction` entries for the geometric, spline and ADSA utilities from `src/contact_angle.py`. Inserted a module docstring and logged this update.

## Entry 63 - Substrate line and contact geometry

**Task:** Implement drawing of a substrate line for contact angle analysis and compute related metrics.

**Summary:** Added `SubstrateLineItem` for interactive line drawing with a new "Draw Substrate" action. Created `physics.contact_geom` with geometry helpers and integrated them into `MainWindow.analyze_drop_image`. Analysis tabs now show base width, radius and apex height. Tests cover the geometry calculations and GUI update.


## Entry 64 - Initialize tool visibility after adding tabs

**Task:** Prevent AttributeError when checking active tab during setup.

**Summary:** Moved the `currentChanged` signal connection in `MainWindow._setup_ui` until after the `contact_tab` is created. This avoids early invocation of `_update_tool_visibility` before the attribute exists, fixing the startup crash.

## Entry 65 - Fix substrate action init

**Task:** Prevent AttributeError for missing draw_substrate_action.

**Summary:** Moved the initial `_update_tool_visibility()` call to after `draw_substrate_action` is created so the action exists before visibility is set.
## Entry 66 - Initialize default event handlers

**Task:** Resolve AttributeError when calling set_substrate_mode during window initialization.

**Summary:** Saved the graphics view's default mouse event handlers before calling `_update_tool_visibility` in `_setup_ui`. This ensures `set_substrate_mode(False)` has access to the handlers. All tests pass.

## Entry 67 - Contact tab substrate button

**Task:** Add a "Draw Substrate Line" button to the contact angle tab and require a substrate line before analysis.

**Summary:** Introduced `substrate_button` in `AnalysisTab` and connected it to a new handler in `MainWindow` that enables substrate line drawing. `_run_analysis` now shows a warning if no line is defined when running contact-angle analysis. Added corresponding GUI tests. All tests pass.

## Entry 68 - Fix substrate line signal

**Task:** Remove Qt signal from SubstrateLineItem to avoid runtime errors on PySide6 versions without QObject support.

**Summary:** Replaced the Qt `Signal` with a simple callback-based `CallbackSignal` class in `items.py`. Updated `SubstrateLineItem` to instantiate this custom signal and handle move notifications. All tests pass.
