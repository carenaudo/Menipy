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


## Entry 69 - Sessile-drop analysis

**Task:** Implement robust sessile-drop analysis with new GUI tab.

**Summary:** Added `image_proc.sessile` with `analyze_sessile` to compute contact line, droplet mask, apex and symmetry axis. Created a CA improved tab in the GUI and wrote tests for contact line points, mask cleanup, apex alignment and metric baselines. All tests pass.

## Entry 70 - Remove example image

**Task:** Delete large binary test image and corresponding tests that blocked PR creation.

**Summary:** Removed `doc/Images/example sessile 1.png` and tests relying on it. The sessile-drop analysis module and GUI remain. All other tests pass.

## Entry 71 - Activate CA improved tab buttons

**Task:** Fix the CAImproved tab where the buttons performed no actions.

**Summary:** Connected the tab's buttons to substrate drawing and new detection/analysis handlers. Implemented `_ca_detect` and `_ca_analyze` in `MainWindow` using `analyze_sessile`, and updated `_update_tool_visibility` to show the substrate tool on this tab. Added result display in the tab's label. All tests pass.
\n## Entry 72 - Improved sessile algorithm\n\n**Task:** Enhance CA improved segmentation and metrics following new workflow.\n\n**Summary:** Reworked `analyze_sessile` to apply bilateral filtering, exclusive threshold masks, and droplet-side filtering. The function now extracts the largest contour, computes apex and symmetry axis, and returns overlay and metrics. All tests pass.

## Entry 73 - Remove CA improved tab

**Task:** Remove the CAImproved tab and related code.

**Summary:** Deleted the CAImprovedTab widget, the sessile image processing module and corresponding handlers in `MainWindow`. Updated GUI exports and tests to account for four tabs.

## Entry 74 - Contact angle alt geometry

**Task:** Add improved contact-angle tab using substrate-based geometry helpers.

**Summary:** Implemented `geom_metrics_alt` and associated utilities in `geometry_alt.py` for polyline intersections and side checks. Updated `MainWindow` to use these metrics when the new tab is active and enhanced the overlay to draw contact segments. Added documentation and tests for the new helpers.

## Entry 75 - Fix ContactAngleTabAlt import

**Task:** Resolve import error for ContactAngleTabAlt in the GUI.

**Summary:** Removed `ContactAngleTabAlt` from the controls import list in `gui/main_window.py` and kept the direct import from `contact_angle_tab_alt`. All tests pass.


## Entry 76 - Robust substrate geometry

**Task:** Refactor the contact-angle alternative implementation so helper lines align with the drawn substrate regardless of tilt.

**Summary:** Added `mirror_filter`, `find_substrate_intersections`, `apex_point` and `split_contour_by_line` in `geometry_alt.py` and rewrote `geom_metrics_alt` to use them. Updated `MainWindow` and docs to reflect the new workflow and added `tests/test_geometry.py` for the helpers. All tests pass.
## Entry 77 - Drop side selection\n\n**Task:** Allow choosing droplet side in alternative contact-angle workflow and auto-detect when not specified.\n\n**Summary:** Added a "Select Drop Side" button in `ContactAngleTabAlt` and new side-selection mode in `MainWindow`. Side choice is stored in `_keep_above` and passed to `geom_metrics_alt`, which now computes area on both sides when `keep_above` is `None`. Updated `geometry_alt.py` accordingly and added tests for automatic side detection. All tests pass.

## Entry 78 - Automatic substrate detection button

**Task:** Add a "Detect Substrate Line" button to the Contact Angle tab and wire it to the Hough-based detection routine.

**Summary:** Inserted a new button in `MainWindow` that runs `detect_substrate_line` on the drop ROI and draws the resulting line. Added `_detect_substrate_line` helper and a GUI test verifying the line orientation. Updated imports and signal connections. All tests pass.

## Entry 79 - Contact angle alt overlay fix

**Task:** Keep substrate line appearance when analyzing and ensure alt tab uses latest detection logic.

**Summary:** Stopped drawing the contact line overlay in `analyze_drop_image` so the dashed substrate item remains light blue after analysis. All tests pass.

## Entry 80 - Preserve substrate line and document alt tab

**Task:** Confirm the alternative contact-angle tab uses the apex projection fix and keeps the dashed substrate line after analysis.

**Summary:** Added a GUI test checking that `_run_analysis("contact-angle-alt")` leaves the same `SubstrateLineItem` with a dashed pen. Updated `README.md` to mention the *Contact Angle (Alt)* tab and clarified in `doc/contact_angle_alt.md` that `geom_metrics_alt` provides the projected symmetry axis. All tests continue to pass.
## Entry 81 - Robust contact region

**Task:** Implement robust contact-region extraction and mode-specific metrics.

**Summary:** Added `region.close_droplet` with half-plane filtering, intersection clustering and mask closing. Implemented `metrics_sessile` and `metrics_pendant` to compute diameter, apex, height and volume via the new helper. Created `tests/test_contact_region.py` with regression cases. All tests pass.

## Entry 82 - Contact tab side selection

**Task:** Let users choose the droplet side in the regular Contact Angle workflow.

**Summary:** Added a "Select Drop Side" button to `contact_tab` in `MainWindow` and wired it to `_select_side_button_clicked`. Updated the tab widget test to expect the extra tab and created a GUI test for the new button. All tests pass.


## Entry 83 - Remove Contact Angle Alt Tab

**Task:** Remove the temporary Contact Angle Alt tab now that the main contact angle tab works.

**Summary:** Deleted `contact_angle_tab_alt.py` and references to the alternative tab in `MainWindow`, exports and GUI tests. Updated documentation and README accordingly. All tests pass.

## Entry 84 - Highlight contact points

**Task:** Add overlay markers for the substrate contact points P1 and P2.

**Summary:** Extended `draw_drop_overlay` with a ``contact_pts`` parameter to draw yellow dots at the intersection points. Updated `MainWindow.analyze_drop_image` to pass the contact points from ``compute_drop_metrics`` and adjusted the GUI test. All tests pass.

## Entry 85 - Start CODEX refactor plan

**Task:** Begin phase 0 of CODEX_REFACTOR_PLAN by creating a discovery tool.

**Summary:** Added `scripts/generate_legacy_map.py` to analyze import
relationships in the legacy `src` directory and produce
`docs/legacy_map.html`. Executed the script to generate the file and
verified that existing tests still pass.

## Entry 86 - Scaffold refactor package

**Task:** Continue with phase 1 of CODEX_REFACTOR_PLAN by creating the src_alt skeleton.

**Summary:** Added the `src_alt/optical_goniometry` package with stub modules and a minimal CLI. Created an import test under `src_alt/tests` and ensured all tests pass.

## Entry 87 - Rename new package and begin utilities port

**Task:** Rename the refactored package from `optical_goniometry` to `menipy` and start phase 2 of the refactor plan.

**Summary:** All stubs were moved under `src_alt/menipy` and import tests updated. Implemented initial utilities by porting calibration routines, image loader, and a preprocessing function. Added parity tests to verify these functions match the legacy implementations. All tests pass.

## Entry 88 - Port detection modules

**Task:** Continue phase 3 by porting calibration and detection code to the new package.

**Summary:** Implemented needle, substrate and droplet detection algorithms in `menipy.detection`. Added parity tests comparing against the legacy modules to ensure identical results. All tests pass.


## Entry 89 - Port analysis modules

**Task:** Continue phase 4 of CODEX_REFACTOR_PLAN by porting drop analysis algorithms.

**Summary:** Added implementations of `compute_drop_metrics` and related helpers in `menipy.analysis.commons`. Exposed wrapper functions for pendant and sessile modes and created parity tests verifying results match the legacy `src` modules. Updated detection tests for non-deterministic contact line ordering. All tests pass.

## Entry 90 - UI facade with refactored modules

**Task:** Continue the refactor by wiring the new `menipy` analysis and detection code into a UI facade.

**Summary:** Created a subclass of the legacy `MainWindow` that calls refactored detection and analysis functions. Updated package exports and added a basic GUI instantiation test. Adjusted import tests and stabilized pendant detection parity with a fixed RNG. All tests pass.
## Entry 91 - Plugin hook implementation

**Task:** Implement plugin discovery mechanism as per refactor plan.

**Summary:** Added `menipy.plugins.load_plugins` which queries the `og.analysis` entry point group using `importlib.metadata` and populates a global `PLUGINS` registry. Implemented unit test with patched entry points to ensure plugins are loaded correctly.

## Entry 92 - Example plugin and manager

**Task:** Implement a sample sharpening plugin and GUI menu to enable plugins.

**Summary:** Added `sharpen_filter` in `menipy.sharpen_plugin` and registered it as a built-in plugin. Extended the `MainWindow` facade with a plugin manager dialog listing available plugins. Active plugins are applied after loading images. Updated plugin tests for the built-in filter.

\n## Entry 93 - Remove legacy src

**Task:** Eliminate old src package in favor of the fully featured menipy implementation.\n\n**Summary:** Deleted the legacy `src` tree and renamed `src_alt` to `src`. Reinstated necessary modules (GUI, processing, detection) under `menipy` and updated the test suite to import from the new package. Added `tests/conftest.py` to ensure the package is importable. All tests pass.
## Entry 94 - Verify no legacy code\n\n**Task:** Remove all legacy code leaving only the alt implementation and rename the folder to src.\n\n**Summary:** Reviewed the repository and confirmed the previous migration already deleted the legacy src tree and renamed `src_alt` to `src`. All modules now reside under `src/menipy`, with alt geometry utilities in `detectors/geometry_alt.py`. No additional legacy code was present. Test suite executed successfully.

## Entry 95 - GUI entry point

**Task:** Ensure running the package launches the new GUI implementation.

**Summary:** Updated `src/__main__.py` to invoke `menipy.gui.main` and changed the console script in `setup.py` so the `menipy` command opens the graphical interface by default.

## Entry 96 - Remove unused files

**Task:** Clean up remaining unused modules.

**Summary:** Deleted obsolete `io` and `preprocessing` packages, the unused `properties2.py` module, and placeholder UI subpackages. Tests continue to pass, confirming these modules were not required.

## Entry 97 - Relative import for src

**Task:** Fix ModuleNotFoundError when executing `python -m src`.

**Summary:** Modified `src/__main__.py` to use a relative import of `menipy.gui` so running the package with `python -m src` resolves the internal package correctly. Tests still pass.

## Entry 98 - Verify imports

**Task:** Check for remaining import errors after exposing `main()` entry point.

**Summary:** Ran the test suite (53 passed) and executed `python -m src`. The command failed due to missing system library `libEGL.so.1`, indicating a runtime environment issue rather than a Python import error. No additional import errors were found.

## Entry 99 - Refactored UI default

**Task:** Replace legacy GUI exports with the refactored version.

**Summary:** Renamed `gui/main_window.py` to `_legacy_main_window.py` and updated `ui.main_window` to import from the new private module. `menipy.gui.__init__` now exposes `ui.MainWindow`, ensuring the plugin menu is always available. All tests pass.

## Entry 100 - Modularize main window

**Task:** Remove legacy implementation and import the refactored window.

**Summary:** Replaced `_legacy_main_window.py` with `base_window.py` containing a
`BaseMainWindow` class. The refactored UI now subclasses this base class and all
imports use the new module. The old file was deleted.

## Entry 101 - Fix package imports

**Task:** Resolve `ModuleNotFoundError` when launching the GUI with
`python -m src`.

**Summary:** Changed absolute imports in `ui/main_window.py` to package
relative imports so the module works when the project is executed as a
module without installation. Tests pass and running `python -m src`
fails only due to missing GUI libraries, confirming the import issue is
fixed.


## Entry 102 - Add analysis reset button

**Task:** Provide a clear way to return the viewer to the unmodified image.

**Summary:** Added a `Clear Analysis` button below the image view that calls a new `clear_analysis` method. This routine restores the original image, removes detection and analysis overlays, and resets related state. Tests were updated with a new case ensuring the button works. All tests pass.

## Entry 103 - Fix drawing after clear

**Task:** Resolve crash when drawing needle ROI after using Clear Analysis.

**Summary:** Updated `_display_image` to reset ROI and overlay references so cleared items do not leave dangling Qt objects. Tests pass.

## Entry 104 - Reset all overlays on clear

**Task:** Ensure Clear Analysis fully resets drawing modes and calibration items.

**Summary:** Updated `clear_analysis` to disable drawing modes, remove calibration items, and reset internal state. Added a regression test verifying that all overlays are gone and drawing works after clearing.

## Entry 105 - Reset metrics on clear

**Task:** Ensure analysis metrics and result overlays fully reset when clearing.

**Summary:** Added `clear_metrics` helpers to GUI panels and updated `clear_analysis` to reset scale and metrics displays. New tests verify metric labels return to defaults after clearing. All tests pass.
## Entry 106 - Separate pendant overlays and remove stray axis line

**Task:** Adjust overlay drawing so pendant and sessile analyses have distinct visuals and remove unintended red axis line on pendant drops.**

**Summary:** Updated `BaseMainWindow` and `ui.main_window` to only compute and draw the axis line for contact-angle analyses. Pendant overlays now receive center and contact lines only in pendant mode. This eliminates the diagonal red line and keeps sessile visuals unchanged. All tests pass (53 passed).
## Entry 107 - Refactor overlay pipelines

**Task:** Fully separate pendant and sessile analysis drawing and geometry logic.

**Summary:** Introduced new `pipelines` package with dedicated geometry and drawing modules for pendant and sessile workflows. Updated GUI to call these pipelines when analyzing images, preventing cross-mode overlay contamination. All 53 tests still pass.

## Entry 108 - Fix analyze_drop_image indentation

**Task:** Resolve IndentationError when executing `python -m src`.

**Summary:** Added missing `if mode == "pendant":` line in `ui/main_window.py` to correctly open the analysis branch. Tests run successfully (53 passed).
## Entry 109 - Fix overlay import path

**Task:** Resolve `ModuleNotFoundError` for `src.menipy.pipelines.overlay` when executing `python -m src`.

**Summary:** Updated pendant and sessile drawing pipelines to import `draw_drop_overlay` from `menipy.gui` rather than a non-existent local module. All tests pass (53 passed).
## Entry 110 - Correct relative import depth

**Task:** Address `ModuleNotFoundError` when running the GUI due to incorrect relative imports in drawing pipelines.

**Summary:** Updated `pendant/drawing.py` and `sessile/drawing.py` to import `draw_drop_overlay` from `menipy.gui` using `...gui` so the package resolves correctly. All tests pass (53 passed).

## Entry 111 - Break GUI import cycle

**Task:** Resolve circular import error when running `python -m src`.

**Summary:** Updated pendant and sessile drawing pipelines to import
`draw_drop_overlay` from `menipy.gui.overlay` rather than the package root.
This prevents `gui/__init__` from re-importing the pipelines during
initialization. All tests still pass (53 passed).
## Entry 112 - Fix substrate line offset

**Task:** Resolve ValueError when analyzing sessile drops due to mismatched substrate line coordinates.

**Summary:** Adjusted `analyze_drop_image` in `ui/main_window.py` and `gui/base_window.py` to subtract the ROI origin from the substrate line before calling `analyze_sessile`. This ensures the line intersects the local contour. All tests pass (53 passed).

## Entry 113 - Pendant drop surface metrics

**Task:** Fix droplet surface area calculation and add projected area and needle area metrics.

**Summary:** Updated `compute_drop_metrics` to derive projected area from the filled mask, compute the needle contact area and subtract it from the surface of revolution. The true surface area function now handles duplicate z values. Added `needle_area_mm2` to the returned metrics and adjusted tests to check for the new fields. All tests pass (53 passed).

## Entry 114 - Symmetry side area metrics

**Task:** Report surface and projected areas for each side of the drop and expose missing distances in the GUI.

**Summary:** `compute_drop_metrics` now splits the droplet mask about the apex to calculate left and right projected areas and surfaces of revolution. It returns these along with their mean. The `AnalysisTab` UI gained labels for the new metrics and now displays the apex-to-diameter and needle-to-diameter distances. `MainWindow` passes the additional values when updating the panel. Tests updated to verify the new keys. All tests pass (53 passed).

## Entry 115 - Debug surface area steps

**Task:** Add print statements to display intermediate values when computing left and right surface areas.

**Summary:** Updated `compute_drop_metrics` in `analysis/commons.py` to print the profile points and resulting surface areas for each side. All tests pass (53 passed).
