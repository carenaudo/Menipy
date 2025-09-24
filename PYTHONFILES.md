# Python Files Overview

This document summarizes each Python file in the repository so contributors can quickly locate functionality.

## Root-level Files
- `setup.py`: Packaging configuration for the Menipy package.
- `src/__init__.py`: Package initializer for top-level source directory.
- `src/__main__.py`: CLI entry point for launching the GUI.

## scripts
- `scripts/generate_legacy_map.py`: Utility for scanning imports and generating a legacy module map.

## docs
- `docs/conf.py`: Sphinx configuration for building project documentation.

## src/menipy Package
- `src/menipy/__init__.py`: Initializes the Menipy library.
- `src/menipy/batch.py`: Batch image processing orchestrator with `_iter_image_paths` and `run_batch`.
- `src/menipy/cli.py`: Command-line interface exposing the `main` entry.
- `src/menipy/contact_angle.py`: Contact-angle analysis utilities such as `height_width_ca`, `circle_fit_ca`, and `adsa_ca` along with derived property helpers.
- `src/menipy/gui.py`: Launches the Menipy GUI via `main`.
- `src/menipy/plugins.py`: Lightweight plugin loader through `load_plugins`.
- `src/menipy/sharpen_plugin.py`: Implements a simple `sharpen_filter` plugin.
- `src/menipy/utils.py`: Miscellaneous helpers including `log`.

### Calibration
- `src/menipy/calibration/__init__.py`: Calibration utilities namespace.
- `src/menipy/calibration/calibrator.py`: `Calibration` class and helpers for pixel/mm conversions and auto-calibration.

### Detection
- `src/menipy/detection/__init__.py`: Stub package for detection utilities.
- `src/menipy/detection/droplet.py`: Droplet detection algorithms with `Droplet`, `SessileDroplet`, `PendantDroplet`, and helpers like `detect_droplet`.
- `src/menipy/detection/needle.py`: Needle detection via `detect_vertical_edges`.
- `src/menipy/detection/substrate.py`: Substrate line detection with `clip_line_to_roi` and `detect_substrate_line`.

### Detectors (Alt Geometry)
- `src/menipy/detectors/__init__.py`: Package placeholder.
- `src/menipy/detectors/geometry_alt.py`: Alternate geometry helpers such as `trim_poly_between`, `symmetry_axis`, and `geom_metrics_alt`.

### Physics
- `src/menipy/physics/__init__.py`: Physics utilities namespace.
- `src/menipy/physics/contact_geom.py`: Geometry helpers for contact analysis (`line_params`, `contour_line_intersections`, etc.).

### Metrics
- `src/menipy/metrics/__init__.py`: Metrics package stub.
- `src/menipy/metrics/metrics.py`: Statistical utilities with `compute_statistics`.

### Models
- `src/menipy/models/__init__.py`: Model utilities package.
- `src/menipy/models/drop_extras.py`: Extra droplet metrics like `worthington_number` and `surface_area_mm2`.
- `src/menipy/models/geometry.py`: Geometric fitting (`fit_circle`, `horizontal_intersections`).
- `src/menipy/models/physics.py`: Young–Laplace solver (`young_laplace_ode`, `solve_young_laplace`).
- `src/menipy/models/properties.py`: Property calculations such as `droplet_volume` and `estimate_surface_tension`.
- `src/menipy/models/surface_tension.py`: Pendant-drop surface tension calculations (`jennings_pallas_beta`, `surface_tension`).

### Analysis
- `src/menipy/analysis/__init__.py`: Drop analysis package.
- `src/menipy/analysis/commons.py`: Shared contour helpers (`extract_external_contour`, `compute_drop_metrics`).
- `src/menipy/analysis/drop.py`: Entry points for droplet metric calculations.
- `src/menipy/analysis/pendant.py`: Pendant drop analysis via `compute_metrics`.
- `src/menipy/analysis/plotting.py`: Contour plotting utilities (`save_contour_sides_image`, etc.).
- `src/menipy/analysis/sessile.py`: Sessile drop analysis (`compute_metrics`, `contact_points_from_spline`, smoothing helpers).
- `src/menipy/analysis/sessile_alt.py`: Alternative sessile analysis with similar helpers.

### Pipelines
- `src/menipy/pipelines/__init__.py`: High-level analysis pipelines.
- `src/menipy/pipelines/pendant/__init__.py`: Pendant pipeline namespace.
- `src/menipy/pipelines/pendant/drawing.py`: Drawing helpers for pendant analysis (`draw_overlays`).
- `src/menipy/pipelines/pendant/geometry.py`: Pendant analysis pipeline (`analyze`, `HelperBundle`, `PendantMetrics`).
- `src/menipy/pipelines/sessile/__init__.py`: Sessile pipeline namespace.
- `src/menipy/pipelines/sessile/drawing.py`: Drawing overlays for sessile analysis.
- `src/menipy/pipelines/sessile/drawing_alt.py`: Alternate sessile drawing helpers.
- `src/menipy/pipelines/sessile/geometry.py`: Sessile analysis pipeline (`analyze`, `HelperBundle`, `SessileMetrics`).
- `src/menipy/pipelines/sessile/geometry_alt.py`: Alternative sessile pipeline with contour filtering, contact detection, and `analyze`.

### Processing
- `src/menipy/processing/__init__.py`: Processing package placeholder.
- `src/menipy/processing/classification.py`: Detects drop mode via `classify_drop_mode`.
- `src/menipy/processing/detection.py`: Image-based droplet detection (`Droplet` classes and `detect_droplet`).
- `src/menipy/processing/geometry.py`: ROI geometry helper `clip_line_to_roi`.
- `src/menipy/processing/metrics.py`: Droplet metric helpers (`metrics_sessile`, `metrics_pendant`).
- `src/menipy/processing/reader.py`: Image loading through `load_image`.
- `src/menipy/processing/region.py`: Region utilities (`close_droplet`).
- `src/menipy/processing/segmentation.py`: Thresholding, contour extraction, and `ml_segment`.
- `src/menipy/processing/substrate.py`: Substrate detection with `detect_substrate_line`.

### GUI
- `src/menipy/gui/__init__.py`: Makes the `gui` package runnable, launching the application via `app.main`.
- `src/menipy/gui/app.py`: Contains the `main` function that bootstraps the `QApplication` and `MainWindow`.
- `src/menipy/gui/mainwindow.py`: The main window shell, responsible for UI setup and delegating all logic to controllers.
- `src/menipy/gui/calibration_dialog.py`: Interactive `CalibrationDialog` for scale selection.
- `src/menipy/gui/logging_bridge.py`: Utilities to stream Python's `logging` messages into a Qt widget.
- `src/menipy/gui/plugins_panel.py`: Controller for the plugins dock and menu actions.
- `src/menipy/gui/sop_controller.py`: Helper class for managing Standard Operating Procedure (SOP) steps and UI.
- `src/menipy/gui/items.py`: Graphics items such as `SubstrateLineItem`.
- `src/menipy/gui/overlay.py`: Rendering helpers like `draw_drop_overlay`.
- `src/menipy/gui/controllers/main_controller.py`: The central controller that orchestrates all services, view models, and other controllers.
- `src/menipy/gui/controllers/pipeline_controller.py`: Manages the execution of analysis pipelines and updates the GUI with results.
- `src/menipy/gui/controllers/setup_panel_controller.py`: Controller for the "Setup" panel, managing user inputs for pipeline selection, source, and SOPs.
- `src/menipy/gui/panels/preview_panel.py`: A helper class that encapsulates interactions with the image preview area.
- `src/menipy/gui/panels/results_panel.py`: A helper class for displaying analysis results in a table.
- `src/menipy/gui/services/camera_service.py`: A Qt-friendly service for capturing live video from a camera in a non-blocking background thread.
- `src/menipy/gui/services/pipeline_runner.py`: A background service for running analysis pipelines in a separate thread.
- `src/menipy/gui/services/settings_service.py`: Manages loading and saving of application settings.
- `src/menipy/gui/services/sop_service.py`: Manages loading and saving of Standard Operating Procedures (SOPs).
- `src/menipy/gui/views/image_view.py`: `ImageView` widget wrapping `QGraphicsView` for displaying images and overlays.

### UI
- `src/menipy/ui/__init__.py`: UI components package.
- `src/menipy/ui/main_window.py`: Contains the `MainWindow` class that was previously used, now kept for reference or potential reuse. The primary `MainWindow` is in `gui/mainwindow.py`.
- `src/menipy/ui/views/step_item_widget.py`: A custom widget for displaying a single step in the SOP list.
- `src/menipy/ui/views/ui_main_window.py`: The compiled Python UI class generated from `main_window_split.ui`.

## tests
- `tests/__init__.py`: Test package marker.
- `tests/conftest.py`: Pytest fixtures.
- `tests/test_alt_workflow.py`: Validates alternative sessile workflow.
- `tests/test_analysis.py`: Regression tests for analysis helpers.
- `tests/test_batch.py`: Ensures batch processing operates correctly.
- `tests/test_classification.py`: Tests drop mode classification.
- `tests/test_contact_angle.py`: Verifies contact-angle calculations.
- `tests/test_contact_angle_alt.py`: Tests alternate contact-angle geometry helpers.
- `tests/test_contact_geom.py`: Checks geometric intersection helpers.
- `tests/test_contact_region.py`: Tests droplet region helpers and metrics.
- `tests/test_drop_extras.py`: Validates extra droplet metrics.
- `tests/test_drop_metrics.py`: Regression tests for apex detection and area calculations.
- `tests/test_geometry.py`: Validates geometry metrics and apex calculations.
- `tests/test_gui.py`: GUI integration tests covering widgets and workflows.
- `tests/test_models.py`: Tests geometric and physical models.
- `tests/test_plotting.py`: Ensures plotting utilities save expected images.
- `tests/test_processing.py`: Exercises segmentation, detection, and ML toggle.
- `tests/test_sessile_metrics.py`: Confirms sessile metrics calculations.
- `tests/test_sessile_segment.py`: Checks segment filtering for sessile analysis.
- `tests/test_spline_contact.py`: Validates spline-based contact detection.
- `tests/test_substrate.py`: Verifies substrate detection helpers.
- `tests/test_surface_tension.py`: Tests surface tension calculations.
- `tests/test_utils.py`: Calibration utility tests.

## Other
- `tests/test_drop_metrics.py`, etc.: Additional targeted tests listed above ensure correct functionality across modules.
