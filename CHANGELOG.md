# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Auto-Calibration Wizard**
  - New 1-click auto-calibration wizard for automatic detection of ROI, needle, substrate, and drop regions.
  - Accessible via the **🎯 Auto-Calibrate** button in the Setup Panel's Calibration section.
  - Pipeline-specific detection strategies:
    - **Sessile**: CLAHE + adaptive thresholding, gradient-based substrate detection, contact points at substrate
    - **Pendant**: Otsu thresholding, needle shaft line analysis, apex detection at drop bottom
  - Modal wizard dialog with live preview, region checkboxes, and confidence scores.
  - Detected regions drawn as colored overlays (ROI=yellow, Needle=blue, Substrate=magenta, Drop=green, Contact=red).
  - Files added:
    - `src/menipy/common/auto_calibrator.py` (detection engine)
    - `src/menipy/gui/dialogs/calibration_wizard_dialog.py` (wizard UI)
    - `tests/test_auto_calibrator.py` (25 unit tests)
  - Files modified:
    - `src/menipy/gui/views/setup_panel.ui` (added button)
    - `src/menipy/gui/controllers/setup_panel_controller.py` (signal wiring)
    - `src/menipy/gui/main_controller.py` (wizard launch + result handling)

- **Detection Plugins**
  - New plugin architecture for modular detection algorithms.
  - **Function-based plugins** (return values directly):
    - `detect_needle.py` - Sessile (top-border contour) and pendant (shaft line analysis)
    - `detect_roi.py` - Sessile (drop+substrate box) and pendant (needle-to-apex box)
    - `detect_substrate.py` - Gradient-based and Hough line detection
    - `detect_drop.py` - Sessile (adaptive threshold) and pendant (Otsu threshold)
    - `detect_apex.py` - Top of dome (sessile) or bottom of drop (pendant)
  - **Stage-based preprocessor plugins** (operate on ctx, follow pipeline pattern):
    - `preproc_detect_substrate.py` - Detects substrate line, stores in ctx
    - `preproc_detect_needle.py` - Detects needle region, stores in ctx
    - `preproc_detect_drop.py` - Detects drop contour, stores in ctx
    - `preproc_detect_roi.py` - Computes ROI from detected features
    - `preproc_auto_detect.py` - Chains all detection steps in correct order
  - Added 5 detector registries: `needle_detectors`, `roi_detectors`, `substrate_detectors`, `drop_detectors`, `apex_detectors`
  - Unit tests: 51 total (25 calibrator + 14 function plugins + 13 stage plugins)
  - Files modified:
    - `src/menipy/common/registry.py` (new registries and register functions)
    - `src/menipy/pipelines/sessile/preprocessing.py` (uses auto_detect plugin)
    - `src/menipy/pipelines/pendant/preprocessing.py` (uses auto_detect plugin)


- **Preprocessing: "Fill Holes" option**
  - Adds a preprocessing option to fill interior holes in the region of interest (ROI) mask and remove spurious contour points close to the contact line. This helps produce cleaner droplet contours for downstream geometry and fitting stages.
  - GUI: available in the Preprocessing Configuration dialog under the new "Fill Holes" page (checkbox + numeric parameters).
  - Default behavior: disabled. Parameters include `max_hole_area` (default 500 px), `remove_spurious_near_contact` (default true), and `proximity_px` (default 5 px).
  - Implementation notes: prefers `skimage.morphology` (remove_small_holes/remove_small_objects) when available; falls back to OpenCV contour/connected-component based filling and small-object removal when `skimage` is not present.
  - Files touched:
    - `src/menipy/models/config.py` (new `FillHolesSettings` and `fill_holes` field in `PreprocessingSettings`)
    - `src/menipy/common/preprocessing_helpers.py` (new `fill_holes(context)` helper)
    - `src/menipy/common/preprocessing.py` (calls `fill_holes` after `crop_to_roi`)
    - `src/menipy/gui/dialogs/preprocessing_config_dialog.py` (adds UI page and bindings)

### Fixed

- **Sessile Pipeline Overlay Stage**:
  - Fixed `UnboundLocalError` in the overlay stage caused by a redundant `import cv2` statement that shadowed the module-level import.
  - Fixed contour drawing by properly reshaping the contour array to the expected OpenCV format `(n, 1, 2)` for `cv2.drawContours`.
  - Added validation for contour data to prevent crashes when contour is empty or has an invalid shape.
  - Files touched:
    - `src/menipy/pipelines/sessile/stages.py` (fixed overlay method contour handling)

- **Results History - Phase 4 UI Enhancements**:
  - Added measurement sequence numbering to pipeline overlays - displays "Measurement #X" on preview images with semi-transparent background
  - Enhanced status bar messages to show measurement count after analysis completes (e.g., "Analysis complete - 5 measurements recorded")
  - Updated all pipeline overlays to display measurement numbers: sessile, pendant, oscillating, captive_bubble, capillary_rise
  - Added `measurement_id` and `measurement_sequence` fields to Context model for tracking
  - Files touched:
    - `src/menipy/models/context.py` (added measurement tracking fields)
    - `src/menipy/pipelines/base.py` (pass measurement fields to Context)
    - `src/menipy/gui/controllers/pipeline_controller.py` (populate measurement tracking, enhanced status messages)
    - `src/menipy/pipelines/*/stages.py` (all pipeline overlays updated)

### Changed

- Edge-detection & Preview overlays
  - The Edge Detection controller no longer bakes overlay graphics (contours / contact-point circles) into preview images. Instead it emits the raw preview image and metadata (contour points and detected contact points). The `MainController` now draws overlays using the `ImageView` overlay API which makes overlays easier to update, remove, and style.
  - Geometry configuration preview can now request the "preprocessed" image: when the geometry dialog's "Use preprocessed image for preview" option is enabled, the main controller requests a one-shot preprocessing preview and uses that image as the source for edge-detection previews.
  - Visuals: detected contours are shown in red; detected contact points are shown as distinct colored markers.
  - Files touched:
    - `src/menipy/gui/controllers/edge_detection_controller.py` (emit contour + contact points as metadata)
    - `src/menipy/gui/main_controller.py` (draw overlays using ImageView, one-shot preprocessing preview for geometry dialog)



## [0.2.0] - 2025-09-22

### Added

- **Developer Documentation**:
  - Guide for adding new analysis pipelines (`developer_guide_pipelines.md`).
  - Guide for creating new plugins (`developer_guide_plugins.md`).

### Fixed
- **Pipeline Discovery**: Corrected the dynamic pipeline discovery mechanism to properly find and load pipelines. The developer guide was also updated to reflect the correct file structure for creating new pipelines.

### Changed

- **Major Architectural Refactoring**:
  - Migrated core logic into a modular `menipy` package with a modern, extensible architecture.
  - Introduced a `pipelines` architecture to separate the logic for different analysis types (pendant, sessile). This makes the system more maintainable and extensible.
  - Introduced a `plugins` architecture to allow for easy extension of the core functionality.

### Removed

- **Legacy Code**: Deleted the original, monolithic `src/menipy/analysis`, `src/menipy/calibration`, and other `zold_*` directories in favor of the new `menipy` package structure.

## [0.1.0] - 2025-09-20

This changelog was created based on the development history tracked in `CODEXLOG.md`. The project is currently in its initial development phase.

### Added

- **Initial GUI Framework**:
  - Core application window with a two-panel layout for image viewing and controls.
  - Image loading from files and basic display.
  - Zoom controls (slider, fit-to-window, actual size).
  - Status bar for user feedback.

- **Analysis & Processing Core**:
  - **Image Segmentation**: Initial support for Otsu and adaptive thresholding.
  - **Contour Detection**: Find and overlay droplet contours on the image viewer.
  - **Droplet Classification**: Heuristics to classify droplets as "pendant" or "sessile".
  - **Batch Processing**: CLI tool to process a directory of images and export results to CSV.

- **Calibration Tools**:
  - **Manual Calibration**: Draw a line of known length to set the pixel-to-mm scale.
  - **Automatic Calibration**: Automatically detect needle diameter from a user-drawn ROI using Canny edge detection and Hough transforms.

- **Analysis Pipelines & Tabs**:
  - **Pendant Drop Analysis**:
    - Dedicated tab for analyzing hanging drops.
    - Computes surface tension, volume, surface area (total and per-side), apex radius, and Bond number.
    - Option to save side-profile plots of the droplet contour.
  - **Sessile Drop / Contact Angle Analysis**:
    - Dedicated tab for analyzing droplets on a substrate.
    - Interactive tool to draw or automatically detect the substrate line.
    - Manual and spline-based automatic detection of contact points.
    - Computes contact angle, base diameter, and droplet height.
  - **Detection Test Tab**:
    - An interface for experimenting with different image segmentation algorithms and filters.

- **Plugin System**:
  - **Filter Plugins**: Extensible system to add custom image filters (e.g., sharpening, brightness) that appear in the "Detection Test" tab.
  - **Plugin Manager**: GUI dialog to discover, activate, and deactivate installed plugins.
  - CLI for managing plugin directories.

- **Data Export**:
  - **Save Annotated Image**: Export the current view with all overlays (contours, lines, points) to an image file.
  - **Save CSV**: Export all user-set parameters and computed metrics to a CSV file.

- **Developer Documentation**:
  - Guide for adding new analysis pipelines (`doc/developer_guide_pipelines.md`).
  - Guide for creating new image filter plugins (`doc/developer_guide_plugins.md`).

### Changed

- **Major Architectural Refactoring**:
  - Migrated core logic into a modular `menipy` package.
  - Introduced a `pipelines` architecture to separate the logic for different analysis types (pendant, sessile). This makes the system more maintainable and extensible.
  - Replaced monolithic UI logic with calls to the appropriate pipeline based on the active tab.

- **GUI Layout**:
  - Reorganized the main window into a multi-tabbed interface for different workflows (Calibration, Pendant Drop, Contact Angle, etc.).
  - Refactored the main window into a modern, splittable layout with dedicated panels for setup, image preview, and results.

- **Analysis Algorithms**:
  - **Volume Calculation**: Improved to use a more accurate solid-of-revolution integration.
  - **Apex Detection**: Made more robust to handle cases with multiple points at the extremum.
  - **Sessile Drop Analysis**: Enhanced with advanced geometry helpers for contour cleaning, substrate intersection, and side-selection.

- **Logging**:
  - Implemented a "Log" tab in the GUI that displays detailed, per-stage logs from the analysis pipelines for easier debugging.

### Fixed

- **GUI Signal Handling**: Resolved a crash during "Save CSV" operations by correcting signal/slot connections in PySide6.
- **Overlay Drawing**:
  - Fixed an issue where analysis overlays from one mode would incorrectly appear in another.
  - Corrected coordinate offsets for substrate lines drawn within an ROI.
  - Ensured all overlays are properly cleared when loading a new image or using the "Clear Analysis" button.
- **Import Errors**: Resolved various `ModuleNotFoundError` and `ImportError` issues that occurred during the architectural refactor.

### Removed

- **Legacy Code**: Deleted the original, monolithic `src` directory in favor of the new `menipy` package structure.
- **Redundant GUI Tabs**: Removed several temporary and duplicative "Contact Angle" tabs, consolidating functionality into a single, improved workflow.
- **Unused `matplotlib` Dependency**: Removed `matplotlib` as a core dependency, making it optional for specific plotting features.
