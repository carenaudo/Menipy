# Python Files Overview

This document summarizes each Python file in the repository.

## Root-level Files

- `.\__init__.py`: Refactored Menipy package stubs.
- `.\__main__.py`: Entry point for running Menipy as a module.

## menipy

- `menipy\__init__.py`: Top-level menipy package.
- `menipy\__main__.py`: Run menipy as a module: `python -m menipy` -> GUI entrypoint.
- `menipy\cli.py`: Command-line interface for running Menipy pipelines.

## menipy\cli

- `menipy\cli\__init__.py`: Command-line interface (console-script entrypoint).

## menipy\common

- `menipy\common\__init__.py`: (no docstring)
- `menipy\common\acquisition.py`: Common acquisition utilities for loading images from files or cameras.
- `menipy\common\edge_detection.py`: Edge detection utilities and pipeline stage logic.
- `menipy\common\geometry.py`: Common, low-level geometric utilities.
- `menipy\common\metrics.py`: Common metrics calculations for droplet analysis.
- `menipy\common\optimization.py`: Optimization stage utilities for refining solver results.
- `menipy\common\outputs.py`: Output formatting and normalization for pipeline results.
- `menipy\common\overlay.py`: Overlay drawing utilities for visualizing analysis results on images.
- `menipy\common\physics.py`: Physics constants and parameter management.
- `menipy\common\plugin_db.py`: Database abstraction for managing plugins.
- `menipy\common\plugin_loader.py`: Utilities to apply registered plugin functions into pipeline instances.
- `menipy\common\plugins.py`: Plugin discovery, loading, and management utilities.
- `menipy\common\preprocessing.py`: Preprocessing pipeline stage implementation.
- `menipy\common\preprocessing_helpers.py`: (no docstring)
- `menipy\common\registry.py`: Central registry for pipeline components and plugins.
- `menipy\common\scaling.py`: Scaling stage utilities for pixel-to-mm conversion.
- `menipy\common\solver.py`: Solver stage utilities for fitting physical models to contours.
- `menipy\common\units.py`: Unit registry configuration using Pint.
- `menipy\common\validation.py`: Validation stage utilities for quality assurance of results.
- `menipy\common\zold_detection.py`: Deprecated/Empty file.

## menipy\gui

- `menipy\gui\__init__.py`: GUI package initialization.
- `menipy\gui\__main__.py`: GUI entry point for running Menipy GUI directly.
- `menipy\gui\app.py`: Main GUI application setup and initialization.
- `menipy\gui\logging_bridge.py`: Qt logging bridge utilities for streaming Python logs into Qt widgets.
- `menipy\gui\main_controller.py`: Main window controller coordinating GUI components.
- `menipy\gui\main_window.py`: Main window implementation (legacy - check if still used).
- `menipy\gui\mainwindow.py`: Main window class for Menipy GUI.
- `menipy\gui\overlay.py`: Overlay rendering and display utilities.
- `menipy\gui\plugins_panel.py`: Plugin dock and menu controller for the main window.
- `menipy\gui\zold_drawing_alt.py`: Legacy drawing utilities (deprecated).

## menipy\gui\controllers

- `menipy\gui\controllers\edge_detection_controller.py`: Controller for edge detection configuration and execution.
- `menipy\gui\controllers\pipeline_controller.py`: Pipeline execution helper for Menipy GUI.
- `menipy\gui\controllers\preprocessing_controller.py`: Controller for preprocessing stage configuration.
- `menipy\gui\controllers\setup_panel_controller.py`: Controller for the setup panel widgets.
- `menipy\gui\controllers\sop_controller.py`: SOP management helper for Menipy GUI.

## menipy\gui\dialogs

- `menipy\gui\dialogs\acquisition_config_dialog.py`: Dialog for configuring image acquisition settings.
- `menipy\gui\dialogs\edge_detection_config_dialog.py`: Dialog for edge detection method configuration.
- `menipy\gui\dialogs\geometry_config_dialog.py`: Geometry configuration dialog for Menipy.
- `menipy\gui\dialogs\overlay_config_dialog.py`: Dialog for overlay visualization settings.
- `menipy\gui\dialogs\physics_config_dialog.py`: Dialog for editing physics parameters with unit support.
- `menipy\gui\dialogs\plugin_manager_dialog.py`: Dialog for managing and configuring plugins.
- `menipy\gui\dialogs\preprocessing_config_dialog.py`: Dialog for preprocessing settings configuration.

## menipy\gui\helpers

- `menipy\gui\helpers\image_marking.py`: Interactive image marking and annotation tools.

## menipy\gui\panels

- `menipy\gui\panels\__init__.py`: GUI panels package.
- `menipy\gui\panels\discover.py`: Panel discovery and registration utilities.
- `menipy\gui\panels\preview_panel.py`: Live preview panel for image display and interaction.
- `menipy\gui\panels\results_panel.py`: Results panel helper for Menipy GUI.
- `menipy\gui\panels\setup_panel.py`: Controller for the setup panel widgets.

## menipy\gui\resources

- `menipy\gui\resources\menipy_icons_rc.py`: Generated Qt resource file for icons (auto-generated).

## menipy\gui\services

- `menipy\gui\services\camera_service.py`: Qt-friendly camera capture service for Menipy GUI.
- `menipy\gui\services\image_convert.py`: Image format conversion utilities for Qt.
- `menipy\gui\services\pipeline_runner.py`: Service for running pipelines in the GUI context.
- `menipy\gui\services\plugin_service.py`: Plugin management service for GUI.
- `menipy\gui\services\settings_service.py`: Application settings persistence service.
- `menipy\gui\services\sop_service.py`: Standard Operating Procedure (SOP) management service.

## menipy\gui\viewmodels

- `menipy\gui\viewmodels\plugins_vm.py`: View model for plugin management UI.
- `menipy\gui\viewmodels\results_vm.py`: View model for results display.
- `menipy\gui\viewmodels\run_vm.py`: View model for pipeline run management.

## menipy\gui\views

- `menipy\gui\views\__init__.py`: GUI views package.
- `menipy\gui\views\image_view.py`: Custom image viewer widget.
- `menipy\gui\views\step_item_widget.py`: Widget for displaying pipeline step items.
- `menipy\gui\views\ui_main_window.py`: UI definition for main window (likely auto-generated).

## menipy\math

- `menipy\math\jurin.py`: Placeholder for Jurin's law (capillary rise) calculations.
- `menipy\math\rayleigh_lamb.py`: Placeholder for Rayleigh-Lamb oscillation frequency calculations.
- `menipy\math\young_laplace.py`: Placeholder for Young-Laplace equation solver implementation.

## menipy\models

- `menipy\models\__init__.py`: Model utilities for Menipy.
- `menipy\models\config.py`: Configuration models for pipeline settings and physical parameters.
- `menipy\models\context.py`: Context model for sharing state between pipeline stages.
- `menipy\models\datatypes.py`: Common data types and structures for analysis records and preprocessing state.
- `menipy\models\drop_extras.py`: Additional droplet property calculations (Worthington number, curvature, surface area, etc.).
- `menipy\models\fit.py`: Data models for numerical fitting results and configuration.
- `menipy\models\frame.py`: Frame model for representing images with metadata, calibration, and timing information.
- `menipy\models\geometry.py`: Geometric models for contours, geometry landmarks, and spatial features.
- `menipy\models\physics.py`: Physical modeling utilities.
- `menipy\models\properties.py`: Basic geometric and physical property calculations for droplets.
- `menipy\models\result.py`: Data models for final, high-level analysis outputs.
- `menipy\models\state.py`: Runtime state models for Menipy (migrated from datatypes.py).
- `menipy\models\surface_tension.py`: Surface tension and related pendant-drop calculations.
- `menipy\models\typing.py`: Type aliases for numpy arrays used throughout Menipy.
- `menipy\models\unit_types.py`: Pydantic-Pint unit types for physical quantities.

## menipy\pipelines

- `menipy\pipelines\__init__.py`: This package contains the core analysis pipelines for different measurement modes.
- `menipy\pipelines\base.py`: Base pipeline class with template method pattern for stage-based execution.
- `menipy\pipelines\discover.py`: Centralized pipeline discovery.
- `menipy\pipelines\runner.py`: Simple runner for executing pipelines non-interactively.

## menipy\pipelines\capillary_rise

- `menipy\pipelines\capillary_rise\__init__.py`: (no docstring)
- `menipy\pipelines\capillary_rise\acquisition.py`: STUB: Capillary Rise Pipeline - Acquisition Stage
- `menipy\pipelines\capillary_rise\edge_detection.py`: STUB: Capillary Rise Pipeline - Edge Detection Stage
- `menipy\pipelines\capillary_rise\geometry.py`: STUB: Capillary Rise Pipeline - Geometry Stage
- `menipy\pipelines\capillary_rise\optimization.py`: STUB: Capillary Rise Pipeline - Optimization Stage
- `menipy\pipelines\capillary_rise\outputs.py`: STUB: Capillary Rise Pipeline - Outputs Stage
- `menipy\pipelines\capillary_rise\overlay.py`: STUB: Capillary Rise Pipeline - Overlay Stage
- `menipy\pipelines\capillary_rise\physics.py`: STUB: Capillary Rise Pipeline - Physics Stage
- `menipy\pipelines\capillary_rise\preprocessing.py`: STUB: Capillary Rise Pipeline - Preprocessing Stage
- `menipy\pipelines\capillary_rise\scaling.py`: STUB: Capillary Rise Pipeline - Scaling Stage
- `menipy\pipelines\capillary_rise\solver.py`: STUB: Capillary Rise Pipeline - Solver Stage
- `menipy\pipelines\capillary_rise\stages.py`: (no docstring)
- `menipy\pipelines\capillary_rise\validation.py`: STUB: Capillary Rise Pipeline - Validation Stage

## menipy\pipelines\captive_bubble

- `menipy\pipelines\captive_bubble\__init__.py`: (no docstring)
- `menipy\pipelines\captive_bubble\acquisition.py`: STUB: Captive Bubble Pipeline - Acquisition Stage
- `menipy\pipelines\captive_bubble\edge_detection.py`: STUB: Captive Bubble Pipeline - Edge Detection Stage
- `menipy\pipelines\captive_bubble\geometry.py`: STUB: Captive Bubble Pipeline - Geometry Stage
- `menipy\pipelines\captive_bubble\optimization.py`: STUB: Captive Bubble Pipeline - Optimization Stage
- `menipy\pipelines\captive_bubble\outputs.py`: STUB: Captive Bubble Pipeline - Outputs Stage
- `menipy\pipelines\captive_bubble\overlay.py`: STUB: Captive Bubble Pipeline - Overlay Stage
- `menipy\pipelines\captive_bubble\physics.py`: STUB: Captive Bubble Pipeline - Physics Stage
- `menipy\pipelines\captive_bubble\preprocessing.py`: STUB: Captive Bubble Pipeline - Preprocessing Stage
- `menipy\pipelines\captive_bubble\scaling.py`: STUB: Captive Bubble Pipeline - Scaling Stage
- `menipy\pipelines\captive_bubble\solver.py`: STUB: Captive Bubble Pipeline - Solver Stage
- `menipy\pipelines\captive_bubble\stages.py`: (no docstring)
- `menipy\pipelines\captive_bubble\validation.py`: STUB: Captive Bubble Pipeline - Validation Stage

## menipy\pipelines\oscillating

- `menipy\pipelines\oscillating\__init__.py`: (no docstring)
- `menipy\pipelines\oscillating\acquisition.py`: STUB: Oscillating Pipeline - Acquisition Stage
- `menipy\pipelines\oscillating\edge_detection.py`: STUB: Oscillating Pipeline - Edge Detection Stage
- `menipy\pipelines\oscillating\geometry.py`: STUB: Oscillating Pipeline - Geometry Stage
- `menipy\pipelines\oscillating\optimization.py`: STUB: Oscillating Pipeline - Optimization Stage
- `menipy\pipelines\oscillating\outputs.py`: STUB: Oscillating Pipeline - Outputs Stage
- `menipy\pipelines\oscillating\overlay.py`: STUB: Oscillating Pipeline - Overlay Stage
- `menipy\pipelines\oscillating\physics.py`: STUB: Oscillating Pipeline - Physics Stage
- `menipy\pipelines\oscillating\preprocessing.py`: STUB: Oscillating Pipeline - Preprocessing Stage
- `menipy\pipelines\oscillating\scaling.py`: STUB: Oscillating Pipeline - Scaling Stage
- `menipy\pipelines\oscillating\solver.py`: STUB: Oscillating Pipeline - Solver Stage
- `menipy\pipelines\oscillating\stages.py`: (no docstring)
- `menipy\pipelines\oscillating\validation.py`: STUB: Oscillating Pipeline - Validation Stage

## menipy\pipelines\pendant

- `menipy\pipelines\pendant\__init__.py`: (no docstring)
- `menipy\pipelines\pendant\acquisition.py`: STUB: Pendant Pipeline - Acquisition Stage
- `menipy\pipelines\pendant\drawing.py`: STUB: Pendant Pipeline - Drawing Stage
- `menipy\pipelines\pendant\edge_detection.py`: STUB: Pendant Pipeline - Edge Detection Stage
- `menipy\pipelines\pendant\geometry.py`: (no docstring)
- `menipy\pipelines\pendant\metrics.py`: (no docstring)
- `menipy\pipelines\pendant\optimization.py`: STUB: Pendant Pipeline - Optimization Stage
- `menipy\pipelines\pendant\outputs.py`: STUB: Pendant Pipeline - Outputs Stage
- `menipy\pipelines\pendant\overlay.py`: STUB: Pendant Pipeline - Overlay Stage
- `menipy\pipelines\pendant\physics.py`: STUB: Pendant Pipeline - Physics Stage
- `menipy\pipelines\pendant\preprocessing.py`: STUB: Pendant Pipeline - Preprocessing Stage
- `menipy\pipelines\pendant\scaling.py`: STUB: Pendant Pipeline - Scaling Stage
- `menipy\pipelines\pendant\solver.py`: STUB: Pendant Pipeline - Solver Stage
- `menipy\pipelines\pendant\stages.py`: (no docstring)
- `menipy\pipelines\pendant\validation.py`: STUB: Pendant Pipeline - Validation Stage

## menipy\pipelines\sessile

- `menipy\pipelines\sessile\__init__.py`: (no docstring)
- `menipy\pipelines\sessile\acquisition.py`: STUB: Sessile Pipeline - Acquisition Stage
- `menipy\pipelines\sessile\drawing.py`: STUB: Sessile Pipeline - Drawing Stage
- `menipy\pipelines\sessile\edge_detection.py`: STUB: Sessile Pipeline - Edge Detection Stage
- `menipy\pipelines\sessile\geometry.py`: (no docstring)
- `menipy\pipelines\sessile\metrics.py`: (no docstring)
- `menipy\pipelines\sessile\optimization.py`: STUB: Sessile Pipeline - Optimization Stage
- `menipy\pipelines\sessile\outputs.py`: STUB: Sessile Pipeline - Outputs Stage
- `menipy\pipelines\sessile\overlay.py`: STUB: Sessile Pipeline - Overlay Stage
- `menipy\pipelines\sessile\physics.py`: STUB: Sessile Pipeline - Physics Stage
- `menipy\pipelines\sessile\preprocessing.py`: STUB: Sessile Pipeline - Preprocessing Stage
- `menipy\pipelines\sessile\scaling.py`: STUB: Sessile Pipeline - Scaling Stage
- `menipy\pipelines\sessile\solver.py`: STUB: Sessile Pipeline - Solver Stage
- `menipy\pipelines\sessile\stages.py`: (no docstring)
- `menipy\pipelines\sessile\validation.py`: STUB: Sessile Pipeline - Validation Stage

## menipy\viz

- `menipy\viz\plots.py`: Placeholder for plotting utilities.
