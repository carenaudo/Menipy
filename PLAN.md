# Menipy Development Plan

This document outlines the development plan for Menipy, a Python-based toolkit for droplet shape analysis. It is intended to be used by the AI agent to guide the implementation of new features and the maintenance of the existing codebase.

## 1. Tech Stack

-   **Language**: Python 3.9+
-   **GUI**: PySide6
-   **Image I/O & Processing**: OpenCV (cv2), scikit-image
-   **Numerical Computing**: NumPy, SciPy
-   **Data Handling**: pandas, pydantic
-   **Plotting**: Matplotlib
-   **Packaging & Testing**: setuptools, pytest, pytest-qt, flake8
-   **Documentation**: Sphinx

## 2. Architecture Overview

The Menipy application is built on a modular architecture based on **pipelines** and **plugins**.

-   **Pipelines:** A pipeline is a sequence of stages that process an image to extract droplet properties. Each stage is a Python function that takes a `Context` object as input and modifies it. The `Context` object carries data through the pipeline. Pipelines are discovered automatically from the `src/menipy/pipelines/` directory.
-   **Plugins:** Plugins provide a way to extend the functionality of Menipy with new algorithms for any pipeline stage. Plugins are discovered and registered by the `PluginDB` and `registry` modules.

## 3. Directory Structure

```
.
├── doc/                    # Reference documentation for the AI agent
│   ├── droplet_description.md
│   ├── physics_models.md
│   ├── numerical_methods.md
│   ├── image_processing.md
│   └── gui_design.md
├── src/
│   └── menipy/             # Main application package
│       ├── __main__.py     # Main entry point
│       ├── cli.py          # Command-line interface
│       ├── common/         # Common components for pipeline stages
│       │   ├── __init__.py
│       │   ├── acquisition.py
│       │   ├── edge_detection.py
│       │   ├── geometry.py
│       │   ├── optimization.py
│       │   ├── outputs.py
│       │   ├── overlay.py
│       │   ├── physics.py
│       │   ├── plugin_db.py
│       │   ├── plugin_loader.py
│       │   ├── plugins.py
│       │   ├── preprocessing.py
│       │   ├── registry.py
│       │   ├── scaling.py
│       │   ├── solver.py
│       │   └── validation.py
│       ├── gui/            # PySide6 GUI components
│       │   ├── __init__.py
│       │   ├── app.py
│       │   ├── main_controller.py
│       │   ├── mainwindow.py
│       │   ├── panels/
│       │   │   ├── __init__.py
│       │   │   ├── plugin_manager_panel.py  # Proposed
│       │   │   └── results_panel.py         # Proposed
│       │   └── widgets/
│       │       └── ...
│       ├── math/           # Mathematical models
│       │   ├── __init__.py
│       │   ├── young_laplace.py
│       │   └── ...
│       ├── models/         # Data models (pydantic)
│       │   ├── __init__.py
│       │   ├── datatypes.py
│       │   └── ...
│       ├── pipelines/      # Analysis pipelines
│       │   ├── __init__.py
│       │   ├── base.py
│       │   ├── discover.py
│       │   ├── pendant/
│       │   │   └── __init__.py
│       │   ├── sessile/
│       │   │   └── __init__.py
│       │   ├── captive_bubble/   # Proposed
│       │   │   └── __init__.py
│       │   └── oscillating/      # Proposed
│       │       └── __init__.py
│       └── viz/            # Visualization and plotting
│           ├── __init__.py
│           └── plots.py
├── plugins/                # Custom plugins
│   ├── __init__.py
│   ├── preproc_blur.py
│   └── edge_canny.py       # Proposed
├── data/
│   └── samples/            # Sample images for tests
│       ├── pendant_drop.png
│       └── sessile_drop.png
├── tests/                  # Pytest suites
│   ├── __init__.py
│   ├── test_pipelines.py
│   ├── test_plugins.py
│   └── ...
├── requirements.txt        # Project dependencies
├── setup.py                # Package metadata
├── AGENTS.md               # AI agent descriptions
└── PLAN.md                 # This development plan
```

## 4. Development Roadmap

The development of Menipy will proceed in the following phases.

### Phase 1: Core Infrastructure (Completed)

-   [x] **Project Scaffolding:** Set up the directory structure, `setup.py`, and `requirements.txt`.
-   [x] **Pipeline Architecture:** Implement the base pipeline (`PipelineBase`) and the `Context` object.
-   [x] **Plugin System:** Implement the plugin discovery and registration mechanism.
-   [x] **GUI Skeleton:** Create the main window with a basic layout for image display and controls.
-   [x] **Basic Pipelines:** Implement initial pipelines for pendant and sessile drops.

### Phase 2: Feature Implementation

-   [ ] **Implement All Pipeline Stages:** Ensure that there are plugin implementations for all pipeline stages defined in `PipelineBase`.
    -   [ ] **Acquisition:** Load an image from a file or camera.
    -   [ ] **Preprocessing:** Prepare the image for analysis (e.g., cropping, resizing, filtering).
    -   [ ] **Edge Detection:** Detect the outline of the droplet in the image.
    -   [ ] **Geometry:** Analyze the geometric properties of the droplet's shape.
    -   [ ] **Scaling:** Calibrate the image from pixels to physical units (e.g., mm).
    -   [ ] **Physics:** Apply physical models to the droplet's shape (e.g., Young-Laplace equation).
    -   [ ] **Solver:** Solve the physical equations to determine properties like surface tension.
    -   [ ] **Optimization:** Optimize the parameters of the physical model to best fit the observed shape.
    -   [ ] **Outputs:** Generate and save the results of the analysis (e.g., as a CSV file).
    -   [ ] **Overlay:** Draw the results of the analysis on top of the original image.
    -   [ ] **Validation:** Perform checks to ensure the quality and correctness of the results.
-   [ ] **GUI Enhancements:**
    -   [ ] **Plugin Manager:** Create a GUI panel to view, activate, and deactivate plugins.
    -   [ ] **Pipeline Controls:** Allow the user to select and configure pipelines from the GUI.
    -   [ ] **Interactive Plotting:** Improve the interactivity of plots (zoom, pan, etc.).
    -   [ ] **Results Display:** Create a dedicated panel to display the results of the analysis in a structured format.
-   [ ] **New Pipelines:**
    -   [ ] **Captive Bubble:** Create a new pipeline for captive bubble experiments.
    -   [ ] **Oscillating Drop:** Implement a pipeline for analyzing oscillating drops.
-   [ ] **Command-Line Interface (CLI):**
    -   [ ] Enhance the CLI to allow running pipelines and generating reports from the command line.
-   [ ] **Testing and CI:**
    -   [ ] Increase test coverage for all components.
    -   [ ] Set up a CI/CD pipeline using GitHub Actions to run tests automatically.

### Phase 3: Documentation and Polishing

-   [ ] **User Documentation:** Write comprehensive user documentation explaining how to use the Menipy application.
-   [ ] **Developer Documentation:** Improve the developer documentation, especially for creating new pipelines and plugins.
-   [ ] **Packaging and Distribution:** Create binary packages for Windows, macOS, and Linux.
