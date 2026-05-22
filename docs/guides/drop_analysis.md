# Drop Analysis Module

This document describes the workflow and algorithms used by the Menipy application.

## 1. Overview

Menipy uses a flexible, pipeline-driven architecture to perform automated measurements on droplet images. Instead of a fixed interface, the user selects an analysis pipeline (e.g., for pendant or sessile drops), configures its stages, and executes it. The analysis is performed on a selected source, which can be a single image, a batch of images, or a live camera feed.

## 2. User Workflow

The main workflow is organized around the top workflow bar, the left setup rail,
the central preview, and the results/diagnostics area:

1.  **Select Pipeline**: Use the analysis buttons in the top workflow bar
    (`Sessile`, `Pendant`, `Osc.`, `Capillary`, `Captive`) to choose the active
    pipeline.

2.  **Select Source Mode**: Use the source buttons in the workflow bar:
    *   **File**: Analyze one image file.
    *   **Batch**: Process all compatible images in a folder.
    *   **Camera**: Acquire frames from a connected camera.

3.  **Select Source and Preview**: Browse for an image/folder or choose the
    camera, then click **Preview** in the setup rail to load the source into the
    preview panel.

4.  **Calibrate and Configure**: Use **Calibrate** to run the auto-calibration
    wizard, and use the setup rail or **Advanced** workflow controls for pipeline
    settings and stage configuration.

5.  **Run Analysis**: Click **Run Analysis** to execute the pipeline on the
    selected source.

6.  **Review Results**: The preview updates with overlays such as ROI,
    contours, contact points, fit lines, or model profiles. The Results area
    displays numerical metrics, timings, residuals, and diagnostics.

### Scientific Step Testing

For development and scientific review, Menipy provides a stage-testing panel:

1. Choose **View -> Focus -> Science**.
2. Click the **Test** button that appears next to the pipeline selection buttons.
3. The left rail switches from the setup panel to **Step Test**.
4. Select a stage and click **Run Stage**.

The test panel lists the selected pipeline's executable stages except
`acquisition`. It runs the requested stage with prerequisite stages included, so
testing `profile_fitting`, for example, automatically runs the earlier stages
needed to build the context. Before the test run, Menipy silently runs
auto-calibration on the selected source and merges detected ROI, needle,
substrate/contact line, contour, contact points, and apex into the test context.
Missing detections are reported as inline prerequisite warnings instead of
opening the calibration wizard.

Configuration changes made in the test panel are sandboxed. They affect test
runs only until **Apply** is clicked. **Discard** restores the sandbox from the
current live settings. Editable test configuration currently covers
`preprocessing`, `contour_extraction`, `geometric_features`, `physics`, and
`overlay`; other stages can still be run and inspected with their current
inputs.

## 3. Algorithms

The core logic for the analysis is distributed across several modules in the `src/menipy/` directory:

- **Needle Detection**: Implemented in `src/menipy/detection/needle.py`. The ROI is blurred, edges are detected with Canny, and vertical lines are fit using a Hough transform to locate the needle sides and determine its width.

- **Substrate Detection**: Implemented in `src/menipy/detection/substrate.py`. For sessile drops, this module can automatically detect the baseline using a RANSAC or Hough transform.

- **Contour and Metric Extraction**: Core helpers are located in `src/menipy/analysis/commons.py` and `src/menipy/common/edge_detection.py`. These modules handle the extraction of the external contour using methods like `cv2.findContours` and subsequent morphological filtering. They also compute the fundamental geometric metrics that are passed to the physics solvers.
