# Drop Analysis Module

This document describes the workflow and algorithms used by the Menipy application.

## 1. Overview

Menipy uses a flexible, pipeline-driven architecture to perform automated measurements on droplet images. Instead of a fixed interface, the user selects an analysis pipeline (e.g., for pendant or sessile drops), configures its stages, and executes it. The analysis is performed on a selected source, which can be a single image, a batch of images, or a live camera feed.

## 2. User Workflow

The main user workflow is centered around the "Setup" panel:

1.  **Select Source Mode**: Choose the data source using the radio buttons:
    *   **Single Image**: For analyzing one image file.
    *   **Batch**: For processing all images in a folder.
    *   **Camera**: For live analysis from a connected camera.

2.  **Select Source**: Provide the path to the image or folder, or select the camera ID.

3.  **Select Pipeline**: Choose the desired analysis method (e.g., "Pendant Drop", "Sessile Drop") from the "Pipeline" dropdown menu.

4.  **Configure Pipeline Stages**: The list of steps for the selected pipeline (e.g., `acquisition`, `preprocessing`, `edge_detection`, `geometry`) is displayed. Each stage can be configured individually by clicking its associated "config" button, which opens a detailed dialog for that stage's parameters.

5.  **Run Analysis**: Click the **"Run All"** button to execute the entire pipeline on the selected source.

6.  **View Results**: After the pipeline completes, the image preview panel will update with graphical overlays (like contours and fitted lines), and the "Results" panel will display the final numerical metrics (e.g., surface tension, contact angle).

## 3. Algorithms

The core logic for the analysis is distributed across several modules in the `src/menipy/` directory:

- **Needle Detection**: Implemented in `src/menipy/detection/needle.py`. The ROI is blurred, edges are detected with Canny, and vertical lines are fit using a Hough transform to locate the needle sides and determine its width.

- **Substrate Detection**: Implemented in `src/menipy/detection/substrate.py`. For sessile drops, this module can automatically detect the baseline using a RANSAC or Hough transform.

- **Contour and Metric Extraction**: Core helpers are located in `src/menipy/analysis/commons.py` and `src/menipy/common/edge_detection.py`. These modules handle the extraction of the external contour using methods like `cv2.findContours` and subsequent morphological filtering. They also compute the fundamental geometric metrics that are passed to the physics solvers.