# How `test_sessile_qt.py` Works

This script performs a standalone test of the Sessile Drop Pipeline, replicating the logic used in the main application but with enhanced debug visualization using PySide6 (Qt).

## Overview

1.  **Loads an Image**: Reads the image file specified in the command line.
2.  **Auto-Calibration**: Runs the `AutoCalibrator` to detect the substrate line, needle, and initial drop contour.
3.  **Pipeline Analysis**: Passes the image and detected features (or manual overrides) to the `SessilePipeline`.
4.  **Visualization**: Displays the result in a Qt window, drawing overlays (contour, baseline, apex, contact points) using `QPainter`.

## Detailed Steps

### 1. Initialization and Auto-Calibration (`run_test`)
*   **CLI Parsing**: The script starts by parsing command-line arguments:
    *   `image_path`: Path to the image.
    *   `--substrate-y`: (Optional) Manual Y-coordinate for the substrate.
    *   `--margin-fraction` & `--block-size`: Parameters for detection tuning.
*   **Auto-Calibration**: It executes `menipy.common.auto_calibrator.run_auto_calibration` on the image. This mirrors the "Calibration Wizard" in the main app, automatically finding:
    *   `substrate_line`: The horizontal baseline.
    *   `drop_contour`: The initial edge points of the drop.
    *   `needle_rect`: The location of the needle.

### 2. Manual Substrate Override logic
*   If the user provides `--substrate-y <value>`:
    *   The script **overrides** the auto-detected substrate line with the manual value.
    *   **CRITICAL**: It intentionally **skips injecting the auto-detected contour**. This forces the pipeline to re-run edge detection (`Canny`) and properly clip the new contour against the *manual* substrate line. This prevents "stale" contours (clipped at the old auto-detected Y) from being used.
*   If no override is provided (`substrate_y is None`):
    *   The script **injects** the auto-detected `drop_contour` directly into the pipeline context. This effectively disables the pipeline's internal edge detection and uses the pre-calibrated result, ensuring consistency with the wizard.
*   **Pipeline Configuration**:
    *   `auto_detect_features` is set to `False` to prevent the pipeline from ignoring the manual/injected data and running its own auto-detection from scratch.
    *   `px_per_mm` is set to a default (133.0) for metric calculation.

### 3. Running the Pipeline
*   The `SessilePipeline` is instantiated and `run(**pipeline_kwargs)` is called.
*   The pipeline stages execute in order:
    1.  **Acquisition**: Loads the image into the Context.
    2.  **Preprocessing**: Skipped (since we disabled auto-detect), preserving manual inputs.
    3.  **Edge Detection**: Runs only if no contour was injected (i.e., in manual override mode).
    4.  **Geometry**:
        *   Calculates the Apex.
        *   **Clips Contour**: Calls `clip_contour_to_substrate` to trim points below the substrate line.
        *   **Tail Trimming**: The refined logic removes "horizontal tails" where the contour incorrectly tracks the flat substrate surface.
    5.  **Scaling, Physics, Solver**: Fits the Young-Laplace model to the clean contour.

### 4. Visualization (`ResultWidget`)
*   A custom `QWidget` is created to display the results.
*   `paintEvent`:
    *   **Image**: Draws the original image scaled to the window.
    *   **Contour**: Draws the final processing contour in **Green**.
    *   **Baseline**: Draws the substrate line (manual or detected) in **Blue Dashed** line, with a text label for its Y-coordinate.
    *   **Apex**: Marks the drop apex with a **Red Cross** and coordinate label.
    *   **Contact Points**: Marks Left/Right contact points in **Magenta** with labels.
    *   **Metrics**: Displays calculated values (Contact Angle, Volume, etc.) in a semi-transparent overlay box.

## How to Use

**Basic Run (Auto-Detection):**
```bash
python scripts/test_sessile_qt.py "path/to/image.png"
```

**Manual Substrate Override:**
```bash
python scripts/test_sessile_qt.py "path/to/image.png" --substrate-y 300
```
This forces the baseline to Y=300 and re-detects the drop above that line.
