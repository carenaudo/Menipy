# Menipy User Guide

Menipy is a Python toolkit for droplet shape analysis. It provides image processing algorithms and a PySide6 GUI to estimate properties such as surface tension and contact angles from droplet images.

This guide explains how to set up the environment and launch the application.

## Package Installation (Recommended)

1. Ensure Python 3.10 or newer is installed.
2. Install Menipy and its dependencies in editable mode:
   ```bash
   pip install -e .
   ```
   This command installs Menipy from `pyproject.toml` and registers the `menipy` console script.
3. Run the application from the command line:
   ```bash
   menipy
   ```

## Quick start

If you just want to run Menipy quickly, use one of these options depending on whether
you installed the package or are running from the source tree.

- Installed (recommended after `pip install -e .` or a normal install):
   ```bash
   menipy
   ```

- From source without installing (development fallback):
   ```bash
   python -m src
   ```

## Using a Virtual Environment

Alternatively, you can create an isolated environment manually.

1. Create and activate a virtual environment in the project directory:
   ```bash
   python -m venv .venv
   # On Linux/macOS
   source .venv/bin/activate
   # On Windows (cmd.exe)
   .venv\Scripts\activate.bat
   # On Windows (PowerShell)
   # Run this in PowerShell:  . .venv\Scripts\Activate.ps1
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Launch the program directly:
   ```bash
   # If you have not installed the package, run the package entry module from source
   python -m src
   ```

Both approaches will start the Menipy GUI (once fully implemented) and allow you to analyze droplet images.


## Developer Plugins

See [Developer Guide: Plugins](docs/guides/developer_guide_plugins.md) for the full plugin protocol, migration guide, and examples for all plugin types.

## Interface Overview

When you launch Menipy, the main window displays an image view on the left and a set of controls on the right. The control area includes:

- **Zoom slider** – adjust the magnification of the loaded image.
- **Parameter panel** – enter air density, liquid density, and surface tension values.
- **Metrics panel** – shows calculated interfacial tension, Wo number, droplet volume, contact angle, height, and diameter.

Load an image using the File menu and press **Process** to apply segmentation and compute metrics. Contours and model fits are overlaid on the image. Choose **File → Save Annotated Image** to export the view with overlays.

## Preprocessing: "Fill Holes"

Purpose

- The "Fill Holes" preprocessing option fills small interior holes in the segmented region-of-interest (ROI) mask and removes spurious contour points that lie very close to the substrate baseline/contact line. This yields cleaner contour geometry and improves the robustness of contact-point detection and subsequent model fits (polynomial, circle, and Young–Laplace).

Where to find it

- Open the Preprocessing Configuration dialog from the detection / preprocessing panel. A new page named "Fill Holes" appears in the left navigation list.

Controls and defaults

- Enable (checkbox): turns the Fill Holes stage on or off. Default: off.
- Max hole area (`max_hole_area`): integer pixel area. Holes smaller than this value will be filled. Default: 500 px.
- Remove spurious near contact (`remove_spurious_near_contact`): when enabled, the algorithm will remove small connected components that overlap (or are within `proximity_px`) of the detected contact-line mask. Default: true.
- Proximity (`proximity_px`): number of pixels around the contact line used to consider a small component spurious. Default: 5 px.

Implementation notes

- The implementation prefers `skimage.morphology.remove_small_holes` and `remove_small_objects` when scikit-image is installed. If scikit-image is not available, an OpenCV-based fallback is used (contour filling + connected components filtering).
- The stage runs immediately after ROI cropping in the preprocessing pipeline, so all downstream stages (grayscale conversion, filtering, contact-line detection, edge finding) operate on the cleaned mask/image.

When to enable

- Enable this option when the automatic segmentation produces ROI masks with interior holes or noisy contour artifacts near the substrate. It is particularly useful for sessile-drop images where reflections, lighting variation, or imaging artifacts create small false holes inside the droplet mask.

Troubleshooting

- If enabling the stage unexpectedly removes legitimate small features, increase `max_hole_area` or temporarily disable `remove_spurious_near_contact` to test behavior. If results differ between environments, ensure `skimage` is installed for the preferred morphological implementation.

---

## Preview overlays and geometry preview improvements

- The edge-detection stage no longer bakes overlay graphics (contours and contact-point circles) into preview images. Instead, controllers emit the raw preview image plus overlay metadata (fields named `contour_xy` and `contact_points`). The main preview UI draws overlays using the `ImageView` overlay API. This makes overlays easier to update, remove, and style without modifying the image pixels.

- The Geometry Configuration dialog has a new checkbox labeled "Use preprocessed image for preview". When enabled, the dialog requests a one-shot preprocessed image and uses it as the source for geometry/edge-detection previews. This lets you inspect how the chosen geometry detector behaves on the same preprocessed image that the pipeline will use during analysis.

These changes improve UX (faster overlay updates) and make it straightforward to export the raw image separately from drawn overlays.

---

## Auto-Calibration Wizard

Menipy includes a **1-click auto-calibration wizard** for automatic detection of ROI, needle, substrate, and drop regions.

### Accessing the Wizard

1. Load an image using the Browse button
2. Click **Preview** to display the image
3. Click the **🎯 Auto-Calibrate** button in the Calibration section of the Setup Panel

### Wizard Controls

| Control | Description |
|---------|-------------|
| **Detect** | Runs automatic detection on the current image |
| **Region Checkboxes** | Enable/disable specific regions (ROI, Needle, Substrate, Drop) |
| **Confidence Score** | Shows overall detection confidence (0-100%) |
| **Apply All** | Accepts results and draws overlays on preview |
| **Cancel** | Closes wizard without applying changes |

### Pipeline-Specific Detection

The wizard automatically selects the appropriate detection strategy based on the selected pipeline:

#### Sessile Drop Detection
- **Segmentation**: CLAHE contrast enhancement + adaptive thresholding
- **Substrate**: Gradient-based detection in left/right image margins
- **Needle**: Contour touching top border of image
- **Drop**: Largest centered contour with convex hull smoothing
- **Contact Points**: Where drop meets substrate baseline
- **ROI**: Bounding box around drop + substrate

#### Pendant Drop Detection
- **Segmentation**: Otsu thresholding (optimal for high-contrast silhouettes)
- **Needle**: Shaft line analysis at top of contour, walks down to find deviation
- **Drop**: Largest centered contour (min 5% of image area)
- **Contact Points**: Where drop contour deviates from needle shaft
- **Apex**: Bottom of drop (maximum Y coordinate)
- **ROI**: Bounding box from needle to apex

### Overlay Colors

When calibration is applied, detected regions are drawn as overlays:

| Region | Color |
|--------|-------|
| ROI | Yellow |
| Needle | Blue |
| Substrate | Magenta |
| Drop Contour | Green |
| Contact Points | Red |
| Apex | Red (cross marker) |

### Technical Details

The auto-calibration module is located at `src/menipy/common/auto_calibrator.py` and provides:

- `AutoCalibrator` class - Main detection engine
- `CalibrationResult` dataclass - Container for detection results
- `run_auto_calibration()` - Convenience function for one-shot detection

Unit tests are available in `tests/test_auto_calibrator.py` (25 tests covering sessile and pendant detection).

---

## Detection Plugins

The detection algorithms are also available as modular plugins for use in pipelines or custom scripts.

### Available Plugins

| Plugin | File | Registered Names |
|--------|------|------------------|
| Needle | `plugins/detect_needle.py` | `sessile`, `pendant` |
| ROI | `plugins/detect_roi.py` | `sessile`, `pendant`, `auto` |
| Substrate | `plugins/detect_substrate.py` | `gradient`, `hough` |
| Drop | `plugins/detect_drop.py` | `sessile`, `pendant` |
| Apex | `plugins/detect_apex.py` | `sessile`, `pendant`, `auto` |

### Quick Usage

```python
import sys
sys.path.insert(0, 'plugins')
import detect_needle

from menipy.common.registry import NEEDLE_DETECTORS

# Detect needle
needle_rect = NEEDLE_DETECTORS['sessile'](image)
# Returns: (x, y, width, height)
```

### High-Level Helper

For convenience, use `auto_detect_features()` to run all detections at once:

```python
from menipy.common.detection_helpers import auto_detect_features

# Detect all features for sessile drop
features = auto_detect_features(image, pipeline="sessile")

# features dict contains:
# - substrate_line: ((x1, y1), (x2, y2))
# - needle_rect: (x, y, w, h)
# - drop_contour: Nx2 array
# - contact_points: ((left), (right))
# - apex_point: (x, y)
# - roi_rect: (x, y, w, h)
```

### Pipeline Integration

Detection plugins are automatically invoked during the preprocessing stage of sessile and pendant pipelines. Detected features are stored in the pipeline context:

- `ctx.substrate_line` - Detected substrate baseline
- `ctx.needle_rect` - Detected needle bounding box
- `ctx.detected_contour` - Detected drop contour
- `ctx.contact_points` - Detected contact points
- `ctx.apex_point` - Detected apex
- `ctx.detected_roi` - Detected ROI

To disable auto-detection, set `ctx.auto_detect_features = False` before running the pipeline.

### Example Script

See `examples/detection_plugins_example.py` for a complete working example demonstrating:
- Using individual detector plugins
- Using the `auto_detect_features()` helper
- Visualizing detection results

---

## GUI Architecture Reorganization

In order to keep the codebase highly maintainable, modern, and modular, Menipy's PySide6 GUI was refactored. The legacy, redundant files under the obsolete `panels/` package and old stacked workflow structures were decommissioned and removed. 

The active panels and views are now consolidated directly under `src/menipy/gui/views/`:
* **`src/menipy/gui/views/preview_panel.py`**: Houses the live image view, overlays manager, and scene coordinates rendering.
* **`src/menipy/gui/views/results_panel.py`**: Renders metric tables, history records, and exports.
* **`src/menipy/gui/views/main_window.py`**: Wireframe container for the visual widgets.

The logical packaging of `src/menipy/gui/` consists of:
* **`views/`**: Renders widgets, scene overlays, and loaded `.ui` layouts.
* **`controllers/`**: Coordinates pipeline triggers, SOP states, camera streaming, and user events (e.g. `main_controller.py`, `setup_panel_controller.py`).
* **`dialogs/`**: Houses modals and multi-step configurations (e.g. `advanced_workflow_dialog.py`, `calibration_wizard_dialog.py`).
* **`services/` & `helpers/`**: Deals with database connectivity, pipeline invocation wrappers, logging bridges, and color themes.

---

## Consolidated Command-Line Interface (`adsa`)

The Menipy toolkit features a headless-friendly CLI designed with full feature parity to the GUI. The CLI is registered under the console command `adsa` (and `menipy-cli`).

### Command Usage Overview

```bash
adsa [options]
```

### Argument & Flag Options

#### 1. Pipeline & Input Control
* `--pipeline`: Droplet shape analysis pipeline to run. Choices: `sessile`, `oscillating`, `capillary_rise`, `pendant`, `captive_bubble` (default: `sessile`).
* `--image`: Path to an individual input image file.
* `--camera`: Camera index (e.g., `0`) to stream frame acquisition in real-time.
* `--frames`: Number of frames to acquire when capturing from a `--camera` source (default: `1`).
* `--input-dir`, `-i`: Directory path containing images for batch processing.
* `--glob`, `-g`: Comma-separated list of glob patterns to filter images inside `--input-dir` (default: `*.png,*.jpg,*.jpeg`).

#### 2. Geometry & Auto-Calibration
* `--auto-calibrate`, `-a`: Enable automatic calibration. The CLI runs the baseline `AutoCalibrator` engine on-the-fly to detect the ROI, needle, and substrate if coordinates are not provided.
* `--roi`: Define a manual ROI bounding box as `x,y,w,h` integers.
* `--needle`: Define a manual needle bounding box as `x,y,w,h` integers.
* `--contact-line` (or `--baseline`): Define manual baseline endpoints as `x1,y1,x2,y2` integers.

#### 3. Output Management
* `--out` (or `--output-dir`, `-o`): Target output folder path (default: `./out`).
* `--no-overlay`: Skip overlay drawing stages.
* **Outputs Generated**:
  * **Single Image / Camera Mode**: Generates `preview.png` (preprocessed image), `overlay.png` (with colorized vector annotations), and `results.json` (raw metrics and logs).
  * **Batch Directory Mode**: Collision-free naming is used (`<basename>_preview.png`, `<basename>_overlay.png`, and `<basename>_results.json`). Additionally compiles all metrics into a consolidated CSV sheet at `<out>/results.csv`.

#### 4. SQLite Materials & Needle Database
* `--material` (or `--fluid-name`): Queries the materials database (`menipy_materials.sqlite`) to fetch fluid density (`rho1`).
* `--needle-name`: Queries the needle database to extract outer diameter (in mm) by gauge (e.g. `22G`) or name.
* `--needle-diameter`: Directly override the outer needle diameter in mm (defaults to `0.72` mm).
* `--materials-db`: Specify a custom materials SQLite database path.

#### 5. Standard Operating Procedure (SOP) Loading
* `--sop`, `-s`: Specify a profile name in the local database or a custom JSON SOP file path. It automatically loads prep, edge, and optimizer stages to standard values.

#### 6. Algorithmic Overrides
* `--preprocessing-method`: Override the preprocessing stage (e.g. `blur`, `clahe`).
* `--edge-detection-method`: Override the edge detector stage (e.g. `canny`, `sobel`).

#### 7. Extensible SQLite Plugin Management
* `--plugins`: Plugin search directory path.
* `--db`: SQLite database plugin state path (default: `adsa_plugins.sqlite`).
* `--activate`: Activate a plugin dynamically using `<name>:<kind>`.
* `--deactivate`: Deactivate a plugin dynamically using `<name>:<kind>`.
* **Subcommands**:
  * `adsa plugins set-dirs <dirs>`: Update scan paths in the active plugins database.

---

### Batch Mode Output Schema (`results.csv`)

When running in batch mode (`--input-dir`), all metric fields generated dynamically by the active pipeline stages are parsed, collected, and exported into `results.csv`. This sheet includes:
* `image_path`: Location of the source image.
* `pipeline`: The analysis pipeline utilized.
* `qa_ok`: Quality check boolean flag from the pipeline's validation stage.
* Dynamic pipeline-specific headers (e.g. `contact_angle_left`, `contact_angle_right`, `droplet_volume`, `interfacial_tension`, etc.).

---

### SOP Configuration Schema

A JSON SOP configuration profile consists of the active stages list and specific stage parameter maps:

```json
{
  "include_stages": [
    "acquisition",
    "preprocessing",
    "feature_detection",
    "contour_extraction",
    "contour_refinement",
    "calibration",
    "geometric_features",
    "physics",
    "profile_fitting",
    "compute_metrics",
    "overlay",
    "validation"
  ],
  "params": {
    "preprocessing": {
      "method": "clahe",
      "clip_limit": 2.0,
      "tile_grid_size": [8, 8]
    },
    "contour_extraction": {
      "method": "canny",
      "threshold1": 50,
      "threshold2": 150
    }
  }
}
```

---

### Headless & CI/CD Portability

The consolidated CLI is completely isolated from GUI dependencies. It uses a custom `NumpyEncoder` to serialize NumPy floating/integer arrays directly to standard JSON without requiring a display thread or crashing on server environments without visual desktops (e.g. Docker, head-free Linux virtual machines).
