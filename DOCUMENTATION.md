# Menipy User Guide

Menipy is an alpha-stage toolkit for droplet and meniscus shape analysis from
images. This guide is the main user-facing reference for installing Menipy,
launching the GUI, running a first analysis, and using the `adsa` command-line
tool.

For a short project overview, see [README.md](README.md).

## Known Limitations

Menipy is not production-ready. Treat all measurements as experimental unless
you have independently validated the workflow, image acquisition setup,
calibration, and numerical method for your use case.

Practical caveats:

- Image quality, lighting, reflections, and region selection can strongly affect
  detected contours and fitted metrics.
- Auto-calibration is a convenience feature, not a guarantee of physical
  accuracy.
- Experimental pipelines such as `oscillating`, `capillary_rise`, and
  `captive_bubble` may expose CLI or GUI paths before they are fully validated.
- Scientific outputs should be checked against reference images, manual
  measurements, or trusted software before publication or operational use.

## Install

Menipy requires Python 3.10 or newer.

### Editable Install

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e .
```

For development and test tools, install the optional extras:

```bash
pip install -e ".[dev,test]"
```

## Launch the GUI

After installation, start Menipy with:

```bash
menipy
```

Fallback module entry point:

```bash
python -m menipy.gui.app
```

The GUI opens an image preview area, setup controls, pipeline controls, overlays,
and results panels.

## First GUI Workflow

1. Launch the GUI with `menipy`.
2. Load an image from the setup or file controls.
3. Select the intended analysis mode, usually sessile or pendant.
4. Configure calibration manually or use auto-calibration.
5. Run preview or analysis.
6. Inspect the overlay and metrics.
7. Export annotated images, overlays, or result files as needed.

For pendant and contact-angle workflows, see
[docs/guides/drop_analysis.md](docs/guides/drop_analysis.md).

## Calibration

Calibration converts image pixels to physical units. The GUI supports manual
calibration and automatic detection workflows.

### Manual Calibration

Use manual calibration when you know the physical length represented by a line
or region in the image.

1. Enable calibration mode in the setup controls.
2. Draw the required line or region on the image.
3. Enter the known physical length or needle diameter.
4. Apply calibration and confirm that the scale value is shown before running
   analysis.

### Auto-Calibration Wizard

Auto-calibration attempts to detect the ROI, needle, substrate, drop contour,
contact points, and apex depending on the selected pipeline.

1. Load an image.
2. Preview it in the GUI.
3. Click the Auto-Calibrate button in the calibration section.
4. Click Detect in the wizard.
5. Review the colored overlays and confidence indicators.
6. Toggle any detected regions that should not be applied.
7. Click Apply All to use the detected regions.

Overlay colors used by the calibration workflow:

| Region | Color |
| --- | --- |
| ROI | Yellow |
| Needle | Blue |
| Substrate | Magenta |
| Drop contour | Green |
| Contact points | Red |
| Apex | Red cross marker |

Auto-calibration currently uses pipeline-specific strategies:

- Sessile: CLAHE contrast enhancement, adaptive thresholding, substrate
  detection, drop contour extraction, and contact-point detection.
- Pendant: Otsu thresholding, needle shaft analysis, drop contour extraction,
  contact-point detection, and apex detection.

The core implementation is in `src/menipy/common/auto_calibrator.py`, with
coverage in `tests/test_auto_calibrator.py`.

## Preprocessing and Preview Overlays

Preprocessing options clean and transform the image before contour detection.
The "Fill Holes" option can fill small interior holes in the segmented ROI and
remove small artifacts near the substrate/contact line.

Typical controls:

- Enable: turns the Fill Holes stage on or off.
- Max hole area: fills holes smaller than the configured pixel area.
- Remove spurious near contact: removes small components near the detected
  contact line.
- Proximity: controls the pixel distance used for near-contact cleanup.

Preview overlays are drawn by the GUI overlay layer rather than baked into the
image pixels. This allows contours, contact points, and other visual markers to
be updated or hidden without modifying the underlying preview image.

The Geometry Configuration dialog can use the preprocessed image for preview so
that geometry and edge-detection previews match the image used by the pipeline.

## Run the CLI

The command-line interface is `adsa`.

```bash
adsa --help
```

Supported pipeline choices:

- `sessile`
- `pendant`
- `oscillating`
- `capillary_rise`
- `captive_bubble`

Sessile and pendant are the primary documented user workflows. Treat the other
pipelines as experimental unless validated for your application.

### Single Image

```bash
adsa --pipeline sessile --image "data/samples/prueba sesil 2.png" --auto-calibrate --out ./out
```

### Batch Directory

```bash
adsa --pipeline sessile --input-dir data/samples --glob "*.png" --auto-calibrate --out ./out
```

### Manual Geometry

```bash
adsa --pipeline sessile --image "data/samples/prueba sesil 2.png" --roi 50,40,500,420 --contact-line 80,420,520,420 --out ./out
```

### Common CLI Options

| Option | Purpose |
| --- | --- |
| `--pipeline` | Select the pipeline to run. |
| `--image` | Analyze one image file. |
| `--input-dir`, `-i` | Analyze matching images in a directory. |
| `--glob`, `-g` | Filter batch images with comma-separated glob patterns. |
| `--camera` | Acquire frames from a camera index. |
| `--frames` | Number of frames to acquire from a camera. |
| `--auto-calibrate`, `-a` | Enable automatic geometry detection. |
| `--roi` | Provide manual ROI as `x,y,w,h`. |
| `--needle` | Provide manual needle region as `x,y,w,h`. |
| `--contact-line`, `--baseline` | Provide manual baseline endpoints as `x1,y1,x2,y2`. |
| `--out`, `--output-dir`, `-o` | Set the output directory. |
| `--no-overlay` | Skip overlay image generation. |
| `--material`, `--fluid-name` | Look up fluid density from a materials database. |
| `--needle-name` | Look up needle outer diameter by name or gauge. |
| `--needle-diameter` | Override needle diameter in mm. |
| `--materials-db` | Use a custom materials SQLite database path. |
| `--sop`, `-s` | Load a Standard Operating Procedure profile or JSON file. |
| `--preprocessing-method` | Override preprocessing method. |
| `--edge-detection-method` | Override edge detection method. |
| `--plugins` | Set plugin scan directories. |
| `--db` | Set plugin SQLite database path. |
| `--activate` | Activate a plugin as `name:kind`. |
| `--deactivate` | Deactivate a plugin as `name:kind`. |

### CLI Outputs

Single-image and camera runs write outputs under the selected `--out`
directory. Typical outputs include preview images, overlay images, and JSON
results.

Batch directory mode uses collision-free per-image names and writes a combined
`results.csv` with common fields such as:

- `image_path`
- `pipeline`
- `qa_ok`
- pipeline-specific metric columns

## SOP Configuration

The CLI can load a Standard Operating Procedure with `--sop`. SOP files are JSON
documents that declare enabled stages and stage-specific parameters.

Example:

```json
{
  "include_stages": [
    "acquisition",
    "preprocessing",
    "feature_detection",
    "contour_extraction",
    "calibration",
    "geometric_features",
    "physics",
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

## Detection Plugins

Detection algorithms are available as modular plugins for pipelines and custom
scripts.

Common detector plugin files:

| Plugin | File | Registered names |
| --- | --- | --- |
| Needle | `plugins/detect_needle.py` | `sessile`, `pendant` |
| ROI | `plugins/detect_roi.py` | `sessile`, `pendant`, `auto` |
| Substrate | `plugins/detect_substrate.py` | `gradient`, `hough` |
| Drop | `plugins/detect_drop.py` | `sessile`, `pendant` |
| Apex | `plugins/detect_apex.py` | `sessile`, `pendant`, `auto` |

High-level helper:

```python
from menipy.common.detection_helpers import auto_detect_features

features = auto_detect_features(image, pipeline="sessile")
```

For the full plugin protocol, see
[docs/guides/developer_guide_plugins.md](docs/guides/developer_guide_plugins.md).

## GUI Architecture Notes

The active GUI package is organized under `src/menipy/gui/`:

- `views/`: widgets, image preview, results panel, and loaded `.ui` layouts.
- `controllers/`: user events, pipeline triggers, setup state, camera flow, and
  SOP coordination.
- `dialogs/`: calibration, preprocessing, physics, plugin, material, and
  settings dialogs.
- `services/` and `helpers/`: pipeline wrappers, settings, plugin services,
  logging bridge, image conversion, icons, and theme helpers.

Application logs are forwarded to the in-app Log tab. If icon resource warnings
appear for `:/icons/...`, rebuild GUI resources from the project root:

```bash
python tools/build_resources.py
```

The application also falls back to SVG files under
`src/menipy/gui/resources/icons/` when the compiled resource is unavailable.

## Where to Go Next

User guides:

- [docs/guides/drop_analysis.md](docs/guides/drop_analysis.md)
- [docs/guides/image_processing.md](docs/guides/image_processing.md)

Scientific background:

- [docs/guides/base_information.md](docs/guides/base_information.md)
- [docs/guides/physics_models.md](docs/guides/physics_models.md)
- [docs/guides/numerical_methods.md](docs/guides/numerical_methods.md)

Developer docs:

- [docs/guides/developer_guide_pipelines.md](docs/guides/developer_guide_pipelines.md)
- [docs/guides/developer_guide_plugins.md](docs/guides/developer_guide_plugins.md)
- [docs/guides/developer_guide_utilities.md](docs/guides/developer_guide_utilities.md)

Result contracts:

- [docs/contracts/sessile_results.md](docs/contracts/sessile_results.md)
- [docs/contracts/pendant_results.md](docs/contracts/pendant_results.md)
- [docs/contracts/oscillating_results.md](docs/contracts/oscillating_results.md)
- [docs/contracts/capillary_rise_results.md](docs/contracts/capillary_rise_results.md)
- [docs/contracts/captive_bubble_results.md](docs/contracts/captive_bubble_results.md)
- [docs/contracts/results_panel_integration.md](docs/contracts/results_panel_integration.md)

Planning and design references:

- [PLAN.md](PLAN.md)
- [docs/UnitPLAN.md](docs/UnitPLAN.md)
- [docs/ux_redesign2.md](docs/ux_redesign2.md)
- [docs/adsa_ui_mockup.md](docs/adsa_ui_mockup.md)
- [archive/2026-05-cleanup/](archive/2026-05-cleanup/)
