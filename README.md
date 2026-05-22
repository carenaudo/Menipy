# Menipy

Menipy is a Python-based toolkit designed for droplet and meniscus shape analysis from images. It provides a modern, unified PySide6 graphical interface alongside a highly robust command-line interface with full feature parity.

Development of Menipy is AI-driven, utilizing specialized agent workflows with minimal human coding intervention.

## Objectives and Scope

- Provide a clear, extensible skeleton for droplet and meniscus shape analysis from images.
- Structure the work as pipelines where each step represents a concrete stage of image analysis (loading, preprocessing, segmentation, contour extraction, geometry fitting, metrics, and reporting).
- Grow a toolbox of measurements and utilities, expanding beyond the most common metrics typically implemented.

## GUI and CLI Invocation

Menipy provides registered console scripts for standard execution (once installed via `pip install -e .`):

- **Launch the GUI**: Run `menipy` (or fallback: `python -m menipy.gui.app`)
- **Run the CLI**: Run `adsa` (or fallbacks: `menipy-cli` / `python -m menipy.cli`)

## Origins: “Vibe Coding” Prototype

The original intention of this repository was to explore the feasibility of prototyping applications using primarily “vibe coding” (AI-assisted development with minimal manual coding). You can find the original branch demonstrating this approach here:

- Min-Human-Interaction branch: https://github.com/carenaudo/Menipy/tree/Min-Human-Interaction

## Maintainer

- Maintainer: Carlos Renaudo (PLAPIQUI — Planta Piloto de Ingeniería Química, Universidad Nacional del Sur/CONICET)
- Group: Tecnología de Partículas, PLAPIQUI

The implementation roadmap is tracked in `PLAN.md`, while active reference material lives under `docs/guides/` and `docs/contracts/`.

<img width="1909" height="1028" alt="image" src="https://github.com/user-attachments/assets/45fd8f53-98cc-4c6d-bf59-0e0687c5c7fe" />

# ​​ Disclaimer

**This software is under development and is *not production-ready***. It is *not intended to replace any commercial or non-commercial tools or software*. The values reported by this software **may be incorrect or inaccurate**, and **the developers assume no responsibility** whatsoever for how the software is used or for any consequences arising from its use.

Use at your own risk.

## Repository Overview

- **PLAN.md** – a high-level roadmap for the current implementation phases.
- **docs/guides/** – user and developer guides (workflows, methods, and feature usage).
- **docs/contracts/** – pipeline results schemas and results panel integration contracts.
- **docs/history/** – historical planning logs and archived process notes.
- **CHANGELOG.md** – A log of notable changes for each version.


For developers: see [Developer Guide: Plugins](docs/guides/developer_guide_plugins.md) for up-to-date instructions on writing and migrating plugins for all supported types (edge, solver, acquisition, optimizer, physics, output, overlayer, scaler, validator, and more).

## Human Interaction

Human involvement in this project is intentionally kept light. The main role of
the user will be testing the GUI and submitting prompts as tasks to CODEX or as
issues on GitHub. Feedback from these manual tests will guide further automated
iterations of the tool.

## Calibration Workflow

1. Enable **Calibration Mode** in the parameter panel.
2. Toggle **Manual Calibration** on to draw a line between two points, or off to
   draw the needle region for automatic detection.
3. Click **Calibrate** to compute the pixel-to-mm scale. The resulting value is
   shown in the parameter panel.
   Automatic mode detects the two vertical edges of the needle using Canny edge
   detection and a Hough transform to measure their separation.

## Auto-Calibration Wizard

Menipy includes a **1-click auto-calibration wizard** for automatic detection of ROI, needle, and drop regions.

### How to Use

1. Load an image using the Browse button
2. Click **Preview** to display the image
3. Click the **🎯 Auto-Calibrate** button in the Calibration section
4. In the wizard dialog:
   - Click **Detect** to run automatic detection
   - Review detected regions (shown with colored overlays)
   - Toggle region checkboxes to enable/disable specific detections
   - Click **Apply All** to accept results

### Automatic Detection Features

| Region | Sessile Drop | Pendant Drop |
|--------|-------------|--------------|
| **ROI** | Bounding box around drop + substrate | Bounding box from needle to apex |
| **Needle** | Contour touching top border | Shaft lines at top of contour |
| **Substrate** | Gradient-based detection in margins | N/A |
| **Drop Contour** | Largest centered contour with convex hull | Largest centered contour |
| **Contact Points** | Where drop meets substrate | Where drop deviates from needle shaft |
| **Apex** | N/A | Bottom of drop (max Y) |

### Detection Algorithms

- **Sessile Pipeline**: Uses CLAHE contrast enhancement + adaptive thresholding for robust detection across varied lighting conditions
- **Pendant Pipeline**: Uses Otsu thresholding (optimal for high-contrast silhouettes) + shaft line analysis for needle detection

The wizard automatically selects the appropriate detection strategy based on the selected pipeline.

## Command-Line Interface (CLI)

Menipy features a robust, consolidated command-line tool `adsa` designed to process droplets in headless or automated environments (e.g., Docker, CI/CD pipelines) with full feature-parity matching the GUI.

### Key Capabilities

1. **Auto-Calibration Fallback**: If coordinates are omitted, the CLI automatically detects the ROI, needle, and substrate on-the-fly.
2. **Materials SQLite DB Lookup**: Seamlessly queries fluid densities and outer needle gauge/diameters from the `menipy_materials.sqlite` database.
3. **Standard Operating Procedure (SOP) Loading**: Load SOP parameters from active JSON configuration files.
4. **Directory Batch Processing**: Scan directories via glob filters, generate collision-free visuals (`_preview.png`, `_overlay.png`), and compile measurements into a unified `results.csv`.

### Example CLI Commands

- **Single Image Analysis (with Auto-Calibration and Material/Needle DB Lookup)**:
  ```bash
  adsa --pipeline sessile --image data/samples/prueba_sesil_2.png --auto-calibrate --material "Water (25°C)" --needle-name 22G --out ./single_out
  ```
- **Batch Processing of a Directory**:
  ```bash
  adsa --pipeline sessile --input-dir data/samples --glob "*.png" --auto-calibrate --out ./batch_out
  ```

## Image Processing

Segmented masks are cleaned with morphological operations and reduced to the largest external contour. This ensures internal artifacts do not affect volume or fitting calculations.

## Drop Analysis

The GUI includes a **Drop Analysis** tab for pendant and contact-angle modes. After defining
needle and drop regions, the application detects the needle width, extracts the
outer drop contour and overlays key features such as the apex and symmetry axis.
Metrics including height, diameter, volume and surface tension are displayed in
the panel. Detailed instructions are available in `docs/guides/drop_analysis.md`.

### Science-Mode Pipeline Step Testing

For method development and diagnostic review, choose **View -> Focus -> Science**
in the GUI. A **Test** button appears next to the analysis-type buttons. It
opens a left-rail Step Test panel where individual pipeline stages can be run
with their prerequisites, excluding `acquisition`. Test runs use the selected
source, silently auto-detect calibration prerequisites, and keep configuration
changes sandboxed until **Apply** is clicked.

## Developer note: runtime resources and GUI logging

During recent refactor work, the application now emits clear per-stage pipeline logs and a status message that is shown in the GUI status bar. If you run the GUI and see missing icon warnings for paths like `:/icons/...`, run the resource build helper:

- In the project root, run the build script to produce `icons.rcc` (requires PySide6 tools):
   ```bash
   python tools/build_resources.py
   ```

After producing `icons.rcc`, the application will register it automatically at startup and QIcon lookups for `:/icons/...` should resolve. If you prefer a quick workaround, the GUI will fall back to the on-disk SVG files under `src/menipy/gui/resources/icons/` when the compiled resource is not available.

Logs: the application forwards Python logging to the in-app Log tab. Look there for per-stage messages like:
```
[pipeline:<name> ctx=0x...] Completed stage: <stage> (<ms> ms) | frames=N preview=M contour=K results_keys=L
```

## Developer Documentation

For developers looking to extend Menipy's functionality, we provide guides on common tasks:

- [**Adding a New Analysis Pipeline**](docs/guides/developer_guide_pipelines.md): A step-by-step guide on how to create and integrate a new analysis workflow into the application.
- [**Adding a New Image Filter Plugin**](docs/guides/developer_guide_plugins.md): A guide on creating and integrating custom image filters using the plugin system.
- [**Creating Utility Plugins**](docs/guides/developer_guide_utilities.md): How to create utility plugins for image testing and analysis (accessible via Utilities menu).

## Contributing

We welcome contributions! Please see our [**Contributing Guide**](CONTRIBUTING.md) for:
- Setting up your development environment
- Running tests and linting
- Code style guidelines
- How to submit pull requests
