# Menipy User Guide

Menipy is a Python toolkit for droplet shape analysis. It provides image processing algorithms and a PySide6 GUI to estimate properties such as surface tension and contact angles from droplet images.

This guide explains how to set up the environment and launch the application.

## Using `setup.py`

1. Ensure Python 3.9 or newer is installed.
2. Install Menipy and its dependencies in editable mode:
   ```bash
   pip install -e .
   ```
   This command uses `setup.py` to register the `menipy` console script.
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

- From source without installing:
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

