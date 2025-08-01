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

## Using a Virtual Environment

Alternatively, you can create an isolated environment manually.

1. Create and activate a virtual environment in the project directory:
   ```bash
   python -m venv .venv
   # On Linux/macOS
   source .venv/bin/activate
   # On Windows
   # .venv\Scripts\activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Launch the program directly:
   ```bash
   python src/main.py
   ```

Both approaches will start the Menipy GUI (once fully implemented) and allow you to analyze droplet images.

## Interface Overview

When you launch Menipy, the main window displays an image view on the left and a set of controls on the right. The control area includes:

- **Zoom slider** – adjust the magnification of the loaded image.
- **Parameter panel** – enter air density, liquid density, and surface tension values.
- **Metrics panel** – shows calculated interfacial tension, Wo number, droplet volume, contact angle, height, and diameter.

Load an image using the File menu and press **Process** to apply segmentation and compute metrics. Contours and model fits are overlaid on the image. Choose **File → Save Annotated Image** to export the view with overlays.
