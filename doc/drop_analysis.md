# Drop Analysis Module

This document describes the workflow and algorithms used by the Drop Analysis tab.

## 1. Overview

The Drop Analysis tab provides automated measurement of pendant and contact-angle droplets.
It detects the needle, extracts the outer droplet contour, and reports geometric
metrics such as height, diameter and volume. The analysis is performed on
regions of interest drawn by the user on the main image canvas.

## 2. User Workflow

1. Load an image using the existing Open action.
2. Choose the analysis **Method**: `pendant` or `contact-angle`.
3. Press **Needle Region** and draw a blue rectangle around the needle.
4. Press **Detect Needle** to fit the needle axis. The length is calculated in
   pixels and combined with the known needle length in mm to obtain the
   pixel-to-mm scale.
5. Press **Drop Region** and draw a green rectangle enclosing the droplet.
6. Press **Analyze Image** to extract the external contour and overlay the
   symmetry axis, apex, and max diameter line. Results are displayed in the
   panel.

## 3. Algorithms

- **Needle Detection** – implemented in `analysis/needle.py`. The ROI is blurred,
  edges are detected with Canny, and vertical lines are fit to locate the needle
  sides. The horizontal distance between these lines yields the needle width.
- **Drop Contour and Metrics** – implemented in `analysis/drop.py`. The outer
  contour is obtained using morphological filtering and `cv2.findContours`.
  Metrics such as height, diameter, volume, contact angle and surface tension
  are computed assuming axial symmetry.

For additional examples, see `examples/pendant_demo.png`.

