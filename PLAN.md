# Project Plan

A concise, step-by-step guideline for the CODEX agent to implement a Python-based GUI application for droplet shape analysis.\
Based on the “Development Plan for a Python-Based Droplet Shape Analysis Tool” fileciteturn0file0.

---

## Tech Stack

- **Language**: Python 3.9+
- **GUI**: PySide6
- **Image I/O & Processing**: OpenCV (cv2), scikit-image
- **Numerical Computing**: NumPy, SciPy
- **Data Handling**: pandas
- **Plotting & Overlays**: Matplotlib (optional), PySide6 QGraphicsView
- **Packaging & Testing**: setuptools, pytest
- **Optional (ML)**: TensorFlow / PyTorch for advanced segmentation

---

## Software Requirements

- **OS**: Windows, macOS, or Linux
- **Python** ≥ 3.9
- **Dependencies** listed in `requirements.txt`
- **Build**: `setup.py` with entry point `src/main.py`
- **Testing**: pytest with coverage reports

---

## Directory Structure

```
.
├── doc/                    # Reference docs (markdown) for CODEX to verify:
│   ├── droplet_description.md   # • Droplet definition for image detection
│   ├── physics_models.md   # • Young–Laplace, ADSA equations
│   ├── numerical_methods.md# • ODE integration, optimization
│   ├── image_processing.md # • thresholding, contour extraction
│   └── gui_design.md       # • layout, controls, dialogs
├── src/
│   ├── main.py             # Entry point
│   ├── gui/                # PySide6 widgets & views
│   ├── processing/         # Image I/O, segmentation, contour detection
│   ├── models/             # Geometric & physical model implementations
│   ├── utils/              # Calibration, I/O helpers
│   └── batch.py            # Batch‐mode orchestration
├── data/
│   └── samples/            # Sample images for tests
├── tests/                  # pytest suites
│   ├── test_processing.py
│   ├── test_models.py
│   └── test_gui.py
├── requirements.txt        # Pin exact versions
├── setup.py                # Package metadata & entry point
├── AGENTS.md                # Agents descriptions
└── PLAN.md                 # This plan
```

---

## CODEX Agent Tasks

1. **Load & Validate Documentation**

   - Read all `.md` files in `/doc/` to extract algorithmic details and required equations.

2. **Project Scaffold**

   - Create directory structure as above.
   - Generate `requirements.txt` and `setup.py` with placeholders.

3. **Environment Setup**

   - Initialize a Python 3.9+ virtual environment.
   - Install dependencies in `requirements.txt`.

4. **Image Preprocessing Module**

   <!-- Completed by Codex -->

   - Implement `processing/reader.py` (image loading).
   - Implement `processing/segmentation.py` with Otsu, adaptive, morphology.

5. **Geometric & Physical Models**

   <!-- Completed by Codex -->

   - In `models/geometry.py` implement circle‐fit, ellipse‐fit, polynomial tangent.
   - In `models/physics.py` implement Young–Laplace ODE solver and ADSA optimizer.

6. **Property Calculators**

   <!-- Completed by Codex -->

   - In `models/properties.py` compute:
     - Surface tension (γ) from fitted shape
     - Contact angle (θ) via circle/ellipse or half-angle
     - Volume via numerical integration and spherical‐cap formula
     - Height & diameter from contour

7. **GUI Skeleton**

   <!-- Completed by Codex -->

   - In `gui/main_window.py` build two‐panel layout: image view + controls using PySide6.
   - Stub controls: open file/folder, calibration dialog, model selector, “Process” button.

8. **Integration & Overlays**

   <!-- Completed by Codex -->

   - Hook processing modules into GUI actions.
   - Draw overlays (raw contour, model curve) using QGraphicsView or Matplotlib canvas.
   - Add “Save annotated image” feature.

9. **Calibration Features**

   <!-- Completed by Codex -->

   - Implement `utils/calibration.py` for pixel-to-mm entry and on-image line measurement dialog.

10. **Batch Mode**

    <!-- Completed by Codex -->

    - In `batch.py`, iterate over directory, apply default segmentation + model(s), aggregate results into a pandas DataFrame and export to CSV.

11. **Testing & Validation**

    <!-- Completed by Codex -->

    - Write pytest tests covering `processing/`, `models/`, and core GUI logic.
    - Include sample images in `data/samples/` for CI.

12. **Packaging & CI**

    <!-- Completed by Codex -->

    - Finalize `setup.py` entry point (`src.main:main`).
    - Configure GitHub Actions (or equivalent) to run tests on push.

13. **Optional ML Plugin**

    <!-- Completed by Codex -->

    - Scaffold a toggle in GUI to enable ML-based segmentation (TensorFlow/PyTorch).
    - Leave as future extension.

14. **UI Enhancements & Parameter Panel**
14.1 **Project review**
    - Review the documentation folder /doc/
    - Analyze if /src/models/geometry.py, /src/models/physics.py and /src/models/properties.py need to be modified
    - Analyze if image processing need to be updated or exended
14.2 **Zoom Control**
    <!-- Completed by Codex -->
    - In `gui/controls.py`, implement a **zoom slider** that adjusts the scale of the image view (QGraphicsView or Matplotlib canvas) and its overlays.
14.3 **Parameter Panel**
    <!-- Completed by Codex -->
    - Add a **Parameter Panel** in `gui/main_window.py` or `gui/controls.py` for user inputs:
      - Air density
      - Liquid density
      - Surface tension
14.4 **Metrics Panel**
    <!-- Completed by Codex -->
    - Add a **Metrics Panel** in `gui/main_window.py` or `gui/controls.py` for calculated values:
    - Extend the **Metrics Panel** to display calculated values:
      - Interfacial tension (IFT) in mN/m
      - Wo number
      - Volume (µL)
      - Contact angle (θ)
      - Height & diameter
14.5 **Update Documentation**
    <!-- Completed by Codex -->
    - Update the DOCUMENTATION.m.

15 **Interactive Calibration Box**
    <!-- Completed by Codex -->
    - UI Mode Toggle
      -Add a “Calibration Mode” checkbox in the parameter panel.
      -When enabled, left-click+drag on the image to draw the blue “needle box.”
    - Data Capture
      -Store the box as pixel coordinates 𝑥1,𝑦1,𝑥2,𝑦2 for downstream processing.
    - Visual Feedback
      -Immediately render the blue rectangle in the QGraphicsView overlay, editable by drag handles.

16 **Pixel-to-mm Calibration**
    <!-- Completed by Codex -->
    -Manual Mode
      -In Calibration Mode, allow drawing a line between two points inside the blue box.
      -Let the user enter the real length (mm) in the parameter panel.
      -Compute scale = length_mm / pixel_distance.
   -Automatic Mode
      -Run edge detection (e.g. Canny + vertical Hough lines) within the blue box.
      -Identify the two roughly parallel vertical needle edges and measure their pixel separation.
      -Derive scale automatically and display it in the panel.
    -UI Switch
      -Radio buttons to switch between Manual / Automatic calibration.

17 **Region-of-Interest for Volume**
   <!-- Completed by Codex -->
   -ROI Drawing
      -After calibration, enable an “ROI Mode” to draw a green box around the droplet’s shadow.
      -Store ROI coordinates and overlay in the scene.
   -Constrained Contour Extraction
      -Restrict all image-processing (thresholding, contour find) to pixels inside the green ROI.
   -Volume Integration
      -Compute the solid-of-revolution volume using only the contour segments within that ROI, applying the calibrated scale.
18 **Apex & Contact-Point Marking**
   <!-- Completed by Codex -->
   -Apex Detection
      -Detect the highest point on the extracted droplet contour.
      -Mark it with a yellow QGraphicsEllipseItem.
   -Contact-Point Detection
      -For pendant drops, find where the contour meets the top of the blue box; for sessile, where it meets the substrate baseline.
      -Draw a light-blue horizontal QGraphicsLineItem at the contact location.
  -Live Updates
      -Any time the contour is re-computed (e.g. after ROI change), update these markers.

19 **Improvement 1: External Contour Only**
   <!-- Completed by Codex -->
   -Detect only the external droplet contour
     -Discard all internal contours (e.g., bright interior region) before volume and fitting routines.

20 **Missing Feature 1: “Calculate” & “Draw” Buttons**
   -Calculate
     -Compute surface tension for pendant drops or contact angle for sessile drops.
     -For pendant: find γ such that the curvature at the apex (red line) produces the observed max-diameter radius (blue line).
     -Allow each fitting method to return slightly different γ/θ values.
   -Draw
     -Use the Young–Laplace equation (or chosen approximation) to generate the predicted droplet profile from the apex outward.
     -Base the model on either user-entered fluid properties or the values computed by Calculate.
   -UI Cleanup
     -Remove the standalone “Calibration” button—calibration workflows remain in the parameter panel.

21 *Missing Feature 2: CSV Export**
   -Add a “Save CSV” button
     -Exports all user parameters and computed results (γ, θ, volume, dimensions) in comma-separated format.

22 *Missing Feature 3: Save Annotated Image**
   -Add a “Save Image” button
     -Open a file-save dialog for naming.
     -Save the current view—including calibration box, ROI, apex & contact markers, and model overlay—as a single image file.

23 **Propose enhancements**
   -Evaluate adding real-time metric updates when parameters change.
   -Investigate integrating true ML-based segmentation models.
   -Consider 3D droplet reconstruction for pendant drops.
   -Provide a command-line interface for batch operations.
   -Expand unit tests to cover new functionality as it is added.

*End of CODEX agent plan.*
