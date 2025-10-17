# GUI Design

This document outlines the user interface architecture and components using PySide6.

## 1. Framework Choice

- **PySide6**  
  Official Qt for Python bindings with up-to-date support. Use `QtWidgets`, `QtCore`, and `QtGui` modules for the main window, graphics views, and event handling. QGraphicsView provides an interactive canvas for overlays, and `QtChart` or embedding Matplotlib can be used for plots.

## 2. Main Window Layout

- **Two-Panel Design**  
  - **Left**: Image display area with zoom/pan using `QGraphicsView` inside a `QSplitter`.  
  - **Right/Bottom**: Controls and outputs inside `QWidget` containers with `QVBoxLayout` or `QFormLayout`.
- **Controls**  
  - File/Open (single) via `QAction` and `QFileDialog`  
  - Folder/Open (batch) via `QAction`  
  - Segmentation settings (threshold sliders with `QSlider`, algorithm dropdown with `QComboBox`)  
  - Calibration input (draw pixel-to-mm line with mouse events on `QGraphicsScene` or manual entry via `QLineEdit`)  
  - Model selection (checkboxes for Young–Laplace, Circle, Ellipse, Polynomial using `QCheckBox`)  
  - “Process” button (`QPushButton`)

### Configuration Dialogs

Menipy utilizes dedicated dialogs for configuring various pipeline stages, accessible via the setup panel. These dialogs provide granular control over the parameters of each stage.

-   **Preprocessing Configuration Dialog**: Allows users to adjust settings for image cropping, resizing, filtering, background subtraction, and normalization. It features a stacked widget interface with a navigation list for different categories of settings.
-   **Edge Detection Configuration Dialog**: Provides extensive controls for the edge detection stage. Users can select from various algorithms (Canny, Threshold, Sobel, Scharr, Laplacian) and fine-tune their parameters. It also includes options for Gaussian blur preprocessing, contour refinement (minimum/maximum length), and specific controls for detecting fluid-droplet and solid-droplet interfaces. The dialog is structured with a stacked widget for easy navigation between general settings, algorithm-specific parameters, and interface detection options.

## 3. Image Display & Overlays

- Use `QGraphicsScene` with `QGraphicsPixmapItem` for the image and `QGraphicsPathItem` for contours.  
- Contour (solid line) and model fit (dashed line) are drawn using `QPen` settings.  
- Provide checkboxes or actions to toggle overlay visibility in a toolbar.
- **Predicted Y–L Profile Overlay**  
  - Once the physics model returns the optimal γ and ΔP, generate its (r(z)) curve.  
  - Convert model coordinates into scene coordinates and draw via a second `QGraphicsPathItem` (dashed).  
  - Use distinct pen style (e.g. width 2, dashed) so users can compare measured contour vs. theory.

## 4. Output Panel

- Display numerical results with units in `QTableWidget` or `QTreeView`:
  - Surface tension (mN/m)  
  - Contact angle (°)  
  - Volume (µL)  
  - Height & diameter (mm)
- For batch mode, show a `QProgressBar` and use a `QPlainTextEdit` or `QListWidget` for log messages and error flags per image.

## 5. Batch Mode

- Progress bar updates via signals and slots.  
- Option to skip overlay rendering for speed, controlled by a `QCheckBox`.

## 6. Saving & Export

- Save annotated images using `QImage.save()` combined with drawn overlays.  
- Export results table and logs to CSV using Python’s `csv` module or `pandas.DataFrame.to_csv()`.

## 7. Future Enhancements

- Real-time parameter adjustment sliders that emit signals to trigger live overlay updates.  
- Optional ML segmentation toggle integrating with an external service or local model, controlled via `QAction` in the menu.

---

*Note: All Qt object names should follow a consistent naming convention (e.g., `mainWindow`, `graphicsView`, `processButton`).*