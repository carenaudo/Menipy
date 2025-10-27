# DRY Plan — Preferred Packages by Task

Goal: Map Menipy’s pipeline tasks to mature, well-supported Python packages to avoid reinventing the wheel, with emphasis on numerical stages. Each item lists recommended libraries and typical use in this codebase.

## Image I/O, Preprocessing, Segmentation
- scikit-image: filters (denoise, gaussian, threshold), morphology (remove_small_holes, remove_small_objects), measure (label), transform (Hough, rescale), exposure; robust, well-documented.
- OpenCV (opencv-python): fast I/O (imread/imwrite), Canny, dilation/erosion, contour finding; excellent performance and broad algorithm coverage.
- imageio + tifffile: broad image format support (TIFF stacks, 16-bit microscopy) when needed beyond OpenCV.

Use: edge detection and mask cleanup before contour extraction. Prefer skimage morphology; fall back to OpenCV when needed.

## Contours, Geometry, Fitting
- scikit-image.measure: regionprops, find_contours (on binary masks), RANSAC, CircleModel/EllipseModel for robust circle/ellipse fits near apex or footprint.
- NumPy: vectorized geometry, distances, projections; the baseline for array math.
- SciPy (scipy.spatial, scipy.optimize): distance metrics; optimize least_squares for param fits; KDTree/ConvexHull if needed.
- Shapely (optional): robust 2D computational geometry (segment/line intersections, buffers, distances) when precision and edge cases matter.

Use: fitting local circle at apex (pendant), footprint circle/ellipse (sessile), robust contact-line intersections, and contour simplification via skimage.measure.approximate_polygon when needed.

## Smoothing, Curvature, Interpolation
- SciPy.signal: savgol_filter for curvature-friendly smoothing; detrend; median filters.
- SciPy.interpolate: UnivariateSpline/LsqUnivariateSpline for smooth derivatives; parametric spline fits to contour sections.
- csaps (optional): cubic smoothing splines with simple smoothing parameter; helpful for curvature stability.

Use: smooth contour slices before curvature or derivative estimations to stabilize contact angle metrics.

## Optimization and Model Fitting
- SciPy.optimize: least_squares (robust loss: soft_l1/huber), curve_fit for simpler models; differential_evolution for global search when needed.
- lmfit (optional): higher-level parameter management, bounds, composites, and robust fitting; convenient but optional if SciPy suffices.

Use: fit geometric/physical parameters (e.g., apex radius R0, baseline tilt, polynomial/circle near contact) with robust losses.

## ODE/BVP Solvers (Young–Laplace, Capillarity)
- SciPy.integrate: solve_bvp for boundary-value Y–L shape; solve_ivp for shooting integrations.
- mpmath (optional): high-precision special cases if double precision is insufficient.

Use: implement ADSA-style boundary value solves, with scipy.optimize driving parameter identification; keep robust loss to handle noisy contours.

## Signal Processing (Oscillating Drops)
- SciPy.signal: find_peaks, peak_widths, welch periodogram, spectrogram, butterworth/chebyshev filters, hilbert transform for envelope/instantaneous phase.
- NumPy.fft: fast FFT for frequency estimation; windowing via SciPy.signal.windows.
- statsmodels (optional): ARMA/ARIMA or lowess for detrending and noise modeling.

Use: estimate frequency, damping (decaying sinusoid fits), Q factor from radius/area time series.

## Units, Physical Constants, Error Propagation
- Pint + pint-pydantic: modern units with Pydantic integration; clearer than unum and widely used.
- CoolProp (optional): thermophysical properties for fluids/air; interpolate density/viscosity vs T/P.
- uncertainties (optional): propagate measurement uncertainties through formulas; report error bars.

Use: standardize units across pipelines and GUI; attach units to calibration (px_per_mm), densities, g, surface tension; compute derived quantities with units.

## Data I/O and Tables
- pandas: CSV/Excel ingestion (read_csv, read_excel), tidy results tables, long-to-wide transforms.
- openpyxl/xlrd: engines for Excel when needed.

Use: load batch inputs; export results tables; interop with GUI results panel and CLI.

## Plotting and Visualization
- matplotlib: static plots for debugging and docs.
- pyqtgraph (optional): fast, interactive plots in PySide6 GUI (time-series, spectra, live cursors).

Use: preview results, oscillation traces, residual plots for fits.

## GUI and Overlays
- PySide6: main GUI framework (already used).
- qimage2ndarray (optional): fast NumPy ↔ QImage conversions if needed.

Use: keep drawing in Qt; avoid OpenCV for overlays beyond prototyping; maintain separation of data and rendering.

## Plugin/Extension System
- importlib.metadata entry points: discover external plugins by group name.
- pluggy (optional): mature plugin hooks and lifecycle management.

Use: define hook specs for stages (edge_detection, geometry, solver); allow third-party algorithms.

## Performance and Parallelism
- numba: JIT hotspots (distance loops, curvature scoring) when vectorization isn’t enough.
- joblib: simple parallel map for batch images; cache intermediate results.
- numpy strides/broadcasting: prefer vectorization before JIT.

Use: speed up curvature/contact scoring, iterative solvers over frames, and batch processing.

## Testing and QA
- pytest, pytest-qt: unit/GUI tests (already present).
- hypothesis (optional): property-based tests for geometry/solvers (e.g., invariances, monotonicity, bounds).

Use: ensure numerical stability and guard against regressions in fits and BVPs.

---

# Concrete Mappings to Menipy Tasks

1) Edge Detection and Contour Extraction
- Primary: scikit-image (filters, morphology, feature.canny), OpenCV (Canny, findContours).
- Optional: skimage.measure.find_contours on binary masks; approximate_polygon for simplification.

2) Contact Points and Baseline/Axis Estimation
- Primary: NumPy for projections/distances; SciPy.signal.savgol_filter for smoothing prior to curvature.
- Optional: Shapely for robust line–contour intersections and distances when geometry gets tricky.

3) Apex Radius and Local Geometry Fits
- Primary: scikit-image.measure.CircleModel + RANSAC; SciPy.optimize.least_squares.
- Optional: SciPy.interpolate.UnivariateSpline for smooth local curvature and derivatives.

4) ADSA/Young–Laplace Solver and Parameter Identification
- Primary: SciPy.integrate.solve_bvp / solve_ivp; SciPy.optimize.least_squares with robust loss.
- Optional: lmfit for model composition and constraints; mpmath for precision edge cases.

5) Volume/Area by Revolution and Surfaces
- Primary: NumPy vectorization; SciPy.integrate.trapezoid/simpson for line integrals.
- Optional: Shapely for projected areas/intersections if needed; scikit-image.measure.marching methods for 3D (future extensions).

6) Oscillating Drop Analysis
- Primary: SciPy.signal (find_peaks, hilbert, welch), NumPy.fft.
- Optional: statsmodels for trend/noise modeling; lmfit for decaying-sinusoid fits.

7) Units, Calibration, and Physical Properties
- Primary: Pint (+ pint-pydantic) for units; NumPy for conversions.
- Optional: CoolProp for temperature/pressure-dependent densities/viscosities; uncertainties for error bars.

8) Data Import/Export and Reporting
- Primary: pandas (CSV/Excel), matplotlib (static charts).
- Optional: seaborn for styled diagnostics; openpyxl for Excel specifics.

9) GUI/Overlay Rendering
- Primary: PySide6; retain overlays as metadata + Qt painting.
- Optional: pyqtgraph for high-FPS time-series.

10) Performance/Batch Processing
- Primary: NumPy vectorization; joblib for parallel batches.
- Optional: numba for inner loops (geometry scoring), especially in contact-point detection.

---

# Adoption Notes and Migration Hints
- Prefer scikit-image morphology and measure utilities over bespoke implementations in low-level geometry helpers; keep OpenCV as fast fallback.
- Standardize on SciPy.optimize.least_squares with robust loss for geometric and physical parameter fits; avoid custom optimizers unless necessary.
- For BVPs, start with solve_bvp; only fall back to shooting (solve_ivp + root finding) if conditioning issues arise.
- Consider migrating from unum to Pint to align with Pydantic ecosystem and common scientific Python workflows.
- For complex geometric predicates near the contact line, adopt Shapely to remove brittle float comparisons.
- For oscillation metrics, keep a clean split: preprocessing → smoothing → peak/envelope detection → parametric fit; surface these as independent, testable functions.

