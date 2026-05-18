# Numerical Methods

This document outlines the algorithms and numerical techniques for solving and fitting droplet models.

## 1. ODE Integration

- **SciPy ODE Solvers**  
  Use `scipy.integrate.solve_ivp` (e.g., RK4) to integrate the dimensionless Young–Laplace equation from the apex. :contentReference[oaicite:6]{index=6}

## 2. Parameter Optimization

- **Surface Tension Fitting**  
  Pendant drops first compute a calibrated geometric Jennings-Pallas estimate
  from apex curvature and maximum droplet diameter. The pendant pipeline then
  runs a strict Young-Laplace fit in millimetres with
  `scipy.optimize.least_squares`; if residual and status gates pass, this
  strict fit populates the public surface-tension result. If it fails, the
  geometric estimate remains public and the strict fit is retained as
  diagnostics.
- **Curve Fitting**  
  - **Circle**: Least-squares fitting is used to determine apex curvature.
  - **Spline Fitting**: Local splines are used for tangent estimation at the contact points, which is more robust than simple polynomial regression.

## 3. Numerical Integration for Volume

- **Rotational Volume**  
  Discrete trapezoidal integration of the profile is implemented with
  `scipy.integrate.trapezoid`:  
  %%
    V \approx \sum_{i} \pi\,r(y_i)^2\,\Delta y
  %%
  where \(r(y)\) is half the width at pixel row \(y\). :contentReference[oaicite:8]{index=8}
- **Spherical Cap Formula**  
  %%
    V = \frac{\pi\,h\,(3r^2 + h^2)}{6}
  %%

## 4. Morphological and Filtering Operations

- While primarily in the image-processing pipeline, certain numerical cleanup (e.g., area filtering of small contours) ensures robust input to model fitting. :contentReference[oaicite:9]{index=9}


## 5. Pendant Contour-Driven Y-L Fitting Algorithm

1. **Calibrate Coordinates**
   Convert clipped pendant contour pixels into apex-centered millimetres with
   `x_mm = (x_px - axis_x) / px_per_mm` and
   `z_mm = (apex_y - y_px) / px_per_mm`.
2. **Solve Y-L ODE**  
   Integrate the dimensionless axisymmetric Young-Laplace system from the apex
   using `scipy.integrate.solve_ivp`, then scale by fitted `r0_mm`.
3. **Distance Metric**  
   Compare the observed calibrated contour to the model profile with KD-tree
   nearest-neighbor lookup and normal-projection residuals in millimetres.
4. **Parameter Optimization**  
   Fit `[r0_mm, beta, x_offset_mm, z_offset_mm]` with
   `scipy.optimize.least_squares`, seeded from the Jennings-Pallas estimate.
5. **Reporting Gate**
   Use strict Young-Laplace values publicly only when the optimizer succeeds,
   parameters are finite and not pinned to bounds, model height covers the
   observed contour, and `rmse_mm <= max(0.05, 0.03 * diameter_mm)`.

## 6. Current Library-Backed Numerical Swaps

- Normal-projection residual matching in the common solver uses
  `scipy.spatial.cKDTree` for nearest-neighbor lookup while preserving the
  previous residual shape and sign convention.
- Volume and surface-area integrations use `scipy.integrate.trapezoid` instead
  of direct NumPy trapezoid calls.
- Young-Laplace model integration and least-squares optimization already use
  SciPy (`solve_ivp` and `least_squares`). Circle/contact-angle fitting,
  baseline detection, and most contour heuristics remain custom.
