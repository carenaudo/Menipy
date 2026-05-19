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
  strict fit populates the public surface-tension result. The pipeline also
  runs registered pendant approximation plugins by default so scientific users
  can compare strict ADSA-style results against independent approximate
  estimates.
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
   using `scipy.integrate.solve_ivp`, then scale by fitted `r0_mm`. For the
   pendant image coordinate convention, the hydrostatic term is evaluated as
   `2 - beta*z - sin(phi)/r`.
3. **Distance Metric**  
   Compare the observed calibrated contour to the model profile with KD-tree
   nearest-neighbor lookup and normal-projection residuals in millimetres.
4. **Parameter Optimization**  
   Fit `[r0_mm, beta, x_offset_mm, z_offset_mm]` with
   `scipy.optimize.least_squares`, seeded from the Jennings-Pallas estimate.
5. **Physical Branch Truncation**
   Model generation stops at the observed pendant contact/needle height before
   nodoid or unduloid branches can be plotted. The fit records a
   `strict_fit_stop_reason`, usually `height_cutoff` for accepted sample fits.
6. **Reporting Gate**
   Use strict Young-Laplace values publicly only when the optimizer succeeds,
   parameters are finite and not pinned to bounds, model height covers the
   observed contour, and `rmse_mm <= max(0.05, 0.03 * diameter_mm)`.

## 6. Pendant Approximation Plugins

Pendant approximations implement the plugin contract
`fn(ctx, profile_mm, physics) -> dict`. The built-in plugins are active by
default and report comparison keys without replacing a passing strict
Young-Laplace result.

- **Selected-plane approximation** uses the classic pendant selected-plane
  idea: measure the equatorial diameter `d_e`, a selected-plane diameter `d_k`
  at distance `k*d_e` from the apex, compute `S_k = d_k / d_e`, and estimate
  the shape factor `H` from a Young-Laplace lookup. The method is useful for
  comparison but can be unavailable or low-confidence when the selected plane
  lies outside the measured branch. Berry et al. describe selected-plane
  methods in the historical pendant-drop workflow and contrast them with
  modern full-profile fitting: https://doi.org/10.1016/j.jcis.2015.05.012.
- **Multi-selected-plane approximation** repeats the selected-plane estimate
  at several default planes (`k = 0.6, 0.7, 0.8, 0.9, 1.0`) and reports the
  median and spread. Jůza et al. discuss selected and multiple selected planes
  using `d_e`, `d_k`, `S_k`, and `H`: https://link.springer.com/article/10.1007/s00396-025-05513-5.
- **Volume-apex lookup approximation** uses the measured volume, measured
  height, and apex radius to infer the Young-Laplace beta from a lookup table
  generated with the same pendant ODE convention. Yeow et al. describe the
  pendant volume plus apex-curvature route and its limitations:
  https://doi.org/10.1016/j.colsurfa.2007.07.025.

Fallback order when strict Y-L fails is: multi-selected-plane,
selected-plane, volume-apex lookup, then the legacy Jennings-Pallas geometric
estimate. Each fallback must have an `ok` status before it can become public.

## 7. Current Library-Backed Numerical Swaps

- Normal-projection residual matching in the common solver uses
  `scipy.spatial.cKDTree` for nearest-neighbor lookup while preserving the
  previous residual shape and sign convention.
- Volume and surface-area integrations use `scipy.integrate.trapezoid` instead
  of direct NumPy trapezoid calls.
- Young-Laplace model integration and least-squares optimization already use
  SciPy (`solve_ivp` and `least_squares`). Circle/contact-angle fitting,
  baseline detection, and most contour heuristics remain custom.
