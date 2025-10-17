# Numerical Methods

This document outlines the algorithms and numerical techniques for solving and fitting droplet models.

## 1. ODE Integration

- **SciPy ODE Solvers**  
  Use `scipy.integrate.solve_ivp` (e.g., RK4) to integrate the dimensionless Young–Laplace equation from the apex. :contentReference[oaicite:6]{index=6}

## 2. Parameter Optimization

- **Surface Tension Fitting**  
  The current implementation uses a direct estimation method based on the apex curvature and maximum droplet diameter. A full optimization routine to fit the entire contour by minimizing the mean square distance (e.g., using `scipy.optimize.minimize`) is a planned enhancement.
- **Curve Fitting**  
  - **Circle**: Least-squares fitting is used to determine apex curvature.
  - **Spline Fitting**: Local splines are used for tangent estimation at the contact points, which is more robust than simple polynomial regression.

## 3. Numerical Integration for Volume

- **Rotational Volume**  
  Discrete trapezoidal integration of the profile:  
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


## 5. Planned: Contour-Driven Y–L Fitting Algorithm

*This section describes the target algorithm for a full Axisymmetric Drop Shape Analysis (ADSA) implementation, which is not yet fully integrated into the main analysis pipeline.*

1. **Resample & Smooth**  
   Uniformly sample the detected contour in arc-length and apply a small moving-average to reduce noise.  
2. **Solve Y-L ODE**  
   For a given (γ, ΔP), integrate  
   %%
     \frac{dψ}{ds} = \Deltaρ\,g\,r(s)\,\cosψ(s) - \frac{ΔP}{γ}
   %%
   with initial condition at the apex.  
3. **Distance Metric**  
   Compute  
   %%
     E = \sum_i \bigl\|\,\bigl(r_i,z_i\bigr) - \bigl(r_{\rm model}(s_i),z_{\rm model}(s_i)\bigr)\,\bigr\|^2
   %%
4. **Parameter Optimization**  
   Use `scipy.optimize.minimize` (e.g. Nelder–Mead) on (γ, ΔP) to drive E→min.  
5. **Generate Predicted Curve**  
   Once optimal, compute (r(z)) over full z range and pass to GUI for plotting.
