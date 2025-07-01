# Physics Models

This document describes the mathematical models used to analyze pendant and sessile droplets.

## 1. Young–Laplace Equation (Axisymmetric Drop Shape)

- **Governing Principle**  
  Balance of capillary pressure and hydrostatic pressure:  
  \[
    \Delta P = \gamma\,\kappa
  \]  
  where \(\gamma\) is surface tension and \(\kappa\) is curvature.  
- **Capillary Length**  
  \(\ell_c = \sqrt{\frac{\gamma}{\Delta\rho\,g}}\)  
  quantifies the relative importance of surface tension to gravity. :contentReference[oaicite:0]{index=0}
- **Bashforth–Adams Equation**  
  Dimensionless form of the Young–Laplace ODE, solved via numerical integration from the drop apex. :contentReference[oaicite:1]{index=1}

## 2. Simple Geometric Models

- **Circular (Spherical Cap) Fit**  
  Approximates small sessile drops as spherical caps. Contact angle:  
  \[
    \theta = 2\,\arctan\!\bigl(\tfrac{h}{r}\bigr)
  \]  
  where \(h\) is height and \(r\) half the base width. :contentReference[oaicite:2]{index=2}
- **Ellipse Fit**  
  Fits an ellipse to the sessile drop profile (useful for contact angles up to ~130°). :contentReference[oaicite:3]{index=3}
- **Polynomial (Tangent) Fit**  
  Local polynomial fit near the contact point to estimate the tangent slope and hence \(\theta\). :contentReference[oaicite:4]{index=4}

## 3. Axisymmetric Drop Shape Analysis (ADSA)

- **Full ADSA**  
  Combines Young–Laplace numerical solution with optimization to fit \(\gamma\) (and contact angle for sessile drops) to the detected contour.  
- **Low-Bond ADSA**  
  Perturbation-based approximation for small Bond numbers, fitting both the drop and its reflection to extract contact angle without manual contact-point selection. :contentReference[oaicite:5]{index=5}


## 4. Contour-Based Young–Laplace Fitting

- **Detected Contour Input**  
  Use the image-processed droplet contour (ordered (x,y) points) as data for fitting.  
- **Profile Parameterization**  
  Convert contour into axisymmetric coordinates r(s) vs. z(s), where s is arc-length.  
- **Fluid Properties**  
  Explicitly include liquid density ρₗ and gas density ρᵍ in the Bond number  
  \[
    Bo = \frac{(ρₗ - ρᵍ)\,g\,R^2}{γ}
  \]  
  and in the hydrostatic term of the Young–Laplace equation.  
- **Optimization Loop**  
  1. Guess γ and ΔP.  
  2. Numerically integrate the Y-L ODE for that guess.  
  3. Compute the point-to-curve distance between model and detected contour.  
  4. Update γ, ΔP to minimize the total squared distance.  
- **Predicted Profile Generation**  
  Once γ and ΔP converge, generate the full theoretical profile (r(z)) and export it for overlay.