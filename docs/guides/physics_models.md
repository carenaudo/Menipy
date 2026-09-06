# Physics Models

This document describes the mathematical models used to analyze pendant and sessile droplets.

## 1. Young–Laplace Equation (Axisymmetric Drop Shape)

- **Governing Principle**  
  Balance of capillary pressure and hydrostatic pressure:  
  [\n    \Delta P = \gamma\,\kappa
  ]  
  where \(\gamma\) is surface tension and \(\kappa\) is curvature.  
- **Capillary Length**  
  \(\ell_c = \sqrt{\frac{\gamma}{\Delta\rho\,g}}\)
  quantifies the relative importance of surface tension to gravity. :contentReference[oaicite:0]{index=0}
- **Bashforth–Adams Equation**  
  Dimensionless form of the Young–Laplace ODE, solved via numerical integration from the drop apex. :contentReference[oaicite:1]{index=1}

## 2. Geometric Contact Angle Models

Menipy provides three primary geometric contact angle models, all generalized to compute both acute ($\theta \le 90^\circ$) and obtuse ($\theta > 90^\circ$) angles seamlessly:

- **Spherical Cap Method (`spherical_cap`)**  
  Approximates small, capillary-dominated sessile drops ($\text{Bo} \ll 1$) as spherical caps:  
  $$\theta = 2 \arctan\left(\frac{h}{r}\right)$$  
  where $h$ is the drop height above the substrate and $r$ is half the base footprint width. Valid for $0^\circ \le \theta \le 180^\circ$.

- **Circle Fit Method (`circle_fit`)**  
  Fits a circle $(C, R)$ to contour points along the drop flank. The signed contact angle is computed via radial projection onto the inward substrate unit vector $\hat{\mathbf{u}}$ and apex-normal unit vector $\hat{\mathbf{n}}$:  
  $$\theta = \text{arctan2}(-r_u, r_n)$$  
  where $r_u = \hat{\mathbf{r}} \cdot \hat{\mathbf{u}}$ and $r_n = \hat{\mathbf{r}} \cdot \hat{\mathbf{n}}$. Because the circle center height relative to the substrate determines $r_n = -h_C/R$, the formulation naturally yields acute angles when the center is below the substrate ($h_C < 0$), exactly $90^\circ$ when the center is on the substrate ($h_C = 0$), and obtuse angles when the center is above the substrate ($h_C > 0$).

- **Tangent Polynomial / SVD Method (`tangent`)**  
  Fits a local tangent line near the contact point using weighted SVD without spherical curvature assumptions. The tangent is oriented upward into the droplet phase ($\hat{\mathbf{t}} \cdot \hat{\mathbf{n}} \ge 0$), and the angle is evaluated as $\theta = \text{arctan2}(\hat{\mathbf{t}} \cdot \hat{\mathbf{n}}, \, \hat{\mathbf{t}} \cdot \hat{\mathbf{u}})$.

- **Curved Substrate Slope Correction**  
  When drops reside on non-planar surfaces (e.g. fibers, lenses, cavities), the apparent angle measured against the chord is corrected for local substrate slope $\alpha_{\text{sub}}$:  
  $$\theta_{\text{intrinsic}} = \theta_{\text{apparent}} - \alpha_{\text{sub}}$$  
  See [`docs/guides/curved_substrates_and_baseline_detection.md`](curved_substrates_and_baseline_detection.md) and [`docs/guides/contact_angle_geometry.md`](contact_angle_geometry.md) for full derivations.

## 3. Axisymmetric Drop Shape Analysis (ADSA)

- **Full ADSA (Numerical Young–Laplace ODE)**  
  Integrates the non-linear Bashforth–Adams ODE system using numerical Runge–Kutta integration (`solve_ivp`, RK45) from the drop apex. Standard ADSA fits apex radius $R_0$ and shape parameter $\beta$ to minimize pointwise or normal-projection residuals against the drop silhouette.

- **Low-Bond Axisymmetric Drop Shape Analysis (LB-ADSA)**  
  For sessile drops under weak-to-moderate gravity ($\text{Bo} \in [0.001, 0.25]$), Menipy implements the first-order analytical perturbation theory of the Young–Laplace equation (Stalder et al., EPFL, 2010; *Colloids and Surfaces A*, 364, 72–81).  
  In polar coordinates centered at the apex center of curvature $(0, R_0)$:
  $$R(\alpha, R_0, \text{Bo}) = R_0 \left( 1 + \frac{1}{3} \text{Bo} \cdot \left[ \cos\alpha \left( \frac{1}{2} + \ln\left(\frac{2}{1 + \cos\alpha}\right) \right) - \frac{1}{2} \right] \right)$$
  where $\alpha \in [0, \pi)$ is the polar angle measured from the downward apex normal.  
  
  **Key Analytical Properties:**
  1. *Closed-form radial residuals*: Every experimental point $(X_i, Z_i)$ projects directly along the ray $\alpha_i = \arctan2(|X_i|, R_0 - Z_i)$ with distance $D_i = \sqrt{X_i^2 + (R_0 - Z_i)^2}$, yielding residuals $\rho_i = D_i - R(\alpha_i, R_0, \text{Bo})$ in $\mathcal{O}(1)$ time without ODE integration.
  2. *Analytical profile tangents*: The local tangent angle with the horizontal substrate is:
     $$\theta(\alpha) = \arctan2\left(R \sin\alpha - R' \cos\alpha, \, R \cos\alpha + R' \sin\alpha\right)$$
     where $R'(\alpha) = \frac{1}{3} \text{Bo} R_0 \sin\alpha \left( \frac{\cos\alpha}{1 + \cos\alpha} + \ln(1 + \cos\alpha) - \frac{1}{2} - \ln 2 \right)$.
  3. *Full angular range*: Supports both acute ($\theta \le 90^\circ$) and obtuse ($\theta > 90^\circ$) droplets seamlessly.
  4. *Simultaneous surface tension recovery*: When fluid density difference $\Delta\rho$ is known:
     $$\gamma = \frac{\Delta\rho \, g \, R_0^2}{\text{Bo}} \quad [\text{mN/m}]$$
     converging in $< 10\text{ ms}$ with guaranteed numerical stability.


## 4. Contour-Based Young–Laplace Fitting

- **Detected Contour Input**  
  Use the image-processed droplet contour (ordered (x,y) points) as data for fitting.  
- **Profile Parameterization**  
  Convert contour into axisymmetric coordinates r(s) vs. z(s), where s is arc-length.  
- **Fluid Properties**  
  Explicitly include liquid density ρₗ and gas density ρᵍ in the Bond number  
  [\n    Bo = \frac{(ρₗ - ρᵍ)\,g\,R^2}{γ}
  ]
  and in the hydrostatic term of the Young–Laplace equation.  
- **Optimization Loop**  
  1. Guess γ and ΔP.  
  2. Numerically integrate the Y-L ODE for that guess.  
  3. Compute the point-to-curve distance between model and detected contour.  
  4. Update γ, ΔP to minimize the total squared distance.  
- **Predicted Profile Generation**  
  Once γ and ΔP converge, generate the full theoretical profile (r(z)) and export it for overlay.

## 5. Captive Bubble Analysis *(Planned)*

- **Governing Principle**: The physics of a captive bubble (a bubble under a solid surface in a liquid) is analogous to an inverted pendant drop. The same Young-Laplace equation governs the shape, but the role of gravity is inverted (buoyancy pushes the bubble upwards).
- **Implementation Status**: This feature is planned. A `CaptiveBubbleGeometry` data model exists, but the full analysis pipeline is not yet implemented.

## 6. Capillary Rise Analysis *(Planned)*

- **Governing Principle**: This method measures surface tension by analyzing the height a liquid rises in a narrow capillary tube. The height (h) is determined by the balance of surface tension forces and gravity, as described by **Jurin's Law**:
  [\n    h = \frac{2\gamma \cos\theta}{\rho g r}
  ]
  where \(\theta\) is the contact angle, \(\rho\) is the liquid density, \(g\) is gravity, and \(r\) is the tube radius.
- **Implementation Status**: This feature is planned but not yet implemented.
