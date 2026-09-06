# Contact Angle Geometry and Vector Formulation Guide

This document describes the mathematical and geometric foundation of Menipy's contact angle estimation routines implemented in [`src/menipy/common/geometry.py`](file:///d:/programacion/Menipy/src/menipy/common/geometry.py).

---

## 1. Physical and Geometric Definition

The contact angle $\theta$ of a liquid droplet on a solid substrate is conventionally defined as the angle measured **through the liquid phase** between the solid substrate and the liquid-gas interface at the three-phase contact line:

- **Acute Droplets ($\theta < 90^\circ$):** Wetting / hydrophilic regime. The droplet base represents the widest footprint, and the droplet flank ascends **inward** into the liquid body.
- **Right-Angle Droplets ($\theta = 90^\circ$):** Hemispherical cap. The contact line is vertical at the interface.
- **Obtuse Droplets ($\theta > 90^\circ$):** Non-wetting / hydrophobic / superhydrophobic regime. The droplet bulges **outward** past the contact point as it ascends from the substrate.

Historically, naive contact angle implementations computed $\theta = \arctan(dy/dx)$ or $\arccos(|\hat{\mathbf{t}} \cdot \hat{\mathbf{s}}|)$, restricting the output to $[0^\circ, 90^\circ]$ and erroneously folding obtuse angles back into acute angles (e.g., measuring $145^\circ$ as $35^\circ$). Menipy resolves this across all methods through a **continuous, signed vector projection** in an orthonormal substrate frame.

---

## 2. Orthonormal Substrate Coordinate Frame

For any contact point $\mathbf{p} = (x_p, y_p)$ on a straight line or curved substrate profile, `_substrate_frame` constructs an orthonormal basis $(\hat{\mathbf{u}}, \hat{\mathbf{n}})$:

$$\hat{\mathbf{u}} = \text{inward\_substrate\_vec}, \quad \hat{\mathbf{n}} = \text{normal\_vec}$$

1. **Substrate Tangent $\hat{\mathbf{s}}$:**
   - On a straight substrate line $\mathbf{p}_1 \to \mathbf{p}_2$, $\hat{\mathbf{s}} = \frac{\mathbf{p}_2 - \mathbf{p}_1}{\|\mathbf{p}_2 - \mathbf{p}_1\|}$, oriented toward $+x$.
   - On a curved `SubstrateProfile` (e.g. circular arc, spline), $\hat{\mathbf{s}}$ is evaluated locally from the tangent angle $\alpha(x_p, y_p)$.
2. **Substrate Normal $\hat{\mathbf{n}}$:**
   - $\hat{\mathbf{n}}$ is orthogonal to $\hat{\mathbf{s}}$ and oriented toward the droplet apex ($\mathbf{rel} \cdot \hat{\mathbf{n}} > 0$).
3. **Inward Orientation $\hat{\mathbf{u}}$:**
   - $\hat{\mathbf{u}} = \text{sgn}(\text{median}(s_{\text{all}})) \cdot \hat{\mathbf{s}}$, ensuring that $\hat{\mathbf{u}}$ always points along the substrate toward the interior of the droplet.

In this coordinate frame, any 2D displacement relative to the contact point has coordinates $(s, h)$ where $s = \mathbf{rel} \cdot \hat{\mathbf{u}}$ (inward distance) and $h = \mathbf{rel} \cdot \hat{\mathbf{n}}$ (height above substrate).

---

## 3. Flank Point Selection (`_contact_branch_points`)

Extracting the correct contour points on the droplet flank is crucial, especially in noisy experimental images containing substrate reflections, baseline textures, or dispensing needle shadows:

```python
radius = float(window_px) * factor
distances = np.linalg.norm(rel_all, axis=1)
s_in = s_all * inward_sign

inward_mask = (distances <= radius) & (h_all > 0.5) & (s_in >= -1.0)
if np.count_nonzero(inward_mask) >= 3:
    mask = inward_mask
else:
    outward_ok = (s_in < -1.0) & (h_all > 1.5) & (h_all >= np.abs(s_in) * 0.25)
    mask = (distances <= radius) & (inward_mask | outward_ok)
```

### Inward-First Discrimination Logic
- **For Acute Drops ($\theta \le 90^\circ$):** The flank ascends with $s_{\text{in}} \ge 0$. Substrate texture or reflection artifacts outside the droplet have $s_{\text{in}} < -1.0$. Because `inward_mask` gathers sufficient points ($\ge 3$, typically $10-40$ points), outward points are strictly excluded, preventing substrate noise from dragging down the estimated contact angle.
- **For Obtuse Drops ($\theta > 90^\circ$):** The droplet overhanging flank ascends with $s_{\text{in}} < 0$. Because `inward_mask` contains fewer than 3 points, the selection automatically activates the outward ascending branch (`outward_ok`).
- **Baseline and Vertical Closure Filtering:** Closed contour segments with $h \approx \text{const}$ or vertical closure artifacts are detected and suppressed.

---

## 4. Circle Fit Contact Angle (`estimate_contact_angle_circle_fit`)

A circle $(C, R)$ is fitted to the local flank points via linear least squares (`fit_circle`).

Let $\mathbf{v} = \mathbf{p} - C$ be the vector from the circle center to the contact point $\mathbf{p}$. The unit radial vector is:

$$\hat{\mathbf{r}} = \frac{\mathbf{p} - C}{\|\mathbf{p} - C\|}$$

Projecting $\hat{\mathbf{r}}$ onto the basis $(\hat{\mathbf{u}}, \hat{\mathbf{n}})$:

$$r_u = \hat{\mathbf{r}} \cdot \hat{\mathbf{u}}, \quad r_n = \hat{\mathbf{r}} \cdot \hat{\mathbf{n}}$$

Because $C$ lies inside the droplet body and $\mathbf{p}$ is on its boundary, $\hat{\mathbf{r}}$ points away from the droplet interior along $\hat{\mathbf{u}}$, so $r_u < 0$.

The unit tangent vector $\hat{\mathbf{t}}$ perpendicular to $\hat{\mathbf{r}}$ that points in the ascending direction ($t_n > 0$) is uniquely:

$$\hat{\mathbf{t}} = (t_u, t_n) = (r_n, -r_u)$$

The contact angle inside the droplet phase is therefore:

$$\theta = \text{arctan2}(t_n, t_u) = \text{arctan2}(-r_u, r_n)$$

### Center Height Sign Discrimination
Since $p$ lies on the substrate ($h_p = 0$), $r_n = \frac{h_p - h_C}{R} = -\frac{h_C}{R}$:
- **Acute Droplet ($\theta < 90^\circ$):** Circle center is below substrate ($h_C < 0 \implies r_n > 0$). Thus $t_u > 0 \implies \theta \in (0^\circ, 90^\circ)$.
- **Hemisphere ($\theta = 90^\circ$):** Circle center is on the substrate line ($h_C = 0 \implies r_n = 0$). Thus $t_u = 0 \implies \theta = 90^\circ$.
- **Obtuse Droplet ($\theta > 90^\circ$):** Circle center is above substrate ($h_C > 0 \implies r_n < 0$). Thus $t_u < 0 \implies \theta \in (90^\circ, 180^\circ)$.

This formulation is continuous, analytical, and exact across the entire domain $(0^\circ, 180^\circ)$.

---

## 5. Tangent Polynomial / SVD Method (`estimate_contact_angle_tangent`)

In the tangent method, a line is fitted to weighted local flank points via Singular Value Decomposition (SVD):

1. **SVD Fit:**
   $$\mathbf{v} = \arg\min_{\|\mathbf{v}\|=1} \sum_i w_i \left((\mathbf{x}_i - \mathbf{p}) \times \mathbf{v}\right)^2$$
   where weights decay with distance: $w_i = \frac{1}{\max(d_i, 1.0)^p}$.
2. **Ascending Orientation:**
   Ensure $\hat{\mathbf{t}}$ points upward into the droplet phase:
   $$\hat{\mathbf{t}} = \mathbf{v} \cdot \text{sgn}(\mathbf{v} \cdot \hat{\mathbf{n}})$$
3. **Signed Angle Computation:**
   $$t_u = \hat{\mathbf{t}} \cdot \hat{\mathbf{u}}, \quad t_n = \hat{\mathbf{t}} \cdot \hat{\mathbf{n}}$$
   $$\theta = \text{arctan2}(t_n, t_u)$$

---

## 6. Multi-Model Selector (`auto_residual`)

In [`src/menipy/pipelines/sessile/metrics.py`](file:///d:/programacion/Menipy/src/menipy/pipelines/sessile/metrics.py), `contact_angle_method="auto_residual"` evaluates both the tangent fit and circle fit residuals $\text{RMSE}_{\text{tangent}}$ and $\text{RMSE}_{\text{circle}}$:

- The tangent model is preferred as the baseline local model.
- If the circle fit is valid and materially improves the residual ($\text{RMSE}_{\text{circle}} \le 0.8 \cdot \text{RMSE}_{\text{tangent}}$), `circle_fit` is selected.
- Diagnostics (`selector_diagnostics`) report the chosen method, individual residuals, and justification.

---

## 7. Apparent vs. Intrinsic Angles on Curved Substrates

On curved substrates (cylinders, spheres, fibers, lenses), the contact angle measured relative to the chord is the **apparent** angle ($\theta_{\text{apparent}}$). The true thermodynamic **intrinsic** contact angle is corrected for local substrate inclination:

$$\theta_{\text{intrinsic}} = \theta_{\text{apparent}} - \alpha_{\text{sub}}$$

where $\alpha_{\text{sub}}$ is evaluated via `SubstrateProfile.eval_tangent_angle_deg(x, y)` at each contact point.

---

## 8. Parametric B-Spline Tangent Contact Angles (`fit_bspline_snake`)

Following the DropSnake methodology established by Stalder et al. (2006, 2010) and Brigger et al. (2000), Menipy provides continuous parametric cubic B-spline fitting for sub-pixel droplet profile smoothing and contact angle evaluation ([`src/menipy/math/active_contour.py`](file:///d:/programacion/Menipy/src/menipy/math/active_contour.py)).

### 8.1 Continuous Parametric Curve Formulation
An open droplet interface with $N$ vertices is represented as a cubic B-spline curve $\mathbf{C}(u) = (x(u), y(u))^\top$ with normalized curve parameter $u \in [0, 1]$:

$$\mathbf{C}(u) = \sum_{i=0}^{m} N_{i, 3}(u) \, \mathbf{P}_i$$

where $N_{i, 3}(u)$ are the cubic B-spline basis functions evaluated via the Cox-de Boor recursion algorithm over the knot vector $\mathbf{U}$, and $\mathbf{P}_i \in \mathbb{R}^2$ are the control points fitted via SciPy's FITPACK algorithms (`scipy.interpolate.splprep`).

### 8.2 Analytical Tangents and Local Curvature
Because the B-spline basis possesses $C^2$ continuity everywhere along the profile, exact analytical derivatives of any order are computed directly from the spline coefficients:

$$\mathbf{C}'(u) = \left( \frac{dx}{du}, \frac{dy}{du} \right)^\top, \quad \mathbf{C}''(u) = \left( \frac{d^2 x}{du^2}, \frac{d^2 y}{du^2} \right)^\top$$

The signed local curvature $\kappa(u)$ along the droplet interface is analytically:

$$\kappa(u) = \frac{x'(u) y''(u) - y'(u) x''(u)}{\left( x'(u)^2 + y'(u)^2 \right)^{3/2}}$$

### 8.3 Coordinate-Invariant Vector Dot Product Contact Angles
Let $\hat{\mathbf{u}}_{\text{sub}}$ be the unit direction vector along the substrate line, oriented from the left contact point toward the right contact point (pointing along $+x$ in standard horizontal geometry).

At the three-phase contact lines, the unit interface tangents ascending into the liquid droplet phase are:

$$\hat{\mathbf{t}}_{\text{left}} = \frac{\mathbf{C}'(0)}{\|\mathbf{C}'(0)\|}, \quad \hat{\mathbf{t}}_{\text{right}} = -\frac{\mathbf{C}'(1)}{\|\mathbf{C}'(1)\|}$$

Using signed vector projections onto the inward substrate vectors ($\hat{\mathbf{u}}_{\text{sub}}$ at left, $-\hat{\mathbf{u}}_{\text{sub}}$ at right):

$$\theta_{\text{left}} = \arccos\left( \hat{\mathbf{u}}_{\text{sub}} \cdot \hat{\mathbf{t}}_{\text{left}} \right)$$

$$\theta_{\text{right}} = \arccos\left( -\hat{\mathbf{u}}_{\text{sub}} \cdot \hat{\mathbf{t}}_{\text{right}} \right)$$

### 8.4 Invariance Across All Wetting Regimes
Unlike naive slope-based formulations $\theta = \arctan(dy/dx)$ that collapse obtuse angles into acute angles, the vector dot product formulation is continuous, signed, and exact across all geometric regimes:
- **Acute Droplets ($\theta < 90^\circ$):** $\hat{\mathbf{u}}_{\text{sub}} \cdot \hat{\mathbf{t}}_{\text{left}} > 0 \implies \theta \in (0^\circ, 90^\circ)$.
- **Hemispherical Droplets ($\theta = 90^\circ$):** $\hat{\mathbf{u}}_{\text{sub}} \cdot \hat{\mathbf{t}}_{\text{left}} = 0 \implies \theta = 90^\circ$.
- **Obtuse / Superhydrophobic Droplets ($\theta > 90^\circ$):** $\hat{\mathbf{u}}_{\text{sub}} \cdot \hat{\mathbf{t}}_{\text{left}} < 0 \implies \theta \in (90^\circ, 180^\circ)$.
- **Tilted Substrates:** Rotational invariance holds automatically because $\hat{\mathbf{u}}_{\text{sub}}$ reflects the true inclination vector of the tilted solid plate.

### 8.5 Addressing Active Contour Corner Blunting Near Substrates
When active contours evolve against discrete image gradients, symmetric 2D Gaussian scale-space blurring at the three-phase contact line corner can pull the immediate endpoint slightly inward. To ensure rigorous experimental contact angle measurement, Menipy's contour refinement stage ([`src/menipy/common/contour_refinement.py`](file:///d:/programacion/Menipy/src/menipy/common/contour_refinement.py)) combines B-spline sub-pixel profile evolution with weighted polynomial flank projection (`estimate_contact_angle_tangent`), sampling the unblurred droplet flank ($h > 1.0\text{ px}$) to yield sub-degree accuracy.