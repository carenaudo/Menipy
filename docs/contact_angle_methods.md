# Contact Angle Estimation Methods

Menipy supports multiple complementary methods for contact angle estimation on sessile droplets. Each method has distinct mathematical formulations, geometric assumptions, and sensitivity characteristics.

---

## 1. Summary of Available Methods

| Method Identifier | Applicable Droplet Geometry | Contact Angle Range | Strengths | Limitations |
|---|---|---|---|---|
| `spherical_cap` | Small, capillary-dominated drops ($\text{Bo} \ll 1$) | $0^\circ \le \theta \le 180^\circ$ | Highly robust to contour noise; closed-form solution from apex height $h$ and radius $r$. | Inaccurate when gravity flattens large droplets ($\text{Bo} > 0.1$). |
| `circle_fit` | Small to medium drops; circular contact region | $0^\circ \le \theta \le 180^\circ$ | Exact, continuous signed radial formulation; immune to local pixel step noise. | Assumes local circular curvature near the contact line. |
| `tangent` | Arbitrary droplet shapes (flattened, tilted, dynamic) | $0^\circ \le \theta \le 180^\circ$ | Strictly local model via weighted SVD; no spherical assumption. | More sensitive to pixel quantization noise at the contact point. |
| `auto_residual` *(default)* | General / automated analysis | $0^\circ \le \theta \le 180^\circ$ | Evaluates both `tangent` and `circle_fit` residuals; chooses the best fit automatically. | Performs two fits per contact point. |
| `lbadsa` | Gravity-affected sessile drops ($\text{Bo} \in [0.001, 0.25]$) | $0^\circ \le \theta \le 180^\circ$ | First-order Young–Laplace perturbation model; simultaneously measures $\theta$ and surface tension $\gamma$ with sub-pixel closed-form ray projections. | First-order approximation accurate for $\text{Bo} \le 0.25$. |

---

## 2. Mathematical Formulations

### 2.1 Spherical Cap Method (`spherical_cap`)
For small droplets where surface tension dominates gravity, the droplet profile is well-approximated by a spherical cap. Given apex height $h$ and contact radius $r = w/2$:

$$\theta = 2 \arctan\left(\frac{h}{r}\right)$$

This formula is valid for both acute ($\theta < 90^\circ \implies h < r$) and obtuse ($\theta > 90^\circ \implies h > r$) droplets.

### 2.2 Circle Fit Method (`circle_fit`)
Fits a circle to flank points near the contact point $\mathbf{p}$. The unit radial vector from center to contact point is projected onto the inward substrate vector $\hat{\mathbf{u}}$ and apex-normal vector $\hat{\mathbf{n}}$:

$$r_u = \frac{\mathbf{p} - C}{\|\mathbf{p} - C\|} \cdot \hat{\mathbf{u}}, \quad r_n = \frac{\mathbf{p} - C}{\|\mathbf{p} - C\|} \cdot \hat{\mathbf{n}}$$

$$\theta = \text{arctan2}(-r_u, r_n)$$

The sign of $r_n$ automatically distinguishes acute ($r_n > 0 \implies \theta < 90^\circ$) from obtuse ($r_n < 0 \implies \theta > 90^\circ$) regimes without heuristic conditionals.

### 2.3 Tangent Polynomial / SVD Method (`tangent`)
Performs a weighted line fit on local flank points within distance $W_{\text{px}}$ of the contact point $\mathbf{p}$:

$$\mathbf{v} = \arg\min_{\|\mathbf{v}\|=1} \sum_i w_i \left((\mathbf{x}_i - \mathbf{p}) \times \mathbf{v}\right)^2, \quad w_i = \frac{1}{\max(d_i, 1.0)^4}$$

The tangent is oriented upward into the droplet phase ($\hat{\mathbf{t}} \cdot \hat{\mathbf{n}} \ge 0$), yielding:

$$\theta = \text{arctan2}(\hat{\mathbf{t}} \cdot \hat{\mathbf{n}}, \, \hat{\mathbf{t}} \cdot \hat{\mathbf{u}})$$

### 2.4 Auto-Residual Selection (`auto_residual`)
In `src/menipy/pipelines/sessile/metrics.py`:
1. Both `tangent_angle_at_point_pure` and `circle_fit_angle_at_point` are computed along with their respective Root Mean Square Errors ($\text{RMSE}$).
2. If $\text{RMSE}_{\text{circle}} \le 0.8 \cdot \text{RMSE}_{\text{tangent}}$ and the circle fit is valid, `circle_fit` is chosen.
3. Otherwise, the local `tangent` fit is retained.
4. Per-contact diagnostics are stored under `ctx.results["selector_diagnostics"]`.

### 2.5 Low-Bond ADSA Method (`lbadsa`)
Analytical first-order perturbation of the Young–Laplace equation (Stalder et al., EPFL, 2010):

$$R(\alpha, R_0, \text{Bo}) = R_0 \left( 1 + \frac{1}{3} \text{Bo} \cdot \left[ \cos\alpha \left( \frac{1}{2} + \ln\left(\frac{2}{1 + \cos\alpha}\right) \right) - \frac{1}{2} \right] \right)$$

1. Fits $R_0$, $\text{Bo}$, and apex translation $(\Delta X, \Delta Z)$ using robust non-linear least squares with $\mathcal{O}(1)$ ray projection residuals:
   $$\rho_i = \sqrt{X_i^2 + (R_0 - Z_i)^2} - R(\arctan2(|X_i|, R_0 - Z_i), R_0, \text{Bo})$$
2. Computes the baseline intersection $Z(\alpha_c) = H$ and evaluates the analytical tangent angle $\theta(\alpha_c)$.
3. Calculates surface tension if fluid density $\Delta\rho$ is known: $\gamma = \frac{\Delta\rho \, g \, R_0^2}{\text{Bo}}$.

---

## 3. Configuration & CLI Usage

In the PySide6 GUI, select the desired method in the **Sessile Settings** panel. In the headless CLI, pass the `--contact-angle-method` flag:

```powershell
uv run adsa --pipeline sessile --image drop.png --contact-angle-method auto_residual
uv run adsa --pipeline sessile --image drop.png --contact-angle-method lbadsa
uv run adsa --pipeline sessile --image drop.png --contact-angle-method circle_fit
uv run adsa --pipeline sessile --image drop.png --contact-angle-method tangent
```

---

## 4. API Reference

- [`fit_lbadsa_drop`](file:///d:/programacion/Menipy/src/menipy/common/lbadsa_solver.py): Fits the LB-ADSA model and extracts $R_0$, $\text{Bo}$, $\theta_{\text{left}}$, $\theta_{\text{right}}$, $\gamma$, and overlay points.
- [`contact_angle_lbadsa`](file:///d:/programacion/Menipy/src/menipy/math/lbadsa.py): Analytical contact angle root solver for baseline height $H$.
- [`radius_lbadsa`](file:///d:/programacion/Menipy/src/menipy/math/lbadsa.py): Analytical perturbation radius function $R(\alpha)$.
- [`estimate_contact_angle_circle_fit`](file:///d:/programacion/Menipy/src/menipy/common/geometry.py#L640): Fits circle to flank points and computes signed contact angle and RMSE.
- [`estimate_contact_angle_tangent`](file:///d:/programacion/Menipy/src/menipy/common/geometry.py#L425): Fits local tangent via polynomial or weighted SVD.
- [`tangent_angle_at_point_pure`](file:///d:/programacion/Menipy/src/menipy/common/geometry.py#L740): SVD tangent fit without legacy circular fallback.
- [`compute_sessile_metrics`](file:///d:/programacion/Menipy/src/menipy/pipelines/sessile/metrics.py#L180): High-level stage metric computation executing the selected method and performing apparent-to-intrinsic substrate slope corrections.

