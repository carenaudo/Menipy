# Curved Substrates and Robust Baseline Detection Guide

## 1. Physical Motivation & Scientific Foundations

In standard sessile drop goniometry, droplets are assumed to rest on an optically flat, horizontal plane. However, in practical materials science, chemical engineering, and biological applications, droplets often reside on **curved substrates** (such as cylindrical fibers, spherical lenses, curved wires, pill capsules, or convex/concave coatings).

### 1.1 Apparent vs. Intrinsic (Thermodynamic) Contact Angle

When a droplet rests on a curved surface, the contact angle measured directly against a chord or horizontal line is an **apparent contact angle** ($\theta_{\text{apparent}}$). To evaluate true surface energy or wettability (Young's equation), the contact angle must be measured relative to the **local tangent plane** of the curved solid substrate at each contact point:

$$\theta_{\text{intrinsic}} = \theta_{\text{apparent}} - \alpha_{\text{sub}}$$

where $\alpha_{\text{sub}}$ is the local substrate slope (tangent angle) in the droplet frame at the contact point.

- **Convex Substrates (Cylinders / Spheres / Fibers):** The substrate curves downward away from the apex ($y_{\text{sub}}(x) < y_{\text{apex}}$). At the left contact point, the local surface slopes upward ($\alpha_L > 0$), making the intrinsic contact angle larger than the apparent angle measured against a horizontal chord. At the right contact point ($\alpha_R < 0$), the local slope shifts the contact line accordingly.
- **Concave Substrates (Cavities / Wells / Lens Inverts):** The surface curves upward around the droplet, yielding $\alpha_L < 0$ and $\alpha_R > 0$.

### 1.2 Scientific References

1. **Extrand, C. W., & Moon, M. W. (2008).** *Indirect Measurement of Contact Angles on Curved Surfaces*. **Langmuir**, 24(17), 9470–9473. DOI: [10.1021/la801091m](https://doi.org/10.1021/la801091m)
   - Demonstrates that contact angles measured on curved surfaces must be corrected for local surface inclination $\alpha$ at the three-phase contact line, recovering the flat thermodynamic contact angle.
2. **Carroll, B. J. (1976).** *The accurate measurement of contact angle, phase volume, and surface area of drops on cylindrical fibers*. **Journal of Colloid and Interface Science**, 57(3), 488–495. DOI: [10.1016/0021-9797(76)90227-7](https://doi.org/10.1016/0021-9797(76)90227-7)
   - Provides analytical relationships between apparent drop profile coordinates and intrinsic contact angles on cylindrical substrates.
3. **Chau, T. T. (2009).** *A review of techniques for measuring contact angles of curved surfaces*. **Colloids and Surfaces A: Physicochemical and Engineering Aspects**, 347(1-3), 3–9.

---

## 2. Robust Substrate Baseline Detection Algorithm

Baseline detection can fail or become ambiguous in images with:
- Poor contrast or uneven illumination.
- Stage tilt or camera misalignment ($0.5^\circ - 5^\circ$).
- Lens vignetting / dark corner fixtures.
- Droplet reflections mimicking false horizontal edges.

Menipy introduces `detect_sessile_substrate_robust` (`src/menipy/common/sessile_detection.py`), replacing unconstrained Hough transforms with a **physically bounded, bilateral sector row-gradient analysis**:

### 2.1 Algorithmic Pipeline

1. **Physical Band Search ($[0.55 h, 0.90 h]$):**
   Excludes the upper image (needle, drop apex) and extreme bottom margin (sample holder clamps) to prevent drifting into fixture borders.
2. **Central Row-Gradient Anchor:**
   Excludes the outer 8% margins ($x \in [0.08 w, 0.92 w]$) where vignetting occurs. Convolves the row-averaged vertical intensity profile with a smoothing filter and identifies the primary step gradient $y_{\text{anchor}}$.
3. **Raw Dynamic Range & Contrast Scoring:**
   Measures peak-to-peak variation $\text{ptp}(I_{\text{raw}})$ in the search band. If $\text{ptp} < 25$ intensity counts, a low-contrast penalty factor $f_{\text{contrast}} = \min(1.0, \text{ptp} / 25.0)$ is applied to prevent CLAHE from over-amplifying noise into false confident edges.
4. **Bilateral Sector Tilt Refinement:**
   Searches a $\pm 10$ px window around $y_{\text{anchor}}$ in left ($[0.08 w, 0.30 w]$) and right ($[0.70 w, 0.92 w]$) clear sectors:
   $$\theta_{\text{tilt}} = \arctan\left(\frac{y_R - y_L}{x_R - x_L}\right)$$
   A tilt is accepted only if $0.5^\circ \le |\theta_{\text{tilt}}| \le 5.0^\circ$, avoiding noisy pixel quantization fluctuations on true flat baselines.
5. **Confidence Metric $Q$:**
   $$Q = \mathrm{clip}\left(\left(0.65 \cdot q_{\text{contrast}} + 0.35 \cdot q_{\text{bilateral}}\right) \cdot f_{\text{contrast}},\, 0.25,\, 0.98\right)$$
   - **$Q \ge 0.75$ ("confident"):** Automatic baseline is reliable; analysis proceeds without interruption.
   - **$0.45 \le Q < 0.75$ ("doubtful"):** Baseline detected with uncertainty; automated warning banner is displayed with quick drawing actions.
   - **$Q < 0.45$ ("failed"):** Detection failed or ambiguous; fallback horizontal line assigned and prominent manual intervention requested.

---

## 3. Curved Substrate Interactive Drawing & UX Workflow

When a sample is curved or when baseline detection is doubtful, Menipy provides an intuitive, seamless manual workflow:

### 3.1 Three-Point Arc Drawing (`DRAW_ARC`)

In `ImageView` (`src/menipy/gui/views/image_view.py`), users can activate **Curved Substrate** mode from the Mark menu or warning banner:
1. **Click 1 ($P_1$):** Place the left substrate anchor point.
2. **Click 2 ($P_2$):** Place the right substrate anchor point.
3. **Click 3 ($P_3$):** Click and drag the crest/trough handle to set substrate curvature.

The live view renders the circular arc interactively as the cursor moves, ensuring immediate visual feedback. Once placed, the arc is converted to a `SubstrateProfile` (`type="circle_arc"`), calculating the exact radius of curvature $R$, center $(x_c, y_c)$, and convexity direction.

### 3.2 Automated Doubtful Warning Banner

When `AutoCalibrator` or `detect_sessile_substrate_robust` reports $Q < 0.75$:
- A dismissible banner appears at the top of the **Preview Panel** (`src/menipy/gui/views/preview_panel.py`).
- Displays the confidence percentage with a color-coded pill:
  - Green ($\ge 75\%$): Confident
  - Amber ($50\% - 75\%$): Doubtful
  - Red ($< 50\%$): Failed
- Includes quick-action buttons:
  - `[✏ Draw Baseline]`: Initiates 2-point straight line drawing.
  - `[⌒ Draw Curved Arc]`: Initiates 3-point curved arc drawing.
- Dismisses automatically when the user manually draws a line or arc.

---

## 4. Data Models & Result Schema

### 4.1 `SubstrateProfile` Model (`src/menipy/models/geometry.py`)

```python
class SubstrateProfile(BaseModel):
    type: Literal["line", "circle_arc", "polynomial", "spline"] = "line"
    points: list[tuple[float, float]] = Field(default_factory=list)
    parameters: dict[str, float] = Field(default_factory=dict)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    user_defined: bool = False

    def eval_y(self, x: float) -> float | None: ...
    def eval_tangent_angle_deg(self, x: float, y: float | None = None) -> float: ...
    def to_chord(self) -> tuple[tuple[float, float], tuple[float, float]]: ...
    def sample_points(self, num_points: int = 50, n_points: int | None = None) -> list[tuple[float, float]]: ...
```

### 4.2 Sessile Result Contract Fields (`docs/contracts/sessile_results.md`)

When executing the sessile pipeline on curved or flat substrates, the following keys are populated:

| Key | Type | Description |
|---|---|---|
| `theta_left_deg` | `float` | Intrinsic left contact angle corrected for local substrate slope ($\theta_L = \theta_{\text{app}, L} - \alpha_L$). |
| `theta_right_deg` | `float` | Intrinsic right contact angle corrected for local substrate slope ($\theta_R = \theta_{\text{app}, R} - \alpha_R$). |
| `theta_left_apparent_deg` | `float` | Apparent left angle measured relative to chord before slope correction. |
| `theta_right_apparent_deg` | `float` | Apparent right angle measured relative to chord before slope correction. |
| `substrate_profile` | `dict` | Serialized `SubstrateProfile` (`type`, `parameters`, `confidence`). |
| `substrate_radius_mm` | `float | None` | Calibrated radius of curvature $R_{\text{sub}}$ in mm (positive convex, negative concave). |
| `substrate_curvature_inv_mm` | `float | None` | Substrate curvature $\kappa_{\text{sub}} = 1 / R_{\text{sub}}$ in $\text{mm}^{-1}$. |
| `substrate_warning` | `bool` | True when detection confidence $Q < 0.75$. |
| `substrate_quality` | `float` | Confidence score $Q \in [0.0, 1.0]$. |
