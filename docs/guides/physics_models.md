# Physics Models

This document describes the physical and mathematical models used to analyze droplet and bubble shapes across Menipy's analysis pipelines.

---

## 1. Young–Laplace Equation (Axisymmetric Drop Shape)

### 1.1. Governing Principle

The static equilibrium of a curved fluid interface separating two fluid phases (liquid and gas, or two immiscible liquids) is governed by the Young–Laplace equation:

$$\Delta P = \gamma \, \kappa = \gamma \left( \frac{1}{R_1} + \frac{1}{R_2} \right)$$

where:
- $\Delta P$ is the capillary pressure jump across the interface ($\text{Pa}$).
- $\gamma$ is the interfacial or surface tension ($\text{mN/m}$ or $\text{N/m}$).
- $R_1$ and $R_2$ are the principal radii of curvature of the interface ($\text{m}$).
- $\kappa = \frac{1}{R_1} + \frac{1}{R_2}$ is the total mean curvature ($\text{m}^{-1}$).

In the presence of gravity, the pressure jump varies hydrostatically along the vertical coordinate $z$:

$$\Delta P(z) = \Delta P_0 \pm \Delta\rho \, g \, z$$

where:
- $\Delta P_0 = \frac{2\gamma}{R_0}$ is the capillary pressure at the apex ($z = 0$), where $R_1 = R_2 = R_0$.
- $\Delta\rho = \rho_{\text{heavy}} - \rho_{\text{light}}$ is the fluid density difference ($\text{kg/m}^3$).
- $g$ is the acceleration due to gravity ($9.80665\text{ m/s}^2$).

---

### 1.2. Capillary Length

The relative importance of surface tension forces compared to gravitational hydrostatic forces is parameterized by the **capillary length** $\ell_c$:

$$\ell_c = \sqrt{\frac{\gamma}{\Delta\rho \, g}}$$

For pure water at $20^\circ\text{C}$ in air ($\gamma \approx 72.8\text{ mN/m}$, $\Delta\rho \approx 998\text{ kg/m}^3$), $\ell_c \approx 2.7\text{ mm}$.
- Drops with characteristic radius $R \ll \ell_c$ ($\text{Bo} \ll 1$) are dominated by surface tension and adopt nearly spherical shapes.
- Drops with $R \gg \ell_c$ ($\text{Bo} \gg 1$) are flattened into puddles by gravity.

---

### 1.3. Bashforth–Adams Dimensionless Formulation

The axisymmetric Young–Laplace ODE system is non-dimensionalized by scaling with the apex radius $R_0$:

$$\frac{d\bar{r}}{d\bar{s}} = \cos\phi, \quad \frac{d\bar{z}}{d\bar{s}} = \sin\phi, \quad \frac{d\phi}{d\bar{s}} = 2 \pm \beta \bar{z} - \frac{\sin\phi}{\bar{r}}$$

where the dimensionless shape factor (Bond number) is:

$$\beta = \frac{\Delta\rho \, g \, R_0^2}{\gamma} = \left( \frac{R_0}{\ell_c} \right)^2$$

---

## 2. Geometric Contact Angle Models

Menipy provides three geometric contact angle models, all generalized to compute both acute ($\theta \le 90^\circ$) and obtuse ($\theta > 90^\circ$) angles seamlessly:

### 2.1. Spherical Cap Method (`spherical_cap`)

Approximates small, capillary-dominated sessile drops ($\text{Bo} \ll 1$) as spherical caps:

$$\theta = 2 \arctan\left(\frac{h}{r}\right)$$

where $h$ is the drop height above the substrate and $r$ is half the base footprint width. Valid for $0^\circ \le \theta \le 180^\circ$.

---

### 2.2. Circle Fit Method (`circle_fit`)

Fits a circle $(C, R)$ to contour points along the drop flank. The signed contact angle is computed via radial projection onto the inward substrate unit vector $\hat{\mathbf{u}}$ and apex-normal unit vector $\hat{\mathbf{n}}$:

$$\theta = \text{arctan2}(-r_u, r_n)$$

where $r_u = \hat{\mathbf{r}} \cdot \hat{\mathbf{u}}$ and $r_n = \hat{\mathbf{r}} \cdot \hat{\mathbf{n}}$. Because the circle center height relative to the substrate determines $r_n = -h_C/R$:
- Center below substrate ($h_C < 0$): $\theta < 90^\circ$ (acute).
- Center on substrate ($h_C = 0$): $\theta = 90^\circ$.
- Center above substrate ($h_C > 0$): $\theta > 90^\circ$ (obtuse).

---

### 2.3. Tangent Polynomial / SVD Method (`tangent`)

Fits a local tangent line near the contact point using weighted singular value decomposition (SVD) without assuming spherical curvature. The tangent vector is oriented into the droplet phase ($\hat{\mathbf{t}} \cdot \hat{\mathbf{n}} \ge 0$), and the angle is evaluated as:

$$\theta = \text{arctan2}(\hat{\mathbf{t}} \cdot \hat{\mathbf{n}}, \, \hat{\mathbf{t}} \cdot \hat{\mathbf{u}})$$

---

### 2.4. Curved Substrate Slope Correction

When drops reside on non-planar surfaces (e.g. fibers, lenses, cavities), the apparent angle measured against the chord is corrected for the local substrate tangent slope $\alpha_{\text{sub}}$:

$$\theta_{\text{intrinsic}} = \theta_{\text{apparent}} - \alpha_{\text{sub}}$$

See [`docs/guides/curved_substrates_and_baseline_detection.md`](curved_substrates_and_baseline_detection.md) and [`docs/guides/contact_angle_geometry.md`](contact_angle_geometry.md) for full derivations.

---

## 3. Axisymmetric Drop Shape Analysis (ADSA)

### 3.1. Full ADSA (Numerical Young–Laplace ODE)

Integrates the non-linear Bashforth–Adams ODE system using numerical Runge–Kutta integration (`solve_ivp`, RK45) from the drop apex. Standard ADSA fits apex radius $R_0$ and shape parameter $\beta$ to minimize orthogonal normal-projection residuals against the drop silhouette:

$$\min_{R_0, \beta} \sum_{i=1}^N \rho\left( d_i(R_0, \beta) \right)$$

where $d_i$ is the Euclidean distance from contour point $\mathbf{x}_i$ to the nearest point on the model curve, evaluated using `scipy.spatial.cKDTree`, and $\rho$ is the robust Soft $L_1$ loss.

---

### 3.2. Low-Bond Axisymmetric Drop Shape Analysis (LB-ADSA)

For sessile drops under weak-to-moderate gravity ($\text{Bo} \in [0.001, 0.25]$), Menipy implements the first-order analytical perturbation theory of the Young–Laplace equation (Stalder et al., EPFL, 2010; *Colloids and Surfaces A*, 364, 72–81).

In polar coordinates centered at the apex center of curvature $(0, R_0)$:

$$R(\alpha, R_0, \text{Bo}) = R_0 \left( 1 + \frac{1}{3} \text{Bo} \cdot \left[ \cos\alpha \left( \frac{1}{2} + \ln\left(\frac{2}{1 + \cos\alpha}\right) \right) - \frac{1}{2} \right] \right)$$

where $\alpha \in [0, \pi)$ is the polar angle measured from the downward apex normal.

**Key Analytical Properties:**
1. **Closed-form radial residuals**: Every experimental point $(X_i, Z_i)$ projects directly along the ray $\alpha_i = \arctan2(|X_i|, R_0 - Z_i)$ with distance $D_i = \sqrt{X_i^2 + (R_0 - Z_i)^2}$, yielding residuals $\rho_i = D_i - R(\alpha_i, R_0, \text{Bo})$ in $\mathcal{O}(1)$ time without ODE integration.
2. **Analytical profile tangents**: The local tangent angle with the horizontal substrate is:
   $$\theta(\alpha) = \arctan2\left(R \sin\alpha - R' \cos\alpha, \, R \cos\alpha + R' \sin\alpha\right)$$
   where $R'(\alpha) = \frac{1}{3} \text{Bo} R_0 \sin\alpha \left( \frac{\cos\alpha}{1 + \cos\alpha} + \ln(1 + \cos\alpha) - \frac{1}{2} - \ln 2 \right)$.
3. **Full angular range**: Supports both acute ($\theta \le 90^\circ$) and obtuse ($\theta > 90^\circ$) droplets seamlessly.
4. **Simultaneous surface tension recovery**: When fluid density difference $\Delta\rho$ is known:
   $$\gamma = \frac{\Delta\rho \, g \, R_0^2}{\text{Bo}} \quad [\text{mN/m}]$$
   converging in $< 10\text{ ms}$ with guaranteed numerical stability.

---

## 4. Contour-Based Young–Laplace Fitting

1. **Detected Contour Input**: The discrete droplet contour $(x_i, y_i)$ is extracted using sub-pixel edge detection.
2. **Profile Parameterization**: The contour is transformed into axisymmetric coordinates $r(s)$ vs. $z(s)$ centered on the detected symmetry axis.
3. **Fluid Properties**: Liquid density $\rho_l$ and gas/ambient density $\rho_g$ are specified to define the density difference $\Delta\rho = \rho_l - \rho_g$.
4. **Optimization Loop**:
   - Seed initial estimates $R_0$ and $\beta$ using Jennings–Pallas geometric formulas.
   - Numerically integrate the theoretical Young–Laplace ODE with adaptive RK45.
   - Compute normal orthogonal distances via spatial $k$-d tree.
   - Minimize the robust Soft $L_1$ loss using the Trust Region Reflective algorithm.
5. **Quality Gates**: The result is published only if the optimizer status indicates convergence, the model height covers the drop silhouette, and $\text{RMSE}_{\text{mm}} \le \max(0.05, 0.03 \cdot d_e)$.

---

## 5. Captive Bubble Analysis

A captive bubble consists of a gas bubble pinned beneath a flat solid substrate (ceiling) immersed in a surrounding liquid.

### 5.1. Inverted Buoyancy Mechanics

The physics of a captive bubble is mathematically isomorphic to an inverted pendant drop or sessile drop under inverted gravity:
- The surrounding liquid has density $\rho_{\text{liquid}} > \rho_{\text{gas}}$.
- The density difference $\Delta\rho = \rho_{\text{liquid}} - \rho_{\text{gas}} > 0$ generates an upward buoyancy force.
- The bubble apex is the lowest point of the bubble ($y_{\text{apex}} = \max(y)$ in image coordinates).
- Coordinate transformation into the axisymmetric reference frame:
  $$z = y_{\text{apex}} - y$$
  such that $z = 0$ at the apex and $z > 0$ increases upward toward the ceiling substrate at $y = y_{\text{ceiling}}$.

---

### 5.2. Bubble vs. Liquid Contact Angles

In captive bubble goniometry, optical detection observes the gas bubble contour at the ceiling. The interface forms two complementary contact angles:

1. **Bubble Contact Angle ($\theta_{\text{bubble}}$)**:
   The angle measured through the gas bubble phase:
   $$\theta_{\text{bubble}} \approx 2 \arctan\left(\frac{h_{\text{depth}}}{r_{\text{eq}}}\right)$$
2. **Liquid Contact Angle ($\theta_{\text{liquid}}$)**:
   Standard thermodynamic contact angles are defined through the denser liquid phase:
   $$\theta_{\text{liquid}} = 180^\circ - \theta_{\text{bubble}}$$

This allows captive bubble measurements to characterize hydrophilic substrates ($\theta_{\text{liquid}} < 90^\circ \implies \theta_{\text{bubble}} > 90^\circ$) and hydrophobic substrates ($\theta_{\text{liquid}} > 90^\circ \implies \theta_{\text{bubble}} < 90^\circ$) while submerged, preventing droplet evaporation.

---

## 6. Capillary Rise & Wilhelmy Plate

### 6.1. Jurin's Law for Capillary Rise

When a narrow circular tube of inner radius $r$ is immersed in a wetting liquid, capillary forces pull the liquid upwards until balanced by the weight of the elevated liquid column (Jurin, 1718):

$$2\pi r \gamma \cos\theta = \Delta\rho \, g \, (\pi r^2 h)$$

Solving for surface tension $\gamma$:

$$\gamma = \frac{\Delta\rho \, g \, h \, r}{2 \cos\theta}$$

or for capillary rise height $h$:

$$h = \frac{2\gamma \cos\theta}{\Delta\rho \, g \, r}$$

---

### 6.2. Lord Rayleigh 3rd-Order Meniscus Volume Correction

In real capillary tubes of finite radius, the liquid in the curved meniscus above the meniscus apex contributes to the hydrostatic weight. Lord Rayleigh (1915) derived an exact asymptotic correction for the effective height $h_{\text{eff}}$:

$$h_{\text{eff}} = h + \frac{r}{3} - 0.1288 \left(\frac{r^2}{h}\right) + 0.1312 \left(\frac{r^3}{h^2}\right)$$

In Menipy (`src/menipy/math/jurin.py`), this 3rd-order correction is applied whenever $r/h \le 1.0$, replacing raw height $h$ with $h_{\text{eff}}$ to guarantee high-accuracy surface tension recovery.

---

### 6.3. Exact Vertical Wilhelmy Plate Meniscus

For a vertical plate immersed in a liquid bath, the 2D Young–Laplace equation possesses an exact analytical first integral. The meniscus elevation $h$ at the plate surface is:

$$h = \sqrt{2} \, \ell_c \sqrt{1 - \sin\theta}$$

where $\ell_c = \sqrt{\frac{\gamma}{\Delta\rho g}}$ is the capillary length. Inverting this expression recovers the contact angle $\theta$ directly from the measured meniscus height:

$$\sin\theta = 1 - \frac{\Delta\rho \, g \, h^2}{2\gamma} \implies \theta = \arcsin\left(1 - \frac{\Delta\rho \, g \, h^2}{2\gamma}\right)$$

---

## 7. Interfacial Dilational Rheology

In dynamic pendant drop or oscillating bubble tensiometry, the droplet volume and surface area are subjected to periodic sinusoidal variations at frequency $f = \omega / (2\pi)$:

$$A(t) = A_0 + \Delta A \sin(\omega t)$$

The dynamic response of the surfactant monolayer causes a periodic oscillation in interfacial tension with amplitude $\Delta\gamma$ and phase shift $\delta = \phi_\gamma - \phi_A$:

$$\gamma(t) = \gamma_0 + \Delta\gamma \sin(\omega t + \delta)$$

---

### 7.1. Complex Dilational Modulus

The interfacial dilational viscoelasticity is defined as the change in surface tension relative to fractional area strain (Lucassen-Reynders & Lucassen, 1969; Miller et al., 2000):

$$E = \frac{d\gamma}{d\ln A} = A_0 \frac{\Delta\gamma}{\Delta A} e^{i\delta} = E' + i E''$$

- **Dilational Elasticity (Storage Modulus $E'$)**:
  $$E' = |E| \cos\delta$$
  quantifies reversible elastic energy stored in the interface during expansion/compression.
- **Dilational Loss Modulus ($E''$)**:
  $$E'' = |E| \sin\delta$$
  quantifies irreversible energy dissipation due to surfactant diffusion and relaxation.
- **Surface Dilational Viscosity ($\eta_d$)**:
  $$\eta_d = \frac{E''}{\omega} = \frac{E''}{2\pi f} \quad [\text{mN}\cdot\text{s/m}]$$

---

### 7.2. Rayleigh–Lamb Droplet Resonance Tensiometry

For a free droplet or levitated drop undergoing natural capillary shape oscillations without mechanical forcing, surface tension is determined from the fundamental quadrupole ($n = 2$) natural resonance frequency (Lord Rayleigh, 1879; Lamb, 1932):

$$\omega_2^2 = (2\pi f_2)^2 = \frac{24 \gamma}{(3\rho_{\text{drop}} + 2\rho_{\text{medium}}) R_0^3}$$

$$\gamma = \left( \frac{3\rho_{\text{drop}} + 2\rho_{\text{medium}}}{24} \right) (2\pi f_2)^2 R_0^3$$

---

## 8. Hydrodynamic Dynamic Wetting (Cox–Voinov)

During droplet spreading or dynamic contact angle measurements (e.g., expanding/contracting sessile drops or moving liquid fronts), viscous dissipation near the three-phase contact line alters the apparent contact angle.

### 8.1. Viscous Dissipation & The Cox–Voinov Law

The dynamic contact angle $\theta_d$ depends on the contact line velocity $v_{\text{CL}}$ through the Cox–Voinov relation (Cox, 1986; Voinov, 1976):

$$\theta_d^3 = \theta_0^3 \pm 9 \, \text{Ca} \, \ln\left( \frac{L}{\ell_m} \right)$$

where:
- $\text{Ca} = \frac{\mu |v_{\text{CL}}|}{\gamma}$ is the dimensionless Capillary number.
- $\mu$ is dynamic viscosity ($\text{Pa}\cdot\text{s}$).
- $v_{\text{CL}} = \frac{dr_{\text{contact}}}{dt}$ is the contact line velocity ($\text{m/s}$).
- $L / \ell_m$ is the ratio of macroscopic drop size to the microscopic slip cut-off scale ($\sim 1\text{ nm}$).
- $+$ corresponds to advancing contact lines ($v_{\text{CL}} > 0$).
- $-$ corresponds to receding contact lines ($v_{\text{CL}} < 0$).

---

### 8.2. Zero-Velocity Asymptotic Extrapolation

By plotting $\theta_d^3$ against contact line velocity $v_{\text{CL}}$, linear regression extracts the true static thermodynamic equilibrium contact angle $\theta_0$ as $v_{\text{CL}} \to 0$:

$$\theta_0 = \lim_{v_{\text{CL}} \to 0} \left( \theta_d^3(v_{\text{CL}}) \right)^{1/3}$$

This removes hydrodynamic viscous distortion from high-speed video contact angle measurements.

---

## 9. Droplet Retention & Sliding on Tilted Substrates (Furmidge)

### 9.1. Contact Angle Hysteresis & Retention Force

When a droplet rests on an inclined plate, gravity causes the droplet to deform asymmetrically:
- The downhill front forms the **advancing contact angle** $\theta_A$.
- The uphill rear forms the **receding contact angle** $\theta_R$.

The difference $\Delta\theta = \theta_A - \theta_R$ is the **contact angle hysteresis**. The resulting net lateral capillary retention force $F_{\text{ret}}$ opposing downward sliding is described by the Furmidge relation (Furmidge, 1962):

$$F_{\text{ret}} = \gamma \, w \left( \cos\theta_R - \cos\theta_A \right)$$

where $w$ is the droplet contact base width ($w = 2 r_{\text{contact}}$).

---

### 9.2. Critical Sliding Angle Onset

As the tilt angle $\alpha$ of the substrate increases, the gravitational force acting parallel to the surface increases:

$$F_g = m \, g \sin\alpha$$

The droplet remains pinned as long as $F_g \le F_{\text{ret}}$. Sliding begins at the **critical sliding angle** $\alpha_{\text{crit}}$:

$$m \, g \sin\alpha_{\text{crit}} = \gamma \, w \left( \cos\theta_R - \cos\theta_A \right)$$

$$\alpha_{\text{crit}} = \arcsin\left( \frac{\gamma \, w (\cos\theta_R - \cos\theta_A)}{m \, g} \right)$$

Menipy monitors temporal video sequences during substrate tilting, detecting the sliding transition and reporting $\alpha_{\text{crit}}$, retention force $F_{\text{ret}}$, and the critical advancing and receding contact angles.

---

## 10. References & Academic Bibliography

1. **Jurin, J. (1718).** "An account of some experiments shown before the Royal Society; with an enquiry into the cause of the ascent and suspension of water in capillary tubes." *Phil. Trans. R. Soc.*, 30(355), 739–747. [DOI: 10.1098/rstl.1717.0026](https://doi.org/10.1098/rstl.1717.0026)
2. **Rayleigh, Lord (1879).** "On the capillary phenomena of jets." *Proc. R. Soc. Lond.*, 29(196-199), 71–97. [DOI: 10.1098/rspl.1879.0015](https://doi.org/10.1098/rspl.1879.0015)
3. **Rayleigh, Lord (1915).** "On the theory of the capillary tube." *Proc. R. Soc. Lond. A*, 92(637), 184–195. [DOI: 10.1098/rspa.1915.0006](https://doi.org/10.1098/rspa.1915.0006)
4. **Furmidge, C. G. L. (1962).** "Studies at interfaces. I. The sliding of liquid drops on solid surfaces and a theory for spray retention." *J. Colloid Sci.*, 17(4), 309–324. [DOI: 10.1016/0095-8522(62)90011-9](https://doi.org/10.1016/0095-8522(62)90011-9)
5. **Lucassen-Reynders, E. H., & Lucassen, J. (1969).** "Properties of capillary waves." *Adv. Colloid Interface Sci.*, 2(4), 347–395. [DOI: 10.1016/0001-8686(69)80006-0](https://doi.org/10.1016/0001-8686(69)80006-0)
6. **Voinov, O. V. (1976).** "Hydrodynamics of wetting." *Fluid Dyn.*, 11(5), 714–721. [DOI: 10.1007/BF01012963](https://doi.org/10.1007/BF01012963)
7. **Cox, R. G. (1986).** "The dynamics of the spreading of liquids on a solid surface. Part 1. Viscous flow." *J. Fluid Mech.*, 131, 1–46. [DOI: 10.1017/S0022112086000032](https://doi.org/10.1017/S0022112086000032)
8. **Loglio, G., Tesei, U., & Cini, R. (1988).** "Measurement of interfacial dilatational properties by a dynamic method." *J. Colloid Interface Sci.*, 126(2), 486–492. [DOI: 10.1016/0021-9797(88)90150-6](https://doi.org/10.1016/0021-9797(88)90150-6)
9. **Miller, R., et al. (2000).** "Interfacial dilatational rheology by oscillating bubble/drop methods." *Colloids Surf. A*, 175(1-2), 125–134. [DOI: 10.1016/S0927-7757(00)00525-7](https://doi.org/10.1016/S0927-7757(00)00525-7)
10. **Stalder, A. F., et al. (2010).** "Low-bond axisymmetric drop shape analysis for surface tension and contact angle measurements of sessile drops." *Colloids Surf. A*, 364(1-3), 72–81. [DOI: 10.1016/j.colsurfa.2010.04.040](https://doi.org/10.1016/j.colsurfa.2010.04.040)

