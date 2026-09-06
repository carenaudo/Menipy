# Numerical Methods Guide

This document provides a comprehensive mathematical and algorithmic specification of the numerical techniques, differential equation solvers, parameter optimization strategies, signal processing routines, and sub-pixel edge refinement algorithms implemented across Menipy.

---

## 1. Young–Laplace Axisymmetric ODE Systems

### 1.1. Differential Geometry of Fluid Interfaces

The static equilibrium shape of an axisymmetric fluid interface (pendant drop, sessile drop, or captive bubble) is governed by the Young–Laplace equation, balancing capillary pressure across the curved interface against gravitational hydrostatic pressure:

$$\Delta P(z) = \Delta P_0 \pm \Delta\rho \, g \, z = \gamma \left( \kappa_1 + \kappa_2 \right)$$

where:
- $\gamma$ is the interfacial/surface tension ($\text{mN/m}$ or $\text{N/m}$).
- $\Delta\rho = \rho_{\text{dense}} - \rho_{\text{light}}$ is the fluid density difference ($\text{kg/m}^3$).
- $g$ is the gravitational acceleration ($9.80665\text{ m/s}^2$).
- $\Delta P_0 = \frac{2\gamma}{R_0}$ is the Laplace pressure jump at the drop apex, where the two principal radii of curvature are equal ($R_1 = R_2 = R_0$).
- $\kappa_1$ and $\kappa_2$ are the meridian and azimuthal principal curvatures.

In cylindrical coordinates $(r, z)$ parameterized by arc-length $s$ measured from the apex ($s = 0$), the geometric tangents and curvatures are:

$$\frac{dr}{ds} = \cos\phi, \quad \frac{dz}{ds} = \sin\phi$$

$$\kappa_1 = \frac{d\phi}{ds}, \quad \kappa_2 = \frac{\sin\phi}{r}$$

where $\phi(s)$ is the turning angle (the angle between the interface tangent and the horizontal radial plane).

Substituting these into the Young–Laplace equation yields the classical Bashforth–Adams system of first-order non-linear ordinary differential equations:

$$\frac{dr}{ds} = \cos\phi$$

$$\frac{dz}{ds} = \sin\phi$$

$$\frac{d\phi}{ds} = \frac{2}{R_0} \pm \frac{\Delta\rho \, g}{\gamma} z - \frac{\sin\phi}{r}$$

---

### 1.2. Dimensionless Formulation & Bond Number

To ensure numerical stability across droplet sizes from nanolitres to millilitres, Menipy non-dimensionalizes the ODE system by scaling all spatial coordinates with the apex radius of curvature $R_0$:

$$\bar{r} = \frac{r}{R_0}, \quad \bar{z} = \frac{z}{R_0}, \quad \bar{s} = \frac{s}{R_0}$$

The dimensionless shape factor (Bond number $\beta$ or $\text{Bo}$) is defined as:

$$\beta = \frac{\Delta\rho \, g \, R_0^2}{\gamma}$$

In dimensionless coordinates, the system simplifies to:

$$\frac{d\bar{r}}{d\bar{s}} = \cos\phi$$

$$\frac{d\bar{z}}{d\bar{s}} = \sin\phi$$

$$\frac{d\phi}{d\bar{s}} = 2 \pm \beta \bar{z} - \frac{\sin\phi}{\bar{r}}$$

Once integrated, dimensional coordinates are reconstructed by direct multiplication: $r = R_0 \bar{r}$, $z = R_0 \bar{z}$.

---

### 1.3. Apex Singularity Resolution ($s \to 0$)

At the apex ($s = 0$), both $\phi = 0$ and $r = 0$, causing an apparent $0/0$ division by zero in the azimuthal curvature term $\frac{\sin\phi}{r}$.

By applying L'Hôpital's rule along the meridian curve:

$$\lim_{s \to 0} \frac{\sin\phi(s)}{r(s)} = \lim_{s \to 0} \frac{\frac{d}{ds}[\sin\phi]}{\frac{dr}{ds}} = \lim_{s \to 0} \frac{\cos\phi \, \frac{d\phi}{ds}}{\cos\phi} = \frac{d\phi}{ds}(0)$$

Substituting this identity into the turning angle ODE at $s = 0$ ($z = 0$):

$$\frac{d\phi}{ds}(0) = \frac{2}{R_0} - \frac{d\phi}{ds}(0) \implies 2 \frac{d\phi}{ds}(0) = \frac{2}{R_0} \implies \frac{d\phi}{ds}(0) = \frac{1}{R_0}$$

In dimensionless variables ($\bar{R}_0 = 1$):

$$\lim_{\bar{s} \to 0} \frac{\sin\phi}{\bar{r}} = 1, \quad \frac{d\phi}{d\bar{s}}(0) = 2 - 1 = 1$$

In Menipy's implementations (`src/menipy/math/young_laplace.py` and `src/menipy/pipelines/pendant/strict_young_laplace.py`), numerical evaluation protects against divergence using an adaptive numerical threshold:

```python
if r < 1e-12:
    sin_phi_over_r = 1.0 / R0_mm  # or 1.0 in dimensionless form
else:
    sin_phi_over_r = np.sin(phi) / r
```

---

### 1.4. Coordinate Conventions across Drop Geometries

Menipy supports three distinct physical configurations:

| Geometry | Apex Location | Gravity / Buoyancy Vector | Hydrostatic Term $\frac{d\phi}{ds}$ | Coordinate Transform |
| :--- | :--- | :--- | :--- | :--- |
| **Pendant Drop** | Lowest point (tip) | $+z$ downwards from apex | $2 - \beta z - \frac{\sin\phi}{r}$ | $z = y_{\text{apex}} - y$ |
| **Sessile Drop** | Highest point (summit) | $+z$ downwards towards substrate | $2 + \beta z - \frac{\sin\phi}{r}$ | $z = y - y_{\text{apex}}$ |
| **Captive Bubble** | Lowest point (buoyant apex)| $+z$ upwards towards solid ceiling | $2 + \beta z - \frac{\sin\phi}{r}$ | $z = y_{\text{apex}} - y, \Delta\rho = \rho_{\text{liq}} - \rho_{\text{gas}} > 0$ |

---

### 1.5. Numerical Integration with SciPy `solve_ivp`

The system is integrated as an initial value problem (IVP) with initial conditions at the apex:

$$\mathbf{y}(0) = [r(0), z(0), \phi(0)]^T = [0, 0, 0]^T$$

Menipy utilizes the adaptive Runge–Kutta Dormand–Prince order 5(4) integrator (`method="RK45"` in `scipy.integrate.solve_ivp`):
- **Relative Tolerance (`rtol`)**: $10^{-5}$
- **Absolute Tolerance (`atol`)**: $10^{-6}$
- **Adaptive Step Limiting**: `max_step = s_max / 150.0` to guarantee smooth, dense profile curves for KD-tree matching.

#### Terminal Event Handlers

Numerical integration terminates dynamically when physically meaningful boundary conditions are reached:
1. `hit_axis`: Triggers if the profile curls back toward the central axis ($r \le 10^{-6}\text{ mm}$ for $s > 0.1$), indicating droplet necking and pinch-off.
2. `hit_target_h`: Triggers when the integrated height reaches the experimental drop height ($z = h_{\text{target}}$).
3. `hit_overhang`: Triggers if the profile exceeds an overhang turning angle ($\phi \ge 175^\circ$), preventing unphysical multi-valued branches.

---

## 2. Nonlinear Parameter Optimization & Profile Matching

### 2.1. Parameter Vector and State Space

In full Axisymmetric Drop Shape Analysis (ADSA), the objective is to optimize the physical parameters so that the theoretical Young–Laplace curve matches the extracted drop silhouette.

For pendant droplets, the optimized parameter vector is:

$$\mathbf{p} = [R_0, \beta, x_0, z_0]^T$$

where $(x_0, z_0)$ represent rigid image-space translation offsets between the optical detector coordinate system and the drop symmetry axis.

For sessile droplets:

$$\mathbf{p} = [R_0, \text{Bo}]^T \quad \text{or} \quad [R_0, \text{Bo}, x_0]^T$$

---

### 2.2. Orthogonal Distance Metric via `scipy.spatial.cKDTree`

Point-to-point Euclidean matching between discrete pixel points and interpolated profile points introduces discretization errors. Menipy projects each experimental contour point $\mathbf{x}_i = (x_i, z_i)$ orthogonally onto the theoretical Young–Laplace profile curve $\mathbf{y}_{\text{model}}(s)$:

$$d_i = \min_{s} \|\mathbf{x}_i - \mathbf{y}_{\text{model}}(s)\|_2$$

To compute this efficiently across thousands of contour points per video frame, Menipy constructs a spatial $k$-d tree (`scipy.spatial.cKDTree`) from the model profile points:
1. Nearest-neighbor query: `distances, indices = kdtree.query(contour_points, k=1)`.
2. Computational complexity: $\mathcal{O}(M \log M)$ construction and $\mathcal{O}(N \log M)$ evaluation, where $N$ is the number of contour points and $M$ is the number of model profile points.
3. Sign convention: Signed normal residuals are preserved, where positive residuals correspond to experimental points outside the model envelope.

---

### 2.3. Robust Loss Estimation: Soft $L_1$ Loss

Experimental contours frequently contain local perturbations caused by dust particles, illumination glare, reflections, or nozzle edges. A standard least-squares loss ($L_2$ norm) is overly sensitive to these outliers:

$$L_2(r_i) = \frac{1}{2} r_i^2$$

Menipy employs the robust **Soft $L_1$ loss** (Huber-like smooth approximation):

$$\rho(z) = 2 \left( \sqrt{1 + z} - 1 \right), \quad z = \left( \frac{r_i}{\sigma} \right)^2$$

For small residuals ($z \ll 1$), $\rho(z) \approx z$ (quadratic $L_2$ behavior). For large residuals ($z \gg 1$), $\rho(z) \approx 2\sqrt{z}$ (linear $L_1$ behavior), effectively bounding the gradient influence of spurious edge artifacts.

---

### 2.4. Trust Region Reflective (`trf`) Optimization

The optimization problem is solved using `scipy.optimize.least_squares` with `method="trf"`:
- **Box Constraints**:
  - Apex radius: $R_0 \in [0.05\text{ mm}, 50.0\text{ mm}]$
  - Bond number: $\beta, \text{Bo} \in [-10.0, 10.0]$
  - Spatial offsets: bounded by $\pm 2.0\text{ mm}$ around initial geometric estimates.
- **Physical Branch Cutoff**: The model profile integration is strictly truncated at the physical boundary (needle cannula edge or solid substrate baseline). Fits where the model height fails to span the drop contour are rejected (`strict_fit_stop_reason != "height_cutoff"`).

---

## 3. Pendant Contour-Driven Fitting & Approximation Plugins

### 3.1. Calibrated Axis Frame & Profile Envelope

The pendant pipeline executes a multi-step sequence to extract public surface tension results:
1. **Calibrate Coordinates**: Convert clipped pendant contour pixels into apex-centered millimetres using an orthonormal radial/axial frame:
   $$x_{\text{mm}} = \frac{x_{\text{px}} - x_{\text{axis}}}{\text{px\_per\_mm}}, \quad z_{\text{mm}} = \frac{y_{\text{apex}} - y_{\text{px}}}{\text{px\_per\_mm}}$$
2. **Jennings–Pallas Geometric Initialization**: Estimate apex curvature and maximum equatorial diameter $d_e$ to seed initial parameters $[R_0, \beta]$.
3. **Strict Young–Laplace Fitting**: Run `scipy.optimize.least_squares` using the RK45 ODE solver and cKDTree orthogonal projection residuals in millimetres.
4. **Reporting Gate**: Accept strict Young–Laplace values publicly only when:
   - The optimizer converges with status $> 0$.
   - Parameters are finite and not pinned against artificial bounds.
   - Profile height spans the observed drop silhouette (`strict_fit_stop_reason == "height_cutoff"`).
   - $\text{RMSE}_{\text{mm}} \le \max(0.05, 0.03 \cdot d_e)$.

---

### 3.2. Pendant Approximation Plugins

For rapid comparison or fallback diagnostics, Menipy includes three independent pendant approximation plugins adhering to the plugin contract `fn(ctx, profile_mm, physics) -> dict`:

- **Selected-Plane Approximation**: Measures the equatorial diameter $d_e$ and selected-plane diameter $d_k$ at vertical distance $k \cdot d_e$ ($k = 1.0$) from the apex. The ratio $S_k = d_k / d_e$ indexes into precomputed Young–Laplace shape factor tables $1/H(S_k)$, yielding:
  $$\gamma = \frac{\Delta\rho \, g \, d_e^2}{H}$$
  (Berry et al., 2015; [DOI: 10.1016/j.jcis.2015.05.012](https://doi.org/10.1016/j.jcis.2015.05.012)).
- **Multi-Selected-Plane Approximation**: Evaluates selected-plane estimates across multiple planes ($k \in \{0.6, 0.7, 0.8, 0.9, 1.0\}$) and reports the median and interquartile spread (Jůza et al., 2025; [DOI: 10.1007/s00396-025-05513-5](https://doi.org/10.1007/s00396-025-05513-5)).
- **Volume–Apex Lookup Approximation**: Inverts the measured drop volume, total drop height, and apex curvature against a 2D lookup grid generated from the dimensionless pendant ODE (Yeow et al., 2007; [DOI: 10.1016/j.colsurfa.2007.07.025](https://doi.org/10.1016/j.colsurfa.2007.07.025)).

---

## 4. Low-Bond Perturbation Theory (LB-ADSA)

### 4.1. Mathematical Formulation

For small sessile droplets where surface tension dominates gravity ($\text{Bo} \ll 1$, typically $\text{Bo} \in [0.001, 0.25]$), Menipy implements the analytical first-order perturbation solution of the Young–Laplace equation established by Stalder et al. (EPFL, 2010).

In polar coordinates centered at the apex center of curvature $(0, R_0)$, the radial profile $R(\alpha)$ as a function of the polar angle $\alpha \in [0, \pi)$ is expressed in closed form:

$$R(\alpha, R_0, \text{Bo}) = R_0 \left( 1 + \frac{1}{3} \text{Bo} \cdot f(\cos\alpha) \right)$$

where:

$$f(u) = u \left( \frac{1}{2} + \ln\left(\frac{2}{1 + u}\right) \right) - \frac{1}{2}, \quad u = \cos\alpha$$

---

### 4.2. Analytical Derivatives and Contact Angles

The derivative with respect to polar angle is computed analytically:

$$\frac{dR}{d\alpha} = \frac{1}{3} \text{Bo} \, R_0 \sin\alpha \left( \frac{\cos\alpha}{1 + \cos\alpha} + \ln(1 + \cos\alpha) - \frac{1}{2} - \ln 2 \right)$$

At the apex ($\alpha = 0$), $\sin\alpha = 0$, ensuring $\left.\frac{dR}{d\alpha}\right|_{\alpha=0} = 0$ identically.

The local tangent contact angle with the horizontal substrate is derived directly:

$$\theta(\alpha) = \arctan2\left( R(\alpha)\sin\alpha - R'(\alpha)\cos\alpha, \; R(\alpha)\cos\alpha + R'(\alpha)\sin\alpha \right)$$

---

### 4.3. Computational Complexity and Stability

| Feature | Full ADSA (Runge–Kutta ODE) | Low-Bond ADSA (Stalder Perturbation) |
| :--- | :--- | :--- |
| **Equation Type** | Non-linear ODE initial value problem | Closed-form analytical formula |
| **Residual Evaluation** | $\mathcal{O}(M)$ ODE integration + KD-Tree lookup | $\mathcal{O}(1)$ direct projection per point |
| **Residual Formula** | Orthogonal normal distance $d_i$ | Radial difference $\rho_i = D_i - R(\alpha_i)$ |
| **Convergence Time** | $50 - 250\text{ ms}$ per drop | $< 5\text{ ms}$ per drop |
| **Angular Range** | $0^\circ \le \theta \le 180^\circ$ | $0^\circ \le \theta \le 170^\circ$ |
| **Validity Regime** | Any Bond number $\text{Bo}$ | $\text{Bo} \le 0.25$ |

---

## 5. Variational Active Contours (DropSnake)

Menipy provides an energy-minimizing active contour (snake) engine in `src/menipy/math/active_contour.py` based on the variational principles of Kass et al. (1988) and Stalder et al. (2006):

$$E_{\text{snake}}(v) = \int_0^1 \left[ E_{\text{int}}(v(s)) + E_{\text{ext}}(v(s)) \right] ds$$

### 5.1. Energy Functional Components

1. **Internal Energy (Regularization)**:
   $$E_{\text{int}}(v(s)) = \frac{1}{2} \left[ \alpha \left| \frac{dv}{ds} \right|^2 + \beta \left| \frac{d^2v}{ds^2} \right|^2 \right]$$
   - $\alpha$ controls tension / elasticity (penalizes contour elongation).
   - $\beta$ controls rigidity / flexural stiffness (penalizes sharp bending).

2. **External Energy (Image Attraction)**:
   $$E_{\text{ext}}(v(s)) = w_{\text{edge}} E_{\text{edge}} + w_{\text{flux}} E_{\text{flux}} + w_{\text{balloon}} E_{\text{balloon}}$$
   - Edge potential: $E_{\text{edge}} = -\|\nabla (G_\sigma * I)(v(s))\|^2$ attracts the snake to maximum image intensity gradients.
   - Balloon pressure (Cohen, 1991): $F_{\text{balloon}} = w_{\text{balloon}} \, \hat{\mathbf{n}}(s)$ inflates or deflates the snake toward the interface.

---

### 5.2. Substrate-Constrained Sliding Line Boundary Conditions

Unlike traditional closed snakes, sessile droplets intersect a solid substrate. Menipy enforces a **sliding line boundary condition** (`SnakeBoundaryCondition.SLIDING_LINE`):
- Contact points $\mathbf{v}(0)$ and $\mathbf{v}(1)$ are constrained to slide freely along the detected baseline line $\mathbf{p}_1 + \lambda (\mathbf{p}_2 - \mathbf{p}_1)$.
- At each iteration, normal components perpendicular to the substrate are projected out:
  $$\Delta \mathbf{v}_{\text{contact}} = (\Delta \mathbf{v} \cdot \hat{\mathbf{u}}_{\text{sub}}) \hat{\mathbf{u}}_{\text{sub}}$$
  where $\hat{\mathbf{u}}_{\text{sub}}$ is the unit vector parallel to the substrate.
- This allows contact lines to settle into their true mechanical equilibrium without artificial pinning.

---

## 6. Interfacial Dilational Rheology & Fourier Analysis

When analyzing oscillating pendant drops or bubbles, the interfacial area undergoes harmonic perturbations, inducing dynamic changes in surface tension:

$$A(t) = A_0 + \Delta A \sin(\omega t)$$

$$\gamma(t) = \gamma_0 + \Delta \gamma \sin(\omega t + \delta)$$

where $\omega = 2\pi f$ is the angular frequency ($\text{rad/s}$) and $\delta = \phi_\gamma - \phi_A$ is the phase shift.

---

### 6.1. Harmonic Regression and Spectral Peak Discovery

To extract the fundamental frequency $f$, amplitude, and phase shift without distortion, Menipy executes a two-stage signal processing pipeline (`src/menipy/math/rheology.py`):

1. **Windowed FFT Frequency Discovery**:
   - The DC offset is removed: $\tilde{y}(t) = y(t) - \bar{y}$.
   - A Hann window is applied to eliminate spectral leakage:
     $$w_k = 0.5 \left( 1 - \cos\left(\frac{2\pi k}{N - 1}\right) \right)$$
   - The signal is zero-padded to the next power of two: $N_{\text{fft}} = 2^{\lceil \log_2 N \rceil + 1}$.
   - The real fast Fourier transform (`np.fft.rfft`) identifies the dominant frequency peak:
     $$k^* = \arg\max_{k \ge 1} |X_k|, \quad f_{\text{init}} = \frac{k^*}{N_{\text{fft}} \Delta t}$$

2. **Nonlinear Least-Squares Sinusoidal Fit**:
   - The signal is fitted to the 4-parameter harmonic model:
     $$y(t) = y_0 + A \sin(2\pi f t + \phi)$$
   - Optimization using `scipy.optimize.curve_fit` with parameter bounds:
     $$f \in [0.5 f_{\text{init}}, 1.5 f_{\text{init}} + 10.0], \quad \phi \in [-\pi, \pi]$$

---

### 6.2. Viscoelastic Dilational Moduli

From the fitted amplitudes and phase difference, Menipy computes the complex dilational modulus $\tilde{E}$ (Lucassen-Reynders & Lucassen, 1969; Loglio et al., 1988):

$$|E| = A_0 \frac{\Delta\gamma}{\Delta A} \quad [\text{mN/m}]$$

- **Dilational Elasticity (Storage Modulus $E'$)**:
  $$E' = |E| \cos\delta$$
- **Dilational Viscous Dissipation (Loss Modulus $E''$)**:
  $$E'' = |E| \sin\delta$$
- **Surface Dilational Viscosity ($\eta_d$)**:
  $$\eta_d = \frac{E''}{\omega} = \frac{E''}{2\pi f} \quad [\text{mN}\cdot\text{s/m}]$$

---

### 6.3. Rayleigh–Lamb Quadrupole Droplet Resonance

For freely levitated or vibrating droplets, surface tension is determined from the natural quadrupole oscillation frequency ($n = 2$) (Lord Rayleigh, 1879; Lamb, 1932):

$$\omega_n^2 = \frac{n(n - 1)(n + 2) \gamma}{\rho_1 R_0^3 + \left( \frac{n}{n + 1} \right) \rho_2 R_0^3}$$

For the fundamental quadrupole mode ($n = 2$):

$$\omega_2^2 = (2\pi f_2)^2 = \frac{24 \gamma}{3\rho_1 R_0^3 + 2\rho_2 R_0^3}$$

Inverting for surface tension:

$$\gamma = \left( \frac{3\rho_1 + 2\rho_2}{24} \right) (2\pi f_2)^2 R_0^3 \quad [\text{N/m}]$$

---

## 7. Hydrodynamic Dynamic Wetting & Contact Line Extrapolation

### 7.1. Cox–Voinov Contact Angle Law

During dynamic wetting or dewetting, the apparent dynamic contact angle $\theta_d$ deviates from the static equilibrium contact angle $\theta_0$ due to viscous dissipation in the wedge near the moving contact line (Cox, 1986; Voinov, 1976):

$$\theta_d^3 = \theta_0^3 \pm 9 \, \text{Ca} \, \ln\left( \frac{L}{\ell_m} \right)$$

where:
- $\text{Ca} = \frac{\mu |v_{\text{CL}}|}{\gamma}$ is the dimensionless Capillary number.
- $\mu$ is dynamic viscosity ($\text{Pa}\cdot\text{s}$).
- $v_{\text{CL}}$ is the contact line velocity ($\text{m/s}$ or $\text{mm/s}$).
- $L$ is the macroscopic length scale (droplet capillary length $\ell_c$).
- $\ell_m$ is the microscopic cut-off length scale (molecular slip length $\sim 1\text{ nm}$).
- $+$ applies to advancing contact lines ($v_{\text{CL}} > 0$).
- $-$ applies to receding contact lines ($v_{\text{CL}} < 0$).

---

### 7.2. Linear Least-Squares Extrapolation to Zero Velocity

Menipy linearizes the Cox–Voinov relation:

$$\theta_d^3 = a + b \cdot v_{\text{CL}}$$

where:
- Dependent variable: $y_i = \theta_{d,i}^3$ (with $\theta$ in radians).
- Independent variable: $x_i = v_{\text{CL},i}$.
- Model intercept: $a = \theta_0^3$.
- Model slope: $b = 9 \left( \frac{\mu}{\gamma} \right) \ln\left( \frac{L}{\ell_m} \right)$.

The static equilibrium contact angle $\theta_0$ is recovered by inverting the intercept:

$$\theta_0 = a^{1/3} \quad [\text{rad}] = \frac{180}{\pi} a^{1/3} \quad [^\circ]$$

When viscosity $\mu$ and surface tension $\gamma$ are provided, Menipy computes the microscopic length ratio parameter:

$$\ln\left( \frac{L}{\ell_m} \right) = \frac{|b| \, \gamma}{9 \mu}$$

---

### 7.3. Furmidge Droplet Retention Force & Tilting Plate Analysis

For a sessile droplet on an inclined substrate tilted at angle $\alpha$, the pinning force per unit base width $f_{\text{ret}}$ and the total lateral retention force $F_{\text{ret}}$ opposing gravitational sliding are governed by the Furmidge relation (Furmidge, 1962):

$$f_{\text{ret}} = \gamma \left( \cos\theta_R - \cos\theta_A \right) \quad [\text{mN/m}]$$

$$F_{\text{ret}} = \gamma \, w \left( \cos\theta_R - \cos\theta_A \right) \quad [\mu\text{N}]$$

where:
- $\theta_A$ is the advancing (downhill) contact angle.
- $\theta_R$ is the receding (uphill) contact angle.
- $w$ is the droplet contact base diameter / width ($w = 2 r_{\text{contact}}$).

#### Critical Sliding Angle Prediction

Gravitational force pulling the droplet down the incline is $F_g = m g \sin\alpha$. Sliding commences at the critical sliding angle $\alpha_{\text{crit}}$ where gravity balances lateral retention:

$$m g \sin\alpha_{\text{crit}} = \gamma \, w \left( \cos\theta_R - \cos\theta_A \right)$$

$$\alpha_{\text{crit}} = \arcsin\left( \frac{F_{\text{ret}}}{m g} \right)$$

In video sequences, Menipy tracks the substrate baseline angle across all frames, identifies sliding onset when contact line velocity exceeds $0.05\text{ mm/s}$, and records $\alpha_{\text{crit}}$, $\theta_A$, and $\theta_R$ at the critical frame.

---

## 8. Curvilinear Sub-Pixel Edge Detection (Steger Algorithm)

To achieve measurement reproducibility beyond pixel resolution ($< 0.05\text{ px}$), Menipy implements normal gradient profiling based on Steger's unbiased curvilinear edge detector (Steger, 1998; Alvarez et al., 2008) in `plugins/auto_subpixel_edge.py`.

```
                    Contour Point p_i
                           •
   ------------------------|------------------------> Contour Tangent t_i
                           |
                           |  Normal n_i = (-t_y, t_x)
                           v
        •      •      •    •    •      •      •       Sample grid along n_i
       s=-3   s=-2   s=-1 s=0  s=+1   s=+2   s=+3
```

### 8.1. Tangent & Normal Field Computation

For discrete contour points $\mathbf{p}_i = (x_i, y_i)$:
1. Central difference tangents:
   $$\mathbf{t}_i = \frac{\mathbf{p}_{i+1} - \mathbf{p}_{i-1}}{2}$$
2. Unit normal vectors directed outwards:
   $$\hat{\mathbf{n}}_i = \begin{bmatrix} -t_{y,i} / \|\mathbf{t}_i\| \\ t_{x,i} / \|\mathbf{t}_i\| \end{bmatrix}$$

---

### 8.2. Normal Intensity Profiling via Bilinear Remapping

A 1D spatial coordinate grid is constructed perpendicular to each contour point:

$$\mathbf{p}_i(s) = \mathbf{p}_i + s \, \hat{\mathbf{n}}_i, \quad s \in [-s_{\max}, +s_{\max}]$$

Intensity profiles $I_i(s)$ are extracted from the smoothed grayscale image using OpenCV's optimized bilinear interpolation (`cv2.remap` with `INTER_LINEAR` and `BORDER_REFLECT`).

---

### 8.3. Parabolic Peak Refinement

1. The 1D directional gradient magnitude along the normal is computed:
   $$g_i(s) = \left| \frac{\partial I_i}{\partial s} \right|$$
2. The discrete maximum sample index $k^* = \arg\max_k g_i(s_k)$ is located.
3. Sub-pixel continuous peak offset $\delta s$ is calculated by fitting a 3-point parabola through $(s_{k-1}, g_{k-1})$, $(s_k, g_k)$, and $(s_{k+1}, g_{k+1})$:
   $$\delta s = \frac{g_{k-1} - g_{k+1}}{2 \left( g_{k-1} - 2g_k + g_{k+1} \right)} \cdot \Delta s$$
4. The refined sub-pixel vertex position is updated:
   $$\mathbf{p}_{i, \text{sub}} = \mathbf{p}_i + (s_{k^*} + \delta s) \hat{\mathbf{n}}_i$$

---

## 9. Specular Reflection Cusp & Baseline Identification

On specular reflective substrates (e.g., polished silicon wafers, glass, mirror-finish metals), an optical reflection of the droplet appears below the physical substrate. The true droplet contour and its inverted reflection merge at the three-phase contact line, forming an hourglass waist.

```
       /        \
      /  Droplet \
     (            )
      \          /
------->  Neck  <------- Substrate Baseline Line (y = y_cusp)
      /          \
     ( Reflection )
      \          /
```

Menipy identifies this contact baseline using the reflection cusp algorithm (`src/menipy/common/geometry.py`):

1. **Horizontal Profile Width Function**:
   The droplet is sliced into vertical bins $y_k \in [y_{\min} + 0.35 h, y_{\max} - 0.05 h]$ of height $\Delta y = 3\text{ px}$. For each bin, the horizontal width is:
   $$w(y_k) = \max(x|_{y_k}) - \min(x|_{y_k})$$

2. **Moving Average Smoothing**:
   $$w_{\text{smooth}}(y) = w(y) * K_{\text{box}}$$

3. **Necking Cusp Invariant**:
   The reflection baseline corresponds to the interior local minimum of width:
   $$\frac{dw}{dy} = 0, \quad \frac{d^2w}{dy^2} > 0$$

4. **Prominence Filter**:
   To prevent false positives from image noise, the neck must exhibit a relative prominence $> 3\%$:
   $$\frac{w_{\text{above}} - w_{\min}}{w_{\min}} > 0.03 \quad \text{and} \quad \frac{w_{\text{below}} - w_{\min}}{w_{\min}} > 0.03$$
   When detected, the baseline is established at $y = y_{\text{cusp}}$ with high confidence.

---

## 10. Discrete Volumetric & Surface Area Quadrature

### 10.1. Axisymmetric Solid of Revolution Integration

Drop volume and interfacial surface area are evaluated using discrete trapezoidal quadrature (`scipy.integrate.trapezoid`):

- **Rotational Volume**:
  $$V = \pi \int_{z_0}^{z_{\text{base}}} r(z)^2 dz \approx \pi \sum_{i=1}^{N-1} \frac{r_i^2 + r_{i+1}^2}{2} (z_{i+1} - z_i)$$
- **Surface Area**:
  $$S = 2\pi \int r(s) ds \approx 2\pi \sum_{i=1}^{N-1} \frac{r_i + r_{i+1}}{2} \sqrt{(r_{i+1} - r_i)^2 + (z_{i+1} - z_i)^2}$$

---

### 10.2. Lord Rayleigh 3rd-Order Meniscus Volume Correction

In capillary rise experiments, liquid rises to height $h$ in a tube of radius $r$. For finite tubes, liquid in the meniscus contributes to the hydrostatic head. Lord Rayleigh (1915) derived the 3rd-order asymptotic correction for the effective rise height $h_{\text{eff}}$:

$$h_{\text{eff}} = h + \frac{r}{3} - 0.1288 \left( \frac{r^2}{h} \right) + 0.1312 \left( \frac{r^3}{h^2} \right)$$

Surface tension is then computed using Jurin's Law (`src/menipy/math/jurin.py`):

$$\gamma = \frac{\Delta\rho \, g \, h_{\text{eff}} \, r}{2 \cos\theta}$$

---

### 10.3. Exact Vertical Wilhelmy Plate Meniscus Equation

For a vertical Wilhelmy plate, the exact first integral of the 2D Young–Laplace equation provides the relation between meniscus rise height $h$ and contact angle $\theta$:

$$h = \sqrt{2} \, \ell_c \sqrt{1 - \sin\theta}$$

where $\ell_c = \sqrt{\frac{\gamma}{\Delta\rho g}}$ is the capillary length. Inverting for contact angle:

$$\sin\theta = 1 - \frac{\Delta\rho \, g \, h^2}{2\gamma} \implies \theta = \arcsin\left( 1 - \frac{\Delta\rho \, g \, h^2}{2\gamma} \right)$$

---

## 11. Droplet Apex Detection & Sub-Pixel Summit Refinement (`menipy.math.apex`)

Accurate location of the droplet apex (summit) is fundamental to Axisymmetric Drop Shape Analysis (ADSA). The apex coordinates define the coordinate origin $(r=0, z=0)$ for Young–Laplace ODE integration, establish the optical symmetry axis, and directly govern the apex radius of curvature $R_0 = 1/\kappa_0$.

### 11.1. Crest Discretization & Multi-Point Median Centering (`detect_apex_flat`)

On discrete pixel grids or for droplets with low Bond numbers, the crown of the drop is nearly horizontal ($dy/dx \approx 0$). Multiple discrete contour points share the exact same extreme coordinate:
- For sessile drops and capillary rise menisci: $y_{\min} = \min_i y_i$
- For pendant drops and captive bubbles: $y_{\max} = \max_i y_i$

Standard `np.argmin` or `np.argmax` implementations deterministically return the first (leftmost) index encountering the extreme value, biasing the summit toward the droplet flank by multiple pixels:

$$x_{\text{naive}} = \arg\min_i y_i \implies \text{leftmost edge of flat crest}$$

Menipy resolves this bias by identifying all discrete points belonging to the crest plateau ($|y_i - y_{\text{ext}}| \le \epsilon$):
- If multiple points share the extreme plateau ($N \ge 2$), the horizontal apex coordinate is computed as the geometric median/centroid:

$$x_{\text{apex}} = \text{median}(\{x_i \mid |y_i - y_{\text{ext}}| \le \epsilon\}), \quad y_{\text{apex}} = y_{\text{ext}}$$

This centers the apex on symmetric drops, eliminating discretization bias.

### 11.2. Planar Tilted Substrates & Normal Projection (`detect_apex_normal`)

When a sessile drop sits on an inclined plate (tilted at an angle $\alpha$ relative to horizontal), the highest point in image coordinates ($\min y$) does not coincide with the physical apex. Due to gravity and inclination, image vertical $\min(y)$ is skewed toward the uphill contact flank.

The true physical summit is defined as the contour point maximizing the perpendicular distance from the substrate chord $\mathbf{p}_1 \to \mathbf{p}_2$:

$$h_{\perp, i} = (\mathbf{p}_i - \mathbf{p}_1) \cdot \hat{\mathbf{n}}$$

where $\hat{\mathbf{n}}$ is the unit inward normal perpendicular to the substrate unit tangent $\hat{\mathbf{u}} = \frac{\mathbf{p}_2 - \mathbf{p}_1}{\|\mathbf{p}_2 - \mathbf{p}_1\|}$:

$$\hat{\mathbf{n}} = (-u_y, u_x) \quad \text{such that } \hat{\mathbf{n}} \text{ points into the droplet fluid}$$

The apex coordinate is reconstructed along the substrate frame:

$$\mathbf{p}_{\text{apex}} = \mathbf{p}_1 + \text{median}(\{u_j\}) \, \hat{\mathbf{u}} + h_{\max} \, \hat{\mathbf{n}}$$

### 11.3. Curved Substrates: Radial & Elevation Clearance (`detect_apex_curved_substrate`)

For drops perched on non-planar surfaces (convex cylinders, spherical beads, or cylindrical fibers), the apex corresponds to the maximum clearance from the substrate boundary:
- **Convex Circular Arcs (Cylinders / Spheres / Fibers)**: Given substrate center $\mathbf{c}_{\text{sub}}$ and radius $R_{\text{sub}}$, the radial clearance is:

  $$d_i = \|\mathbf{p}_i - \mathbf{c}_{\text{sub}}\| - R_{\text{sub}}$$

  The apex maximizes $d_i$ along the droplet crown.
- **Concave Cavities / Microwells**: For captive bubbles or sessile drops inside curved wells, clearance is inverted:

  $$d_i = R_{\text{sub}} - \|\mathbf{p}_i - \mathbf{c}_{\text{sub}}\|$$

- **General Polynomial Substrates**: For arbitrary $y_{\text{sub}}(x)$, clearance is evaluated as elevation difference $|y_{\text{sub}}(x_i) - y_i|$.

### 11.4. Continuous Sub-Pixel Refinement & Asymmetry Quantification (`refine_apex_polynomial`)

Contour coordinates in digital images have finite pixel resolution. Menipy refines candidate apex locations to continuous sub-pixel coordinates by transforming local contour vertices into the substrate-aligned Frenet frame $(\xi, \eta)$:

$$\xi_i = (\mathbf{p}_i - \mathbf{p}_{\text{init}}) \cdot \hat{\mathbf{u}}, \quad \eta_i = (\mathbf{p}_i - \mathbf{p}_{\text{init}}) \cdot \hat{\mathbf{n}}$$

A local polynomial of order 2 or 3 is fitted within a search radius $w$:

$$\eta(\xi) = a\xi^2 + b\xi + c \quad (+ \, d\xi^3)$$

1. **Continuous Sub-Pixel Peak**:
   Setting $\frac{d\eta}{d\xi} = 0$:
   - For parabolic fit ($d=0$): $\xi^* = -\frac{b}{2a}$
   - For cubic fit: the stationary root satisfying negative second derivative $6d\xi^* + 2a < 0$.
   The refined coordinate in image space is:

   $$\mathbf{p}_{\text{refined}} = \mathbf{p}_{\text{init}} + \xi^* \hat{\mathbf{u}} + \eta(\xi^*) \hat{\mathbf{n}}$$

2. **Apex Radius of Curvature ($R_0$)**:
   The apex curvature $\kappa_0$ of the profile is directly determined by the second derivative at the summit:

   $$\kappa_0 = \left| \frac{d^2\eta}{d\xi^2} \right|_{\xi^*} = |2a|, \quad R_0 = \frac{1}{\kappa_0} = \frac{1}{|2a|}$$

3. **Asymmetric Tangential Shift**:
   The coordinate $\xi^*$ quantifies tangential summit displacement due to contact angle hysteresis, windage, or substrate heterogeneity.

---

## 12. References & Bibliography

1. **Bashforth, F., & Adams, J. C. (1883).**  
   *An Attempt to Test the Theories of Capillary Action by Comparing the Theoretical and Measured Forms of Drops of Fluid.* Cambridge University Press.
2. **Rotenberg, Y., Boruvka, L., & Neumann, A. W. (1983).**  
   "Determination of surface tension and contact angle from the shapes of axisymmetric fluid interfaces."  
   *Journal of Colloid and Interface Science*, 93(1), 169–183. [DOI: 10.1016/0021-9797(83)90396-X](https://doi.org/10.1016/0021-9797(83)90396-X)
3. **Song, B., & Springer, J. (1996).**  
   "Determination of interfacial tension from the profile of a pendant drop using computer aided image processing: 1. Theoretical."  
   *Colloids and Surfaces A: Physicochemical and Engineering Aspects*, 112(1), 41–52. [DOI: 10.1016/0927-7757(95)03478-2](https://doi.org/10.1016/0927-7757(95)03478-2)
4. **Berry, J. D., Neeson, M. J., Dagastine, R. R., Chan, D. Y., & Tabor, R. F. (2015).**  
   "Measurement of surface and interfacial tension using pendant drop tensiometry."  
   *Journal of Colloid and Interface Science*, 454, 226–237. [DOI: 10.1016/j.jcis.2015.05.012](https://doi.org/10.1016/j.jcis.2015.05.012)
5. **Extrand, C. W., & Moon, M. W. (2008).**  
   "Indirect Measurement of Contact Angles on Curved Surfaces."  
   *Langmuir*, 24(17), 9470–9473. [DOI: 10.1021/la801091m](https://doi.org/10.1021/la801091m)
6. **Carroll, B. J. (1976).**  
   "The accurate measurement of contact angle, phase volume, and surface area of drops on cylindrical fibers."  
   *Journal of Colloid and Interface Science*, 57(3), 488–495. [DOI: 10.1016/0021-9797(76)90227-7](https://doi.org/10.1016/0021-9797(76)90227-7)
7. **Stalder, A. F., Melchior, T., Müller, M., Sage, D., Blu, T., & Unser, M. (2010).**  
   "Low-bond axisymmetric drop shape analysis for surface tension and contact angle measurements of sessile drops."  
   *Colloids and Surfaces A: Physicochemical and Engineering Aspects*, 364(1-3), 72–81. [DOI: 10.1016/j.colsurfa.2010.04.040](https://doi.org/10.1016/j.colsurfa.2010.04.040)
8. **Stalder, A. F., Kulik, G., Sage, D., Barbieri, L., & Hoffmann, P. (2006).**  
   "A snake-based approach to accurate determination of both contact points and contact angles."  
   *Colloids and Surfaces A: Physicochemical and Engineering Aspects*, 286(1-3), 92–103. [DOI: 10.1016/j.colsurfa.2006.03.008](https://doi.org/10.1016/j.colsurfa.2006.03.008)
9. **Cox, R. G. (1986).**  
   "The dynamics of the spreading of liquids on a solid surface. Part 1. Viscous flow."  
   *Journal of Fluid Mechanics*, 131, 1–46. [DOI: 10.1017/S0022112086000032](https://doi.org/10.1017/S0022112086000032)
10. **Voinov, O. V. (1976).**  
    "Hydrodynamics of wetting."  
    *Fluid Dynamics*, 11(5), 714–721. [DOI: 10.1007/BF01012963](https://doi.org/10.1007/BF01012963)
11. **Furmidge, C. G. L. (1962).**  
    "Studies at interfaces. I. The sliding of liquid drops on solid surfaces and a theory for spray retention."  
    *Journal of Colloid Science*, 17(4), 309–324. [DOI: 10.1016/0095-8522(62)90011-9](https://doi.org/10.1016/0095-8522(62)90011-9)
12. **Lucassen-Reynders, E. H., & Lucassen, J. (1969).**  
    "Properties of capillary waves."  
    *Advances in Colloid and Interface Science*, 2(4), 347–395. [DOI: 10.1016/0001-8686(69)80006-0](https://doi.org/10.1016/0001-8686(69)80006-0)
13. **Loglio, G., Tesei, U., & Cini, R. (1988).**  
    "Measurement of interfacial dilatational properties by a dynamic method."  
    *Journal of Colloid and Interface Science*, 126(2), 486–492. [DOI: 10.1016/0021-9797(88)90150-6](https://doi.org/10.1016/0021-9797(88)90150-6)
14. **Miller, R., et al. (2000).**  
    "Interfacial dilatational rheology by oscillating bubble/drop methods."  
    *Colloids and Surfaces A*, 175(1-2), 125–134. [DOI: 10.1016/S0927-7757(00)00525-7](https://doi.org/10.1016/S0927-7757(00)00525-7)
15. **Rayleigh, Lord (1879).**  
    "On the capillary phenomena of jets."  
    *Proceedings of the Royal Society of London*, 29(196-199), 71–97. [DOI: 10.1098/rspl.1879.0015](https://doi.org/10.1098/rspl.1879.0015)
16. **Rayleigh, Lord (1915).**  
    "On the theory of the capillary tube."  
    *Proceedings of the Royal Society of London. Series A*, 92(637), 184–195. [DOI: 10.1098/rspa.1915.0006](https://doi.org/10.1098/rspa.1915.0006)
17. **Jurin, J. (1718).**  
    "An account of some experiments shown before the Royal Society; with an enquiry into the cause of the ascent and suspension of water in capillary tubes."  
    *Philosophical Transactions of the Royal Society*, 30(355), 739–747. [DOI: 10.1098/rstl.1717.0026](https://doi.org/10.1098/rstl.1717.0026)
18. **Steger, C. (1998).**  
    "An unbiased detector of curvilinear structures."  
    *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 20(2), 113–125. [DOI: 10.1109/34.484400](https://doi.org/10.1109/34.484400)
19. **van der Kooij, H. M., et al. (2016).**  
    "Drop-profile analysis for liquid-on-liquid and liquid-on-solid contact angle goniometry."  
    *Langmuir*, 32(43), 11214–11224. [DOI: 10.1021/acs.langmuir.6b02663](https://doi.org/10.1021/acs.langmuir.6b02663)

