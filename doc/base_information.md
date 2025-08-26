# Axisymmetric Drop Shape Analysis for Surface Tension Measurement

## 1. Differential Equations Governing the Drop Shape

We describe the drop profile using parametric equations in terms of arc length \( s̄ \):

- \( \frac{d\bar{X}}{d\bar{S}} = \cos\phi \)  
- \( \frac{d\bar{Z}}{d\bar{S}} = \sin\phi \)  

These express how the horizontal (`X̄`) and vertical (`Z̄`) coordinates change along the surface, given the tangent angle \( \phi \). These are standard geometric relations for axisymmetric surface profiles :contentReference[oaicite:1]{index=1}.

The curvature of the interface is governed by the dimensionless form of the Young–Laplace equation:

- \( \frac{d\phi}{d\bar{S}} = 2 - \beta \bar{Z} - \frac{\sin\phi}{\bar{X}} \)  

Here, \( \beta = \frac{\Delta\rho\,g\,R_0^2}{\gamma} \) is a dimensionless gravity parameter. This compact form balances curvature, gravitational effects, and geometry — widely used in pendant-drop modeling :contentReference[oaicite:2]{index=2}.

---

## 2. Volume & Surface Area Expressions

For the axisymmetric drop, we compute:

- **Volume**:  
  \( V = \pi \int_{0}^{h} x^2(z)\,dz \)

- **Surface Area**:  
  \( A = 2\pi \int_{0}^{s} x(s)\,ds \)

These standard integrals stem from revolving the profile around the axis, widely used in fluid mechanics and tensiometry.

---

## 3. Optimization: Fitting Theory to Experimental Data

### 3.1 Objective Function

To match theoretical and experimental drop contours:

- \( \chi^2 = \sum_i \left( x_{\text{exp},i} - x_{\text{theo},i} \right)^2 \)

### 3.2 Iterative Solvers

Solution methods such as **Gauss–Newton** or **Levenberg–Marquardt** are commonly employed for non-linear least squares minimization in drop shape fitting — fundamental to **Axisymmetric Drop Shape Analysis (ADSA)** :contentReference[oaicite:3]{index=3}.

### 3.3 Initialization and Convergence

- **Initial guesses**: Use plausible starting values \( \beta_0 \) and \( R_0 \) to seed the solver.
- **Convergence criterion**: Achieve \( \Delta\chi^2 < 10^{-6} \) to ensure precise fits.

---

## 4. Extracting Surface Tension

Once \( \beta \) and \( R_0 \) are determined from the fit:

- **Formula**:  
  \( \gamma = \frac{\Delta\rho \, g \, R_0^2}{\beta} \)

This relation follows directly from the non-dimensional definition of \( \beta \), linking surface tension \( \gamma \) to measurable physical and fit parameters :contentReference[oaicite:4]{index=4}.

---

## 5. Quality Assurance Metrics

To assess the fidelity of fits:

- **RMS error** between experimental and modeled profiles.
- **Worthington number**:  
  \( \text{Wo} = \frac{V}{\tfrac{\pi D^3}{6}} \)  
  A dimensionless comparison of actual volume to a reference spherical volume.
- **Asymmetry factor**:  
  \( \frac{|L_{\text{left}} - L_{\text{right}}|}{L_{\text{avg}}} < 0.01 \)  
  Ensures drop symmetry along the central axis.

These metrics provide diagnostics for profile integrity and fit reliability.

---

## 6. Advanced Refinements & Techniques

- **Tangent method at the apex**:  
  Estimate \( R_0 \) directly by fitting the apex curvature tangent, improving initial guess quality.

- **Perturbation analysis for oscillating drops**:  
  Useful in dynamic or vibrating systems to linearize shape deviations around equilibrium.

- **Dynamic contact angle measurement for sessile drops**:  
  Captures hysteresis and time-dependent wetting behavior — an advanced consideration in surface/tensiometry studies.

---

## 7. Theoretical & Software Foundations

- **Axisymmetric Drop Shape Analysis (ADSA)**:  
  A computational framework matching experimental drop contours to Young–Laplace–derived profiles, applicable to pendant and sessile drops :contentReference[oaicite:5]{index=5}.

- **ImageJ Plugin: Pendent_Drop**:  
  An implementation of ADSA in open-source software, enabling automatic drop profile fitting to extract surface tension, volume, and surface area :contentReference[oaicite:6]{index=6}.

---

## 8. Summary Table

| **Component**            | **Description**                                                                 |
|--------------------------|---------------------------------------------------------------------------------|
| Differential Equations   | Geometrical shape + Young–Laplace in dimensionless form                        |
| Volume & Area            | Classic revolution integrals                                                   |
| Optimization             | Minimize χ² via Gauss–Newton or Levenberg–Marquardt                           |
| Surface Tension Formula  | \( \gamma = (\Delta\rho\,g\,R_0^2) / \beta \)                                  |
| Fit Quality Metrics      | RMS error, Worthington number, asymmetry factor                               |
| Advanced Techniques      | Tangent method, perturbation, dynamic contact angle                            |
| Computational Basis      | ADSA framework and ImageJ plugin implementation                               |

---

###  References

- Geometric derivation: \( dx/ds = \cos\phi \), \( dz/ds = \sin\phi \), and curvature-based ODEs :contentReference[oaicite:7]{index=7}.  
- ADSA methodology & variants (profile, height-diameter, numerical stability) :contentReference[oaicite:8]{index=8}.  
- ImageJ plugin for automated shape analysis and surface tension estimation :contentReference[oaicite:9]{index=9}.  
- Relation between fitted parameters and surface tension: \( \gamma = \Delta\rho\,g\,R_0 / \beta \), consistent with drop shape methodologies :contentReference[oaicite:10]{index=10}.


