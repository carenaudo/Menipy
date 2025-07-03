# Pendant Drop Methods

```{autofunction} src.models.surface_tension.jennings_pallas_beta
```

```{autofunction} src.models.surface_tension.surface_tension
```

```{autofunction} src.models.surface_tension.bond_number
```

```{autofunction} src.models.surface_tension.volume_from_contour
```

## Advanced pendant-drop metrics

- **Worthington number** $\mathrm{Wo} = V / V_{\max}$ with
  \(V_{\max}=\pi \gamma D /(\Delta\rho g)\).
  Values close to 1 indicate imminent detachment.
- **Apex curvature** $\kappa_0 = 2/R_0$.
- **Projected area** $A_{\mathrm{proj}} = \pi (D_e/2)^2$.
- **Surface area** obtained from revolution of the profile.
- **Apparent weight** $W_{\mathrm{app}} = \Delta\rho g V$.

Small drops typically give $\mathrm{Wo}<0.1$ resulting in larger
relative errors.
