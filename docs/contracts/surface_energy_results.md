# Surface Free Energy Results Contract

This contract defines the `surface_energy` pipeline output schema and guarantees. Its
`schema_version` is `1.0`.

## Model and Purpose

Surface Free Energy (SFE) analysis calculates the solid surface energy ($\gamma_S$)
and its dispersive ($\gamma_S^d$) and polar ($\gamma_S^p$) components from measured
contact angles ($\theta$) of test probe liquids with known surface tension components
($\gamma_L, \gamma_L^d, \gamma_L^p$).

Unlike optical shape analysis pipelines, `surface_energy` operates on numerical contact
angle inputs. It supports:
1. **OWRK (Owens–Wendt–Rabel–Kaelble)** linear regression ($y = mx + b$).
2. **Wu (Harmonic Mean)** non-linear two-liquid method.

## Canonical JSON Output

A single run produces a JSON structure conforming to the following layout:

```json
{
  "pipeline": "surface_energy",
  "schema_version": "1.0",
  "substrate": "Glass slide #1",
  "warnings": [],
  "owrk": {
    "gamma_s_dispersive_mN_m": 36.1,
    "gamma_s_polar_mN_m": 9.56,
    "gamma_s_total_mN_m": 45.66,
    "slope": 3.0912,
    "intercept": 6.0087,
    "r_squared": 0.9655,
    "se_gamma_s_dispersive": 7.28,
    "se_gamma_s_polar": 3.614,
    "se_gamma_s_total": 4.927,
    "data_points": [
      {
        "liquid": "Water",
        "contact_angle_deg": 65.3,
        "x": 1.5295,
        "y": 11.0537,
        "y_predicted": 10.7367,
        "residual": 0.317
      },
      {
        "liquid": "Diiodomethane",
        "contact_angle_deg": 42.1,
        "x": 0.0,
        "y": 6.2079,
        "y_predicted": 6.0087,
        "residual": 0.1992
      }
    ],
    "warnings": []
  },
  "wu": {
    "gamma_s_dispersive_mN_m": 39.19,
    "gamma_s_polar_mN_m": 15.35,
    "gamma_s_total_mN_m": 54.54,
    "liquid_names": [
      "Water",
      "Diiodomethane"
    ],
    "contact_angles_deg": [
      65.3,
      42.1
    ],
    "warnings": []
  }
}
```

### Top-Level Attributes
- `pipeline`: `"surface_energy"`.
- `schema_version`: `"1.0"`.
- `substrate`: Substrate label (optional, string or `null`).
- `warnings`: List of global input validation warnings.
- `owrk`: Present when `method` is `"owrk"` or `"both"`.
- `wu`: Present when `method` is `"wu"` or `"both"`.

### OWRK Block Attributes
- `gamma_s_dispersive_mN_m`: Solid dispersive surface energy ($\gamma_S^d = b^2$).
- `gamma_s_polar_mN_m`: Solid polar surface energy ($\gamma_S^p = m^2$).
- `gamma_s_total_mN_m`: Total solid surface free energy ($\gamma_S = \gamma_S^d + \gamma_S^p$).
- `slope`: Rabel regression slope $m = \sqrt{\gamma_S^p}$.
- `intercept`: Rabel regression intercept $b = \sqrt{\gamma_S^d}$.
- `r_squared`: Coefficient of determination $R^2$ (null for $N=2$).
- `se_gamma_s_dispersive`, `se_gamma_s_polar`, `se_gamma_s_total`: First-order Taylor propagated standard errors.
- `data_points`: List of per-liquid evaluation records ($x, y, y_{pred}, \text{residual}$).
- `warnings`: Method-specific diagnostic warnings (e.g. negative slope constraints).

### Wu Block Attributes
- `gamma_s_dispersive_mN_m`: Dispersive surface energy $\gamma_S^d$.
- `gamma_s_polar_mN_m`: Polar surface energy $\gamma_S^p$.
- `gamma_s_total_mN_m`: Total solid surface free energy.
- `liquid_names`: Exactly 2 liquid names used.
- `contact_angles_deg`: Contact angles for the 2 liquids.
- `warnings`: Method-specific warnings.

## Quality Assurance & Rejection Semantics

The pipeline validation stage populates `ctx.qa`:
- `ok`: `true` if regression converges and all values are physically meaningful; `false` if fatal numerical failures or negative intercept occurs without recovery.
- `rejection_reasons`: Populated if `ok == false`.
- If an OWRK slope is negative ($m < 0$), the pipeline flags a warning and applies the physical constraint $\gamma_S^p = 0$ while re-fitting the intercept.
