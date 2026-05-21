# Pendant ITF Models Results

Generated: 2026-05-20T22:49:00

Images analyzed: prueba pend 1.png and gota pendiente 1.png

## prueba pend 1.png

Calibration summary:

- needle_rect: (263, 0, 130, 44)
- roi_rect: (201, 0, 256, 295)
- contact_points: ((np.int64(259), 47), (np.int64(397), 44))
- apex_point: (320, 275)
- Plot: ![prueba pend 1.png contour plot](docs/pendant_itf_plots/prueba_pend_1_png_contour_vs_approx.png)


Global metrics:

| Metric | Value |
|---|---:|
| surface_tension_method | young_laplace_strict |
| surface_tension_mN_m | 69.5618 |
| beta | 0.2909 |
| r0_mm | 1.4373 |
| height_mm | 3.2518 |
| diameter_mm | 2.9280 |
| volume_uL | 17.0449 |
| drop_surface_mm2 | 29.7036 |
| bond_number | 0.2909 |
| worthington_number | 0.4175 |
| strict_fit_success | 1.0000 |
| strict_fit_warning | - |
| strict_fit_stop_reason | height_cutoff |
| strict_rmse_mm | 0.0044 |
| strict_model_coverage_height_mm | 3.2518 |
| strict_observed_height_mm | 3.2518 |
| strict_observed_diameter_mm | 3.5052 |
| strict_x_offset_mm | -0.2363 |
| strict_z_offset_mm | 0.0011 |

Model comparison:

| Model | Status | Surface Tension (mN/m) | Beta | RMSE (mm) | x_offset (mm) | z_offset (mm) | n_left | n_right | arc_quality | Time (ms) | Iterations |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| young_laplace_strict | 1.0000 | 69.5618 | 0.2909 | 0.0044 | -0.2363 | 0.0011 | - | - | - | 1171 | 14 |
| minimize_adsa | ok | 72.3765 | 0.2820 | 0.0103 | -0.2351 | -0.0075 | 120.0000 | 120.0000 | - | 2467 | 11 |
| selected_plane | ok | 84.4086 | 0.2397 | - | - | - | - | - | - | 6841 | - |
| multi_selected_plane | ok | 84.4086 | 0.2397 | - | - | - | - | - | - | 1 | - |
| volume_apex_lookup | - | - | - | - | - | - | - | - | - | 1160 | - |
| **final_selected** | **young_laplace_strict** | **69.5618** | **0.2909** | - | - | - | - | - | - | - | - |

## gota pendiente 1.png

Calibration summary:

- needle_rect: (520, 0, 390, 123)
- roi_rect: (447, 0, 537, 646)
- contact_points: ((np.int64(516), 123), (np.int64(914), 124))
- apex_point: (701, 626)
- Plot: ![gota pendiente 1.png contour plot](docs/pendant_itf_plots/gota_pendiente_1_png_contour_vs_approx.png)


Global metrics:

| Metric | Value |
|---|---:|
| surface_tension_method | young_laplace_strict |
| surface_tension_mN_m | 29.3025 |
| beta | 0.3869 |
| r0_mm | 1.0758 |
| height_mm | 2.3602 |
| diameter_mm | 2.3180 |
| volume_uL | 7.6511 |
| drop_surface_mm2 | 16.6708 |
| bond_number | 0.3869 |
| worthington_number | 0.4449 |
| strict_fit_success | 1.0000 |
| strict_fit_warning | - |
| strict_fit_stop_reason | height_cutoff |
| strict_rmse_mm | 0.0017 |
| strict_model_coverage_height_mm | 2.3602 |
| strict_observed_height_mm | 2.3602 |
| strict_observed_diameter_mm | 2.4682 |
| strict_x_offset_mm | 0.0676 |
| strict_z_offset_mm | 0.0000 |

Model comparison:

| Model | Status | Surface Tension (mN/m) | Beta | RMSE (mm) | x_offset (mm) | z_offset (mm) | n_left | n_right | arc_quality | Time (ms) | Iterations |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| young_laplace_strict | 1.0000 | 29.3025 | 0.3869 | 0.0017 | 0.0676 | 0.0000 | - | - | - | 1363 | 15 |
| minimize_adsa | ok | 29.3031 | 0.3868 | 0.0065 | 0.0677 | -0.0006 | 120.0000 | 120.0000 | - | 1403 | 7 |
| selected_plane | ok | 32.8002 | 0.3456 | - | - | - | - | - | - | 0 | - |
| multi_selected_plane | ok | 32.8002 | 0.3456 | - | - | - | - | - | - | 1 | - |
| volume_apex_lookup | - | - | - | - | - | - | - | - | - | 1119 | - |
| **final_selected** | **young_laplace_strict** | **29.3025** | **0.3869** | - | - | - | - | - | - | - | - |

## Summary

| Image | Final Method | Surface Tension (mN/m) | strict_fit_success | Plot |
|---|---|---:|---:|---|
| prueba pend 1.png | young_laplace_strict | 69.5618 | True | ![plot](docs/pendant_itf_plots/prueba_pend_1_png_contour_vs_approx.png) |
| gota pendiente 1.png | young_laplace_strict | 29.3025 | True | ![plot](docs/pendant_itf_plots/gota_pendiente_1_png_contour_vs_approx.png) |