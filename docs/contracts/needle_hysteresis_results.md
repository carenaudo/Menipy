# Needle-in-Sessile-Drop Hysteresis Results Contract

This contract defines the `needle_hysteresis` pipeline output schema, per-frame records,
and summary invariants. Its `schema_version` is `1.0`.

## Model and Purpose

The needle-in-sessile-drop method (ISO 19403-6 / DIN 55660-6) measures dynamic advancing
contact angles ($\theta_A$), receding contact angles ($\theta_R$), and contact angle
hysteresis ($\Delta\theta = \theta_A - \theta_R$) by continuously inflating and deflating
a sessile droplet through a stationary needle immersed in the droplet apex.

Because the needle punctures the apex, standard Young–Laplace (ADSA) ODE fitting is
unusable. The pipeline fits contact angles using either:
1. **Tangent Method 2**: High-order local polynomial fitting in substrate-aligned coordinates.
2. **CDF (Circumcircle and Difference Fitting)**: Circumcircle baseline with low-order difference polynomial.

## Canonical JSON Schema

The output JSON contains:
- `pipeline`: `"needle_hysteresis"`.
- `schema_version`: `"1.0"`.
- `accepted`: boolean. True if valid fraction $\ge 60\%$ and at least one dynamic state is characterized.
- `rejection_reasons`: list of string codes.
- `summary`:
  - `n_frames`, `n_valid_frames`, `valid_fraction`, `fps`.
  - `velocity_deadband_mm_s`.
  - `base_diameter_initial_mm`, `base_diameter_max_mm`, `base_diameter_final_mm`.
  - `theta_advancing`: object with `median_deg`, `mad_deg`, `ci95_deg`, and `n_frames`.
  - `theta_advancing_deg`: scalar float.
  - `theta_receding`: object with `median_deg`, `mad_deg`, `ci95_deg`, and `n_frames`.
  - `theta_receding_deg`: scalar float.
  - `contact_angle_hysteresis_deg`: $\theta_A - \theta_R$.
  - `advancing_frames_count`, `receding_frames_count`.
  - `advancing_duration_s`, `receding_duration_s`.
  - `advancing_velocity_median_mm_s`, `receding_velocity_median_mm_s`.
- `frames`: array of per-frame records:
  - `frame_index`, `timestamp_s`, `accepted`, `state`.
  - `theta_left_deg`, `theta_right_deg`, `theta_mean_deg`.
  - `base_diameter_mm`, `contact_velocity_mm_s`.
  - `rejection_reasons`.
- `diagnostics`: details on calibration scale, fit method, and per-frame solver RMSE.

## State Machine & Pseudo-Movement Suppression

1. Baseline contact line velocity is evaluated as:
   $$v_{CL}(t) = \frac{1}{2}\frac{d(\text{base\_diameter})}{dt}$$
2. The velocity deadband is computed adaptively from the Median Absolute Deviation of velocity first-differences:
   $$v_{\text{deadband}} = \max(0.01\,\mathrm{mm/s}, 3.0 \times \mathrm{MAD}(\Delta v))$$
3. State transitions:
   - `pinned`: $|v_{CL}| \le v_{\text{deadband}}$
   - `advancing`: $v_{CL} > +v_{\text{deadband}}$ (minimum 3 consecutive frames)
   - `receding`: $v_{CL} < -v_{\text{deadband}}$ (minimum 3 consecutive frames)
4. Pseudo-movement suppression: Contact angle reductions occurring while the contact line remains pinned at the maximum diameter are classified as `pinned`, never `receding`. True receding measurements begin only upon physical contact line retreat.

## CSV Exports

- `results.csv`: One summary row per sequence.
- `results_frames.csv`: Complete time-series with columns:
  `frame_index`, `timestamp_s`, `accepted`, `state`, `theta_left_deg`, `theta_right_deg`, `theta_mean_deg`, `base_diameter_mm`, `contact_velocity_mm_s`, `rejection_reasons`.
