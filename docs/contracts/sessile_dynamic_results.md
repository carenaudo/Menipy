# Dynamic Sessile Results Contract

This contract defines the `sessile_dynamic` pipeline output. Its
`schema_version` is `1.0`; it is independent from the static `sessile` 1.0
contract.

## Acceptance model

A run contains one sequence-level result and auditable per-frame records. A
frame is accepted only when calibrated drop, baseline, contacts and both raw
contact angles pass temporal association. Rejected frames retain diagnostics
but do not acquire interpolated physical values.

The sequence is accepted when it has at least 20 frames, at least 80% accepted
frames, a fixed positive `px_per_mm`, and no invalid run longer than five
frames. A sequence can be accepted without both motion states. Advancing or
receding summaries require five accepted frames in that state; hysteresis is
present only when both summaries exist. A rejected sequence is persisted with
`results={}` through the Phase-A result quarantine.

## Canonical JSON

- `pipeline`: `sessile_dynamic`.
- `schema_version`: `1.0`.
- `accepted`, `rejection_reasons`.
- `metadata`: source type/id/hash, dimensions, FPS, timestamps and frame count.
- `calibration`: fixed scale, source and initial needle samples.
- `summary`: valid fraction, duration, advancing/receding median, MAD, 95%
  deterministic-bootstrap interval, speed, state durations and hysteresis;
  optional Cox–Voinov fields `cox_voinov_theta_advancing_deg`,
  `cox_voinov_theta_receding_deg`, `cox_voinov_theta_equilibrium_deg`, and
  `cox_voinov_hysteresis_deg`; and optional tilting-plate fields
  `tilt_range_deg`, `critical_sliding_angle_deg`, `critical_frame_index`,
  `theta_advancing_critical_deg`, and `theta_receding_critical_deg`.
  The complete per-state Cox–Voinov fit remains in `cox_voinov_advancing` /
  `cox_voinov_receding`, and the complete tilting result remains in
  `tilting_plate`.
- `frames`: frame index/time, acceptance, segment, raw left/right angles,
  baseline, contacts, half-width, seven-frame robust velocity, state,
  predicted ROI, optional contour and diagnostics.

Angles are degrees, coordinates are pixels, time is seconds, contact width is
millimetres and velocity is millimetres per second. Raw angles and contacts are
the promoted scientific series. The robust velocity and any plot smoothing are
classification/visualization data and never replace raw measurements.

## State and recovery semantics

The contact half-width is measured in substrate-aligned coordinates. A
seven-frame robust local regression supplies velocity. The deadband is
`max(0.01 mm/s, 3 * MAD noise)`. Adjacent velocity estimates snap transition
boundaries; advancing/receding states require three consecutive frames.
Rejected frames are `invalid`. Reacquisition after a gap starts a new
`segment_id`; no contour, contact or angle is interpolated across segments.

## Exports

- `results.json`: complete contract, including contours and diagnostics.
- `results.csv`: one stable summary row. Its additive columns include the
  scalar Cox–Voinov and tilting-plate fields above; unavailable optional
  values are empty. Existing columns and their order remain unchanged.
- `results_frames.csv`: one row per frame, with raw geometry and state but no
  contours or masks; nested diagnostics use JSON columns.

Video timestamps are used when strictly monotonic. Otherwise constant-FPS
timestamps are generated from the container FPS. Image sequences require an
explicit FPS and natural filename ordering.

File-backed pipeline acquisition uses a lossless temporary frame store. Only
the first preview image is retained in `ctx.frames`; the excluded
`ctx.sequence_store` supplies frames during analysis and closes on termination.
This storage change does not change the per-frame result or export contract.

## Stable rejection codes

The public codes include `sequence_fps_required`,
`sequence_variable_dimensions`, `sequence_timestamps_not_monotonic`,
`dynamic_missing_calibration`, `dynamic_drop_not_detected`,
`dynamic_contacts_not_detected`, `dynamic_baseline_not_detected`,
`dynamic_contact_jump`, `dynamic_area_jump`, `dynamic_baseline_frame_drift`,
`dynamic_baseline_total_drift`, `dynamic_geometry_failed`,
`dynamic_contact_angle_invalid`, `dynamic_insufficient_frames`,
`dynamic_valid_fraction_low`, and `dynamic_gap_too_long`.

Camera streaming, imposed flow/volume, and elastic-capsule mechanics are not
part of this contract.
