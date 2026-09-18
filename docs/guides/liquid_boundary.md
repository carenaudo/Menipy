# Liquid boundary and measured contours

Calibration previously drew the entire closed segmentation contour. Near a
sessile contact line, that return path could follow bright internal patches or
substrate texture. The intended liquid region is bounded by the free surface
and the solid–liquid contact segment between P1 and P2.

`common/liquid_boundary.py` builds a separate display boundary from the ordered
contour and contact points. It selects the outer cyclic arc on the liquid side,
removes solid-side samples and closes the region with the contact chord. Sessile
liquid is above the chord in image coordinates; pendant liquid is below it.
Tilted straight substrates use projected display endpoints on the actual line.
Consecutive duplicates and exactly collinear intermediate points are removed;
there is no smoothing, convex hull or approximate simplification.

`CalibrationResult.liquid_boundary` is optional and separate from `drop_contour`.
The wizard and applied calibration overlays use the boundary, including its
display contact points. The calibration service updates it after restoring
manual regions, and drawing a new manual region refreshes the display geometry.
Raw measured contours remain diagnostic evidence. `LiquidGeometry` is the
authoritative apex-side result for straight boundaries: it keeps observed
free-surface samples separate from the closed region used for display and area.
The closing contact segment is never passed to edge or profile fitting.

Missing, degenerate, or ambiguous straight contact geometry is unresolved; it
is not closed through a substrate, reflection, or needle mask. Automatic
detection proposes straight lines only. Explicit manual circular profiles keep
their existing local-tangent handling; rough interfaces require user review.

## Validation

Tests cover interior return paths, liquid above/below the contact chord, tilted
lines, reversed input order, idempotence, exact collinear simplification,
duplicate vertices, preservation of measured inputs, cancellation and use by
the main calibration overlay. A synthetic image reproduces a bright patch
touching the substrate; its internal edge disappears from the display boundary.

The final focused calibration, detector, contact-angle, geometry and temporal
run passed 79 tests. A broader run passed 106 tests with 10 detector-conformance
failures; all ten failure identities matched the baseline run with cleanup
disabled. No scientific thresholds were relaxed.

The offscreen wizard renderer produced a before/after comparison under
`.cache/liquid-boundary/fc7b3a40837947bda652cb5a6ca85830/`. The synthetic example
keeps all 404 measured points and draws a 112-point boundary (72% fewer display
points). This is not a solver speedup or a measured frame-rate claim. The user
provided screenshots rather than original image pixels, so validation uses a
synthetic reproduction, not an exact rerun of those source photographs.

Reproduce with:

```powershell
uv run --extra test python tools/preview_liquid_boundary.py
uv run --extra test python tools/check_gui_execution.py tests/test_liquid_boundary.py tests/test_calibration_wizard_dialog.py -q
```
