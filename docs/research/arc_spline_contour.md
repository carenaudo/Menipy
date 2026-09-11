# Arc-spline drop contours: assessment and implementation

Status: implemented as the opt-in sessile contact-angle methods `arc_spline`
and `clothoid_spline` (`src/menipy/common/arc_spline.py`,
`src/menipy/common/clothoid_spline.py`, `src/menipy/math/sessile_box.py`);
pendant drops: the opt-in two-zone clothoid spline `clothoid_zones`
(`src/menipy/common/pendant_spline.py`, `src/menipy/math/pendant_box.py`).
Evidence below comes from the reproducible tools listed at the end.

## Idea

Model the liquid-fluid interface between the contact points P1 and P2 as the
fewest tangent-continuous (G1) circular arcs that follow the observed edge,
drawn from P1 over the apex to P2. The substrate (or needle) between P1 and P2
is not part of the model. Fewer parameters than edge points should reduce the
effect of noise and poor image quality, and the contact angle is the end
tangent of the first and last arc.

## Prior work

General "fewest arcs" curve approximation exists; a drop-specific,
contact-anchored arc spline with a physics prior was not found.

- Rosin & West (1989), *Segmentation of edges into lines and arcs*: threshold
  free line/arc segmentation of edge curves.
- Maier, Janda & Schindler (2012, ICIP), *Minimum description length arc spline
  approximation of digital curves*: minimum number of G1 arcs within a tolerance,
  free breakpoints.
- Maier & Pisinger (2014, CAGD), *Optimal arc spline approximation*: provably
  minimal segment count inside a tolerance channel.
- Jeon, Hwang & Choi (2024), *Reliability-based G1 continuous arc spline
  approximation* (arXiv 2401.09770): covariance-weighted fit, arcs added where a
  chi-squared test fails (road-lane data, not images).
- Drop analysis: the circle (spherical-cap) method; circle plus polynomial
  correction for needle-in-drop images (ACS Omega 2019); HPDSA (Coatings 2016);
  DropSnake B-spline active contours (Stalder et al., 2006); LB-ADSA.

## Method

1. **Frame.** Work in a frame where the P1-P2 chord is horizontal, so tilted
   substrates need no special case. Curved substrates: angles are measured
   against the substrate's local tangent at each contact.
2. **Apex.** The point farthest from the chord has its tangent parallel to the
   chord for any smooth profile (symmetric or not). It is located as the root
   of the derivative of a cubic fitted over a wide window of the top (a single
   highest point wanders by pixels on staircase or noisy edges). Each side is an
   independent chain that starts at the apex with a fixed horizontal tangent
   and ends exactly on its contact point.
3. **Physics prior (box + golden section).** An axisymmetric sessile profile is
   fixed, up to scale, by the Bond number. For each side, the box (half-width
   from the apex axis to the contact, apex height) picks the point on the
   dimensionless profile where `z/x = H/a`; that fixes scale and angle. The only
   free shape parameter is the Bond number, found by a golden-section search
   against the edge points over a process-cached table of dimensionless profiles
   (one vectorized RK4 pass, ~140 ms once). The matched profile's curvature
   `κ(s)` gives the arc count for a tolerance `tol`,
   `N = ∫ (|κ'| / (40.5 tol))^(1/3) ds` (40.5 is the deviation constant of an
   arc against linearly varying curvature), and the joins at equal quantiles of
   `∫ |κ'|^(1/3) ds`. `tol` is at least 0.25 px and at least twice the edge
   noise, which is estimated model-free from second differences (MAD).
4. **Fit.** Nodes (apex and joins) slide along a smoothed copy of the edge and
   may leave it by a bounded normal offset; each arc is the unique one leaving
   its start node with the incoming tangent and passing through its end node.
   Residuals are signed circle distances in a form stable as curvature goes to
   zero; the least-squares Jacobian is analytic (heading recursion
   `ψ_{j+1} = 2α_j − ψ_j`).
5. **Bias correction.** Constant-curvature arcs read the end tangent low where
   curvature keeps rising towards the contact line. The same arc layout is fitted
   to the noise-free matched profile, and its angle error there is subtracted
   ("model" correction). A Richardson step between N and N/2 arcs (bias ∝ 1/N²)
   is the alternative; it amplifies noise and is kept for the blind path.
6. **Adequacy and fallback.** If the physics-sized arcs cannot explain the edge
   down to its noise, the drop is not one Young-Laplace profile, and a blind
   coarse-to-fine search takes over (one arc per side, split the worst arc of
   each side per level, BIC selection). The box-matched Young-Laplace angle is
   still reported as a reference.
7. **Image refinement.** Sample grey-level profiles along the spline normals,
   take the sub-pixel peak of the drop-polarity gradient, refit (two rounds).
   Edges found below the contact line (reflection on dark substrates) and peaks
   on the search boundary are discarded. When the refined edge meets the
   contact line more than 2 px from a given contact point, the contact is moved
   there (segmentations inside the drop put the contacts inside as well).
8. **Quality gate.** Angles outside (0°, 180°), a model below the contact line,
   or a residual far above the edge noise reject the fit: NaN angles, reasons in
   `arc_spline.rejection_reasons`, and the measurement is not accepted.

## Results

### How many arcs a drop needs

Physics-chosen arcs, both sides together, 300 px contact radius, mean over the
four edge-noise models:

| Bond | 30° | 90° | 150° |
|---|---|---|---|
| 0 (spherical cap) | 2.0 | 2.0 | 2.4 |
| 0.5 | 2.2 | 4.2 | 7.8 |
| 2 | 2.8 | 5.6 | 8.4 |
| 8 | 4.1 | 6.4 | 8.5 |

One arc per side is exact for a spherical cap; flattened or high-angle drops
need four to five per side. Noisier edges get fewer arcs (the tolerance scales
with the noise), and the model bias correction keeps the angle unbiased.

### Synthetic Young-Laplace profiles

RMSE in degrees, Bond 0-8, 30-150°, three seeds, random tilt within ±8°:

| Method | clean | binary staircase | σ 0.5 px | σ 1 px | 30° | 90° | 150° | median time |
|---|---|---|---|---|---|---|---|---|
| Arc spline, physics prior + model correction (default) | 0.00 | 0.44 | 0.14 | 0.26 | 0.32 | 0.23 | 0.30 | 93 ms |
| Box-matched Young-Laplace angle (same run) | 0.01 | 0.24 | 0.13 | 0.25 | 0.23 | 0.20 | 0.16 | — |
| Arc spline, blind + Richardson | 0.09 | 0.63 | 0.28 | 0.46 | 0.32 | 0.43 | 0.58 | 310 ms |
| Menipy `tangent` | 0.49 | 19.3 | 52.0 | 79.2 | 1.20 | 42.0 | 81.5 | 0.7 ms |
| Menipy `circle_fit` | 0.09 | 45.9 | 71.4 | 96.0 | 68.9 | 46.4 | 88.8 | 0.7 ms |

For ideal axisymmetric drops the box-matched Young-Laplace angle alone is as
good as the arcs: physics with two parameters is the minimal model there.

The existing `tangent` and `circle_fit` estimators are not meaningful on these
point-cloud contours (Gaussian-scattered points are not a pixel chain); the
rendered-image comparison below is the fair one.

### Drops that are not one Young-Laplace profile

RMSE in degrees, 0.5 px edge noise, three seeds:

| Case | arcs (physics) | arcs (blind) | Young-Laplace box | Menipy tangent | Menipy circle_fit |
|---|---|---|---|---|---|
| Asymmetric sides, 70° / 115° | 0.69 | 0.56 | 0.84 | 12.0 | 27.8 |
| 4 px pinning bump on one flank, 100° | 0.47 | 0.80 | 0.85 | 33.0 | 33.0 |

This is where the arcs earn their place: the shape departs from one
Young-Laplace profile, and the model-free arcs still read the local end tangent.

### Rendered images (anti-aliased, blurred, noisy), known angle

Sixteen renders, 30-130°, tilt ±5°, contact points exact:

| Method | Mean error | RMSE | Max |
|---|---|---|---|
| Menipy `tangent` / `circle_fit` on the mask contour (4 renders) | | | errors from −5.5° to +9.7° |
| Arc spline on the mask contour | −1.26° | 1.35° | 2.11° |
| Arc spline + image refinement | +0.36° | 0.50° | 0.90° |

The mask contour sits 0.43 px inside the true boundary (OpenCV returns boundary
pixel centres); the refined edge is within +0.06 px (σ 0.18 px). The residual
0.5° is edge-localization error of 0.1-0.2 px near the contact, which depends on
the edge orientation against the pixel grid (bicubic sampling was worse).
Contacts displaced 5 px inside the drop are recovered to within 0.2-1.2 px.

### Arcs versus clothoids

Raw end-tangent error (no bias correction), clean profiles:

| Profile | segments per side | 1 | 2 | 3 | 4 | 6 |
|---|---|---|---|---|---|---|
| Bond 2, 90° | arcs | −11.18 | −2.25 | −0.93 | −0.51 | −0.15 |
| | clothoids | −2.08 | −0.24 | −0.04 | −0.11 | diverged |
| Bond 8, 150° | arcs | −52.16 | −10.04 | −4.04 | −1.99 | −1.06 |
| | clothoids | −10.86 | −1.76 | −0.42 | −0.23 | −0.14 |
| Bond 0.5, 120° | arcs | −9.36 | −1.81 | −0.75 | −0.37 | −0.20 |
| | clothoids | −1.56 | −0.18 | −0.07 | −0.01 | −0.10 |

Linear-curvature segments need 3-4× fewer pieces than arcs for the same raw
bias. This table comes from the finite-difference prototype that is still part
of `scripts/benchmark_arc_spline.py` (it takes 1-6 s per fit and diverged once
at six segments); the production version follows.

## Clothoid spline (`clothoid_spline` method)

`src/menipy/common/clothoid_spline.py` is the production version:

- **Segments:** each is the unique G1 Hermite clothoid between two nodes with
  given tangents (Bertolazzi & Frego, 2015): in the normalized frame the heading
  is `Θ(τ) = φ0 (1-τ) + φ1 τ + A (τ²-τ)` and Newton solves `∫ sin Θ = 0` for `A`
  (end-point and end-heading errors ~1e-15). A segment depends only on its two
  nodes, so nothing propagates down the chain and there is nothing to curl.
- **Parameters:** node positions along the smoothed edge plus normal offsets,
  node headings, and the headings at P1/P2, which *are* the contact angles. The
  angle uncertainty comes from the fit covariance (`σ² (JᵀJ)⁻¹`); on 1 px noise
  it predicted the observed scatter within a factor of three.
- **Jacobian:** analytic. Orthogonal residuals at Newton foot points; by the
  envelope theorem `dr/dp = -n · ∂C/∂p`; `∂C/∂p` from implicit differentiation
  of the Hermite solution (the heading `ψ = θ0 (1-τ) + θ1 τ + A (τ²-τ)` does not
  depend on the chord direction, which removes a whole term); generalized Fresnel
  integrals by 12-point Gauss-Legendre quadrature. Checked against finite
  differences (max error 1e-7 on entries up to 38).
- **Sizing:** a G1 Hermite clothoid deviates from a curve with quadratic
  curvature by `|κ''| L⁴ / 384`, so `N = ∫ (|κ''| / (384 tol))^(1/4) ds` and the
  joins sit at equal quantiles of `∫ |κ''|^(1/4) ds` on the box-matched profile.
- **G2 prior:** a soft penalty equalizes curvature across each join and across
  the apex (where the two chains turn in opposite senses).
- **Bias:** the same layout is fitted to the noise-free matched profile and its
  angle error is subtracted, as for arcs. Refinement on the image, contact
  refinement and quality gates are shared with the arc spline.

Results (synthetic Young-Laplace profiles, tilt 5°):

| Case | clothoids: segments, error | arcs: segments, error |
|---|---|---|
| Bond 2, 90°, clean | 4, −0.001° | 8, −0.001° |
| Bond 8, 150°, clean | 6, +0.013° | 12, +0.009° |
| Bond 0.5, 120°, clean | 4, 0.000° | 8, 0.000° |
| Bond 2, 90°, 1 px noise | 4, +0.08/+0.33° (σ 0.26°) | 4, −0.01/+0.18° |
| Rendered images + refinement (16) | RMSE 0.42°, max 0.90° | RMSE 0.50°, max 0.90° |

Half the segments for the same accuracy; slightly better on rendered images;
2-3× slower (80-490 ms per fit, ~750 ms with image refinement), dominated by
the foot-point quadrature. A foot-point bug (curvature term of `f'` scaled by
`L` twice) made noisy fits diverge until fixed; a warm start of the foot points
from the previous evaluation was tried and removed (it inherits foot points
from rejected trial steps).

Full synthetic sweep (same cases as above), RMSE in degrees:

| Method | clean | staircase | σ 0.5 px | σ 1 px | 30° | 90° | 150° | median time |
|---|---|---|---|---|---|---|---|---|
| Clothoids, physics prior | 0.00 | 0.36 | 0.15 | 0.30 | 0.28 | 0.19 | 0.32 | 273 ms |
| Arcs, physics prior | 0.00 | 0.44 | 0.14 | 0.26 | 0.32 | 0.23 | 0.30 | 90 ms |

Segments (both sides) chosen by the physics prior, clothoids versus arcs: 2 vs 2
for spherical caps, 4.0 vs 5.6 at Bond 2 / 90°, 4.8 vs 8.4 at Bond 2 / 150°,
5.4 vs 8.5 at Bond 8 / 150°. Non Young-Laplace drops: asymmetric 70°/115°,
clothoids 0.31° (arcs 0.69°, blind arcs 0.56°); pinning bump, clothoids 0.87°
(arcs 0.47°). Extra segments on a binary-mask staircase are accepted only when
they improve the BIC; without that gate the correlated staircase residual was
followed by more segments and the staircase RMSE was 0.78°.

## Pendant drops: two-zone design study

`scripts/pendant_zones_study.py`, exact pendant Young-Laplace profiles
(`dφ/ds = 2 - Bo z - sin φ / x`, apex at the bottom), equatorial radius 150 px,
needle radius 0.5 R_e (or just above the neck when the neck is wider):

- **Zone 1**, apex → equator (maximum diameter): convex, near-spherical. The
  equator tangent is vertical by definition, the pendant analogue of the apex
  constraint.
- **Zone 2**, equator → needle contacts P1/P2: the neck, with an inflection
  (meridional curvature through zero) for Bond ≥ 0.3.

Segments needed per zone and side at 0.25 px:

| Bond | zone 1 arcs / clothoids | zone 2 arcs / clothoids | inflection in zone 2 |
|---|---|---|---|
| 0.1 | 1.4 / 0.9 | 1.7 / 1.0 | no |
| 0.2 | 1.9 / 1.1 | 2.3 / 1.3 | no |
| 0.3 | 2.2 / 1.2 | 3.1 / 1.6 | yes |
| 0.4 | 2.6 / 1.4 | 2.8 / 1.2 | yes |

A clothoid chain per side with the equator as a node of fixed vertical tangent
(position free) reproduces the side with 2+2 segments to 0.16 px, needle angle
within +0.17-0.33°; 2+3 segments: 0.15 px, +0.05-0.14°.

**Where the Bond number lives.** Anchoring only well-conditioned quantities --
the apex height and the equatorial radius, both extremum *values* -- and
golden-section searching the Bond number:

| Bond | zone 1 alone, 0.5 px / 1 px noise | zones 1+2, 0.5 px / 1 px noise |
|---|---|---|
| 0.1 | 0.103 ± 0.016 / 0.142 ± 0.026 | 0.098 ± 0.003 / 0.103 ± 0.005 |
| 0.2 | 0.210 ± 0.017 / 0.188 ± 0.027 | 0.201 ± 0.002 / 0.197 ± 0.003 |
| 0.3 | 0.317 ± 0.010 / 0.316 ± 0.021 | 0.301 ± 0.001 / 0.301 ± 0.003 |
| 0.4 | 0.388 ± 0.009 / 0.412 ± 0.022 | 0.401 ± 0.000 / 0.397 ± 0.001 |

Zone 1 gives the scale; zone 2 gives the Bond number (5-10× less scatter), which
is why the selected-plane method measures a diameter at height `d_e`, i.e. in
zone 2. A first attempt that pinned the equator *height* (the location of a very
flat maximum of `x(z)`) failed completely under noise: half a pixel in `x` moves
that height by ~`sqrt(R_e)` px. Only extremum values should anchor a box.

## Pendant drops: two-zone clothoid spline (`clothoid_zones`)

Implemented in `math/pendant_box.py` (physics prior), `common/pendant_spline.py`
(model) and `pipelines/pendant/zone_spline.py` (pipeline). The spline reuses the
clothoid residuals and analytic Jacobian of `clothoid_spline._Problem`, whose
layout interface was generalized to nodes with a fixed heading.

1. **Frame.** The symmetry axis, found by reflecting one side across a
   candidate axis (angle, offset) and minimizing its distance to the other
   side (Nelder-Mead). Midpoints of horizontal chords were tried first and
   under-corrected tilt: near the apex the chord midpoints of a tilted,
   near-parabolic drop stay on a vertical line whatever the tilt.
2. **Anchors** (extremum values only): the equatorial radius and axis from
   cubic `x(y)` fits around each side's widest point, the apex height from a
   cubic `y(x)` vertex.
3. **Physics prior.** A pendant table (201 Bond rows, 0-1, vectorized RK4,
   150 ms once per process); for each Bond the profile is scaled to `R_e`,
   placed on the apex height and cut at the needle height; golden-section search
   on both zones' edge points. It gives `Bo`, the apex radius `b`, the surface
   tension `Δρ g b² / Bo`, the clothoids per zone (`∫(|κ''|/(384 tol))^¼ ds`)
   and the joins.
4. **Spline.** Per side apex (horizontal tangent) → zone-1 joins → equator
   (tangent fixed vertical, position free) → zone-2 joins → needle contact
   (heading free: the angle at the needle). Splits BIC-gated as for sessile
   drops. Each contact may slide up to 2 px perpendicular to the axis at fixed
   height (see below).
5. **Bias corrections** from fitting the same layout to the noise-free matched
   profile: the needle angle, and the slope of the Laplace relation below.
6. **Outputs.** Surface tension (box), a second surface tension from the
   spline's curvature -- `κ_m + sin φ / r = 2/b - (Δρ g / γ) z`, a straight line
   in height -- the needle angle per side with covariance sigma, the denoised
   contour, and seeds (apex radius, Bond, axis) for the strict fit.
7. **Image refinement** (`refine_pendant_on_image`): the gradient peak of the
   dark-drop polarity along the spline normals, one round. A binary-mask
   contour traces boundary pixels 0.35 px inside the true edge, which biased
   *every* method (strict included) by -1.8 % at Bo 0.1 and -0.6 % at Bo 0.4.

`scripts/benchmark_pendant_spline.py`, equatorial radius 150 px, 100 px/mm,
5 seeds (surface-tension error %, RMSE with bias in parentheses):

| Edge noise | box | spline curvature | strict | strict on spline, seeded |
|---|---|---|---|---|
| clean | 0.14 (+0.10) | 0.24 (+0.15) | 0.01 | 0.02 |
| 0.5 px | 0.83 (-0.07) | 2.05 (+0.85) | 0.57 (-0.15) | 0.57 (-0.07) |
| 1 px | 1.62 (-0.24) | 1.13 (-0.01) | 1.14 (-0.27) | 1.09 (-0.09) |
| staircase | 0.56 | 0.38 | 0.24 | 0.23 |

| Rendered images, by Bond | strict on mask | box on mask | box refined | curvature refined | strict on spline, seeded |
|---|---|---|---|---|---|
| 0.1 | 1.86 (-1.79) | 2.35 (-2.23) | 0.55 | 1.25 (+1.05) | 0.42 (+0.35) |
| 0.2 | 1.11 (-0.98) | 1.37 (-1.35) | 0.27 | 1.50 (+1.30) | 0.10 |
| 0.3 | 0.57 (-0.26) | 0.88 (-0.88) | 0.21 | 1.25 (+1.11) | 0.09 |
| 0.4 | 0.56 | 0.67 (-0.61) | 0.29 | 0.51 (+0.45) | 0.16 |

**How the method acts on surface tension.** The anchored box (one free
parameter) comes within 1.5× of the strict four-parameter fit's scatter; its
anchors come from local fits, so it uses the data less efficiently. The spline
curvature estimate converges to the truth only when the spline has enough
clothoids (+4.1 % with 1+1 per zone, +0.14 % with 5+5 on clean data): a spline
sized for 0.25 px in *position* flattens the curvature-height relation, a
second derivative. The model-based slope correction removes that bias on
synthetic edges; on rendered images +0.5-1.3 % remains. It is best read as a
consistency check: when box and curvature agree, the drop is one
Young-Laplace profile.

**Strict fit on the denoised contour.** The strict fit's time is 90 % ODE
integration (27-42 `solve_ivp` calls); the number of points does not matter
(827 raw points and 87 spline samples take the same time and give the same
surface tension). What the spline gives the strict fit is its *starting point*:
seeded with the box's apex radius, Bond number and symmetry axis, the fit needs
4 evaluations instead of 10-15 and takes 110-180 ms instead of 360-510 ms, with
the same accuracy (the spline samples every 3 px: 289 points instead of 834).
On rendered images the seeded strict fit on the refined spline is the most
accurate of all.

**Needle angle** (YL `φ` at the contact, 90° vertical): corrected 0.03° / 0.48° /
0.19° / 0.12° RMSE (clean / 0.5 px / 1 px / staircase); the raw heading reads
1.2-1.4° high, so the model correction matters. Covariance sigma 0.2-0.7°.
Rendered images 0.13-0.27°. The contact point dominates: `Context` stores
integer contacts, and forcing the spline through a rounded contact gave
1.20° RMSE (2.24° with the contact 1 px sideways); letting each contact slide
up to 2 px perpendicular to the axis, at fixed height, gives 0.38° / 0.39°
without changing the surface tension.

**Real samples** (commercial instrument reference; auto-detected contour,
contacts and 1.83 mm needle calibration):

| Sample | reference | strict (raw) | box, refined | curvature, refined | strict seeded | needle angle | axis tilt |
|---|---|---|---|---|---|---|---|
| `gota pendiente 1.png` | 29.24 | 29.28 | 29.51 | 29.54 | 29.51 | 110.5° ± 0.1 | -0.1° |
| `prueba pend 1.png` | 70.94 | 69.55 | 70.67 | 70.58 | 70.80 | 122.4° ± 0.7 | 1.5° |

On `prueba pend 1` the spline finds the drop axis 1.5° off the image vertical
that the raw strict fit assumes; the seeded fit on that axis moves from -2.0 %
to -0.2 % of the reference. On `gota pendiente 1` the refined edge moves every
spline estimate +0.9 %.

Pipeline use: `Context.pendant_contour_model = "clothoid_zones"` (GUI:
Pendant settings → Contour model) fits the strict model to the spline and
reports `clothoid_zones`, `needle_angle_deg` and both spline surface tensions;
the opt-in approximator `clothoid_zones` reports them without changing the
strict fit. A rejected spline falls back to the raw contour.

### Sample images (no ground truth)

`tools/compare_arc_spline_samples.py`, full application chain:

| Sample | tangent | circle_fit | lbadsa | arc_spline | notes |
|---|---|---|---|---|---|
| `sessile_3.jpeg` | 49.9/51.1 | 58.6/58.2 | 54.6/54.6 | 56.1/58.4 | physics prior adequate, 4 arcs |
| `sessile_clean_reference.png` | 129.4/133.3 | 129.4/133.3 | 132.1/134.9 | 123.2/129.8 | Young-Laplace box 122.7/126.3 |
| `prueba sesil 2.png` | 92.2/88.1 | 92.2/88.1 | 82.0/82.0 | 81.7/83.0 | contour and contacts ~5 px inside the drop; refinement moved both |
| `gota depositada 1.png` | 39.9/20.3 | 39.9/20.3 | 72.5/72.9 | rejected | detector put P2 in the middle of the drop |
| `sessile_needle_reference.png` | 89.7/96.7 | 89.7/96.7 | 58.5/58.5 | rejected | baseline detected mid-drop, needle cut in the contour |

### Profiling

`tools/profile_arc_spline.py`, per fit, same machine:

| Version | Worst case | Where the time went |
|---|---|---|
| First port (finite-difference Jacobian, all-points-to-all-arcs distances, one split per level) | 21.6 s | 80 % point-to-arc distances inside 16 k residual evaluations |
| Analytic Jacobian, owned-arc signed distances, split both sides per level | 0.42 s | trust-region SVD ~30 %, residual evaluation ~0.23 ms each |
| Physics prior (box + golden section) with model bias correction | 36-218 ms | refit 40-80 %, box search ~15-20 ms per side, model bias fit up to 25 % |
| Clothoids, physics prior | 48-507 ms | refit 70-90 % on noisy edges (100-240 residual evaluations), foot-point quadrature 20-35 %, model bias fit 12-25 %, box search dominates cheap cases |
| Pendant two-zone clothoids (`tools/profile_pendant_spline.py`) | 120-220 ms | least squares 38-65 %, anchored box search 17-32 %, symmetry axis ~15 % (was 24-30 % with two estimates), model bias fit 7-20 %; image refinement +345 ms (one round; two rounds were no more accurate) |
| Strict pendant fit, raw contour vs seeded by the spline | 450-510 ms → 110-180 ms | 90 % ODE integration in both; 13-15 → 3-4 evaluations |

The blind search stays at 160-420 ms; image refinement adds ~120 ms (two rounds;
normal-profile sampling itself is < 1 %).

## Limitations

- Accuracy near the contact is bounded by sub-pixel edge localization
  (~0.1-0.2 px → ~0.5°) and, above all, by contact-point accuracy: a 1 px contact
  error can move the angle by several degrees on a 120 px drop. Contact
  detection failures upstream are rejected, not repaired.
- Needle-in-drop images are not modelled (the interface is interrupted).
- Pendant: a drop without an equator (smaller than the needle allows) is
  rejected; the refraction-index correction of the strict fit is not applied to
  the spline; on rendered images the needle angle keeps a +0.1-0.2° bias and
  the curvature surface tension +0.5-1.3 %.
- Least-squares sliding of the contacts along the contact line
  (`contact_slide_px`) is available but not recommended: it added variance with
  correct contacts (0.49° → 0.76°) and did not recover displaced ones.

## Reproduce

```text
uv run --extra test pytest tests/test_arc_spline.py tests/test_clothoid_spline.py tests/test_sessile_box.py
uv run python scripts/benchmark_arc_spline.py --seeds 3
uv run python scripts/pendant_zones_study.py
uv run --extra test pytest tests/test_pendant_spline.py
uv run python scripts/benchmark_pendant_spline.py --seeds 5
uv run python tools/profile_pendant_spline.py
uv run python tools/profile_arc_spline.py
uv run python tools/compare_arc_spline_samples.py --overlays out/
```
