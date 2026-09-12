# Menipy profiling baseline — 2026-09-12

## Result

All 19 profiling scripts added by commit `386f44e` completed successfully. Their
reference-output assertions all passed: no numerical, persisted-output, table,
geometry, or state mismatch was observed.

The optimized path is materially faster in the contour, temporal, lookup,
envelope, snake-geometry, and ODE-reuse workloads. No statistically meaningful
regression was found. Small slowdowns are retained below for transparency but
are not flagged.

## Median timings

Times are seconds per measured operation, using each script's own paired
repeats. `Δ` is `(optimized / reference - 1) * 100`; negative is faster.

| Script / workload | Reference | Optimized | Δ |
|---|---:|---:|---:|
| adjacent velocity / flat, 3,000 frames | 0.522116 | 0.473541 | -9.3% |
| adjacent velocity / varying, 3,000 frames | 0.674773 | 0.589594 | -12.6% |
| bootstrap percentiles / n=5 | 0.000474 | 0.000402 | -15.3% |
| bootstrap percentiles / n=1,000 | 0.040593 | 0.041066 | +1.2% |
| calibration shaft reuse / sessile | 0.130917 | 0.093617 | -28.5% |
| calibration shaft reuse / pendant | 0.011178 | 0.011270 | +0.8% |
| contour clipping / 1,000 points | 0.011880 | 0.000750 | -93.7% |
| contour clipping / 10,000 points | 0.123132 | 0.006478 | -94.7% |
| cold lookup build / 24×24 grid | 5.851228 | 5.256906 | -10.2% |
| contact crossings / 1,000 points | 0.000695 | 0.000090 | -87.1% |
| contact crossings / 10,000 points | 0.006498 | 0.000178 | -97.3% |
| multi-plane preparation / 300 points | 0.000667 | 0.000341 | -48.8% |
| multi-plane preparation / 10,000 points | 0.003787 | 0.000997 | -73.7% |
| ODE invariants / pendant, 50 integrations | 1.022864 | 1.081360 | +5.7% |
| ODE invariants / pendant fit | 0.859729 | 0.830855 | -3.4% |
| ODE invariants / sessile, 50 integrations | 0.240314 | 0.238610 | -0.7% |
| ODE invariants / sessile fit | 0.095744 | 0.094195 | -1.6% |
| ODE callback reuse / 50 integrations | 0.570704 | 0.537587 | -5.8% |
| ODE callback reuse / full fit | 0.330869 | 0.313709 | -5.2% |
| pendant envelope / 1,000 points | 0.016571 | 0.008322 | -49.8% |
| pendant envelope / 5,000 points | 0.142281 | 0.040863 | -71.3% |
| pendant envelope / 20,000 points | 1.482777 | 0.162260 | -89.1% |
| single-image calibration / sessile | 0.096144 | 0.087440 | -9.1% |
| single-image calibration / pendant | 0.011005 | 0.011133 | +1.2% |
| single-image persisted state / sessile | 1.351853 | 1.357139 | +0.4% |
| single-image persisted state / pendant | 1.054421 | 1.005299 | -4.7% |
| slope convergence / flat, 3,000 frames | 1.575409 | 0.396692 | -74.8% |
| slope convergence / varying, 3,000 frames | 1.551372 | 0.521494 | -66.4% |
| snake forces / default, 80 points | 0.001934 | 0.001977 | +2.2% |
| snake forces / default, 300 points | 0.002754 | 0.002775 | +0.8% |
| snake forces / combined, 80 points | 0.003756 | 0.003310 | -11.9% |
| snake forces / combined, 300 points | 0.005025 | 0.004445 | -11.6% |
| snake geometry / default, 80 points | 0.003173 | 0.001983 | -37.5% |
| snake geometry / default, 300 points | 0.004077 | 0.002727 | -33.1% |
| snake geometry / combined, 80 points | 0.003857 | 0.003267 | -15.3% |
| snake geometry / combined, 300 points | 0.005047 | 0.004403 | -12.8% |
| snake matrix / 80 points | 0.002568 | 0.002103 | -18.1% |
| snake matrix / 200 points | 0.008046 | 0.002381 | -70.4% |
| snake matrix / 300 points | 0.013366 | 0.002693 | -79.8% |
| snake normals / default, 80 points | 0.002684 | 0.001992 | -25.8% |
| snake normals / default, 300 points | 0.003517 | 0.002725 | -22.5% |
| snake normals / combined, 80 points | 0.003342 | 0.003292 | -1.5% |
| snake normals / combined, 300 points | 0.004472 | 0.004631 | +3.6% |

The lookup-RHS audit found 339,318 integration callbacks, of which 312,570
(92.1%) were exact derivative hits. The solver-reuse audit found 18 unique
parameters and 18 total evaluations in each of three checks, hence zero
repeated integrations to reuse in that path.

## Regression and mismatch policy

Timing regressions are flagged only when the optimized median is at least 5%
slower and the paired run deltas pass a two-sided exact sign test at p < 0.05.
With the scripts' six or eight repeats, no measured case satisfies both
conditions. Output mismatches are always flagged regardless of timing; none
occurred because every script assertion completed.

## Reproduction

The scripts were run from repository `D:\programacion\Menipy` with:

```powershell
$env:UV_CACHE_DIR = Join-Path (Get-Location) '.cache\uv-cache'
uv run --extra test python tools/profile_<name>.py
```

Run all 19 names listed in commit `386f44e`'s `tools/profile_*.py` additions.
Each script writes its own JSON under `.cache/` and performs its reference
comparison before writing the report. The consolidated run used Windows 11,
Python 3.14.7, NumPy 2.5.2, and current `HEAD` `9f8cec4`. The worktree was
already dirty before profiling; therefore this is a baseline for that exact
working state, not a clean-tree reconstruction of `386f44e`.

Raw per-script JSON outputs remain in the generated `.cache/` directories and
are intentionally treated as derived output.
