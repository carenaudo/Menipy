"""Needle-in-sessile-drop sequence analysis and contact angle hysteresis engine.

Implements sub-pixel baseline diameter tracking, contact line velocity analysis,
pseudo-movement suppression (Korhonen et al. 2013), and deterministic bootstrap
plateau statistics for advancing and receding contact angles.
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from menipy.common.cancellation import check_cancelled
from menipy.common.needle_drop_detection import detect_needle_drop_features
from menipy.math.needle_profile_fit import fit_needle_contact_angles
from menipy.models.frame import Frame

logger = logging.getLogger(__name__)

BOOTSTRAP_SEED = 20260906


@dataclass
class NeedleHysteresisFrame:
    """Per-frame analysis record for a needle-in-drop sequence."""

    frame_index: int
    timestamp_s: float
    accepted: bool
    state: str  # "pinned", "advancing", "receding", or "invalid"
    theta_left_deg: float | None = None
    theta_right_deg: float | None = None
    theta_mean_deg: float | None = None
    base_diameter_mm: float | None = None
    contact_velocity_mm_s: float | None = None
    rejection_reasons: list[str] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass
class NeedleHysteresisResult:
    """Sequence-level summary of needle-in-drop contact angle hysteresis."""

    schema_version: str = "1.0"
    pipeline: str = "needle_hysteresis"
    accepted: bool = False
    rejection_reasons: list[str] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
    frames: list[NeedleHysteresisFrame] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert result to a plain JSON-serializable dictionary."""
        return {
            "schema_version": self.schema_version,
            "pipeline": self.pipeline,
            "accepted": self.accepted,
            "rejection_reasons": self.rejection_reasons,
            "summary": self.summary,
            "frames": [asdict(f) for f in self.frames],
            "diagnostics": self.diagnostics,
        }


def _robust_local_slope(t: np.ndarray, y: np.ndarray) -> float:
    """Compute local slope using Huber M-estimator."""
    design = np.column_stack([t - np.mean(t), np.ones_like(t)])
    weights = np.ones_like(y)
    coef = np.linalg.lstsq(design, y, rcond=None)[0]
    for _ in range(5):
        residual = y - design @ coef
        scale = max(1e-9, 1.4826 * float(np.median(np.abs(residual - np.median(residual)))))
        normalized = np.abs(residual) / (1.345 * scale)
        weights = np.where(normalized <= 1.0, 1.0, 1.0 / np.maximum(normalized, 1e-12))
        coef = np.linalg.lstsq(design * weights[:, None], y * weights, rcond=None)[0]
    return float(coef[0])


def _bootstrap_stats(values: Sequence[float]) -> dict[str, Any]:
    """Compute median, MAD, and 95% bootstrap confidence interval."""
    arr = np.asarray(values, dtype=float)
    median = float(np.median(arr))
    mad = float(np.median(np.abs(arr - median)))
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = rng.choice(arr, size=(2000, len(arr)), replace=True)
    medians = np.median(samples, axis=1)
    return {
        "median_deg": round(median, 2),
        "mad_deg": round(mad, 2),
        "ci95_deg": [
            round(float(np.percentile(medians, 2.5)), 2),
            round(float(np.percentile(medians, 97.5)), 2),
        ],
        "n_frames": int(len(arr)),
    }


def classify_contact_line_states(
    frames: list[NeedleHysteresisFrame],
) -> float:
    """Classify frames into 'pinned', 'advancing', or 'receding' states.

    Suppresses the pseudo-movement pitfall: receding states are only assigned
    when the baseline diameter is physically shrinking beyond the deadband.
    """
    valid = [f for f in frames if f.accepted and f.base_diameter_mm is not None]
    if not valid:
        return 0.01

    # 1. Compute contact line velocity v_CL(t) = 0.5 * d(diameter)/dt
    for i, frame in enumerate(valid):
        check_cancelled()
        # Look within window of +/- 3 neighbors
        neighbors = [
            other
            for other in valid
            if abs(other.frame_index - frame.frame_index) <= 3
        ]
        if len(neighbors) >= 3:
            t = np.asarray([other.timestamp_s for other in neighbors], dtype=float)
            d = np.asarray([other.base_diameter_mm for other in neighbors], dtype=float)
            # v_CL is radial velocity = 0.5 * diameter change rate
            frame.contact_velocity_mm_s = 0.5 * _robust_local_slope(t, d)
        elif len(valid) > 1:
            # Simple difference fallback
            if i > 0:
                dt = frame.timestamp_s - valid[i - 1].timestamp_s
                dd = frame.base_diameter_mm - valid[i - 1].base_diameter_mm
                frame.contact_velocity_mm_s = (0.5 * dd / dt) if dt > 0 else 0.0
            else:
                dt = valid[1].timestamp_s - frame.timestamp_s
                dd = valid[1].base_diameter_mm - frame.base_diameter_mm
                frame.contact_velocity_mm_s = (0.5 * dd / dt) if dt > 0 else 0.0

    # 2. Compute noise deadband from first differences of velocity
    pairs: list[float] = []
    for left, right in zip(valid, valid[1:]):
        if left.contact_velocity_mm_s is not None and right.contact_velocity_mm_s is not None:
            pairs.append(abs(right.contact_velocity_mm_s - left.contact_velocity_mm_s))
    if len(pairs) >= 3:
        mad = 1.4826 * float(np.median(pairs))
    else:
        mad = 0.005
    deadband = max(0.01, 3.0 * mad)

    # 3. State classification snapped by adjacent local slopes
    for position, frame in enumerate(valid):
        local_slopes: list[float] = []
        if position > 0:
            dt = frame.timestamp_s - valid[position - 1].timestamp_s
            if dt > 0:
                dd = float(frame.base_diameter_mm) - float(valid[position - 1].base_diameter_mm)
                local_slopes.append(0.5 * dd / dt)
        if position + 1 < len(valid):
            dt = valid[position + 1].timestamp_s - frame.timestamp_s
            if dt > 0:
                dd = float(valid[position + 1].base_diameter_mm) - float(frame.base_diameter_mm)
                local_slopes.append(0.5 * dd / dt)

        velocity = float(np.median(local_slopes)) if local_slopes else frame.contact_velocity_mm_s
        if velocity is None or abs(velocity) <= deadband:
            frame.state = "pinned"
        elif velocity > deadband:
            frame.state = "advancing"
        else:
            frame.state = "receding"

    # 4. Debounce short noise bursts (must be >= 3 consecutive frames)
    idx = 0
    while idx < len(frames):
        check_cancelled()
        state = frames[idx].state
        end = idx + 1
        while end < len(frames) and frames[end].state == state:
            end += 1
        if state in ("advancing", "receding") and (end - idx) < 3:
            for p in range(idx, end):
                if frames[p].accepted:
                    frames[p].state = "pinned"
        idx = end

    return deadband


def summarize_hysteresis(
    frames: list[NeedleHysteresisFrame],
    fps: float,
    deadband_mm_s: float,
) -> dict[str, Any]:
    """Calculate advancing/receding plateau metrics and hysteresis."""
    total_frames = len(frames)
    valid_frames = [f for f in frames if f.accepted]
    valid_count = len(valid_frames)

    summary: dict[str, Any] = {
        "n_frames": total_frames,
        "n_valid_frames": valid_count,
        "valid_fraction": round(valid_count / max(1, total_frames), 3),
        "fps": fps,
        "velocity_deadband_mm_s": round(deadband_mm_s, 4),
    }

    if valid_frames:
        diameters = [f.base_diameter_mm for f in valid_frames if f.base_diameter_mm is not None]
        if diameters:
            summary["base_diameter_initial_mm"] = round(diameters[0], 3)
            summary["base_diameter_max_mm"] = round(float(np.max(diameters)), 3)
            summary["base_diameter_final_mm"] = round(diameters[-1], 3)

    # Advancing and Receding plateaus
    for state in ("advancing", "receding"):
        matching = [f for f in frames if f.accepted and f.state == state]
        summary[f"{state}_frames_count"] = len(matching)
        summary[f"{state}_duration_s"] = round(len(matching) / max(1.0, fps), 2)

        if len(matching) >= 5:
            # Discard first 10% of frames to bypass transient depinning acceleration
            trim_start = max(1, int(len(matching) * 0.10))
            plateau_frames = matching[trim_start:]

            angles_l = [f.theta_left_deg for f in plateau_frames if f.theta_left_deg is not None]
            angles_r = [f.theta_right_deg for f in plateau_frames if f.theta_right_deg is not None]
            angles_mean = [
                f.theta_mean_deg for f in plateau_frames if f.theta_mean_deg is not None
            ]

            if angles_mean:
                stats = _bootstrap_stats(angles_mean)
                summary[f"theta_{state}"] = stats
                summary[f"theta_{state}_deg"] = stats["median_deg"]

            if angles_l:
                summary[f"theta_{state}_left"] = _bootstrap_stats(angles_l)
            if angles_r:
                summary[f"theta_{state}_right"] = _bootstrap_stats(angles_r)

            vels = [
                f.contact_velocity_mm_s
                for f in plateau_frames
                if f.contact_velocity_mm_s is not None
            ]
            if vels:
                summary[f"{state}_velocity_median_mm_s"] = round(float(np.median(vels)), 4)

    # Calculate Contact Angle Hysteresis (CAH = theta_A - theta_R)
    if "theta_advancing_deg" in summary and "theta_receding_deg" in summary:
        hysteresis = summary["theta_advancing_deg"] - summary["theta_receding_deg"]
        summary["contact_angle_hysteresis_deg"] = round(hysteresis, 2)

    return summary


def analyze_needle_hysteresis_sequence(
    frames: list[Frame],
    *,
    px_per_mm: float | None = None,
    needle_diameter_mm: float | None = None,
    fit_method: str = "auto",
    fps: float = 10.0,
    check_cancelled=check_cancelled,
) -> NeedleHysteresisResult:
    """Run full needle-in-sessile-drop analysis across a video or image sequence."""
    rejection_reasons: list[str] = []

    # 1. Determine pixel scale
    if px_per_mm is None or px_per_mm <= 0:
        if needle_diameter_mm is not None and needle_diameter_mm > 0:
            scale_samples = []
            for f in frames[:5]:
                check_cancelled()
                det = detect_needle_drop_features(f.image)
                if det.needle_rect and det.needle_rect[2] > 0:
                    scale_samples.append(det.needle_rect[2] / needle_diameter_mm)
            if scale_samples:
                px_per_mm = float(np.median(scale_samples))

    if px_per_mm is None or px_per_mm <= 0:
        rejection_reasons.append("missing_scale_calibration")
        return NeedleHysteresisResult(
            accepted=False,
            rejection_reasons=rejection_reasons,
            summary={"error": "Calibration scale missing or invalid"},
        )

    # 2. Process each frame
    frame_results: list[NeedleHysteresisFrame] = []
    locked_substrate = None
    locked_needle = None

    for idx, f in enumerate(frames):
        check_cancelled()
        ts = idx / max(1.0, fps)
        det = detect_needle_drop_features(
            f.image,
            substrate_line=locked_substrate,
            needle_rect=locked_needle,
        )

        if locked_substrate is None and det.substrate_line is not None:
            locked_substrate = det.substrate_line
        if locked_needle is None and det.needle_rect is not None:
            locked_needle = det.needle_rect

        frame_rej: list[str] = []
        if det.drop_contour is None:
            frame_rej.append("drop_not_detected")
        if det.contact_points is None:
            frame_rej.append("contact_points_not_detected")
        if det.substrate_line is None:
            frame_rej.append("substrate_not_detected")

        if frame_rej:
            frame_results.append(
                NeedleHysteresisFrame(
                    frame_index=idx,
                    timestamp_s=round(ts, 3),
                    accepted=False,
                    state="invalid",
                    rejection_reasons=frame_rej,
                    diagnostics=det.diagnostics,
                )
            )
            continue

        # Fit contact angles
        assert det.drop_contour is not None and det.contact_points is not None and det.substrate_line is not None
        fit_res = fit_needle_contact_angles(
            det.drop_contour,
            det.contact_points,
            det.substrate_line,
            method=fit_method,
            needle_rect=det.needle_rect,
        )

        base_diam_mm = (
            det.base_diameter_px / px_per_mm if det.base_diameter_px is not None else None
        )
        theta_mean = (fit_res.theta_left_deg + fit_res.theta_right_deg) / 2.0

        frame_results.append(
            NeedleHysteresisFrame(
                frame_index=idx,
                timestamp_s=round(ts, 3),
                accepted=True,
                state="pinned",  # will be updated by classifier
                theta_left_deg=fit_res.theta_left_deg,
                theta_right_deg=fit_res.theta_right_deg,
                theta_mean_deg=round(theta_mean, 2),
                base_diameter_mm=round(base_diam_mm, 4) if base_diam_mm is not None else None,
                rejection_reasons=[],
                diagnostics={
                    "fit_method": fit_res.method,
                    "rmse_left_px": fit_res.rmse_left_px,
                    "rmse_right_px": fit_res.rmse_right_px,
                },
            )
        )

    # 3. Classify contact line states
    deadband = classify_contact_line_states(frame_results)

    # 4. Generate summary
    summary = summarize_hysteresis(frame_results, fps, deadband)

    # Acceptance criteria per ISO 19403-6
    has_advancing = "theta_advancing_deg" in summary
    has_receding = "theta_receding_deg" in summary
    valid_fraction = summary.get("valid_fraction", 0.0)

    seq_accepted = bool(valid_fraction >= 0.60 and (has_advancing or has_receding))
    if not seq_accepted:
        if valid_fraction < 0.60:
            rejection_reasons.append("valid_fraction_too_low")
        if not has_advancing and not has_receding:
            rejection_reasons.append("no_advancing_or_receding_plateaus_found")

    return NeedleHysteresisResult(
        accepted=seq_accepted,
        rejection_reasons=rejection_reasons,
        summary=summary,
        frames=frame_results,
        diagnostics={
            "calibration_px_per_mm": px_per_mm,
            "fit_method": fit_method,
            "deadband_mm_s": deadband,
        },
    )


def export_needle_hysteresis_results(
    result: NeedleHysteresisResult,
    out_dir: Path,
) -> None:
    """Export complete results.json, results.csv, and results_frames.csv."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. results.json
    json_path = out_dir / "results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, indent=2)

    # 2. results.csv (Sequence summary row)
    csv_summary_path = out_dir / "results.csv"
    summary_headers = [
        "pipeline",
        "accepted",
        "rejection_reasons",
        "n_frames",
        "n_valid_frames",
        "valid_fraction",
        "theta_advancing_deg",
        "theta_receding_deg",
        "contact_angle_hysteresis_deg",
        "base_diameter_initial_mm",
        "base_diameter_max_mm",
        "base_diameter_final_mm",
    ]
    with open(csv_summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_headers, extrasaction="ignore")
        writer.writeheader()
        row = {
            "pipeline": result.pipeline,
            "accepted": result.accepted,
            "rejection_reasons": ";".join(result.rejection_reasons),
            **result.summary,
        }
        writer.writerow(row)

    # 3. results_frames.csv (Per-frame data series)
    csv_frames_path = out_dir / "results_frames.csv"
    frame_headers = [
        "frame_index",
        "timestamp_s",
        "accepted",
        "state",
        "theta_left_deg",
        "theta_right_deg",
        "theta_mean_deg",
        "base_diameter_mm",
        "contact_velocity_mm_s",
        "rejection_reasons",
    ]
    with open(csv_frames_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=frame_headers, extrasaction="ignore")
        writer.writeheader()
        for frame in result.frames:
            f_dict = asdict(frame)
            f_dict["rejection_reasons"] = ";".join(frame.rejection_reasons)
            writer.writerow(f_dict)

    logger.info(f"Needle hysteresis results exported to {out_dir}")
