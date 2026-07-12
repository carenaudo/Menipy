"""Clean-room temporal association and hysteresis analysis for sessile drops."""

from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from menipy.common.detection_helpers import auto_detect_features
from menipy.models.frame import Frame
from menipy.models.temporal import (
    DynamicSessileResult,
    SequenceMetadata,
    TemporalFrameResult,
)
from menipy.pipelines.sessile.metrics import compute_sessile_metrics

BOOTSTRAP_SEED = 20260711


def _xy(value: Any) -> np.ndarray:
    points = np.asarray(value, dtype=float)
    if points.ndim == 3 and points.shape[-1] == 2:
        points = points.reshape(-1, 2)
    return points.reshape(-1, 2)


def _line_offset_angle(line: tuple[tuple[float, float], tuple[float, float]]) -> tuple[float, float]:
    (x1, y1), (x2, y2) = line
    return float((y1 + y2) / 2.0), float(np.degrees(np.arctan2(y2 - y1, x2 - x1)))


def _polygon_area(points: np.ndarray) -> float:
    return float(abs(cv2.contourArea(np.asarray(points, dtype=np.float32))))


def _predicted_roi(
    contour: np.ndarray,
    contacts: tuple[tuple[float, float], tuple[float, float]],
    velocity: np.ndarray,
    shape: tuple[int, ...],
) -> tuple[int, int, int, int]:
    predicted = contour + velocity.reshape(1, 2)
    x0, y0 = np.floor(np.min(predicted, axis=0)).astype(int)
    x1, y1 = np.ceil(np.max(predicted, axis=0)).astype(int)
    base = max(1.0, abs(contacts[1][0] - contacts[0][0]))
    margin = max(8, int(round(base * 0.08)))
    height, width = shape[:2]
    x0, y0 = max(0, x0 - margin), max(0, y0 - margin)
    x1, y1 = min(width, x1 + margin), min(height, y1 + margin)
    return int(x0), int(y0), int(max(1, x1 - x0)), int(max(1, y1 - y0))


def _robust_slope(t: np.ndarray, values: np.ndarray) -> float:
    """Small deterministic Huber-style local linear regression."""
    design = np.column_stack([t - np.mean(t), np.ones_like(t)])
    weights = np.ones_like(values)
    coef = np.linalg.lstsq(design, values, rcond=None)[0]
    for _ in range(6):
        residual = values - design @ coef
        scale = max(1e-9, 1.4826 * float(np.median(np.abs(residual - np.median(residual)))))
        normalized = np.abs(residual) / (1.345 * scale)
        weights = np.where(normalized <= 1.0, 1.0, 1.0 / np.maximum(normalized, 1e-12))
        coef = np.linalg.lstsq(design * weights[:, None], values * weights, rcond=None)[0]
    return float(coef[0])


def _assign_states(frames: list[TemporalFrameResult]) -> float:
    valid = [frame for frame in frames if frame.accepted and frame.half_width_mm is not None]
    for frame in valid:
        neighbours = [
            other
            for other in valid
            if abs(other.frame_index - frame.frame_index) <= 3
            and other.segment_id == frame.segment_id
        ]
        if len(neighbours) >= 3:
            t = np.asarray([other.timestamp_s for other in neighbours], dtype=float)
            values = np.asarray([other.half_width_mm for other in neighbours], dtype=float)
            frame.contact_velocity_mm_s = _robust_slope(t, values)

    pairs: list[float] = []
    for left, right in zip(valid, valid[1:]):
        dt = right.timestamp_s - left.timestamp_s
        if dt > 0 and right.segment_id == left.segment_id:
            pairs.append((float(right.half_width_mm) - float(left.half_width_mm)) / dt)
    if len(pairs) >= 3:
        # First differences of velocity isolate measurement noise without treating
        # the commanded advancing/receding speed itself as noise.
        values = np.diff(np.asarray(pairs, dtype=float))
        mad = 1.4826 * float(np.median(np.abs(values - np.median(values)))) / np.sqrt(2.0)
    else:
        mad = 0.0
    deadband = max(0.01, 3.0 * mad)
    for position, frame in enumerate(valid):
        local_slopes: list[float] = []
        if position > 0 and valid[position - 1].segment_id == frame.segment_id:
            dt = frame.timestamp_s - valid[position - 1].timestamp_s
            if dt > 0:
                local_slopes.append((float(frame.half_width_mm) - float(valid[position - 1].half_width_mm)) / dt)
        if position + 1 < len(valid) and valid[position + 1].segment_id == frame.segment_id:
            dt = valid[position + 1].timestamp_s - frame.timestamp_s
            if dt > 0:
                local_slopes.append((float(valid[position + 1].half_width_mm) - float(frame.half_width_mm)) / dt)
        # The seven-frame fit remains the reported velocity. The adjacent robust
        # median only snaps state transitions so a plateau is not shifted by half
        # the classification window.
        velocity = float(np.median(local_slopes)) if local_slopes else frame.contact_velocity_mm_s
        frame.state = "pinned" if velocity is None or abs(velocity) <= deadband else ("advancing" if velocity > 0 else "receding")

    # Advancing/receding runs shorter than three frames are not physical states.
    index = 0
    while index < len(frames):
        state = frames[index].state
        end = index + 1
        while end < len(frames) and frames[end].state == state and frames[end].segment_id == frames[index].segment_id:
            end += 1
        if state in {"advancing", "receding"} and end - index < 3:
            for position in range(index, end):
                frames[position].state = "pinned" if frames[position].accepted else "invalid"
        index = end
    return deadband


def _bootstrap_stats(values: Sequence[float]) -> dict[str, float | int | list[float]]:
    array = np.asarray(values, dtype=float)
    median = float(np.median(array))
    mad = float(np.median(np.abs(array - median)))
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = rng.choice(array, size=(2000, len(array)), replace=True)
    medians = np.median(samples, axis=1)
    return {
        "median_deg": median,
        "mad_deg": mad,
        "ci95_deg": [float(np.percentile(medians, 2.5)), float(np.percentile(medians, 97.5))],
        "n_frames": int(len(array)),
    }


def _max_invalid_run(frames: Sequence[TemporalFrameResult]) -> int:
    maximum = current = 0
    for frame in frames:
        current = 0 if frame.accepted else current + 1
        maximum = max(maximum, current)
    return maximum


def _summarize(frames: list[TemporalFrameResult], fps: float) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "n_frames": len(frames),
        "n_valid_frames": sum(frame.accepted for frame in frames),
        "valid_fraction": sum(frame.accepted for frame in frames) / max(1, len(frames)),
        "duration_s": frames[-1].timestamp_s - frames[0].timestamp_s if len(frames) > 1 else 0.0,
        "fps": fps,
    }
    for state in ("advancing", "receding"):
        selected = [frame for frame in frames if frame.accepted and frame.state == state]
        summary[f"{state}_duration_s"] = len(selected) / fps
        velocities = [frame.contact_velocity_mm_s for frame in selected if frame.contact_velocity_mm_s is not None]
        if velocities:
            summary[f"{state}_velocity_median_mm_s"] = float(np.median(velocities))
        for side in ("left", "right"):
            values = [getattr(frame, f"theta_{side}_deg") for frame in selected]
            values = [float(value) for value in values if value is not None]
            if len(values) >= 5:
                summary[f"theta_{state}_{side}"] = _bootstrap_stats(values)
        combined = [
            (float(frame.theta_left_deg) + float(frame.theta_right_deg)) / 2.0
            for frame in selected
            if frame.theta_left_deg is not None and frame.theta_right_deg is not None
        ]
        if len(combined) >= 5:
            stats = _bootstrap_stats(combined)
            summary[f"theta_{state}"] = stats
            summary[f"theta_{state}_deg"] = stats["median_deg"]
    if "theta_advancing_deg" in summary and "theta_receding_deg" in summary:
        summary["contact_angle_hysteresis_deg"] = float(summary["theta_advancing_deg"]) - float(summary["theta_receding_deg"])
    return summary


def analyze_dynamic_sessile(
    frames: list[Frame],
    metadata: SequenceMetadata,
    *,
    px_per_mm: float | None,
    needle_diameter_mm: float | None,
    contact_angle_method: str = "auto_residual",
) -> DynamicSessileResult:
    """Analyze a complete sequence while quarantining invalid frame measurements."""
    detections = [auto_detect_features(frame.image, "sessile") for frame in frames]
    scale_samples: list[float] = []
    if px_per_mm is None and needle_diameter_mm and needle_diameter_mm > 0:
        for detection in detections:
            rect = detection.get("needle_rect")
            if rect and float(rect[2]) > 0:
                scale_samples.append(float(rect[2]) / needle_diameter_mm)
            if len(scale_samples) == 5:
                break
        if scale_samples:
            px_per_mm = float(np.median(scale_samples))

    output: list[TemporalFrameResult] = []
    reference_line: tuple[tuple[float, float], tuple[float, float]] | None = None
    previous_line: tuple[tuple[float, float], tuple[float, float]] | None = None
    previous_contacts: tuple[tuple[float, float], tuple[float, float]] | None = None
    previous_contour: np.ndarray | None = None
    previous_area: float | None = None
    contact_velocity = np.zeros(2, dtype=float)
    lost = 0
    segment = 0

    for frame_index, (frame, detection) in enumerate(zip(frames, detections)):
        result = TemporalFrameResult(frame_index=frame_index, timestamp_s=metadata.timestamps_s[frame_index], segment_id=segment)
        reasons: list[str] = []
        contour_value = detection.get("drop_contour")
        contacts_value = detection.get("contact_points")
        line_value = detection.get("substrate_line")
        if px_per_mm is None or px_per_mm <= 0:
            reasons.append("dynamic_missing_calibration")
        if contour_value is None:
            reasons.append("dynamic_drop_not_detected")
        if contacts_value is None:
            reasons.append("dynamic_contacts_not_detected")
        if line_value is None:
            reasons.append("dynamic_baseline_not_detected")
        if reasons:
            result.rejection_reasons = reasons
            result.diagnostics["detectors"] = detection.get("detector_diagnostics", {})
            output.append(result)
            lost += 1
            continue

        contour = _xy(contour_value)
        contacts = (
            (float(contacts_value[0][0]), float(contacts_value[0][1])),
            (float(contacts_value[1][0]), float(contacts_value[1][1])),
        )
        line = (
            (float(line_value[0][0]), float(line_value[0][1])),
            (float(line_value[1][0]), float(line_value[1][1])),
        )
        if contacts[0][0] >= contacts[1][0]:
            reasons.append("dynamic_contacts_inverted")
        line_offset, line_angle = _line_offset_angle(line)
        if reference_line is None:
            reference_line = line
        if previous_line is not None:
            prior_offset, prior_angle = _line_offset_angle(previous_line)
            if abs(line_offset - prior_offset) > 5.0 or abs(line_angle - prior_angle) > 1.0:
                reasons.append("dynamic_baseline_frame_drift")
        ref_offset, ref_angle = _line_offset_angle(reference_line)
        if abs(line_offset - ref_offset) > 20.0 or abs(line_angle - ref_angle) > 3.0:
            reasons.append("dynamic_baseline_total_drift")

        area = _polygon_area(contour)
        if previous_contacts is not None and lost == 0:
            prior_width = max(1.0, previous_contacts[1][0] - previous_contacts[0][0])
            displacement = max(np.linalg.norm(np.asarray(contacts[index]) - np.asarray(previous_contacts[index])) for index in (0, 1))
            if displacement > 0.10 * prior_width:
                reasons.append("dynamic_contact_jump")
            if previous_area and abs(area - previous_area) / previous_area > 0.25:
                reasons.append("dynamic_area_jump")

        if previous_contour is not None and previous_contacts is not None:
            result.predicted_roi = _predicted_roi(previous_contour, previous_contacts, contact_velocity, frame.image.shape)
        metrics: dict[str, Any] = {}
        if not reasons:
            try:
                metrics = compute_sessile_metrics(
                    contour,
                    px_per_mm=float(px_per_mm),
                    substrate_line=line,
                    contact_points=contacts,
                    auto_detect_baseline=False,
                    auto_detect_apex=True,
                    contact_angle_method=contact_angle_method,
                )
            except Exception as exc:
                reasons.append("dynamic_geometry_failed")
                result.diagnostics["geometry_error"] = str(exc)
        left = metrics.get("theta_left_deg")
        right = metrics.get("theta_right_deg")
        if not reasons and (not isinstance(left, (int, float)) or not isinstance(right, (int, float)) or left <= 0 or right <= 0):
            reasons.append("dynamic_contact_angle_invalid")

        if reasons:
            result.rejection_reasons = reasons
            result.diagnostics["detectors"] = detection.get("detector_diagnostics", {})
            output.append(result)
            lost += 1
            continue

        if lost:
            segment += 1
            result.segment_id = segment
            result.diagnostics["reacquired_after_frames"] = lost
        lost = 0
        if previous_contacts is not None:
            old_center = np.mean(np.asarray(previous_contacts), axis=0)
            new_center = np.mean(np.asarray(contacts), axis=0)
            contact_velocity = new_center - old_center
        result.accepted = True
        result.baseline = line
        result.contacts = contacts
        result.theta_left_deg = float(left)
        result.theta_right_deg = float(right)
        result.half_width_mm = float(np.linalg.norm(np.asarray(contacts[1]) - np.asarray(contacts[0])) / (2.0 * float(px_per_mm)))
        result.contour = [(float(x), float(y)) for x, y in contour]
        result.diagnostics.update({
            "detectors": detection.get("detector_diagnostics", {}),
            "contact_angle": {key: metrics.get(key) for key in ("method", "method_left", "method_right", "contact_angle_fit_rmse_px") if key in metrics},
        })
        output.append(result)
        previous_line, previous_contacts, previous_contour, previous_area = line, contacts, contour, area

    deadband = _assign_states(output)
    summary = _summarize(output, metadata.fps)
    rejection_reasons: list[str] = []
    if len(output) < 20:
        rejection_reasons.append("dynamic_insufficient_frames")
    if summary["valid_fraction"] < 0.80:
        rejection_reasons.append("dynamic_valid_fraction_low")
    maximum_gap = _max_invalid_run(output)
    if maximum_gap > 5:
        rejection_reasons.append("dynamic_gap_too_long")
    if px_per_mm is None or px_per_mm <= 0:
        rejection_reasons.append("dynamic_missing_calibration")
    accepted = not rejection_reasons
    return DynamicSessileResult(
        accepted=accepted,
        rejection_reasons=list(dict.fromkeys(rejection_reasons)),
        metadata=metadata,
        calibration={"px_per_mm": px_per_mm, "method": "explicit" if not scale_samples else "needle_initial_median", "samples": scale_samples},
        summary=summary,
        frames=output,
        diagnostics={"velocity_deadband_mm_s": deadband, "max_invalid_run": maximum_gap, "raw_values_promoted": True, "classification_window_frames": 7},
    )


def export_dynamic_result(result: DynamicSessileResult, output_dir: str | Path) -> dict[str, Path]:
    """Write the canonical JSON, stable summary CSV and long-form frame CSV."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    json_path = root / "results.json"
    summary_path = root / "results.csv"
    frames_path = root / "results_frames.csv"
    json_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    summary_columns = [
        "pipeline", "schema_version", "accepted", "rejection_reasons", "source_id",
        "fps", "n_frames", "n_valid_frames", "valid_fraction", "duration_s",
        "theta_advancing_deg", "theta_receding_deg", "contact_angle_hysteresis_deg",
    ]
    summary_row = {
        "pipeline": result.pipeline,
        "schema_version": result.schema_version,
        "accepted": result.accepted,
        "rejection_reasons": ";".join(result.rejection_reasons),
        "source_id": result.metadata.source_id,
        "fps": result.metadata.fps,
        **{key: result.summary.get(key, "") for key in summary_columns},
    }
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerow(summary_row)
    frame_columns = [
        "frame_index", "timestamp_s", "accepted", "rejection_reasons", "segment_id",
        "state", "theta_left_deg", "theta_right_deg", "half_width_mm",
        "contact_velocity_mm_s", "baseline_x1", "baseline_y1", "baseline_x2",
        "baseline_y2", "contact_left_x", "contact_left_y", "contact_right_x",
        "contact_right_y", "predicted_roi_json", "diagnostics_json",
    ]
    with frames_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=frame_columns)
        writer.writeheader()
        for frame in result.frames:
            baseline = frame.baseline or ((None, None), (None, None))
            contacts = frame.contacts or ((None, None), (None, None))
            writer.writerow({
                "frame_index": frame.frame_index,
                "timestamp_s": frame.timestamp_s,
                "accepted": frame.accepted,
                "rejection_reasons": ";".join(frame.rejection_reasons),
                "segment_id": frame.segment_id,
                "state": frame.state,
                "theta_left_deg": frame.theta_left_deg,
                "theta_right_deg": frame.theta_right_deg,
                "half_width_mm": frame.half_width_mm,
                "contact_velocity_mm_s": frame.contact_velocity_mm_s,
                "baseline_x1": baseline[0][0], "baseline_y1": baseline[0][1],
                "baseline_x2": baseline[1][0], "baseline_y2": baseline[1][1],
                "contact_left_x": contacts[0][0], "contact_left_y": contacts[0][1],
                "contact_right_x": contacts[1][0], "contact_right_y": contacts[1][1],
                "predicted_roi_json": json.dumps(
                    [int(value) for value in frame.predicted_roi]
                    if frame.predicted_roi is not None
                    else None,
                    separators=(",", ":"),
                ),
                "diagnostics_json": json.dumps(frame.diagnostics, separators=(",", ":")),
            })
    return {"json": json_path, "summary_csv": summary_path, "frames_csv": frames_path}
