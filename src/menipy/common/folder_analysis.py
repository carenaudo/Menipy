"""Temporal folder analysis engine for image sequences.

This module provides high-speed, regularized analysis for directories of image files
representing sequential droplet experiment frames (sessile or pendant). It follows the
exact same foundational principles as video sequence analysis:
    1. Natural Alphanumeric Sorting: Ensures human/camera frame ordering
       (e.g., frame_1.png, frame_2.png, frame_10.png).
    2. Physical Invariant Locking on Frame 1: Locks the solid substrate baseline
       in sessile drops and the dispensing cannula geometry/scale in pendant drops,
       preventing redundant whole-frame Hough transforms and template searches.
    3. Localized ROI Prediction & Optical Flow: Tracks contact line displacement
       and restricts processing to a tightly bounded Region of Interest.
    4. Warm-Started Active Contour Evolution: Converges in 5-15 iterations (< 2 ms).
    5. Physical Quality Gating & Fallback: Detects sudden area/contact anomalies,
       resets tracking, and cleanly falls back to cold-start detection.

License & Clean-Room Compliance:
    Authored for Menipy using standard OpenCV (Apache-2.0) and SciPy/NumPy (BSD-3-Clause)
    primitives under Menipy's MIT open-source license.
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from menipy.common.cancellation import check_cancelled
from menipy.common.detection_helpers import auto_detect_features
from menipy.common.sequence_acquisition import IMAGE_SUFFIXES, _natural_key
from menipy.common.temporal_tracking import TemporalDropletTracker
from menipy.pipelines.pendant.metrics import compute_pendant_metrics
from menipy.pipelines.sessile.metrics import compute_sessile_metrics

logger = logging.getLogger(__name__)


@dataclass
class FolderFrameResult:
    """Analysis result for an individual frame within an image folder."""

    index: int
    path: Path
    stem: str
    accepted: bool
    tracked: bool
    rejection_reasons: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    drop_contour: np.ndarray | None = None
    contact_points: tuple[tuple[float, float], tuple[float, float]] | None = None
    apex_point: tuple[float, float] | None = None
    roi_rect: tuple[int, int, int, int] | None = None
    substrate_line: tuple[tuple[float, float], tuple[float, float]] | None = None
    needle_rect: tuple[int, int, int, int] | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass
class FolderAnalysisResult:
    """Consolidated result of a complete image folder analysis."""

    pipeline: str
    folder_path: str
    n_frames: int
    n_valid_frames: int
    valid_fraction: float
    frames: list[FolderFrameResult] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)

    def export_csv(self, out_path: str | Path) -> Path:
        """Export tabular summary CSV for all frames in the folder."""
        target = Path(out_path)
        target.parent.mkdir(parents=True, exist_ok=True)

        if not self.frames:
            target.write_text("", encoding="utf-8")
            return target

        # Extract union of all metric keys
        metric_keys = sorted(
            {k for f in self.frames for k in f.metrics.keys() if isinstance(f.metrics.get(k), (int, float, str, bool))}
        )

        headers = [
            "frame_index",
            "image_name",
            "image_path",
            "accepted",
            "tracked",
            "rejection_reasons",
            *metric_keys,
        ]

        with open(target, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=headers)
            writer.writeheader()
            for f in self.frames:
                row: dict[str, Any] = {
                    "frame_index": f.index,
                    "image_name": f.path.name,
                    "image_path": str(f.path),
                    "accepted": f.accepted,
                    "tracked": f.tracked,
                    "rejection_reasons": ";".join(f.rejection_reasons),
                }
                for mk in metric_keys:
                    val = f.metrics.get(mk)
                    if isinstance(val, (int, float, str, bool)):
                        row[mk] = val
                    else:
                        row[mk] = ""
                writer.writerow(row)

        return target


def discover_image_files(
    folder_or_files: str | Path | list[str | Path],
    glob_patterns: str = "*.png,*.jpg,*.jpeg,*.bmp,*.tif,*.tiff",
) -> list[Path]:
    """Discover image files and sort in natural human/camera order.

    Args:
        folder_or_files: Directory path or list of file paths.
        glob_patterns: Comma-separated glob patterns if directory is provided.

    Returns:
        Naturally sorted list of Path objects.
    """
    if isinstance(folder_or_files, (list, tuple)):
        paths = [Path(p).expanduser().resolve() for p in folder_or_files]
        return sorted((p for p in paths if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES), key=_natural_key)

    root = Path(folder_or_files).expanduser().resolve()
    if root.is_file():
        return [root]

    if not root.is_dir():
        return []

    patterns = [p.strip() for p in glob_patterns.split(",") if p.strip()]
    found: set[Path] = set()
    for pat in patterns:
        found.update(root.glob(pat))

    return sorted((p for p in found if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES), key=_natural_key)


def analyze_image_folder(
    folder_or_files: str | Path | list[str | Path],
    pipeline: str = "sessile",
    *,
    px_per_mm: float | None = None,
    needle_diameter_mm: float | None = None,
    substrate_line: tuple[tuple[float, float], tuple[float, float]] | None = None,
    needle_rect: tuple[int, int, int, int] | None = None,
    roi_rect: tuple[int, int, int, int] | None = None,
    contact_angle_method: str = "auto_residual",
    glob_patterns: str = "*.png,*.jpg,*.jpeg,*.bmp,*.tif,*.tiff",
    use_temporal_tracking: bool = True,
    check_cancelled: Callable[[], None] = check_cancelled,
) -> FolderAnalysisResult:
    """Analyze a folder of image sequence frames with physical invariant locking and tracking.

    Args:
        folder_or_files: Folder path or list of image files.
        pipeline: Analysis pipeline ("sessile" or "pendant").
        px_per_mm: Calibrated spatial scale (pixels per mm).
        needle_diameter_mm: Dispensing needle outer diameter in mm.
        substrate_line: Manual baseline override ((x1, y1), (x2, y2)).
        needle_rect: Manual needle override (x, y, w, h).
        roi_rect: Manual ROI override (x, y, w, h).
        contact_angle_method: Method for sessile contact angle calculation.
        glob_patterns: Comma-separated file globs when folder path is passed.
        use_temporal_tracking: Enable temporal tracking and invariant locking.
        check_cancelled: Cancellation probe.

    Returns:
        FolderAnalysisResult containing per-frame measurements and sequence statistics.
    """
    pipeline = pipeline.lower()
    files = discover_image_files(folder_or_files, glob_patterns=glob_patterns)
    folder_str = str(folder_or_files) if not isinstance(folder_or_files, list) else str(files[0].parent if files else "")

    if not files:
        return FolderAnalysisResult(
            pipeline=pipeline,
            folder_path=folder_str,
            n_frames=0,
            n_valid_frames=0,
            valid_fraction=0.0,
        )

    # Initialize tracker and physical invariants
    tracker = TemporalDropletTracker(pipeline=pipeline) if use_temporal_tracking else None
    locked_substrate = substrate_line
    locked_needle = needle_rect
    locked_scale = px_per_mm

    frame_results: list[FolderFrameResult] = []

    for idx, img_path in enumerate(files):
        check_cancelled()
        img = cv2.imread(str(img_path))
        if img is None:
            logger.warning("Corrupted or unreadable image: %s", img_path.name)
            frame_results.append(
                FolderFrameResult(
                    index=idx,
                    path=img_path,
                    stem=img_path.stem,
                    accepted=False,
                    tracked=False,
                    rejection_reasons=["corrupted_image"],
                )
            )
            continue

        detection: dict[str, Any] | None = None
        is_tracked = False

        # Attempt temporal tracking if initialized
        if tracker is not None and tracker.is_tracking:
            detection = tracker.track_frame(img, dt=0.033)
            if detection is not None:
                is_tracked = True
            else:
                logger.debug("Tracking quality gate failed for %s, falling back to cold start", img_path.name)

        # Cold-start fallback detection if not tracked
        if detection is None:
            detection = auto_detect_features(
                img,
                pipeline,
                detect_needle=(locked_needle is None),
                detect_substrate=(pipeline == "sessile" and locked_substrate is None),
            )

            # Re-apply locked physical invariants if established
            if locked_substrate is not None:
                detection["substrate_line"] = locked_substrate
            if locked_needle is not None:
                detection["needle_rect"] = locked_needle

            # Lock invariants on first valid detection
            if locked_substrate is None and "substrate_line" in detection:
                locked_substrate = detection["substrate_line"]
            if locked_needle is None and "needle_rect" in detection:
                locked_needle = detection["needle_rect"]

            # Estimate scale from needle if not explicitly set
            if locked_scale is None and needle_diameter_mm and needle_diameter_mm > 0:
                rect = detection.get("needle_rect")
                if rect and float(rect[2]) > 0:
                    locked_scale = float(rect[2]) / needle_diameter_mm

            # Initialize tracker for subsequent frames
            if tracker is not None and "drop_contour" in detection and detection["drop_contour"] is not None:
                tracker.initialize(img, detection, scale=locked_scale)

        reasons: list[str] = []
        contour = detection.get("drop_contour")
        contacts = detection.get("contact_points")
        sub_line = detection.get("substrate_line") or locked_substrate
        needle_box = detection.get("needle_rect") or locked_needle

        if contour is None or len(contour) < 4:
            reasons.append("drop_not_detected")
        if pipeline == "sessile":
            if contacts is None or len(contacts) < 2:
                reasons.append("contacts_not_detected")
            if sub_line is None:
                reasons.append("baseline_not_detected")

        metrics: dict[str, Any] = {}
        if not reasons and contour is not None:
            c_array = np.asarray(contour, dtype=float).reshape(-1, 2)
            eff_scale = locked_scale or 1.0

            try:
                if pipeline == "sessile":
                    metrics = compute_sessile_metrics(
                        c_array,
                        px_per_mm=eff_scale,
                        substrate_line=sub_line,
                        contact_points=contacts,
                        auto_detect_baseline=False,
                        auto_detect_apex=True,
                        contact_angle_method=contact_angle_method,
                    )
                    left = metrics.get("theta_left_deg")
                    right = metrics.get("theta_right_deg")
                    if not isinstance(left, (int, float)) or not isinstance(right, (int, float)) or left <= 0 or right <= 0:
                        reasons.append("contact_angle_invalid")
                else:  # pendant
                    apex = detection.get("apex_point")
                    metrics = compute_pendant_metrics(
                        c_array,
                        px_per_mm=eff_scale,
                        needle_diam_mm=needle_diameter_mm,
                        apex=apex,
                    )
                    gamma = metrics.get("surface_tension_mN_m")
                    if gamma is None:
                        gamma = metrics.get("gamma_mN_m")
                    if not isinstance(gamma, (int, float)) or gamma <= 0:
                        reasons.append("surface_tension_invalid")
                    else:
                        metrics["gamma_mN_m"] = float(gamma)
                        metrics["surface_tension_mN_m"] = float(gamma)
            except Exception as exc:
                reasons.append(f"metrics_failed:{exc}")
                logger.debug("Metrics computation failed for %s: %s", img_path.name, exc)

        accepted = len(reasons) == 0
        if not accepted and tracker is not None:
            tracker.reset()

        frame_results.append(
            FolderFrameResult(
                index=idx,
                path=img_path,
                stem=img_path.stem,
                accepted=accepted,
                tracked=is_tracked,
                rejection_reasons=reasons,
                metrics=metrics,
                drop_contour=contour,
                contact_points=contacts,
                apex_point=detection.get("apex_point"),
                roi_rect=detection.get("roi_rect"),
                substrate_line=sub_line,
                needle_rect=needle_box,
                diagnostics=detection.get("detector_diagnostics", {}),
            )
        )

    n_valid = sum(f.accepted for f in frame_results)
    valid_fraction = n_valid / max(1, len(frame_results))

    # Consolidated summary metrics
    summary: dict[str, Any] = {
        "pipeline": pipeline,
        "n_frames": len(frame_results),
        "n_valid_frames": n_valid,
        "valid_fraction": valid_fraction,
        "n_tracked_frames": sum(f.tracked for f in frame_results),
    }

    if pipeline == "sessile":
        valid_thetas = [
            (f.metrics["theta_left_deg"] + f.metrics["theta_right_deg"]) / 2.0
            for f in frame_results
            if f.accepted and "theta_left_deg" in f.metrics and "theta_right_deg" in f.metrics
        ]
        if valid_thetas:
            summary["theta_mean_deg"] = float(np.mean(valid_thetas))
            summary["theta_median_deg"] = float(np.median(valid_thetas))
            summary["theta_std_deg"] = float(np.std(valid_thetas))
    else:
        valid_gammas = [
            f.metrics["gamma_mN_m"]
            for f in frame_results
            if f.accepted and "gamma_mN_m" in f.metrics and f.metrics["gamma_mN_m"] > 0
        ]
        if valid_gammas:
            summary["gamma_mean_mN_m"] = float(np.mean(valid_gammas))
            summary["gamma_median_mN_m"] = float(np.median(valid_gammas))
            summary["gamma_std_mN_m"] = float(np.std(valid_gammas))

    return FolderAnalysisResult(
        pipeline=pipeline,
        folder_path=folder_str,
        n_frames=len(frame_results),
        n_valid_frames=n_valid,
        valid_fraction=valid_fraction,
        frames=frame_results,
        summary=summary,
    )


__all__ = [
    "FolderFrameResult",
    "FolderAnalysisResult",
    "discover_image_files",
    "analyze_image_folder",
]
