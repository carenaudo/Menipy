"""Clean-room geometry prototypes used by the optional Phase-B comparison.

The functions in this module deliberately depend only on image geometry and
SciPy/OpenCV primitives.  They return the same diagnostic envelopes used by
Phase A and never alter a pipeline result unless the caller explicitly opts in.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.optimize import least_squares

from .detection_result import DetectionResult


@dataclass
class BilateralNeedleFit:
    """Robust bilateral shaft fit in image coordinates."""

    left: tuple[float, float]
    right: tuple[float, float]
    rows: np.ndarray
    width_px: float
    width_cv: float
    angle_left_deg: float
    angle_right_deg: float
    rmse_left_px: float
    rmse_right_px: float
    p95_left_px: float
    p95_right_px: float
    inlier_fraction: float
    accepted: bool
    rejection_reasons: list[str] = field(default_factory=list)

    @property
    def centerline(self) -> tuple[tuple[float, float], tuple[float, float]]:
        y0, y1 = float(self.rows.min()), float(self.rows.max())
        return (self.center(y0), self.center(y1))

    def center(self, y: float) -> tuple[float, float]:
        return ((self.left[0] * y + self.left[1] + self.right[0] * y + self.right[1]) / 2.0, y)

    def to_detection(self) -> DetectionResult:
        y0, y1 = int(round(self.rows.min())), int(round(self.rows.max()))
        x_left = self.left[0] * y0 + self.left[1]
        x_right = self.right[0] * y0 + self.right[1]
        x = int(round(min(x_left, x_right)))
        width = int(round(max(self.width_px, 1.0)))
        value = {
            "needle_rect": (x, y0, width, max(1, y1 - y0)),
            "contact_points": (
                tuple(int(round(v)) for v in (self.left[0] * y1 + self.left[1], y1)),
                tuple(int(round(v)) for v in (self.right[0] * y1 + self.right[1], y1)),
            ),
            "left_line": self.left,
            "right_line": self.right,
            "centerline": self.centerline,
        }
        return DetectionResult(
            value=value,
            confidence=1.0 if self.accepted else 0.0,
            accepted=self.accepted,
            rejection_reasons=list(self.rejection_reasons),
            metrics={
                "rows": int(self.rows.size),
                "width_px": float(self.width_px),
                "width_cv": float(self.width_cv),
                "angle_left_deg": float(self.angle_left_deg),
                "angle_right_deg": float(self.angle_right_deg),
                "parallelism_deg": float(abs(self.angle_left_deg - self.angle_right_deg)),
                "rmse_left_px": float(self.rmse_left_px),
                "rmse_right_px": float(self.rmse_right_px),
                "p95_left_px": float(self.p95_left_px),
                "p95_right_px": float(self.p95_right_px),
                "inlier_fraction": float(self.inlier_fraction),
            },
            parameters={"loss": "soft_l1", "min_rows": 20},
        )


def _robust_line(y: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if y.size < 2:
        raise ValueError("not_enough_line_points")
    order = np.argsort(y)
    y = np.asarray(y, dtype=float)[order]
    x = np.asarray(x, dtype=float)[order]
    x0 = np.polyfit(y, x, 1)

    def residual(p: np.ndarray) -> np.ndarray:
        return p[0] * y + p[1] - x

    fit = least_squares(residual, x0=x0, loss="soft_l1", f_scale=1.5, max_nfev=100)
    r = residual(fit.x)
    scale = max(1.0, 1.4826 * float(np.median(np.abs(r - np.median(r)))))
    inliers = np.abs(r) <= 3.0 * scale
    return fit.x, r, inliers


def fit_bilateral_needle_edges(
    rows: np.ndarray,
    left_x: np.ndarray,
    right_x: np.ndarray,
    *,
    min_rows: int = 20,
) -> BilateralNeedleFit:
    """Fit two robust shaft lines from per-row edge samples."""
    rows = np.asarray(rows, dtype=float).reshape(-1)
    left_x = np.asarray(left_x, dtype=float).reshape(-1)
    right_x = np.asarray(right_x, dtype=float).reshape(-1)
    valid = np.isfinite(rows) & np.isfinite(left_x) & np.isfinite(right_x) & (right_x > left_x)
    rows, left_x, right_x = rows[valid], left_x[valid], right_x[valid]
    reasons: list[str] = []
    if rows.size < min_rows:
        reasons.append("needle_insufficient_rows")
    if rows.size < 2:
        dummy = np.array([0.0, 0.0])
        return BilateralNeedleFit((0.0, 0.0), (0.0, 0.0), dummy, 0.0, float("inf"), 0.0, 0.0, float("inf"), float("inf"), float("inf"), float("inf"), 0.0, False, reasons or ["needle_no_edges"])

    lp, lr, li = _robust_line(rows, left_x)
    rp, rr, ri = _robust_line(rows, right_x)
    widths = (rp[0] * rows + rp[1]) - (lp[0] * rows + lp[1])
    width = float(np.median(widths))
    width_cv = float(np.std(widths) / max(abs(width), 1e-9))
    angle_l = float(np.degrees(np.arctan(lp[0])))
    angle_r = float(np.degrees(np.arctan(rp[0])))
    p95_l = float(np.percentile(np.abs(lr), 95))
    p95_r = float(np.percentile(np.abs(rr), 95))
    inlier_fraction = float(np.mean(li & ri))
    if p95_l > 2.5 or p95_r > 2.5:
        reasons.append("needle_residual_gate_failed")
    if abs(angle_l - angle_r) > 1.5:
        reasons.append("needle_non_parallel_edges")
    if width_cv > 0.08:
        reasons.append("needle_width_unstable")
    accepted = not reasons and width > 0
    return BilateralNeedleFit(tuple(lp), tuple(rp), rows, width, width_cv, angle_l, angle_r, float(np.sqrt(np.mean(lr**2))), float(np.sqrt(np.mean(rr**2))), p95_l, p95_r, inlier_fraction, accepted, reasons)


def _edge_samples_from_image(image: np.ndarray, drop_contour: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract stable top-shaft edges from a grayscale/binary image."""
    arr = np.asarray(image)
    if arr.ndim == 3:
        arr = np.mean(arr[..., :3], axis=2)
    arr = arr.astype(float)
    if arr.size == 0:
        return np.empty(0), np.empty(0), np.empty(0)
    lo, hi = float(np.percentile(arr, 1.0)), float(np.percentile(arr, 99.0))
    if hi - lo < 5.0:
        return np.empty(0), np.empty(0), np.empty(0)
    threshold = lo + 0.5 * (hi - lo)
    mask = arr <= threshold
    h, w = mask.shape[:2]
    max_y = int(round(h * 0.60))
    if drop_contour is not None:
        pts = np.asarray(drop_contour).reshape(-1, 2)
        if pts.size:
            max_y = min(max_y, int(np.percentile(pts[:, 1], 45)))
    ys: list[float] = []
    ls: list[float] = []
    rs: list[float] = []
    center = w / 2.0
    for y in range(max_y):
        xs = np.flatnonzero(mask[y])
        if xs.size < 2:
            continue
        # Use the central connected span, avoiding unrelated lateral objects.
        spans = np.split(xs, np.where(np.diff(xs) > 1)[0] + 1)
        span = min((s for s in spans if s.size >= 2), key=lambda s: abs(float(np.mean(s)) - center), default=None)
        if span is None:
            continue
        ys.append(float(y))
        ls.append(float(span[0]))
        rs.append(float(span[-1]))
    return np.asarray(ys), np.asarray(ls), np.asarray(rs)


def detect_bilateral_needle(image: np.ndarray, drop_contour: np.ndarray | None = None, **_: Any) -> DetectionResult:
    """Detect a bilateral needle from an image using robust edge lines."""
    rows, left, right = _edge_samples_from_image(image, drop_contour)
    return fit_bilateral_needle_edges(rows, left, right).to_detection()


@dataclass
class PendantInitialization:
    """Robust pendant axis and Young–Laplace seed."""

    axis_origin_px: tuple[float, float]
    axis_direction_xy: tuple[float, float]
    apex_xy: tuple[float, float]
    r0_seed_mm: float
    beta_seed: float
    coverage: float
    asymmetry: float
    residual_px: float
    accepted: bool
    rejection_reasons: list[str] = field(default_factory=list)

    def to_diagnostics(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "rejection_reasons": list(self.rejection_reasons),
            "axis_origin_px": self.axis_origin_px,
            "axis_direction_xy": self.axis_direction_xy,
            "apex_xy": self.apex_xy,
            "r0_seed_mm": self.r0_seed_mm,
            "beta_seed": self.beta_seed,
            "coverage": self.coverage,
            "asymmetry": self.asymmetry,
            "residual_px": self.residual_px,
        }


def robust_pendant_initializer(contour_px: np.ndarray, px_per_mm: float = 1.0, **_: Any) -> PendantInitialization:
    """Estimate a pendant symmetry axis from paired contour envelopes."""
    xy = np.asarray(contour_px, dtype=float).reshape(-1, 2)
    reasons: list[str] = []
    if xy.shape[0] < 12:
        return PendantInitialization((0.0, 0.0), (0.0, -1.0), (0.0, 0.0), 0.0, 0.3, 0.0, 1.0, float("inf"), False, ["pendant_insufficient_contour"])
    apex = xy[int(np.argmax(xy[:, 1]))]
    y_min, y_max = float(np.min(xy[:, 1])), float(np.max(xy[:, 1]))
    ys: list[float] = []
    mids: list[float] = []
    widths: list[float] = []
    for y in np.linspace(y_min, y_max, 40):
        pts = xy[np.abs(xy[:, 1] - y) <= max(1.0, (y_max - y_min) / 80.0)]
        if pts.shape[0] < 2:
            continue
        xs = np.sort(pts[:, 0])
        ys.append(float(y))
        mids.append(float((xs[0] + xs[-1]) / 2.0))
        widths.append(float(xs[-1] - xs[0]))
    if len(ys) < 8:
        return PendantInitialization(tuple(apex), (0.0, -1.0), tuple(apex), 0.0, 0.3, 0.0, 1.0, float("inf"), False, ["pendant_axis_insufficient_rows"])
    line_params, residual, _ = _robust_line(np.asarray(ys), np.asarray(mids))
    direction = np.asarray([-line_params[0], -1.0], dtype=float)
    direction /= np.linalg.norm(direction)
    radius_px = float(np.percentile(np.asarray(widths), 25.0) / 2.0)
    r0_mm = max(radius_px / max(float(px_per_mm), 1e-9), 0.05)
    aspect = float(np.ptp(xy[:, 0]) / max(np.ptp(xy[:, 1]), 1e-9))
    beta = float(np.clip(0.3 + 0.2 * aspect, 0.02, 4.0))
    midpoint_residual = np.asarray(mids) - (line_params[0] * np.asarray(ys) + line_params[1])
    asymmetry = float(np.std(midpoint_residual) / max(np.ptp(np.asarray(widths)), 1e-9))
    coverage = float(len(ys) / 40.0)
    residual_px = float(np.sqrt(np.mean(residual**2)))
    if coverage < 0.5:
        reasons.append("pendant_axis_low_coverage")
    if residual_px > 3.0:
        reasons.append("pendant_axis_residual_gate_failed")
    if asymmetry > 0.25:
        reasons.append("pendant_asymmetry_high")
    return PendantInitialization(tuple(apex), tuple(direction), tuple(apex), r0_mm, beta, coverage, asymmetry, residual_px, not reasons, reasons)


__all__ = ["BilateralNeedleFit", "PendantInitialization", "fit_bilateral_needle_edges", "detect_bilateral_needle", "robust_pendant_initializer"]
