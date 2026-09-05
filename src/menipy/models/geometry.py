"""Geometric models for contours, geometry landmarks, and spatial features."""

from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import BaseModel, Field, field_validator

from .typing import ContourArray


class Point(BaseModel):
    """Represents a 2D point."""

    x: float
    y: float


class ROI(BaseModel):
    """Represents a rectangular region of interest."""

    x: int
    y: int
    width: int
    height: int


class Needle(BaseModel):
    """Represents the needle."""

    x: int
    y: int
    width: int
    height: int


class ContactLine(BaseModel):
    """Represents the contact line."""

    x1: float
    y1: float
    x2: float
    y2: float


class Contour(BaseModel):
    """
    Detected droplet (or meniscus) boundary.
    Coordinates are in pixels unless `units='mm'` and scaling applied.
    """

    xy: ContourArray = Field(description="array of shape (N, 2) with columns [x, y]")
    closed: bool = Field(default=True)
    units: Literal["px", "mm"] = Field(default="px")
    smoothing: float | None = Field(default=None, ge=0, description="spline/fit λ")
    origin_hint: tuple[float, float] | None = Field(
        default=None, description="optional origin (x0, y0)"
    )

    @field_validator("xy")
    @classmethod
    def _check_xy(cls, arr: np.ndarray) -> np.ndarray:
        if not isinstance(arr, np.ndarray):
            raise TypeError("xy must be a numpy ndarray")
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError("xy must have shape (N, 2)")
        if arr.dtype.kind not in ("f", "i"):
            raise TypeError("xy must be float or int array")
        return arr.astype(np.float64, copy=False)


class Geometry(BaseModel):
    """Geometric landmarks required by solvers."""

    apex_xy: tuple[float, float] | None = None  # pendant/sessile
    axis_x: float | None = None  # symmetry axis x (px or mm)
    baseline_y: float | None = None  # sessile: substrate y
    contact_region_px: tuple[int, int] | None = None  # index range around CL
    tilt_deg: float = Field(default=0.0)


class CaptiveBubbleGeometry(Geometry):
    """Geometry landmarks for captive bubble analysis."""

    ceiling_y: float | None = Field(
        default=None, description="y-coordinate of chamber ceiling"
    )
    cap_depth_px: float | None = Field(
        default=None, description="depth of bubble cap in pixels"
    )


class SubstrateProfile(BaseModel):
    """Represents a flat or curved substrate boundary profile.

    Supports straight lines, circular arcs (cylinders, spheres, fibers, lenses),
    and general polynomials.
    """

    type: Literal["line", "circle_arc", "polynomial", "spline"] = "line"
    points: list[tuple[float, float]] = Field(
        default_factory=list,
        description="Key defining points: endpoints for line, (p1, p2, p3_crest) for arc",
    )
    parameters: dict[str, float] = Field(
        default_factory=dict,
        description="Parameters: center_x, center_y, radius, curvature_inv_px, slope, intercept, etc.",
    )
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)

    @classmethod
    def from_line(
        cls,
        p1: tuple[float, float],
        p2: tuple[float, float],
        confidence: float = 1.0,
    ) -> SubstrateProfile:
        """Create a straight substrate line from two points."""
        p1_f = (float(p1[0]), float(p1[1]))
        p2_f = (float(p2[0]), float(p2[1]))
        if p1_f[0] > p2_f[0]:
            p1_f, p2_f = p2_f, p1_f
        dx = p2_f[0] - p1_f[0]
        dy = p2_f[1] - p1_f[1]
        slope = dy / dx if abs(dx) > 1e-9 else 0.0
        intercept = p1_f[1] - slope * p1_f[0]
        tilt_deg = float(np.degrees(np.arctan2(dy, dx)))
        return cls(
            type="line",
            points=[p1_f, p2_f],
            parameters={
                "slope": slope,
                "intercept": intercept,
                "tilt_deg": tilt_deg,
                "curvature_inv_px": 0.0,
            },
            confidence=float(np.clip(confidence, 0.0, 1.0)),
        )

    @classmethod
    def from_arc(
        cls,
        p1: tuple[float, float],
        p2: tuple[float, float],
        p3: tuple[float, float],
        confidence: float = 1.0,
    ) -> SubstrateProfile:
        """Create a circular arc substrate from 3 points (p1: left, p2: right, p3: crest/curve handle)."""
        p1_f = (float(p1[0]), float(p1[1]))
        p2_f = (float(p2[0]), float(p2[1]))
        p3_f = (float(p3[0]), float(p3[1]))

        # Check collinearity via determinant
        d = 2 * (
            p1_f[0] * (p2_f[1] - p3_f[1])
            + p2_f[0] * (p3_f[1] - p1_f[1])
            + p3_f[0] * (p1_f[1] - p2_f[1])
        )
        if abs(d) < 1e-6:
            # Degenerate: collinear points -> straight line
            return cls.from_line(p1_f, p2_f, confidence=confidence)

        # Center of circle passing through p1, p2, p3
        p1_sq = p1_f[0] ** 2 + p1_f[1] ** 2
        p2_sq = p2_f[0] ** 2 + p2_f[1] ** 2
        p3_sq = p3_f[0] ** 2 + p3_f[1] ** 2

        cx = (
            p1_sq * (p2_f[1] - p3_f[1])
            + p2_sq * (p3_f[1] - p1_f[1])
            + p3_sq * (p1_f[1] - p2_f[1])
        ) / d
        cy = (
            p1_sq * (p3_f[0] - p2_f[0])
            + p2_sq * (p1_f[0] - p3_f[0])
            + p3_sq * (p2_f[0] - p1_f[0])
        ) / d

        radius = float(np.hypot(p1_f[0] - cx, p1_f[1] - cy))
        if radius < 1.0 or radius > 1e6:
            return cls.from_line(p1_f, p2_f, confidence=confidence)

        # Convex flag: if p3_f[1] < cy (in image coords, crest is above center -> convex substrate facing up)
        convex = 1.0 if p3_f[1] < cy else -1.0

        return cls(
            type="circle_arc",
            points=[p1_f, p2_f, p3_f],
            parameters={
                "center_x": float(cx),
                "center_y": float(cy),
                "radius": float(radius),
                "curvature_inv_px": float(1.0 / radius) if radius > 0 else 0.0,
                "convex": convex,
            },
            confidence=float(np.clip(confidence, 0.0, 1.0)),
        )

    def eval_y(self, x: float) -> float | None:
        """Evaluate the substrate y-coordinate at horizontal position x."""
        if self.type == "line":
            slope = self.parameters.get("slope", 0.0)
            intercept = self.parameters.get("intercept", 0.0)
            return float(slope * x + intercept)
        elif self.type == "circle_arc":
            cx = self.parameters.get("center_x", 0.0)
            cy = self.parameters.get("center_y", 0.0)
            r = self.parameters.get("radius", 0.0)
            convex = self.parameters.get("convex", 1.0)
            dx = x - cx
            if abs(dx) > r:
                return None
            dy = float(np.sqrt(max(0.0, r**2 - dx**2)))
            # If convex (> 0, center is below substrate surface in image coords): y = cy - dy
            return float(cy - dy if convex > 0 else cy + dy)
        elif self.type == "polynomial":
            a = self.parameters.get("a", 0.0)
            b = self.parameters.get("b", 0.0)
            c = self.parameters.get("c", 0.0)
            return float(a * x**2 + b * x + c)
        return None

    def eval_tangent_angle_deg(self, x: float, y: float | None = None) -> float:
        """Return the local substrate tangent angle in degrees relative to horizontal.

        Positive angle means tilting downwards to the right (image coordinates dy/dx > 0).
        """
        if self.type == "line":
            return float(self.parameters.get("tilt_deg", 0.0))
        elif self.type == "circle_arc":
            cx = self.parameters.get("center_x", 0.0)
            r = self.parameters.get("radius", 1.0)
            convex = self.parameters.get("convex", 1.0)
            dx = x - cx
            if abs(dx) >= r:
                dx = np.clip(dx, -r * 0.9999, r * 0.9999)
            dy_val = float(np.sqrt(max(1e-9, r**2 - dx**2)))
            # In image coords: if convex > 0, y(x) = cy - sqrt(r^2 - dx^2)
            # dy/dx = - (1 / (2*sqrt)) * (-2 dx) = dx / sqrt(r^2 - dx^2)
            slope = (dx / dy_val) if convex > 0 else -(dx / dy_val)
            return float(np.degrees(np.arctan(slope)))
        elif self.type == "polynomial":
            a = self.parameters.get("a", 0.0)
            b = self.parameters.get("b", 0.0)
            slope = 2 * a * x + b
            return float(np.degrees(np.arctan(slope)))
        return 0.0

    def to_chord(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """Return end chord ((x1, y1), (x2, y2)) for compatibility with substrate_line."""
        if len(self.points) >= 2:
            return (self.points[0], self.points[1])
        return ((0.0, 0.0), (100.0, 0.0))

    def sample_points(
        self, num_points: int = 50, n_points: int | None = None
    ) -> list[tuple[float, float]]:
        """Return sampled points along the substrate profile for rendering or clipping."""
        if n_points is not None:
            num_points = n_points
        if len(self.points) < 2:
            return []
        x_min = min(self.points[0][0], self.points[1][0])
        x_max = max(self.points[0][0], self.points[1][0])
        if abs(x_max - x_min) < 1.0:
            return [self.points[0], self.points[1]]
        xs = np.linspace(x_min, x_max, max(num_points, 10))
        pts: list[tuple[float, float]] = []
        for x in xs:
            y = self.eval_y(float(x))
            if y is not None:
                pts.append((float(x), float(y)))
        return pts

