"""Tests for advanced apex detection methods in menipy.math.apex and plugins."""

from __future__ import annotations

import numpy as np
import pytest

from menipy.common.geometry import refine_apex_curvature
from menipy.common.metrics import find_apex_index
from menipy.math.apex import (
    ApexResult,
    detect_apex,
    detect_apex_curved_substrate,
    detect_apex_flat,
    detect_apex_normal,
    detect_apex_symmetry_axis,
    refine_apex_polynomial,
)
from menipy.models.geometry import SubstrateProfile
from plugins.detect_apex import (
    detect_apex_auto,
    detect_apex_pendant,
    detect_apex_sessile,
)


def test_flat_crest_median_averaging():
    """Verify that multiple points sharing the identical vertical coordinate
    do not cause leftmost-pixel bias; the horizontal coordinate must be the midpoint.
    """
    # Create an arc with a discrete flat top at y = 20.0 from x = 40 to x = 60
    xs_flat = np.linspace(40.0, 60.0, 21)
    ys_flat = np.full_like(xs_flat, 20.0)

    # Flanks descending down to baseline y = 80 (strictly y > 20)
    xs_left = np.linspace(20.0, 39.0, 20)
    ys_left = 21.0 + 59.0 * ((39.0 - xs_left) / 19.0) ** 2

    xs_right = np.linspace(61.0, 80.0, 20)
    ys_right = 21.0 + 59.0 * ((xs_right - 61.0) / 19.0) ** 2

    xy = np.vstack(
        [
            np.column_stack([xs_left, ys_left]),
            np.column_stack([xs_flat, ys_flat]),
            np.column_stack([xs_right, ys_right]),
        ]
    )

    # Naive argmin picks index with leftmost flat point (x = 40.0)
    naive_idx = int(np.argmin(xy[:, 1]))
    assert xy[naive_idx, 0] == pytest.approx(40.0)

    # Our detect_apex_flat must pick the midpoint x = 50.0
    res = detect_apex_flat(xy, mode="sessile", band_px=1.0)
    assert res.point[0] == pytest.approx(50.0, abs=0.5)
    assert res.point[1] == pytest.approx(20.0, abs=0.5)
    assert res.band_points >= 21

    # Master detect_apex with refine=False
    res_master = detect_apex(xy, mode="sessile", refine=False)
    assert res_master.point[0] == pytest.approx(50.0, abs=0.5)


def test_tilted_substrate_perpendicular_distance():
    """Verify that on a tilted substrate, apex is defined as maximum perpendicular
    distance along the substrate normal rather than naive image vertical min(y).
    """
    # Substrate tilted at ~26.56 degrees: from (0, 100) to (200, 0)
    p1 = (0.0, 100.0)
    p2 = (200.0, 0.0)
    line_vec = np.array([200.0, -100.0])
    line_unit = line_vec / np.linalg.norm(line_vec)
    # Normal pointing into the drop (upwards & rightwards)
    normal = np.array([-line_unit[1], line_unit[0]])
    if normal[1] > 0:
        normal = -normal

    # Generate a symmetric spherical cap droplet perched on the tilted substrate
    center_base = np.array(p1) + 0.5 * line_vec  # (100, 50)
    s_coords = np.linspace(-40.0, 40.0, 101)  # chord along plate
    # Dome height perpendicular to plate
    h_max = 30.0
    h_profile = h_max * np.sqrt(np.maximum(0.0, 1.0 - (s_coords / 40.0) ** 2))

    contour_pts = []
    for s, h in zip(s_coords, h_profile):
        pt = center_base + s * line_unit + h * normal
        contour_pts.append(pt)
    xy = np.array(contour_pts)

    # True physical summit is at s = 0 (perpendicular peak), so pt = center_base + 30 * normal
    true_summit = center_base + h_max * normal

    # Naive image vertical min(y) picks an uphill point skewed along the gradient
    naive_idx = int(np.argmin(xy[:, 1]))
    naive_apex = xy[naive_idx]
    # Naive apex is skewed towards the uphill flank (x > true_summit[0])
    assert abs(naive_apex[0] - true_summit[0]) > 5.0

    # detect_apex_normal should recover the true physical summit
    res = detect_apex_normal(xy, baseline=(p1, p2), mode="sessile")
    assert res.point[0] == pytest.approx(true_summit[0], abs=1.0)
    assert res.point[1] == pytest.approx(true_summit[1], abs=1.0)

    # Unified detect_apex dispatcher should also find the normal summit
    res_disp = detect_apex(xy, mode="sessile", baseline=(p1, p2), refine=False)
    assert res_disp.point[0] == pytest.approx(true_summit[0], abs=1.0)
    assert res_disp.point[1] == pytest.approx(true_summit[1], abs=1.0)


def test_asymmetric_droplet_peak_shift():
    """Verify that an asymmetric profile correctly shifts the apex to the true peak."""
    xs = np.linspace(10.0, 90.0, 161)
    peak_x = 65.0
    sigma_left = 25.0
    sigma_right = 12.0
    sigma = np.where(xs < peak_x, sigma_left, sigma_right)
    ys = 60.0 - 40.0 * np.exp(-0.5 * ((xs - peak_x) / sigma) ** 2)
    xy = np.column_stack([xs, ys])

    res = detect_apex(xy, mode="sessile", refine=True, window_px=8.0)
    assert res.point[0] == pytest.approx(peak_x, abs=1.0)
    assert res.point[1] == pytest.approx(20.0, abs=0.5)


def test_curved_substrate_radial_clearance():
    """Verify apex detection on a convex circular substrate (cylinder / fiber)."""
    center_sub = (100.0, 200.0)
    r_sub = 100.0

    sub_profile = SubstrateProfile(
        type="circle_arc",
        parameters={"center_x": 100.0, "center_y": 200.0, "radius": 100.0, "convex": 1.0},
    )

    angles = np.linspace(np.pi * 0.35, np.pi * 0.65, 80)
    r_drop = r_sub + 40.0 * np.sin((angles - np.pi * 0.35) / (np.pi * 0.3) * np.pi)
    xs = center_sub[0] - r_drop * np.cos(angles)
    ys = center_sub[1] - r_drop * np.sin(angles)
    xy = np.column_stack([xs, ys])

    res = detect_apex_curved_substrate(xy, substrate=sub_profile, mode="sessile")
    assert res.point[0] == pytest.approx(100.0, abs=1.0)
    assert res.point[1] == pytest.approx(60.0, abs=1.0)

    # Dispatcher test
    res_disp = detect_apex(xy, mode="sessile", substrate=sub_profile, refine=False)
    assert res_disp.point[0] == pytest.approx(100.0, abs=1.0)
    assert res_disp.point[1] == pytest.approx(60.0, abs=1.0)


def test_subpixel_polynomial_refinement():
    """Verify sub-pixel parabolic vertex estimation and curvature radius R0."""
    true_x0 = 50.37
    true_y0 = 15.22
    a = 0.04  # y = true_y0 + a * (x - true_x0)^2 => R0 = 1 / (2*a) = 12.5 px
    expected_r0 = 1.0 / (2.0 * a)

    xs = np.linspace(35, 65, 31)
    ys = true_y0 + a * (xs - true_x0) ** 2
    xy = np.column_stack([xs, ys])

    res_coarse = detect_apex_flat(xy, mode="sessile")
    res_refined = refine_apex_polynomial(xy, res_coarse, window_px=10.0)

    assert isinstance(res_refined, ApexResult)
    assert res_refined.point[0] == pytest.approx(true_x0, abs=0.05)
    assert res_refined.point[1] == pytest.approx(true_y0, abs=0.05)
    assert res_refined.r0_px is not None
    assert res_refined.r0_px == pytest.approx(expected_r0, rel=0.05)


def test_pendant_apex_tip_detection():
    """Verify pendant apex is at the bottom (y_max) and respects symmetry."""
    xs = np.linspace(50.0, 100.0, 100)
    ys = 180.0 - 0.05 * (xs - 75.0) ** 2
    # Add a flat bottom tip (multiple points sharing y = 180.0)
    flat_mask = np.abs(xs - 75.0) <= 2.0
    ys[flat_mask] = 180.0
    xy = np.column_stack([xs, ys])

    res = detect_apex(xy, mode="pendant", refine=True)
    assert res.point[0] == pytest.approx(75.0, abs=0.5)
    assert res.point[1] == pytest.approx(180.0, abs=0.5)


def test_captive_bubble_tip_detection():
    """Verify captive bubble apex is at the lowest downward point (y_max)."""
    xs = np.linspace(30.0, 90.0, 80)
    ys = 140.0 - 0.03 * (xs - 60.0) ** 2
    xy = np.column_stack([xs, ys])

    res = detect_apex(xy, mode="captive_bubble", refine=True)
    assert res.point[0] == pytest.approx(60.0, abs=0.5)
    assert res.point[1] == pytest.approx(140.0, abs=0.5)


def test_capillary_rise_apex_detection():
    """Verify capillary rise apex is at the meniscus summit (y_min)."""
    xs = np.linspace(20.0, 60.0, 50)
    ys = 40.0 + 0.02 * (xs - 40.0) ** 2
    xy = np.column_stack([xs, ys])

    res = detect_apex(xy, mode="capillary_rise", refine=True)
    assert res.point[0] == pytest.approx(40.0, abs=0.5)
    assert res.point[1] == pytest.approx(40.0, abs=0.5)


def test_symmetry_axis_midpoint_intersection():
    """Verify horizontal slice midpoints and symmetry axis detection."""
    xs_left = np.linspace(30.0, 50.0, 20)
    ys_left = 80.0 - 50.0 * np.sqrt(np.maximum(0.0, 1.0 - ((xs_left - 50.0) / 20.0) ** 2))
    xs_right = np.linspace(50.0, 70.0, 20)
    ys_right = 80.0 - 50.0 * np.sqrt(np.maximum(0.0, 1.0 - ((xs_right - 50.0) / 20.0) ** 2))
    xy = np.vstack([np.column_stack([xs_left, ys_left]), np.column_stack([xs_right, ys_right])])

    res = detect_apex_symmetry_axis(xy, mode="sessile")
    assert res.point[0] == pytest.approx(50.0, abs=1.0)
    assert res.point[1] == pytest.approx(30.0, abs=2.0)


def test_plugin_detect_apex_backward_compatibility():
    """Ensure plugin functions return integer tuples and handle edge cases gracefully."""
    xy = np.array([[20.0, 60.0], [50.0, 20.0], [80.0, 60.0]])

    pt_sessile = detect_apex_sessile(xy)
    assert isinstance(pt_sessile, tuple)
    assert len(pt_sessile) == 2
    assert isinstance(pt_sessile[0], int)
    assert pt_sessile[0] == 50
    assert pt_sessile[1] == 20

    pt_pendant = detect_apex_pendant(xy)
    assert isinstance(pt_pendant, tuple)
    assert pt_pendant[1] == 60  # bottom-most point

    pt_auto = detect_apex_auto(xy)
    assert isinstance(pt_auto, tuple)

    # Empty contour handling: plugins return None
    empty_xy = np.empty((0, 2))
    assert detect_apex_sessile(empty_xy) is None
    assert detect_apex_pendant(empty_xy) is None


def test_refine_apex_curvature_geometry_wrapper():
    """Verify common.geometry.refine_apex_curvature does NOT place apex inside drop."""
    t = np.linspace(np.pi, 0.0, 100)
    # Semicircle radius 30 centered at (100, 100), crest at (100, 70)
    xy = np.column_stack([100.0 + 30.0 * np.cos(t), 100.0 - 30.0 * np.sin(t)])

    apex_pt, conf = refine_apex_curvature(xy)
    assert conf > 0.0
    # Apex must be at the crown (100, 70), NOT at the center (100, 100)!
    assert apex_pt[0] == pytest.approx(100.0, abs=1.0)
    assert apex_pt[1] == pytest.approx(70.0, abs=1.0)


def test_find_apex_index_metrics():
    """Verify common.metrics.find_apex_index returns a valid index close to detect_apex."""
    xs = np.linspace(20.0, 80.0, 61)
    ys = 25.0 + 0.05 * (xs - 50.0) ** 2
    xy = np.column_stack([xs, ys])

    idx = find_apex_index(xy, mode="sessile")
    assert 0 <= idx < len(xy)
    assert xy[idx, 0] == pytest.approx(50.0, abs=1.0)
