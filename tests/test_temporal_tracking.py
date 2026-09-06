"""Unit and integration tests for temporal droplet tracking and invariant locking."""

from __future__ import annotations

import cv2
import numpy as np

from menipy.common.temporal_sessile import analyze_dynamic_sessile
from menipy.common.temporal_tracking import (
    TemporalDropletTracker,
    estimate_contact_velocity_optical_flow,
    predict_droplet_roi,
)
from menipy.models.frame import Frame
from menipy.models.temporal import SequenceMetadata


def make_synthetic_sessile_frame(
    size: tuple[int, int] = (160, 200),
    center_x: float = 100.0,
    substrate_y: float = 120.0,
    radius: float = 35.0,
    theta_deg: float = 65.0,
) -> tuple[np.ndarray, np.ndarray, tuple[tuple[float, float], tuple[float, float]], tuple[tuple[float, float], tuple[float, float]]]:
    """Generate a clean synthetic sessile droplet frame with known geometry."""
    h_img, w_img = size
    img = np.full((h_img, w_img), 230, dtype=np.uint8)

    theta_rad = np.radians(theta_deg)
    center_y = substrate_y + radius * np.cos(theta_rad)

    # Draw droplet disk clipped at substrate
    mask = np.zeros((h_img, w_img), dtype=np.uint8)
    cv2.circle(mask, (int(round(center_x)), int(round(center_y))), int(round(radius)), 255, -1)
    mask[int(round(substrate_y)) :, :] = 0
    img[mask == 255] = 30

    # Draw solid substrate baseline
    cv2.line(img, (0, int(round(substrate_y))), (w_img, int(round(substrate_y))), 80, 2)

    # Calculate exact contact points
    half_w = radius * np.sin(theta_rad)
    c_left = (float(center_x - half_w), float(substrate_y))
    c_right = (float(center_x + half_w), float(substrate_y))
    substrate_line = ((0.0, float(substrate_y)), (float(w_img - 1), float(substrate_y)))

    # Arc points
    angles = np.linspace(-theta_rad, theta_rad, 80)
    arc_x = center_x + radius * np.sin(angles)
    arc_y = center_y - radius * np.cos(angles)
    contour = np.column_stack([arc_x, arc_y])

    return img, contour, (c_left, c_right), substrate_line


def make_synthetic_pendant_frame(
    size: tuple[int, int] = (200, 160),
    needle_center_x: float = 80.0,
    needle_bottom_y: float = 40.0,
    needle_width: float = 20.0,
    drop_radius: float = 30.0,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]]:
    """Generate a synthetic pendant drop hanging from a fixed needle."""
    h_img, w_img = size
    img = np.full((h_img, w_img), 230, dtype=np.uint8)

    # Needle rectangle
    nx = int(needle_center_x - needle_width / 2.0)
    ny = 0
    nw = int(needle_width)
    nh = int(needle_bottom_y)
    needle_rect = (nx, ny, nw, nh)
    cv2.rectangle(img, (nx, ny), (nx + nw, ny + nh), 50, -1)

    # Drop circle center below needle
    center_y = needle_bottom_y + drop_radius * 0.8
    cv2.circle(img, (int(round(needle_center_x)), int(round(center_y))), int(round(drop_radius)), 40, -1)

    # Arc points
    angles = np.linspace(0, 2 * np.pi, 80, endpoint=False)
    x = needle_center_x + drop_radius * np.cos(angles)
    y = center_y + drop_radius * np.sin(angles)
    contour = np.column_stack([x, y])

    return img, contour, needle_rect


# =============================================================================
# 1. ROI Prediction Tests
# =============================================================================


class TestPredictDropletROI:
    def test_predict_roi_basic(self):
        contour = np.array([[50.0, 50.0], [100.0, 50.0], [100.0, 100.0], [50.0, 100.0]])
        contacts = ((50.0, 100.0), (100.0, 100.0))
        velocity = np.array([2.0, 0.0])
        shape = (200, 200)

        x0, y0, w, h = predict_droplet_roi(contour, contacts, velocity, shape, margin_ratio=0.10, min_margin=10)
        assert x0 >= 0
        assert y0 >= 0
        assert x0 + w <= 200
        assert y0 + h <= 200
        # ROI must contain the shifted contour
        assert x0 <= 52.0
        assert y0 <= 50.0
        assert x0 + w >= 102.0
        assert y0 + h >= 100.0

    def test_predict_roi_clips_boundaries(self):
        contour = np.array([[2.0, 2.0], [20.0, 2.0], [20.0, 20.0], [2.0, 20.0]])
        contacts = ((2.0, 20.0), (20.0, 20.0))
        velocity = np.array([-5.0, -5.0])
        shape = (100, 100)

        x0, y0, w, h = predict_droplet_roi(contour, contacts, velocity, shape, margin_ratio=0.2, min_margin=15)
        assert x0 == 0
        assert y0 == 0
        assert w > 0
        assert h > 0

    def test_predict_roi_no_contacts(self):
        contour = np.array([[40.0, 40.0], [80.0, 40.0], [80.0, 80.0], [40.0, 80.0]])
        velocity = np.array([0.0, 0.0])
        shape = (150, 150)

        x0, y0, w, h = predict_droplet_roi(contour, None, velocity, shape, min_margin=15)
        assert x0 == 25
        assert y0 == 25
        assert w == (80 - 40) + 30
        assert h == (80 - 40) + 30


# =============================================================================
# 2. Optical Flow Velocity Estimation Tests
# =============================================================================


class TestOpticalFlowVelocity:
    def test_optical_flow_on_textured_displacement(self):
        prev = np.zeros((100, 100), dtype=np.uint8)
        curr = np.zeros((100, 100), dtype=np.uint8)

        # Draw a small distinct corner feature near contacts
        cv2.circle(prev, (30, 70), 4, 255, -1)
        cv2.circle(prev, (70, 70), 4, 255, -1)

        # Shift by +2 in x, 0 in y
        cv2.circle(curr, (32, 70), 4, 255, -1)
        cv2.circle(curr, (72, 70), 4, 255, -1)

        contacts = ((30.0, 70.0), (70.0, 70.0))
        disp, ok = estimate_contact_velocity_optical_flow(prev, curr, contacts, search_window=(21, 21))

        assert ok
        np.testing.assert_allclose(disp, [2.0, 0.0], atol=0.5)

    def test_optical_flow_fallback_on_blank(self):
        prev = np.full((80, 80), 128, dtype=np.uint8)
        curr = np.full((80, 80), 128, dtype=np.uint8)
        contacts = ((20.0, 50.0), (60.0, 50.0))

        disp, ok = estimate_contact_velocity_optical_flow(prev, curr, contacts)
        assert disp.shape == (2,)


# =============================================================================
# 3. Tracker State & Invariant Locking Tests
# =============================================================================


class TestTemporalDropletTrackerState:
    def test_initialization_locks_invariants(self):
        img, contour, contacts, line = make_synthetic_sessile_frame()
        detection = {
            "drop_contour": contour,
            "contact_points": contacts,
            "substrate_line": line,
            "needle_rect": (90, 0, 20, 30),
        }

        tracker = TemporalDropletTracker(pipeline="sessile", num_nodes=50)
        assert not tracker.is_tracking
        assert tracker.reference_baseline is None

        ok = tracker.initialize(img, detection, scale=25.0)
        assert ok
        assert tracker.is_tracking
        assert tracker.reference_baseline == line
        assert tracker.reference_needle == (90, 0, 20, 30)
        assert tracker.reference_scale == 25.0
        assert tracker.previous_contour is not None
        assert len(tracker.previous_contour) == 50

    def test_initialization_rejects_empty_contour(self):
        img = np.full((100, 100), 200, dtype=np.uint8)
        tracker = TemporalDropletTracker()

        ok = tracker.initialize(img, {"drop_contour": np.array([])})
        assert not ok
        assert not tracker.is_tracking

    def test_reset_preserves_locked_invariants(self):
        img, contour, contacts, line = make_synthetic_sessile_frame()
        tracker = TemporalDropletTracker()
        tracker.initialize(img, {"drop_contour": contour, "substrate_line": line, "contact_points": contacts}, scale=20.0)

        assert tracker.is_tracking
        tracker.reset()

        assert not tracker.is_tracking
        assert tracker.previous_contour is None
        # Invariants must remain locked across resets
        assert tracker.reference_baseline == line
        assert tracker.reference_scale == 20.0

    def test_reset_all_clears_invariants(self):
        img, contour, contacts, line = make_synthetic_sessile_frame()
        tracker = TemporalDropletTracker()
        tracker.initialize(img, {"drop_contour": contour, "substrate_line": line, "contact_points": contacts}, scale=20.0)

        tracker.reset_all()
        assert not tracker.is_tracking
        assert tracker.reference_baseline is None
        assert tracker.reference_scale is None


# =============================================================================
# 4. Temporal Tracking Across Video Frames
# =============================================================================


class TestTemporalTrackingEvolution:
    def test_track_frame_advancing_sessile(self):
        img1, c1, contacts1, line = make_synthetic_sessile_frame(radius=30.0, theta_deg=60.0)
        tracker = TemporalDropletTracker(pipeline="sessile", snake_iterations=15, num_nodes=60)
        tracker.initialize(img1, {"drop_contour": c1, "contact_points": contacts1, "substrate_line": line}, scale=20.0)

        # Frame 2: droplet grew slightly (advancing drop)
        img2, c2, contacts2, _ = make_synthetic_sessile_frame(radius=32.0, theta_deg=62.0)
        tracked = tracker.track_frame(img2, dt=0.033)

        assert tracked is not None
        assert "drop_contour" in tracked
        assert "contact_points" in tracked
        assert "apex_point" in tracked
        assert tracked["substrate_line"] == tracker.reference_baseline

        diag = tracked["detector_diagnostics"]["tracking"]
        assert diag["tracked"] is True
        assert diag["frames_tracked"] == 1

        # Check contact points clipped to substrate line
        sub_y = line[0][1]
        c_pts = tracked["contact_points"]
        assert abs(c_pts[0][1] - sub_y) < 1.5
        assert abs(c_pts[1][1] - sub_y) < 1.5
        assert c_pts[0][0] < c_pts[1][0]

        # Apex should be at or near top
        apex = tracked["apex_point"]
        assert apex[1] < sub_y - 10.0
        assert abs(apex[1] - 103.0) < 3.0

    def test_track_frame_consecutive_steps(self):
        tracker = TemporalDropletTracker(pipeline="sessile", snake_iterations=10, num_nodes=50)

        # Frame 0: init
        img0, c0, contacts0, line = make_synthetic_sessile_frame(radius=28.0)
        tracker.initialize(img0, {"drop_contour": c0, "contact_points": contacts0, "substrate_line": line})

        # Track 5 consecutive frames
        for i in range(1, 6):
            r = 28.0 + 0.5 * i
            img_i, _, _, _ = make_synthetic_sessile_frame(radius=r)
            res = tracker.track_frame(img_i, dt=0.05)
            assert res is not None
            assert tracker.frames_tracked == i

    def test_quality_gate_area_jump_fallback(self):
        img1, c1, contacts1, line = make_synthetic_sessile_frame(radius=25.0)
        tracker = TemporalDropletTracker(pipeline="sessile")
        tracker.initialize(img1, {"drop_contour": c1, "contact_points": contacts1, "substrate_line": line})

        # Sudden jump: radius doubles (area quadruples, > 25% change)
        img2, _, _, _ = make_synthetic_sessile_frame(radius=50.0)
        result = tracker.track_frame(img2, dt=0.033)

        # Quality gate should fail and reset tracker for cold-start reacquisition
        assert result is None
        assert not tracker.is_tracking

    def test_quality_gate_contact_jump_fallback(self):
        img1, c1, contacts1, line = make_synthetic_sessile_frame(center_x=80.0, radius=25.0)
        tracker = TemporalDropletTracker(pipeline="sessile")
        tracker.initialize(img1, {"drop_contour": c1, "contact_points": contacts1, "substrate_line": line})

        # Shift drop center by 30px (> 10% of base width)
        img2, _, _, _ = make_synthetic_sessile_frame(center_x=130.0, radius=25.0)
        result = tracker.track_frame(img2, dt=0.033)

        assert result is None
        assert not tracker.is_tracking


# =============================================================================
# 5. Pendant Drop Tracking
# =============================================================================


class TestPendantDropletTracking:
    def test_pendant_initialization_and_tracking(self):
        img1, contour1, needle1 = make_synthetic_pendant_frame(drop_radius=25.0)
        tracker = TemporalDropletTracker(pipeline="pendant", snake_iterations=10, num_nodes=50)

        ok = tracker.initialize(img1, {"drop_contour": contour1, "needle_rect": needle1}, scale=18.0)
        assert ok
        assert tracker.reference_needle == needle1
        assert tracker.reference_scale == 18.0

        # Frame 2: slightly larger pendant drop
        img2, contour2, _ = make_synthetic_pendant_frame(drop_radius=27.0)
        tracked = tracker.track_frame(img2, dt=0.033)

        assert tracked is not None
        assert "drop_contour" in tracked
        assert "apex_point" in tracked
        assert tracked["needle_rect"] == needle1

        # Pendant apex is the maximum Y point (lowest in image)
        apex = tracked["apex_point"]
        contour = tracked["drop_contour"]
        assert abs(apex[1] - np.max(contour[:, 1])) < 1e-3


# =============================================================================
# 6. Dynamic Sessile Pipeline Integration
# =============================================================================


class TestDynamicSessilePipelineIntegration:
    def test_analyze_dynamic_sessile_with_tracking(self):
        # Create a 15-frame sequence of advancing droplet
        frames: list[Frame] = []
        timestamps = [i * 0.1 for i in range(15)]

        for i in range(15):
            r = 30.0 + i * 0.8  # steadily growing drop
            theta = 60.0 + i * 0.5
            img, _, _, _ = make_synthetic_sessile_frame(radius=r, theta_deg=theta)
            frames.append(Frame(image=img, index=i, timestamp_s=timestamps[i]))

        meta = SequenceMetadata(
            source_type="memory",
            source_id="test_tracking_seq",
            sha256="1" * 64,
            width=200,
            height=160,
            fps=10.0,
            timestamps_s=timestamps,
            frame_count=15,
        )

        # Run with tracking enabled
        result_tracked = analyze_dynamic_sessile(
            frames,
            meta,
            px_per_mm=20.0,
            needle_diameter_mm=None,
            use_temporal_tracking=True,
        )

        assert result_tracked is not None
        assert len(result_tracked.frames) == 15
        assert result_tracked.summary["n_valid_frames"] >= 12
        assert result_tracked.summary["valid_fraction"] > 0.8
        # Sequence is advancing (growing radius)
        assert "advancing_duration_s" in result_tracked.summary
