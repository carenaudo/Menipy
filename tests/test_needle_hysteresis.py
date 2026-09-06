"""Integration tests for needle-in-sessile-drop hysteresis analysis and state machine."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from menipy.common.needle_hysteresis import (
    NeedleHysteresisFrame,
    classify_contact_line_states,
    summarize_hysteresis,
)
from menipy.models.frame import Frame
from menipy.pipelines.discover import PIPELINE_MAP
from menipy.pipelines.needle_hysteresis import NeedleHysteresisPipeline
from menipy.pipelines.runner import PipelineRunner


def test_pipeline_discovery():
    """NeedleHysteresisPipeline must be registered in PIPELINE_MAP."""
    assert "needle_hysteresis" in PIPELINE_MAP
    assert PIPELINE_MAP["needle_hysteresis"] is NeedleHysteresisPipeline


def test_state_machine_advancing_pinned_receding():
    """Test state machine on a known sequence with advancing, pinned, and receding phases."""
    frames = []
    # 0-5: advancing (diameter increases from 2.0 to 3.0 mm)
    for i in range(6):
        frames.append(
            NeedleHysteresisFrame(
                frame_index=i,
                timestamp_s=i * 0.1,
                accepted=True,
                state="pinned",
                base_diameter_mm=2.0 + i * 0.2,  # +2.0 mm/s
                theta_left_deg=80.0,
                theta_right_deg=80.0,
                theta_mean_deg=80.0,
            )
        )
    # 6-11: pinned (diameter constant at 3.0 mm)
    for i in range(6, 12):
        frames.append(
            NeedleHysteresisFrame(
                frame_index=i,
                timestamp_s=i * 0.1,
                accepted=True,
                state="pinned",
                base_diameter_mm=3.0,
                theta_left_deg=70.0 - (i - 6) * 3.0,  # contact angle dropping while pinned!
                theta_right_deg=70.0 - (i - 6) * 3.0,
                theta_mean_deg=70.0 - (i - 6) * 3.0,
            )
        )
    # 12-17: receding (diameter decreases from 3.0 to 2.0 mm)
    for i in range(12, 18):
        frames.append(
            NeedleHysteresisFrame(
                frame_index=i,
                timestamp_s=i * 0.1,
                accepted=True,
                state="pinned",
                base_diameter_mm=3.0 - (i - 11) * 0.2,  # -2.0 mm/s
                theta_left_deg=50.0,
                theta_right_deg=50.0,
                theta_mean_deg=50.0,
            )
        )

    deadband = classify_contact_line_states(frames)
    assert deadband > 0.0

    # Frames 1-4 should be advancing
    advancing_states = [f.state for f in frames[1:5]]
    assert all(s == "advancing" for s in advancing_states)

    # Frames 7-10 MUST be pinned (pseudo-movement suppression!)
    pinned_states = [f.state for f in frames[7:11]]
    assert all(s == "pinned" for s in pinned_states)

    # Frames 13-16 should be receding
    receding_states = [f.state for f in frames[13:17]]
    assert all(s == "receding" for s in receding_states)


def test_summarize_hysteresis():
    """Summary function calculates plateaus and hysteresis cleanly."""
    frames = []
    # 10 advancing frames at 82°
    for i in range(10):
        frames.append(
            NeedleHysteresisFrame(
                frame_index=i,
                timestamp_s=i * 0.1,
                accepted=True,
                state="advancing",
                base_diameter_mm=2.0 + i * 0.1,
                theta_left_deg=82.0,
                theta_right_deg=82.0,
                theta_mean_deg=82.0,
            )
        )
    # 10 receding frames at 54°
    for i in range(10, 20):
        frames.append(
            NeedleHysteresisFrame(
                frame_index=i,
                timestamp_s=i * 0.1,
                accepted=True,
                state="receding",
                base_diameter_mm=3.0 - (i - 9) * 0.1,
                theta_left_deg=54.0,
                theta_right_deg=54.0,
                theta_mean_deg=54.0,
            )
        )

    summary = summarize_hysteresis(frames, fps=10.0, deadband_mm_s=0.02)

    assert pytest.approx(summary["theta_advancing_deg"], abs=0.5) == 82.0
    assert pytest.approx(summary["theta_receding_deg"], abs=0.5) == 54.0
    assert pytest.approx(summary["contact_angle_hysteresis_deg"], abs=0.5) == 28.0
    assert "theta_advancing" in summary
    assert "ci95_deg" in summary["theta_advancing"]


def test_pipeline_runner_with_synthetic_sequence():
    """PipelineRunner can execute needle_hysteresis on in-memory synthetic images."""
    runner = PipelineRunner("needle_hysteresis")

    # Generate 12 synthetic drop frames:
    # A dark circle segment on white background with needle shadow
    frames = []
    for step in range(12):
        img = np.full((300, 400), 255, dtype=np.uint8)
        # Substrate line at y = 220
        img[220:, :] = 120
        # Needle shaft at x in [195, 205], y in [0, 100]
        img[0:100, 196:204] = 30
        # Droplet circle center at (200, 220), radius growing from 60 to 80
        r = int(60 + step * 1.8)
        cv2.circle(img, (200, 220), r, 40, thickness=-1)
        # Re-blank below baseline
        img[220:, :] = 120
        # Re-draw needle
        img[0:100, 196:204] = 30
        frames.append(Frame(image=img))

    ctx = runner.run(
        frames=frames,
        sequence_fps=10.0,
        px_per_mm=20.0,
        needle_diameter_mm=0.4,
    )

    assert ctx.needle_hysteresis_result is not None
    assert ctx.results["pipeline"] == "needle_hysteresis"
    assert "n_frames" in ctx.results
    assert ctx.results["n_frames"] == 12
