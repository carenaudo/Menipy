"""Unit and integration tests for temporal folder analysis engine.

Verifies:
1. Natural alphanumeric ordering of image sequences.
2. Invariant locking (substrate baseline for sessile, cannula for pendant).
3. Warm-started localized active contour tracking across frame images.
4. Quality gate fallback on sudden anomalous jump.
5. Tabular CSV summary export.
6. Cancellation responsiveness.
"""

from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np
import pytest

from menipy.common.cancellation import AnalysisCancelled
from menipy.common.folder_analysis import (
    FolderAnalysisResult,
    analyze_image_folder,
    discover_image_files,
)


def _make_sessile_frame(
    size: tuple[int, int] = (160, 200),
    center_x: float = 100.0,
    substrate_y: float = 120.0,
    radius: float = 35.0,
    theta_deg: float = 65.0,
) -> np.ndarray:
    """Generate a clean synthetic sessile droplet frame."""
    h_img, w_img = size
    img = np.full((h_img, w_img, 3), 230, dtype=np.uint8)

    theta_rad = np.radians(theta_deg)
    center_y = substrate_y + radius * np.cos(theta_rad)

    # Draw droplet disk clipped at substrate
    mask = np.zeros((h_img, w_img), dtype=np.uint8)
    cv2.circle(mask, (int(round(center_x)), int(round(center_y))), int(round(radius)), 255, -1)
    mask[int(round(substrate_y)) :, :] = 0
    img[mask == 255] = (30, 30, 30)

    # Draw solid substrate baseline
    cv2.line(img, (0, int(round(substrate_y))), (w_img, int(round(substrate_y))), (80, 80, 80), 2)
    return img


def _make_pendant_frame(
    size: tuple[int, int] = (200, 160),
    needle_center_x: float = 80.0,
    needle_bottom_y: float = 40.0,
    needle_width: float = 20.0,
    drop_radius: float = 30.0,
) -> np.ndarray:
    """Generate a synthetic pendant drop hanging from a fixed needle."""
    h_img, w_img = size
    img = np.full((h_img, w_img, 3), 230, dtype=np.uint8)

    # Needle rectangle
    nx = int(needle_center_x - needle_width / 2.0)
    ny = 0
    nw = int(needle_width)
    nh = int(needle_bottom_y)
    cv2.rectangle(img, (nx, ny), (nx + nw, ny + nh), (50, 50, 50), -1)

    # Drop circle center below needle
    center_y = needle_bottom_y + drop_radius * 0.8
    cv2.circle(img, (int(round(needle_center_x)), int(round(center_y))), int(round(drop_radius)), (40, 40, 40), -1)
    return img


def test_discover_image_files_natural_order(tmp_path: Path):
    """Ensure natural sorting preserves camera/human sequential order (1, 2, 10, not 1, 10, 2)."""
    names = ["frame_10.png", "frame_1.png", "frame_2.png", "frame_20.jpg", "notes.txt", "frame_3.bmp"]
    for name in names:
        (tmp_path / name).write_bytes(b"dummy")

    files = discover_image_files(tmp_path)
    file_names = [f.name for f in files]

    # notes.txt should be filtered out
    assert "notes.txt" not in file_names
    # Natural order: 1, 2, 3, 10, 20
    expected = ["frame_1.png", "frame_2.png", "frame_3.bmp", "frame_10.png", "frame_20.jpg"]
    assert file_names == expected


def test_discover_image_files_list_input(tmp_path: Path):
    """Ensure discover_image_files handles explicit list of files."""
    f1 = tmp_path / "img_2.png"
    f2 = tmp_path / "img_10.png"
    f3 = tmp_path / "img_1.png"
    for f in (f1, f2, f3):
        f.write_bytes(b"dummy")

    res = discover_image_files([str(f2), str(f1), str(f3)])
    assert [p.name for p in res] == ["img_1.png", "img_2.png", "img_10.png"]


def test_analyze_image_folder_empty(tmp_path: Path):
    """Empty folder returns a zeroed FolderAnalysisResult without crashing."""
    res = analyze_image_folder(tmp_path)
    assert isinstance(res, FolderAnalysisResult)
    assert res.n_frames == 0
    assert res.n_valid_frames == 0
    assert res.valid_fraction == 0.0


def test_analyze_image_folder_sessile(tmp_path: Path, monkeypatch):
    """Analyze a sequential folder of sessile frames with invariant locking."""
    # Create 5 frames of a slowly advancing droplet
    seq_dir = tmp_path / "sessile_seq"
    seq_dir.mkdir()

    for idx in range(5):
        cx = 100.0 + idx * 0.8
        img = _make_sessile_frame(center_x=cx, radius=35.0 + idx * 0.2)
        cv2.imwrite(str(seq_dir / f"frame_{idx:03d}.png"), img)

    # Run folder analysis
    res = analyze_image_folder(
        seq_dir,
        pipeline="sessile",
        px_per_mm=10.0,
    )

    assert res.pipeline == "sessile"
    assert res.n_frames == 5
    assert res.n_valid_frames >= 4
    assert res.valid_fraction >= 0.8
    assert "theta_mean_deg" in res.summary

    # Frame 0 is cold-start; subsequent valid frames should be tracked
    frames = res.frames
    assert not frames[0].tracked  # Frame 0 establishes invariant baseline
    tracked_count = sum(f.tracked for f in frames[1:] if f.accepted)
    assert tracked_count >= 3

    # Locked substrate should remain invariant across frames
    locked_sub = frames[0].substrate_line
    assert locked_sub is not None
    for f in frames:
        if f.accepted:
            assert f.substrate_line == locked_sub


def test_analyze_image_folder_pendant(tmp_path: Path):
    """Analyze a sequential folder of pendant frames with needle and scale locking."""
    seq_dir = tmp_path / "pendant_seq"
    seq_dir.mkdir()

    for idx in range(4):
        rad = 30.0 + idx * 0.5
        img = _make_pendant_frame(drop_radius=rad)
        cv2.imwrite(str(seq_dir / f"pendant_{idx}.png"), img)

    res = analyze_image_folder(
        seq_dir,
        pipeline="pendant",
        needle_diameter_mm=0.72,
        needle_rect=(70, 0, 20, 40),
    )

    assert res.pipeline == "pendant"
    assert res.n_frames == 4
    assert res.n_valid_frames >= 3
    assert res.valid_fraction >= 0.75
    assert "gamma_mean_mN_m" in res.summary

    # Check needle invariant locking
    first_needle = res.frames[0].needle_rect
    assert first_needle is not None
    for f in res.frames:
        if f.accepted:
            assert f.needle_rect == first_needle


def test_analyze_image_folder_fallback_on_jump(tmp_path: Path):
    """Ensure sudden anomaly trips quality gates, resets tracker, and recovers via cold start."""
    seq_dir = tmp_path / "anomaly_seq"
    seq_dir.mkdir()

    # Frame 0: normal
    cv2.imwrite(str(seq_dir / "frame_0.png"), _make_sessile_frame(radius=35.0))
    # Frame 1: normal small change
    cv2.imwrite(str(seq_dir / "frame_1.png"), _make_sessile_frame(radius=35.5))
    # Frame 2: catastrophic jump (+70% radius jump, violates <= 25% area gate)
    cv2.imwrite(str(seq_dir / "frame_2.png"), _make_sessile_frame(radius=60.0))
    # Frame 3: stable at new size
    cv2.imwrite(str(seq_dir / "frame_3.png"), _make_sessile_frame(radius=60.2))

    res = analyze_image_folder(seq_dir, pipeline="sessile", px_per_mm=10.0)
    assert res.n_frames == 4

    # Frame 2 should have failed tracking gate and fallen back to cold-start detection
    # and not crashed the pipeline
    assert res.frames[2].tracked is False or not res.frames[2].accepted


def test_folder_analysis_export_csv(tmp_path: Path):
    """Ensure export_csv generates valid, complete tabular outputs."""
    seq_dir = tmp_path / "export_seq"
    seq_dir.mkdir()

    for idx in range(3):
        img = _make_sessile_frame(center_x=100.0 + idx)
        cv2.imwrite(str(seq_dir / f"frame_{idx}.png"), img)

    res = analyze_image_folder(seq_dir, pipeline="sessile", px_per_mm=10.0)
    csv_target = tmp_path / "out" / "folder_results.csv"
    out_path = res.export_csv(csv_target)

    assert out_path.is_file()
    with open(out_path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    assert len(rows) == 3
    assert rows[0]["image_name"] == "frame_0.png"
    assert "accepted" in rows[0]
    assert "tracked" in rows[0]
    assert "theta_mean_deg" in rows[0] or "theta_left_deg" in rows[0]


def test_analyze_image_folder_cancellation(tmp_path: Path):
    """Ensure cancellation probe stops folder analysis early."""
    seq_dir = tmp_path / "cancel_seq"
    seq_dir.mkdir()

    for idx in range(5):
        img = _make_sessile_frame()
        cv2.imwrite(str(seq_dir / f"frame_{idx}.png"), img)

    call_count = 0

    def cancel_probe():
        nonlocal call_count
        call_count += 1
        if call_count >= 2:
            raise AnalysisCancelled("Analysis cancelled by user")

    with pytest.raises(AnalysisCancelled):
        analyze_image_folder(seq_dir, check_cancelled=cancel_probe)
