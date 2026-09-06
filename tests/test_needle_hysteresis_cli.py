"""CLI integration tests for needle-in-drop contact angle hysteresis."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from menipy.cli import main


def _create_synthetic_sequence_on_disk(folder: Path, n_frames: int = 15) -> None:
    """Write synthetic needle-in-sessile-drop PNG frames to a folder."""
    folder.mkdir(parents=True, exist_ok=True)
    for i in range(n_frames):
        img = np.full((300, 400), 255, dtype=np.uint8)
        # Substrate
        img[220:, :] = 120
        # Needle
        img[0:110, 196:204] = 30
        # Inflate first 8 frames, deflate next 7
        if i < 8:
            r = int(55 + i * 2.5)
        else:
            r = int(55 + 8 * 2.5 - (i - 7) * 2.0)
        cv2.circle(img, (200, 220), r, 40, thickness=-1)
        img[220:, :] = 120
        img[0:110, 196:204] = 30

        cv2.imwrite(str(folder / f"frame_{i:04d}.png"), img)


def test_cli_needle_hysteresis_sequence(tmp_path: Path):
    """Test CLI execution of needle_hysteresis pipeline with full outputs and plots."""
    seq_dir = tmp_path / "seq"
    out_dir = tmp_path / "out_hyst"
    _create_synthetic_sequence_on_disk(seq_dir, n_frames=12)

    args = [
        "--pipeline",
        "needle_hysteresis",
        "--sequence-dir",
        str(seq_dir),
        "--fps",
        "10",
        "--px-per-mm",
        "25.0",
        "--needle-diameter",
        "0.4",
        "--needle-fit-method",
        "auto",
        "--plot",
        "--out",
        str(out_dir),
    ]

    code = main(args)
    assert code == 0

    # Verify JSON result
    json_path = out_dir / "results.json"
    assert json_path.is_file()
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    assert data["pipeline"] == "needle_hysteresis"
    assert "summary" in data
    assert "frames" in data
    assert len(data["frames"]) == 12

    # Verify CSV summary
    csv_path = out_dir / "results.csv"
    assert csv_path.is_file()
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert len(rows) == 1
    assert rows[0]["pipeline"] == "needle_hysteresis"

    # Verify frames CSV
    frames_csv_path = out_dir / "results_frames.csv"
    assert frames_csv_path.is_file()
    with open(frames_csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        frame_rows = list(reader)
    assert len(frame_rows) == 12

    # Verify PNG plots
    assert (out_dir / "hysteresis_timeline.png").is_file()
    assert (out_dir / "hysteresis_loop.png").is_file()


def test_cli_needle_hysteresis_missing_fps(tmp_path: Path, capsys):
    """CLI should error when sequence-dir is provided without positive fps."""
    seq_dir = tmp_path / "seq"
    seq_dir.mkdir(parents=True, exist_ok=True)
    out_dir = tmp_path / "out_err"

    with pytest.raises(SystemExit):
        main([
            "--pipeline",
            "needle_hysteresis",
            "--sequence-dir",
            str(seq_dir),
            "--out",
            str(out_dir),
        ])
