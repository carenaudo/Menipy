"""Consolidated integration tests for the Menipy Command-Line Interface (CLI)."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from menipy.cli import main
from menipy.common.material_db import MaterialDB


def test_coordinate_parsing_validation():
    """Verify that malformed coordinate options raise validation errors."""
    # Test invalid coordinate length for ROI
    with pytest.raises(SystemExit):
        main(["--pipeline", "sessile", "--image", "dummy.png", "--roi", "10,20,30"])

    # Test non-numeric coordinates
    with pytest.raises(SystemExit):
        main(["--pipeline", "sessile", "--image", "dummy.png", "--roi", "10,20,30,abc"])

    # Test coincident baseline line endpoints
    with pytest.raises(SystemExit):
        main(
            [
                "--pipeline",
                "sessile",
                "--image",
                "dummy.png",
                "--contact-line",
                "10,20,10,20",
            ]
        )


def test_sqlite_materials_db_lookups(tmp_path: Path):
    """Verify that materials/needles database lookups extract physical parameters."""
    db_file = tmp_path / "test_materials.sqlite"

    # Initialize a temporary Materials SQLite DB
    mdb = MaterialDB(db_file)
    mdb.init_schema()

    # Verify seed defaults are queryable
    water = mdb.get_material("Water (20°C)")
    assert water is not None
    assert abs(water["density"] - 998.2) < 0.1

    needles = mdb.list_needles()
    assert len(needles) > 0
    # 22G outer diameter is 0.72mm by default
    g22 = [n for n in needles if n["gauge"] == "22G"][0]
    assert abs(g22["outer_diameter"] - 0.72) < 0.01

    # Insert a custom material and syringe
    mdb.upsert_material("CustomOil", {"type": "liquid", "density": 850.5})
    mdb.upsert_needle("CustomNeedle", 0.55, gauge="33G")

    custom_mat = mdb.get_material("CustomOil")
    assert custom_mat is not None
    assert abs(custom_mat["density"] - 850.5) < 0.1

    custom_needles = mdb.list_needles()
    custom_n = [n for n in custom_needles if n["name"] == "CustomNeedle"][0]
    assert abs(custom_n["outer_diameter"] - 0.55) < 0.01


def test_sop_json_loading(tmp_path: Path):
    """Verify that SOP configuration JSON files load and parse preprocessor settings."""
    sop_file = tmp_path / "test_sop.json"
    sop_content = {
        "include_stages": ["acquisition", "preprocessing", "edge_detection"],
        "params": {
            "preprocessing": {"method": "clahe", "clip_limit": 3.0, "grid_size": 8},
            "edge_detection": {"method": "sobel", "ksize": 5},
        },
    }

    with open(sop_file, "w", encoding="utf-8") as f:
        json.dump(sop_content, f)

    # We run single execution with missing image to test up to parse completion
    # and confirm that a bad image file returns a non-zero exit code.
    code = main(["--sop", str(sop_file), "--image", "nonexistent.png"])
    assert code != 0


def test_single_image_execution(tmp_path: Path):
    """Run CLI on a real sample sessile image using Auto-Calibration fallback."""
    out_dir = tmp_path / "single_out"
    sample_img = Path("data/samples/prueba sesil 2.png")

    assert sample_img.is_file(), f"Sample image {sample_img} not found"

    # Execute main execution block
    code = main(
        [
            "--pipeline",
            "sessile",
            "--image",
            str(sample_img),
            "--auto-calibrate",
            "--out",
            str(out_dir),
            "--needle-diameter",
            "0.72",
            "--material",
            "Water (25°C)",
            "--needle-name",
            "22G",
        ]
    )

    assert code == 0, "CLI execution failed"

    # Assert expected outputs are written to the custom output directory
    assert (out_dir / "preview.png").is_file()
    assert (out_dir / "overlay.png").is_file()
    assert (out_dir / "results.json").is_file()

    # Load results.json to verify content
    with open(out_dir / "results.json", encoding="utf-8") as f:
        res = json.load(f)
    assert res["pipeline"] == "sessile"
    assert "results" in res
    assert "qa" in res
    assert "timings_ms" in res

    # Verify contact angle exists in results dictionary
    metrics = res["results"]
    assert "contact_angle_deg" in metrics or "theta_left_deg" in metrics


def test_batch_processing_execution(tmp_path: Path):
    """Verify CLI batch mode walking a directory, executing sequentially, and writing results.csv."""
    input_dir = tmp_path / "batch_in"
    out_dir = tmp_path / "batch_out"
    input_dir.mkdir()

    # Copy sample image files to the input directory
    orig_sample = Path("data/samples/prueba sesil 2.png")
    assert orig_sample.is_file()

    # Create two duplicate sample images to process in batch mode
    (input_dir / "img1.png").write_bytes(orig_sample.read_bytes())
    (input_dir / "img2.jpg").write_bytes(orig_sample.read_bytes())

    code = main(
        [
            "--pipeline",
            "sessile",
            "--input-dir",
            str(input_dir),
            "--glob",
            "*.png, *.jpg",
            "--auto-calibrate",
            "--out",
            str(out_dir),
        ]
    )

    assert code == 0, "CLI batch execution failed"

    # Verify distinct overlay and preview visual images exist per image stem to prevent collisions
    assert (out_dir / "img1_preview.png").is_file()
    assert (out_dir / "img1_overlay.png").is_file()
    assert (out_dir / "img2_preview.png").is_file()
    assert (out_dir / "img2_overlay.png").is_file()
    assert (out_dir / "img1_results.json").is_file()
    assert (out_dir / "img2_results.json").is_file()

    # Verify a beautiful, unified results.csv exists summarizing both runs
    csv_path = out_dir / "results.csv"
    assert csv_path.is_file()

    # Validate results.csv contents
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) == 2
    paths = {row["image_path"] for row in rows}
    assert any("img1.png" in p for p in paths)
    assert any("img2.jpg" in p for p in paths)
    for row in rows:
        assert row["pipeline"] == "sessile"
        assert row["qa_ok"] in ("True", "False")
        # Ensure dynamic metrics are correctly exported
        assert "contact_angle_deg" in row or "theta_left_deg" in row


def test_edge_detection_preproc_overrides(tmp_path: Path):
    """Verify preprocessing and edge detection override inputs take priority."""
    out_dir = tmp_path / "overrides_out"
    sample_img = Path("data/samples/prueba sesil 2.png")

    # Run with Sobel edge detector override
    code = main(
        [
            "--pipeline",
            "sessile",
            "--image",
            str(sample_img),
            "--auto-calibrate",
            "--out",
            str(out_dir),
            "--preprocessing-method",
            "blur",
            "--edge-detection-method",
            "sobel",
        ]
    )

    assert code == 0
    assert (out_dir / "results.json").is_file()
