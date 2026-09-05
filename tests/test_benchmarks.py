"""Tests for benchmark datasets, manifest integrity, author attribution, and calibration standards."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from menipy.common.auto_calibrator import AutoCalibrator
from menipy.common.sessile_detection import detect_sessile_substrate_robust
from menipy.models.geometry import SubstrateProfile
from tools.fetch_benchmarks import compute_sha256, run_benchmark_fetcher

MANIFEST_PATH = Path("data/MANIFEST.json")
ATTRIBUTION_PATH = Path("data/ATTRIBUTION.md")
SAMPLES_DIR = Path("data/samples")


class TestBenchmarkManifest:
    """Test data/MANIFEST.json and data/ATTRIBUTION.md consistency and integrity."""

    def test_manifest_structure_and_schema(self):
        assert MANIFEST_PATH.exists(), "data/MANIFEST.json must exist"
        with open(MANIFEST_PATH, encoding="utf-8") as f:
            manifest = json.load(f)

        assert manifest.get("schema_version") == "1.0"
        assert "datasets" in manifest
        assert len(manifest["datasets"]) >= 8

        for entry in manifest["datasets"]:
            assert "id" in entry
            assert "filename" in entry
            assert "category" in entry
            assert "license" in entry
            assert "destination_dir" in entry
            assert "is_canonical_fixture" in entry

    def test_canonical_fixtures_exist_on_disk(self):
        with open(MANIFEST_PATH, encoding="utf-8") as f:
            manifest = json.load(f)

        canonical_entries = [
            d for d in manifest["datasets"] if d.get("is_canonical_fixture") is True
        ]
        assert len(canonical_entries) >= 5, "Must have at least 5 canonical fixtures"

        for d in canonical_entries:
            dest = Path(d["destination_dir"]) / d["filename"]
            assert dest.exists(), f"Canonical fixture missing: {dest}"

    def test_canonical_fixtures_sha256_checksums(self):
        with open(MANIFEST_PATH, encoding="utf-8") as f:
            manifest = json.load(f)

        for d in manifest["datasets"]:
            if d.get("is_canonical_fixture") and d.get("sha256"):
                dest = Path(d["destination_dir"]) / d["filename"]
                actual_sha = compute_sha256(dest)
                assert (
                    actual_sha == d["sha256"]
                ), f"SHA-256 mismatch for {dest.name}: {actual_sha} != {d['sha256']}"

    def test_all_samples_are_registered_in_manifest(self):
        with open(MANIFEST_PATH, encoding="utf-8") as f:
            manifest = json.load(f)

        registered_filenames = {
            d["filename"] for d in manifest["datasets"] if d["destination_dir"] == "data/samples"
        }

        actual_samples = {
            p.name for p in SAMPLES_DIR.glob("*") if p.is_file() and p.name != ".gitkeep"
        }

        unregistered = actual_samples - registered_filenames
        assert not unregistered, f"Found unregistered files in data/samples/: {unregistered}"

    def test_attribution_documentation_covers_all_categories(self):
        assert ATTRIBUTION_PATH.exists(), "data/ATTRIBUTION.md must exist"
        content = ATTRIBUTION_PATH.read_text(encoding="utf-8")

        assert "OpenDrop" in content
        assert "Berry" in content
        assert "Drop-O-Matic" in content
        assert "Menipy Legacy Lab Captures" in content
        assert "Carroll (1976)" in content
        assert "Extrand & Moon (2008)" in content


class TestBenchmarkDownloader:
    """Test tools/fetch_benchmarks.py downloader and verifier utility."""

    def test_fetcher_verify_only_succeeds(self):
        res = run_benchmark_fetcher(category="all", verify_only=True)
        assert res == 0, "All local benchmark files must verify against MANIFEST.json"


class TestBenchmarkCalibration:
    """Test algorithmic performance against canonical benchmark images."""

    def test_pendant_water_reference_auto_calibration(self):
        img_path = SAMPLES_DIR / "pendant_water_reference.png"
        assert img_path.exists()
        image = cv2.imread(str(img_path))
        assert image is not None

        calib = AutoCalibrator(image, "pendant").detect_all()
        assert calib.needle_rect is not None
        x, y, w, h = calib.needle_rect
        assert y < 10, "Needle shaft must touch top border"
        assert 100 <= w <= 115, f"Needle width should be ~107 px, got {w}"

        # True needle is 1.65 mm -> pixel scale should be ~64.8 px/mm
        px_per_mm = w / 1.65
        assert 60.0 <= px_per_mm <= 70.0, f"Expected ~65 px/mm, got {px_per_mm:.1f}"

        # Apex should be centered horizontally under needle
        needle_center_x = x + w / 2.0
        assert calib.apex_point is not None
        assert abs(calib.apex_point[0] - needle_center_x) < 15.0

    def test_sessile_clean_reference_detection(self):
        img_path = SAMPLES_DIR / "sessile_clean_reference.png"
        assert img_path.exists()
        image = cv2.imread(str(img_path))
        assert image is not None

        line, conf, diag, prof = detect_sessile_substrate_robust(image)
        assert line is not None
        assert conf >= 0.75, f"Expected confident detection, got {conf:.2f}"
        assert diag["status"] == "confident"
        assert diag["warning"] is False

        # Substrate line should be nearly flat across the 1280px width
        y_left, y_right = line[0][1], line[1][1]
        assert abs(y_left - y_right) < 25, "Substrate line should be nearly horizontal"

    def test_sessile_needle_reference_triggers_doubtful_warning(self):
        img_path = SAMPLES_DIR / "sessile_needle_reference.png"
        assert img_path.exists()
        image = cv2.imread(str(img_path))
        assert image is not None

        line, conf, diag, prof = detect_sessile_substrate_robust(image)
        # In this challenging image with dispensing needle obstructing the profile,
        # the robust detector correctly flags ambiguous edge transition
        assert diag["warning"] is True
        assert diag["status"] in ("doubtful", "failed")

    def test_curved_substrate_analytical_extrand_carroll(self):
        # Cylindrical substrate arc with radius R=20.0 mm centered at (100, 100)
        # Droplet contact point at x=110 (10 mm to the right of center)
        cx, cy, r = 100.0, 100.0, 20.0
        # In image coords, convex curved surface: y(x) = cy - sqrt(r^2 - dx^2)
        p1 = (cx - 15.0, cy - float(np.sqrt(r**2 - 15.0**2)))
        p2 = (cx + 15.0, cy - float(np.sqrt(r**2 - 15.0**2)))
        p3 = (cx, cy - r)

        prof = SubstrateProfile.from_arc(p1, p2, p3)
        assert prof.type == "circle_arc"
        assert prof.parameters["radius"] == pytest.approx(r, rel=1e-3)
        assert prof.parameters["center_x"] == pytest.approx(cx, rel=1e-3)
        assert prof.parameters["center_y"] == pytest.approx(cy, rel=1e-3)

        # Local slope at contact point x=110 (dx = 10)
        # dy/dx = dx / sqrt(r^2 - dx^2) = 10 / sqrt(400 - 100) = 10 / sqrt(300) = 1 / sqrt(3) -> 30 deg
        alpha_sub = prof.eval_tangent_angle_deg(110.0)
        assert alpha_sub == pytest.approx(30.0, abs=0.5)

        # Intrinsic contact angle correction: theta_intrinsic = theta_apparent - alpha_sub
        theta_apparent = 95.0
        theta_intrinsic = theta_apparent - alpha_sub
        assert theta_intrinsic == pytest.approx(65.0, abs=0.5)
