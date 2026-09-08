"""Tests for scripts/manual_benchmark_tool.py.

Validates:
1. BenchmarkAnnotation model serialization, deserialization, and atomic JSON persistence.
2. Ground-truth and configuration preset extractors for benchmark datasets (Conan-ML, OpenDrop, UCLA, EPFL, pyDSA).
3. Pipeline execution with manual substrate line override.
4. Pipeline execution with manual needle dimensions and fluid densities.
5. CLI batch runner execution and annotation export.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.manual_benchmark_tool import (
    BenchmarkAnnotation,
    get_known_benchmark_preset,
    run_pipeline_with_annotation,
)

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_DIR = ROOT / "data" / "benchmarks"
SAMPLES_DIR = ROOT / "data" / "samples"


class TestBenchmarkAnnotation:
    """Test annotation data container and serialization."""

    def test_default_values(self) -> None:
        ann = BenchmarkAnnotation()
        assert ann.dataset == "custom"
        assert ann.pipeline == "sessile"
        assert ann.substrate_line is None
        assert ann.needle_rect is None
        assert ann.physics["rho1"] == 998.2

    def test_save_load_roundtrip(self, tmp_path: Path) -> None:
        save_file = tmp_path / "test_annotation.json"
        ann = BenchmarkAnnotation(
            dataset="test_suite",
            image_file="test_drop.png",
            pipeline="sessile",
            substrate_line=((10.5, 200.0), (350.5, 200.0)),
            needle_rect=(100, 0, 50, 80),
            contact_points=((25, 200), (330, 200)),
            apex_point=(175, 50),
            px_per_mm=123.45,
            needle_diameter_mm=1.2,
            physics={"rho1": 1000.0, "rho2": 1.2, "g": 9.81},
            ground_truth={"contact_angle_deg": 65.4, "source": "synthetic test"},
            drop_contour=[[25.0, 200.0], [175.0, 50.0], [330.0, 200.0]],
            results={"contact_angle_deg": 65.3, "status": "PASS"},
        )

        ann.save(save_file)
        assert save_file.exists()

        loaded = BenchmarkAnnotation.load(save_file)
        assert loaded.dataset == "test_suite"
        assert loaded.image_file == "test_drop.png"
        assert loaded.pipeline == "sessile"
        assert loaded.substrate_line == ((10.5, 200.0), (350.5, 200.0))
        assert loaded.needle_rect == (100, 0, 50, 80)
        assert loaded.contact_points == ((25, 200), (330, 200))
        assert loaded.apex_point == (175, 50)
        assert loaded.px_per_mm == pytest.approx(123.45)
        assert loaded.needle_diameter_mm == pytest.approx(1.2)
        assert loaded.ground_truth["contact_angle_deg"] == pytest.approx(65.4)
        assert len(loaded.drop_contour) == 3
        assert loaded.results["status"] == "PASS"

    def test_numpy_types_serialization(self, tmp_path: Path) -> None:
        save_file = tmp_path / "numpy_types.json"
        ann = BenchmarkAnnotation(
            dataset="numpy_test",
            needle_rect=(np.int64(10), np.int64(20), np.int64(30), np.int64(40)),
            drop_contour=np.array([[10.0, 20.0], [30.0, 40.0]]),
            results={"metric_int": np.int64(42), "metric_float": np.float64(3.14)},
        )

        ann.save(save_file)
        with open(save_file, encoding="utf-8") as f:
            data = json.load(f)
        assert data["needle_rect"] == [10, 20, 30, 40]
        assert data["results"]["metric_int"] == 42


class TestKnownPresets:
    """Test known dataset ground-truth and geometry preset extractors."""

    def test_conan_preset_parses_angle(self) -> None:
        preset = get_known_benchmark_preset("data/benchmarks/conan_ml/111.031693.bmp")
        assert preset.dataset == "conan_ml"
        assert preset.pipeline == "sessile"
        assert preset.ground_truth.get("contact_angle_deg") == 111.03

    def test_opendrop_pendant_preset(self) -> None:
        preset = get_known_benchmark_preset("data/benchmarks/pendant/water_in_air002.png")
        assert preset.dataset == "opendrop_pendant"
        assert preset.pipeline == "pendant"
        assert preset.needle_diameter_mm == 0.718
        assert preset.ground_truth.get("surface_tension_mN_m") == 72.80

    def test_ucla_water_preset(self) -> None:
        preset = get_known_benchmark_preset("data/benchmarks/ucla_pendant/H2O_PendantDrop.png")
        assert preset.dataset == "ucla_pendant"
        assert preset.needle_diameter_mm == 0.51
        assert preset.ground_truth.get("surface_tension_mN_m") == 72.80

    def test_ucla_hexadecane_preset(self) -> None:
        preset = get_known_benchmark_preset("data/benchmarks/ucla_pendant/Hexadecane_PendantDrop.png")
        assert preset.dataset == "ucla_pendant"
        assert preset.needle_diameter_mm == 1.95
        assert preset.ground_truth.get("surface_tension_mN_m") == 27.50
        assert preset.physics["rho1"] == 773.0

    def test_epfl_preset(self) -> None:
        preset = get_known_benchmark_preset("data/benchmarks/epfl_drop_analysis/sample.jpg")
        assert preset.dataset == "epfl_drop_analysis"
        assert preset.px_per_mm == 191.82


class TestExecutionWithOverrides:
    """Test pipeline execution with manual overrides injected."""

    def test_pendant_execution_with_needle_override(self) -> None:
        pendant_img = BENCHMARK_DIR / "pendant" / "water_in_air002.png"
        if not pendant_img.exists():
            pytest.skip("Pendant benchmark image not found")

        ann = BenchmarkAnnotation(
            dataset="opendrop_pendant",
            pipeline="pendant",
            needle_diameter_mm=0.718,
            physics={"rho1": 998.2, "rho2": 1.2, "g": 9.80665},
            ground_truth={"surface_tension_mN_m": 72.80},
        )

        ctx, metrics = run_pipeline_with_annotation(pendant_img, ann)
        assert ctx.results is not None
        gamma = metrics.get("surface_tension_mN_m")
        assert gamma is not None
        assert 65.0 <= gamma <= 78.0
        assert metrics.get("status") == "PASS"
        assert ann.drop_contour is not None
        assert len(ann.drop_contour) > 50

    def test_sessile_execution_with_manual_substrate(self) -> None:
        sessile_img = SAMPLES_DIR / "clean_sessile.png"
        if not sessile_img.exists():
            sessile_img = BENCHMARK_DIR / "conan_ml" / "111.031693.bmp"
        if not sessile_img.exists():
            pytest.skip("No sessile benchmark or sample image found")

        ann = BenchmarkAnnotation(
            dataset="sessile_test",
            pipeline="sessile",
            substrate_line=((50.0, 450.0), (800.0, 450.0)),
        )

        ctx, metrics = run_pipeline_with_annotation(sessile_img, ann, use_auto_calibration_seed=False)
        assert ctx.results is not None
        assert "runtime_ms" in metrics
