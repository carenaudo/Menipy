"""Fast CI gates for the opt-in Phase-B geometry prototypes."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from adsa_geometry_conformance import (  # noqa: E402
    evaluate_geometry_case,
    generate_geometry_case,
)

from menipy.common.geometry_prototypes import fit_bilateral_needle_edges
from menipy.pipelines.pendant.strict_young_laplace import (
    model_mm_to_pendant_px,
    pendant_contour_to_model_mm,
)


def test_phase_b_manifest_has_sixty_deterministic_cases():
    path = Path(__file__).parent / "data" / "adsa_geometry_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert manifest["seed"] == 20260710
    assert len(manifest["cases"]) == 60
    assert sum(case["kind"] == "needle" for case in manifest["cases"]) == 24
    assert sum(case["kind"] == "initializer" for case in manifest["cases"]) == 18
    assert sum(case["kind"] == "selector" for case in manifest["cases"]) == 18


def test_bilateral_fit_rejects_unstable_width_and_accepts_clean_lines():
    rows = np.arange(30, dtype=float)
    clean = fit_bilateral_needle_edges(rows, 100 + 0.1 * rows, 140 + 0.1 * rows)
    assert clean.accepted
    unstable = fit_bilateral_needle_edges(rows, 100 + 0.1 * rows, 140 + 0.1 * rows + np.where(rows > 15, 10, 0))
    assert not unstable.accepted
    assert "needle_width_unstable" in unstable.rejection_reasons


def test_pendant_axis_transform_round_trip_for_vertical_and_tilted_axes():
    contour = np.array([[10.0, 20.0], [12.0, 18.0], [8.0, 22.0]])
    for direction in ((0.0, -1.0), (0.12, -0.99)):
        local = pendant_contour_to_model_mm(
            contour, axis_x_px=10, apex_y_px=20, px_per_mm=2,
            axis_origin_px=(10, 20), axis_direction_xy=direction,
        )
        restored = model_mm_to_pendant_px(
            local, axis_x_px=10, apex_y_px=20, px_per_mm=2,
            axis_origin_px=(10, 20), axis_direction_xy=direction,
        )
        assert np.allclose(restored, contour, atol=1e-9)


def test_geometry_cases_are_deterministic_and_invalid_inputs_reject():
    clean_spec = {"id": "needle_sessile_clean_0", "kind": "needle", "pipeline": "sessile", "perturbation": "clean"}
    first = evaluate_geometry_case(generate_geometry_case(clean_spec), clean_spec)
    second = evaluate_geometry_case(generate_geometry_case(clean_spec), clean_spec)
    assert first["accepted"] == second["accepted"]
    invalid_spec = {"id": "needle_invalid", "kind": "needle", "pipeline": "sessile", "perturbation": "severe"}
    invalid = evaluate_geometry_case(generate_geometry_case(invalid_spec), invalid_spec)
    assert invalid["accepted"] is False
