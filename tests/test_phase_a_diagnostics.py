"""Compatibility tests for Phase-A diagnostics and rejected measurements."""

from __future__ import annotations

import numpy as np

from menipy.common.detection_result import DetectionResult, normalize_detection_result
from menipy.common.validation import QACheck, QAResult
from menipy.models.context import Context
from menipy.models.fit import Residuals, SolverInfo
from menipy.models.results import MeasurementResult, build_persisted_analysis


def test_extended_fit_models_remain_backward_compatible():
    solver = SolverInfo(success=True)
    residuals = Residuals(rmse=0.2, max_abs=0.5)
    assert solver.nfev is None and solver.termination_reason is None
    assert residuals.p95_abs is None and residuals.units is None


def test_legacy_measurement_defaults_to_accepted():
    measurement = MeasurementResult.model_validate(
        {
            "id": "legacy",
            "timestamp": "2026-07-10T12:00:00",
            "pipeline": "sessile",
            "results": {"contact_angle_deg": 90.0},
        }
    )
    assert measurement.accepted
    assert measurement.rejection_reasons == []
    assert measurement.diagnostics == {}


def test_rejected_context_quarantines_physical_metrics():
    ctx = Context(results={"surface_tension_mN_m": 72.0, "diagnostics": {"solver": {}}})
    ctx.qa = QAResult(
        ok=False,
        score=0.0,
        checks={
            "fit": QACheck(
                name="Fit",
                passed=False,
                value=0.0,
                threshold=1.0,
                message="failed",
                code="solver_not_converged",
            )
        },
    )
    payload = build_persisted_analysis(ctx)
    assert not payload["accepted"]
    assert payload["results"] == {}
    assert payload["rejection_reasons"] == ["solver_not_converged"]
    assert payload["diagnostics"]


def test_detection_adapter_supports_legacy_and_typed_results():
    contour = np.asarray([[0, 0], [1, 1]])
    legacy = normalize_detection_result(contour, feature="drop")
    missing = normalize_detection_result(None, feature="drop")
    typed = DetectionResult(value=(1, 2, 3, 4), confidence=0.7, accepted=True)
    assert legacy.accepted and legacy.value is contour
    assert not missing.accepted and missing.rejection_reasons == ["drop_not_detected"]
    assert normalize_detection_result(typed, feature="needle") is typed
