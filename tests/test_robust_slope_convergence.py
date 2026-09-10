import numpy as np
import pytest

from menipy.common import temporal_sessile as temporal
from menipy.common.cancellation import check_cancelled


def legacy_slope(t: np.ndarray, values: np.ndarray) -> float:
    """Small deterministic Huber-style local linear regression."""
    design = np.column_stack([t - np.mean(t), np.ones_like(t)])
    weights = np.ones_like(values)
    coef = np.linalg.lstsq(design, values, rcond=None)[0]
    for _ in range(6):
        check_cancelled()
        residual = values - design @ coef
        scale = max(
            1e-9, 1.4826 * float(np.median(np.abs(residual - np.median(residual))))
        )
        normalized = np.abs(residual) / (1.345 * scale)
        weights = np.where(normalized <= 1.0, 1.0, 1.0 / np.maximum(normalized, 1e-12))
        coef = np.linalg.lstsq(design * weights[:, None], values * weights, rcond=None)[
            0
        ]
    return float(coef[0])


@pytest.mark.parametrize("count", [3, 4, 7, 20])
@pytest.mark.parametrize(
    "kind", ["flat", "linear", "noise", "outlier", "duplicate_time"]
)
def test_exact_slopes(count, kind):
    rng = np.random.default_rng(82)
    for _ in range(20):
        t = np.sort(rng.uniform(0, 3, count))
        values = 2 + t * 0.1
        if kind == "flat":
            values[:] = 2
        elif kind == "noise":
            values += rng.normal(0, 0.1, count)
        elif kind == "outlier":
            values[count // 2] += 100
        elif kind == "duplicate_time":
            t[:] = 1
        assert temporal._robust_slope(t, values).hex() == legacy_slope(t, values).hex()


@pytest.mark.parametrize("flat", [True, False])
def test_exact_classification(flat, monkeypatch):
    from tests.test_numerical_repeated_work import frames

    actual = frames(500)
    if flat:
        for frame in actual:
            frame.half_width_mm = 2.0
    expected = [frame.model_copy(deep=True) for frame in actual]
    actual_deadband = temporal._assign_states(actual)
    monkeypatch.setattr(temporal, "_robust_slope", legacy_slope)
    expected_deadband = temporal._assign_states(expected)
    assert actual_deadband == expected_deadband
    assert [f.model_dump() for f in actual] == [f.model_dump() for f in expected]


def test_cancellation_after_converged_solve(monkeypatch):
    from menipy.common.cancellation import (
        AnalysisCancelled,
        CancellationToken,
        cancellation_scope,
    )

    token = CancellationToken()
    original = np.linalg.lstsq
    calls = []

    def solve(*args, **kwargs):
        result = original(*args, **kwargs)
        calls.append(True)
        if len(calls) == 2:
            token.cancel()
        return result

    monkeypatch.setattr(np.linalg, "lstsq", solve)
    with pytest.raises(AnalysisCancelled):
        with cancellation_scope(token):
            temporal._robust_slope(np.arange(7, dtype=float), np.ones(7))
    assert len(calls) == 2
