"""Reference equivalence for fixed residual setup and temporal windows."""

import json
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from menipy.common import solver
from menipy.common import temporal_sessile as temporal
from menipy.models.fit import FitConfig
from menipy.models.temporal import TemporalFrameResult


def legacy_pointwise(obs, model):
    def resample(xy, u):
        seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        s = np.concatenate([[0.0], np.cumsum(seg)])
        v = s / (s[-1] if s[-1] > 0 else 1.0)
        return np.column_stack([np.interp(u, v, xy[:, 0]), np.interp(u, v, xy[:, 1])])

    u = np.linspace(0, 1, min(len(obs), 400))
    return (resample(obs, u) - resample(model, u)).reshape(-1)


def legacy_windows(valid):
    for frame in valid:
        yield (
            frame,
            [
                other
                for other in valid
                if abs(other.frame_index - frame.frame_index) <= 3
                and other.segment_id == frame.segment_id
            ],
        )


def frames(count):
    return [
        TemporalFrameResult(
            frame_index=i,
            timestamp_s=i / 30,
            segment_id=i // 200,
            accepted=i % 47 not in (20, 21, 22),
            half_width_mm=2 + 0.001 * i + 0.02 * np.sin(i / 8),
        )
        for i in range(count)
    ]


@pytest.mark.parametrize("count", [1, 20, 500, 5000])
def test_prepared_pointwise_exact_and_owned(count):
    rng = np.random.default_rng(82)
    obs = rng.normal(size=(count, 2))
    model = rng.normal(size=(100, 2))
    prepared = solver._prepare_pointwise_residual(obs)
    expected = legacy_pointwise(obs, model)
    np.testing.assert_array_equal(prepared(model), expected)
    obs[:] = 123
    np.testing.assert_array_equal(prepared(model), expected)


@pytest.mark.parametrize("weights", [None, [2.0], [1.0, 2.0, 3.0]])
def test_full_pointwise_fit_unchanged(weights):
    x = np.linspace(-1, 1, 100)
    obs = np.column_stack([x, 2 * x * x + 0.5])

    def integrator(params, physics, geometry):
        return np.column_stack([x, params[0] * x * x + params[1]])

    config = FitConfig(x0=[1.0, 0.0], bounds=([-5, -5], [5, 5]), weights=weights)

    def fit():
        result = solver.run(
            SimpleNamespace(contour=SimpleNamespace(xy=obs, units="mm")),
            integrator=integrator,
            config=config,
        )
        return result

    optimized = fit()
    with patch.object(
        solver,
        "_prepare_pointwise_residual",
        lambda obs: lambda model: legacy_pointwise(obs, model),
    ):
        reference = fit()

    # The numerical payload excludes only its measured solve duration.
    def strip_times(value):
        if isinstance(value, dict):
            return {
                key: strip_times(item)
                for key, item in value.items()
                if "time" not in key
            }
        return value

    assert json.dumps(strip_times(optimized), sort_keys=True) == json.dumps(
        strip_times(reference), sort_keys=True
    )


@pytest.mark.parametrize("ordering", ["ordered", "reverse", "duplicates"])
def test_temporal_windows_match_all_pair_search(ordering):
    records = frames(60)
    if ordering == "reverse":
        records.reverse()
    elif ordering == "duplicates":
        records[15].frame_index = records[14].frame_index
    valid = [frame for frame in records if frame.accepted]
    assert list(temporal._velocity_windows(valid)) == list(legacy_windows(valid))


def test_full_temporal_velocities_states_and_deadband_are_exact():
    optimized = frames(500)
    reference = deepcopy(optimized)
    optimized_deadband = temporal._assign_states(optimized)
    with patch.object(temporal, "_velocity_windows", legacy_windows):
        reference_deadband = temporal._assign_states(reference)
    assert optimized_deadband == reference_deadband
    assert [f.model_dump() for f in optimized] == [f.model_dump() for f in reference]
