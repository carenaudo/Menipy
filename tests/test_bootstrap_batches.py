"""Exact bootstrap draw/summary equivalence and bounded sample allocations."""

import hashlib

import numpy as np
import pytest

from menipy.common import temporal_sessile as temporal
from menipy.common.cancellation import (
    AnalysisCancelled,
    CancellationToken,
    cancellation_scope,
)


def legacy_stats(values):
    array = np.asarray(values, dtype=float)
    center = float(np.median(array))
    medians = np.median(
        np.random.default_rng(temporal.BOOTSTRAP_SEED).choice(
            array, size=(2000, len(array)), replace=True
        ),
        axis=1,
    )
    return {
        "median_deg": center,
        "mad_deg": float(np.median(np.abs(array - center))),
        "ci95_deg": [
            float(np.percentile(medians, 2.5)),
            float(np.percentile(medians, 97.5)),
        ],
        "n_frames": len(array),
    }


@pytest.mark.parametrize("count", [1, 5, 17, 131, 257, 1000, 5001])
def test_same_draws_and_exact_statistics(count, monkeypatch):
    values = np.random.default_rng(82).normal(90, 12, count)
    untouched = values.copy()
    expected = legacy_stats(values)
    rng = np.random.default_rng(temporal.BOOTSTRAP_SEED)
    reference_draws = rng.choice(values, size=(2000, count), replace=True)
    expected_digest = hashlib.sha256(reference_draws.tobytes()).hexdigest()
    del reference_draws
    original = np.random.default_rng
    digest = hashlib.sha256()
    sizes = []

    class Recorder:
        def __init__(self, seed):
            self.rng = original(seed)

        def choice(self, *args, **kwargs):
            samples = self.rng.choice(*args, **kwargs)
            digest.update(samples.tobytes())
            sizes.append(samples.size)
            return samples

    monkeypatch.setattr(np.random, "default_rng", Recorder)
    assert temporal._bootstrap_stats(values) == expected
    assert digest.hexdigest() == expected_digest
    assert max(sizes) <= max(count, 262144)
    np.testing.assert_array_equal(values, untouched)


def test_cancellation_between_batches(monkeypatch):
    values = np.arange(1000, dtype=float)
    original = np.random.default_rng
    token = CancellationToken()
    calls = []

    class Cancelling:
        def __init__(self, seed):
            self.rng = original(seed)

        def choice(self, *args, **kwargs):
            calls.append(True)
            samples = self.rng.choice(*args, **kwargs)
            token.cancel()
            return samples

    monkeypatch.setattr(np.random, "default_rng", Cancelling)
    with pytest.raises(AnalysisCancelled):
        with cancellation_scope(token):
            temporal._bootstrap_stats(values)
    assert len(calls) == 1


def test_repeated_values_preserve_percentiles():
    values = np.tile([0.0, 90.0, 90.0, 180.0], 1000)
    assert temporal._bootstrap_stats(values) == legacy_stats(values)
