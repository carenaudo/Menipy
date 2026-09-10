from collections.abc import Sequence

import numpy as np
import pytest

from menipy.common.cancellation import check_cancelled
from menipy.common.temporal_sessile import BOOTSTRAP_SEED, _bootstrap_stats


def separate_percentiles(
    values: Sequence[float],
) -> dict[str, float | int | list[float]]:
    check_cancelled()
    array = np.asarray(values, dtype=float)
    median = float(np.median(array))
    mad = float(np.median(np.abs(array - median)))
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    # Keep the original row-major draw sequence, but retain only a bounded
    # block of samples. A single very long row still needs O(frame_count) space.
    batch_rows = max(1, min(2000, 262144 // max(1, len(array))))
    medians = np.empty(2000, dtype=float)
    for start in range(0, 2000, batch_rows):
        check_cancelled()
        end = min(start + batch_rows, 2000)
        samples = rng.choice(array, size=(end - start, len(array)), replace=True)
        medians[start:end] = np.median(samples, axis=1, overwrite_input=True)
        del samples
    check_cancelled()
    return {
        "median_deg": median,
        "mad_deg": mad,
        "ci95_deg": [
            float(np.percentile(medians, 2.5)),
            float(np.percentile(medians, 97.5)),
        ],
        "n_frames": int(len(array)),
    }


@pytest.mark.parametrize("count", [1, 5, 17, 131, 1000])
@pytest.mark.parametrize("kind", ["noise", "repeated", "extreme"])
def test_exact_statistics(count, kind):
    values = np.random.default_rng(42).normal(90, 10, count)
    if kind == "repeated":
        values = np.round(values)
    elif kind == "extreme":
        values *= 1e100
    original = values.copy()
    assert _bootstrap_stats(values) == separate_percentiles(values)
    np.testing.assert_array_equal(values, original)
