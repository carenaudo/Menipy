import numpy as np
import pytest

from menipy.common import temporal_sessile as temporal
from menipy.common.cancellation import check_cancelled
from menipy.common.temporal_sessile import _robust_slope, _velocity_windows
from menipy.models.temporal import TemporalFrameResult


def legacy_states(frames: list[TemporalFrameResult]) -> float:
    valid = [
        frame for frame in frames if frame.accepted and frame.half_width_mm is not None
    ]
    for frame, neighbours in _velocity_windows(valid):
        check_cancelled()
        if len(neighbours) >= 3:
            t = np.asarray([other.timestamp_s for other in neighbours], dtype=float)
            values = np.asarray(
                [other.half_width_mm for other in neighbours], dtype=float
            )
            frame.contact_velocity_mm_s = _robust_slope(t, values)

    pairs: list[float] = []
    for left, right in zip(valid, valid[1:]):
        check_cancelled()
        dt = right.timestamp_s - left.timestamp_s
        if dt > 0 and right.segment_id == left.segment_id:
            pairs.append((float(right.half_width_mm) - float(left.half_width_mm)) / dt)
    if len(pairs) >= 3:
        # First differences of velocity isolate measurement noise without treating
        # the commanded advancing/receding speed itself as noise.
        values = np.diff(np.asarray(pairs, dtype=float))
        mad = (
            1.4826 * float(np.median(np.abs(values - np.median(values)))) / np.sqrt(2.0)
        )
    else:
        mad = 0.0
    deadband = max(0.01, 3.0 * mad)
    for position, frame in enumerate(valid):
        check_cancelled()
        local_slopes: list[float] = []
        if position > 0 and valid[position - 1].segment_id == frame.segment_id:
            dt = frame.timestamp_s - valid[position - 1].timestamp_s
            if dt > 0:
                local_slopes.append(
                    (
                        float(frame.half_width_mm)
                        - float(valid[position - 1].half_width_mm)
                    )
                    / dt
                )
        if (
            position + 1 < len(valid)
            and valid[position + 1].segment_id == frame.segment_id
        ):
            dt = valid[position + 1].timestamp_s - frame.timestamp_s
            if dt > 0:
                local_slopes.append(
                    (
                        float(valid[position + 1].half_width_mm)
                        - float(frame.half_width_mm)
                    )
                    / dt
                )
        # The seven-frame fit remains the reported velocity. The adjacent robust
        # median only snaps state transitions so a plateau is not shifted by half
        # the classification window.
        velocity = (
            float(np.median(local_slopes))
            if local_slopes
            else frame.contact_velocity_mm_s
        )
        frame.state = (
            "pinned"
            if velocity is None or abs(velocity) <= deadband
            else ("advancing" if velocity > 0 else "receding")
        )

    # Advancing/receding runs shorter than three frames are not physical states.
    index = 0
    while index < len(frames):
        check_cancelled()
        state = frames[index].state
        end = index + 1
        while (
            end < len(frames)
            and frames[end].state == state
            and frames[end].segment_id == frames[index].segment_id
        ):
            check_cancelled()
            end += 1
        if state in {"advancing", "receding"} and end - index < 3:
            for position in range(index, end):
                check_cancelled()
                frames[position].state = (
                    "pinned" if frames[position].accepted else "invalid"
                )
        index = end
    return deadband


@pytest.mark.parametrize(
    "mode", ["varying", "flat", "gaps", "duplicate_time", "reverse", "isolated"]
)
def test_exact_full_states(mode):
    from tests.test_numerical_repeated_work import frames

    actual = frames(500)
    if mode == "flat":
        for frame in actual:
            frame.half_width_mm = 2.0
    elif mode == "gaps":
        actual = actual[::3]
    elif mode == "duplicate_time":
        for frame in actual:
            frame.timestamp_s = 1.0
    elif mode == "reverse":
        actual.reverse()
    elif mode == "isolated":
        for i, frame in enumerate(actual):
            frame.segment_id = i
    expected = [f.model_copy(deep=True) for f in actual]
    assert temporal._assign_states(actual) == legacy_states(expected)
    assert [f.model_dump() for f in actual] == [f.model_dump() for f in expected]


@pytest.mark.parametrize(
    "values",
    [
        [-0.0],
        [0.0],
        [1e-320],
        [-0.0, -0.0],
        [0.0, -0.0],
        [1e308, -1e308],
        [1e-320, -1e-320],
        [float("inf"), 1.0],
        [float("inf"), float("-inf")],
    ],
)
def test_scalar_arithmetic_matches_numpy_bits(values):
    with np.errstate(invalid="ignore", over="ignore"):
        expected = float(np.median(values))
    actual = (
        (0.0 + values[0] + values[1]) / 2.0 if len(values) == 2 else values[0] + 0.0
    )
    if np.isnan(expected):
        assert np.isnan(actual)
    else:
        assert actual.hex() == expected.hex()
