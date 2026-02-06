"""Tests for test preprocessing compose.

Unit tests."""


import numpy as np
from menipy.common import preprocessing
from menipy.models.state import PreprocessingState


def test_compose_grayscale_into_color():
    base = np.zeros((100, 200, 3), dtype=np.uint8)
    roi = np.full((20, 50), 150, dtype=np.uint8)  # grayscale
    state = PreprocessingState()
    state.roi_bounds = (10, 10, 50, 20)

    comp = preprocessing._compose_full_image(base, roi, state)
    assert comp is not None
    assert comp.shape == base.shape
    # Check that the inserted region has the grayscale value across all channels
    sub = comp[10 : 10 + 20, 10 : 10 + 50]
    assert sub.ndim == 3
    assert (sub[..., 0] == 150).all()
    assert (sub[..., 1] == 150).all()
    assert (sub[..., 2] == 150).all()
