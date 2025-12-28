import numpy as np
import pytest

from menipy.models.context import Context
from menipy.models.config import PreprocessingSettings
from menipy.models.state import PreprocessingState
from menipy.common import preprocessing
from menipy.common.preprocessing_helpers import (
    PreprocessingContext,
    crop_to_roi,
    rescale_roi,
    apply_filter,
    subtract_background,
    normalize_intensity,
)


def make_image(width: int = 6, height: int = 4) -> np.ndarray:
    grid = np.arange(width * height, dtype=np.uint8).reshape(height, width)
    return np.stack([grid] * 3, axis=2)


def test_crop_to_roi_generates_mask_and_raw_state():
    image = make_image()
    settings = PreprocessingSettings()
    context = PreprocessingContext(image, settings=settings, roi_bounds=(2, 1, 3, 2))

    crop_to_roi(context)

    assert context.state.raw_roi.shape[:2] == (2, 3)
    assert context.state.roi_mask.shape == (2, 3)
    assert np.all(context.state.roi_mask == 255)
    assert context.state.history[0].name == "crop"


def test_rescale_roi_updates_scale_and_mask(monkeypatch):
    image = make_image(3, 3)
    settings = PreprocessingSettings()
    settings.resize.enabled = True
    settings.resize.target_width = 6
    settings.resize.target_height = 6
    settings.resize.preserve_aspect = True

    context = PreprocessingContext(image, settings=settings, roi_bounds=(0, 0, 3, 3))
    crop_to_roi(context)

    rescale_roi(context)

    assert context.state.working_roi.shape[:2] == (6, 6)
    assert context.state.roi_mask.shape == (6, 6)
    sx, sy = context.state.scale
    assert pytest.approx(sx, rel=1e-6) == 2.0
    assert pytest.approx(sy, rel=1e-6) == 2.0


def test_apply_filter_respects_mask(monkeypatch):
    image = make_image(4, 4)
    settings = PreprocessingSettings()
    settings.filtering.enabled = True
    settings.filtering.method = "gaussian"
    settings.filtering.kernel_size = 3

    context = PreprocessingContext(image, settings=settings, roi_bounds=(0, 0, 4, 4))
    crop_to_roi(context)

    mask = context.state.roi_mask
    mask[:] = 0
    mask[1:3, 1:3] = 255

    def fake_blur(arr, kernel, sigma):  # pragma: no cover - patched in test
        return arr + 17

    monkeypatch.setattr("menipy.common.preprocessing_helpers._gaussian_blur", fake_blur)

    apply_filter(context)

    result = context.state.working_roi
    assert np.all(result[1:3, 1:3] == image[1:3, 1:3] + 17)
    assert np.all(result[0, 0] == image[0, 0])


def test_background_subtraction_uses_strength(monkeypatch):
    image = make_image(4, 4)
    settings = PreprocessingSettings()
    settings.background.enabled = True
    settings.background.mode = "flat"
    settings.background.strength = 0.5

    context = PreprocessingContext(image, settings=settings, roi_bounds=(0, 0, 4, 4))
    crop_to_roi(context)

    subtract_background(context)

    assert context.state.working_roi.dtype == np.uint8
    assert context.state.history[-1].name == "background"


def test_normalization_records_state(monkeypatch):
    image = make_image(4, 4)
    settings = PreprocessingSettings()
    settings.normalization.enabled = True
    settings.normalization.method = "histogram"

    context = PreprocessingContext(image, settings=settings, roi_bounds=(0, 0, 4, 4))
    crop_to_roi(context)

    normalize_intensity(context)

    assert context.state.normalized_roi is not None
    assert context.state.history[-1].name == "normalize"


def test_preprocessing_run_populates_context():
    ctx = Context()
    ctx.frame = make_image(5, 5)
    ctx.roi = (1, 1, 3, 3)

    preprocessing.run(ctx)

    assert isinstance(ctx.preprocessed_state, PreprocessingState)
    assert ctx.preprocessed_roi.shape[:2] == (3, 3)
    assert ctx.preprocessed_history[0]["name"] == "crop"
    assert ctx.preprocessed_settings.crop_to_roi is True
    assert ctx.preprocessed_mask.shape == (3, 3)

    full = ctx.preprocessed
    assert full.shape == ctx.frame.shape
    assert np.array_equal(full[1:4, 1:4], ctx.preprocessed_roi)
