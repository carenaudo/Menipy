"""Tests for test sessile minimal pipeline.

Unit tests."""


import numpy as np
from menipy.models.context import Context
from menipy.pipelines.sessile.preprocessing import run as preprocessing_run
from menipy.pipelines.sessile.scaling import run as scaling_run
from menipy.pipelines.sessile.physics import run as physics_run
from menipy.pipelines.sessile.validation import run as validation_run
from menipy.pipelines.sessile.outputs import run as outputs_run


def make_dummy_image(w=200, h=150):
    # Create a simple synthetic grayscale image with a bright circle representing a droplet
    img = np.zeros((h, w, 3), dtype=np.uint8)
    cv2 = None
    try:
        import cv2 as _cv

        cv2 = _cv
    except Exception:
        pass
    if cv2 is not None:
        cv2.circle(img, (w // 2, h // 2), min(w, h) // 6, (255, 255, 255), -1)
    else:
        # Fallback: place a square
        img[h // 2 - 10 : h // 2 + 10, w // 2 - 10 : w // 2 + 10] = 255
    return img


def test_preprocessing_sets_fields():
    ctx = Context()
    ctx.image = make_dummy_image()
    ctx = preprocessing_run(ctx)
    assert hasattr(ctx, "preprocessed") and ctx.preprocessed is not None
    assert ctx.preprocessed_settings["blur_ksize"] == (5, 5)


def test_scaling_defaults():
    ctx = Context()
    assert ctx.scale == {}
    ctx = scaling_run(ctx)
    assert ctx.scale["px_per_mm"] == 1.0


def test_physics_defaults():
    ctx = Context()
    ctx = physics_run(ctx)
    assert ctx.physics["rho1"] == 1000.0
    assert ctx.physics["g"] == 9.80665


def test_validation_and_outputs():
    ctx = Context()
    # Without fit or geometry, QA should be false
    ctx = validation_run(ctx)
    assert ctx.qa["ok"] is False

    # When geometry present, QA becomes true
    ctx.geometry = object()
    ctx = validation_run(ctx)
    assert ctx.qa["ok"] is True

    # Outputs stage should collect fit params
    ctx.fit = {"param_names": ["R0_mm"], "params": [123.0]}
    ctx = outputs_run(ctx)
    assert ctx.results["R0_mm"] == 123.0
    assert "surface_tension_mN_m" in ctx.results
