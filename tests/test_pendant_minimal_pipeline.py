import numpy as np
from menipy.models.context import Context
from menipy.pipelines.pendant.preprocessing import run as preprocessing_run
from menipy.pipelines.pendant.scaling import run as scaling_run
from menipy.pipelines.pendant.physics import run as physics_run
from menipy.pipelines.pendant.validation import run as validation_run
from menipy.pipelines.pendant.outputs import run as outputs_run


def make_dummy_image(w=160, h=120):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    try:
        import cv2 as _cv

        _cv.circle(img, (w // 2, h // 2), min(w, h) // 5, (255, 255, 255), -1)
    except Exception:
        img[h // 2 - 8 : h // 2 + 8, w // 2 - 8 : w // 2 + 8] = 255
    return img


def test_preprocessing_sets_fields():
    ctx = Context()
    ctx.image = make_dummy_image()
    ctx = preprocessing_run(ctx)
    assert hasattr(ctx, "preprocessed") and ctx.preprocessed is not None
    assert ctx.preprocessed_settings["blur_ksize"] == (5, 5)


def test_scaling_defaults():
    ctx = Context()
    ctx = scaling_run(ctx)
    assert ctx.scale["px_per_mm"] == 1.0


def test_physics_defaults():
    ctx = Context()
    ctx = physics_run(ctx)
    assert ctx.physics["rho1"] == 1000.0
    assert ctx.physics["g"] == 9.80665


def test_validation_and_outputs():
    ctx = Context()
    ctx = validation_run(ctx)
    assert ctx.qa["ok"] is False

    ctx.geometry = object()
    ctx = validation_run(ctx)
    assert ctx.qa["ok"] is True

    ctx.fit = {"param_names": ["R0_mm"], "params": [99.0]}
    ctx = outputs_run(ctx)
    assert ctx.results["R0_mm"] == 99.0
    assert "surface_tension_mN_m" in ctx.results
