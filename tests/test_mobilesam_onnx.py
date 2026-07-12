"""Tests for the optional ONNX-only MobileSAM runtime."""

from __future__ import annotations

import importlib.util

import cv2
import numpy as np
import pytest

from menipy.common.mobilesam_onnx import DEFAULT_MODEL_DIR, MobileSAMOnnx


def test_preprocess_has_expected_shape_and_type() -> None:
    image = np.zeros((279, 471, 3), dtype=np.uint8)
    tensor = MobileSAMOnnx.preprocess(image)
    assert tensor.shape == (1, 3, 1024, 1024)
    assert tensor.dtype == np.float32


@pytest.mark.skipif(
    importlib.util.find_spec("onnxruntime") is None,
    reason="onnxruntime is an optional dependency",
)
def test_box_prompt_segments_sessile_sample() -> None:
    image = cv2.imread("data/samples/gota depositada 1.png")
    assert image is not None
    predictor = MobileSAMOnnx(DEFAULT_MODEL_DIR)
    result = predictor.predict_box(image, (145, 165, 335, 240))
    mask = result.best_mask
    assert mask.shape == image.shape[:2]
    assert 5_000 < int(mask.sum()) < 25_000
    assert mask[205, 235]
    assert not mask[20, 20]
