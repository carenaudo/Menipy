import importlib
import numpy as np
import cv2
import pytest

from menipy.analysis import find_apex_index, save_contour_sides_image


def test_save_contour_sides_image(tmp_path):
    if importlib.util.find_spec("matplotlib") is None:
        pytest.skip("matplotlib not available")

    img = np.zeros((40, 40), dtype=np.uint8)
    cv2.circle(img, (20, 20), 10, 255, -1)
    contour = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0][0].squeeze(1).astype(float)

    apex_idx = find_apex_index(contour, "pendant")
    out_path = tmp_path / "sides.png"
    save_contour_sides_image(contour, apex_idx, str(out_path))

    assert out_path.exists()
    assert out_path.stat().st_size > 0
