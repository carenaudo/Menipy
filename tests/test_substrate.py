import numpy as np
import pytest
from src.processing import detect_substrate_line, SubstrateNotFoundError


def _synthetic_image(angle_deg: float = 0.0, mode: str = "sessile"):
    cv2 = __import__('cv2')
    h, w = 60, 80
    img = np.full((h, w), 255, dtype=np.uint8)
    slope = np.tan(np.deg2rad(angle_deg))
    if mode == "sessile":
        y0 = h - 10
    else:
        y0 = 10
    intercept = y0 - slope * (w / 2)
    x1, y1 = 0, int(round(intercept))
    x2, y2 = w - 1, int(round(slope * (w - 1) + intercept))
    cv2.line(img, (x1, y1), (x2, y2), 0, 2)
    center_y = y1 - 12 if mode == "sessile" else y1 + 12
    cv2.circle(img, (w // 2, center_y), 8, 0, -1)
    mask = np.zeros_like(img)
    cv2.circle(mask, (w // 2, center_y), 8, 255, -1)
    return img, mask


def _theta(p1, p2):
    return np.degrees(np.arctan2(p2[1] - p1[1], p2[0] - p1[0]))


def test_detect_substrate_horizontal():
    img, mask = _synthetic_image(0.0, "sessile")
    p1, p2 = detect_substrate_line(img, mask, "sessile")
    assert abs(_theta(p1, p2)) < 1.0


def test_detect_substrate_tilted_sessile():
    img, mask = _synthetic_image(-15.0, "sessile")
    p1, p2 = detect_substrate_line(img, mask, "sessile")
    assert pytest.approx(_theta(p1, p2), abs=6.0) == -15.0


def test_detect_substrate_pendant():
    img, mask = _synthetic_image(10.0, "pendant")
    p1, p2 = detect_substrate_line(img, mask, "pendant")
    assert pytest.approx(_theta(p1, p2), abs=2.0) == 10.0

