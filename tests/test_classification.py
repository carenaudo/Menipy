import numpy as np
from dataclasses import dataclass

from src.processing import classify_drop_mode

@dataclass
class DummyDroplet:
    contour_px: np.ndarray
    substrate_px: tuple[int, int, int, int] | None
    contact_px: tuple[int, int, int, int] | None
    apex_px: tuple[int, int]


def test_classify_pendant():
    droplet = DummyDroplet(
        contour_px=np.empty((0, 2), dtype=float),
        substrate_px=None,
        contact_px=(0, 0, 40, 0),
        apex_px=(20, 30),
    )
    assert classify_drop_mode(droplet) == "pendant"


def test_classify_sessile():
    droplet = DummyDroplet(
        contour_px=np.empty((0, 2), dtype=float),
        substrate_px=None,
        contact_px=(0, 30, 40, 30),
        apex_px=(20, 0),
    )
    assert classify_drop_mode(droplet) == "sessile"


def test_classify_unknown_short_line():
    droplet = DummyDroplet(
        contour_px=np.empty((0, 2), dtype=float),
        substrate_px=None,
        contact_px=(10, 0, 18, 0),
        apex_px=(14, 30),
    )
    assert classify_drop_mode(droplet) == "unknown"


def test_classify_unknown_gap():
    droplet = DummyDroplet(
        contour_px=np.empty((0, 2), dtype=float),
        substrate_px=None,
        contact_px=(0, 0, 40, 0),
        apex_px=(20, 0),
    )
    assert classify_drop_mode(droplet) == "unknown"
