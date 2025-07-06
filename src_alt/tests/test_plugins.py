import types
from importlib import metadata

import pytest

# Ensure src_alt/ added to path, similar to other tests
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from menipy import plugins


class DummyEP:
    def __init__(self, name, obj):
        self.name = name
        self._obj = obj

    def load(self):
        return self._obj


def test_load_plugins(monkeypatch):
    dummy_obj = object()
    eps = [DummyEP("dummy", dummy_obj)]

    def fake_entry_points(*, group):
        if group == "og.analysis":
            return eps
        return []

    monkeypatch.setattr(metadata, "entry_points", fake_entry_points)

    plugins.PLUGINS.clear()
    plugins.load_plugins()

    assert plugins.PLUGINS["dummy"] is dummy_obj
    from menipy.sharpen_plugin import sharpen_filter
    assert plugins.PLUGINS["sharpen"] is sharpen_filter


def test_sharpen_filter_shape() -> None:
    import numpy as np
    from menipy.sharpen_plugin import sharpen_filter

    img = np.zeros((5, 5, 3), dtype=np.uint8)
    out = sharpen_filter(img)
    assert out.shape == img.shape

