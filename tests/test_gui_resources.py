"""Tests for test gui resources.

Unit tests."""


import pytest
from pathlib import Path

pytest.importorskip("PySide6")


def test_register_qrc_no_rcc(monkeypatch):
    from menipy.gui import app

    # Simulate no rcc files present
    monkeypatch.setattr(Path, "exists", lambda self: False)

    # Should not raise
    app._register_qrc()


def test_register_qrc_with_rcc(monkeypatch, tmp_path):
    pytest.importorskip("PySide6")
    from menipy.gui import app
    from PySide6.QtCore import QResource

    called = {"count": 0}

    def fake_register(path):
        called["count"] += 1
        return True

    # monkeypatch the QResource.registerResource function
    monkeypatch.setattr(QResource, "registerResource", fake_register, raising=False)

    # Simulate that rcc exists
    monkeypatch.setattr(Path, "exists", lambda self: True)

    app._register_qrc()

    assert called["count"] >= 1
