from __future__ import annotations

from types import SimpleNamespace

from PySide6.QtCore import Qt
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QMessageBox

from menipy.gui import main_window as main_window_module
from menipy.gui.panels import results_panel as results_panel_module


class FakeHistory:
    measurements = []

    def add_measurement(self, measurement):
        self.measurements.insert(0, measurement)

    def clear_history(self):
        self.measurements.clear()


def _settings(path):
    return SimpleNamespace(
        selected_pipeline="sessile",
        last_image_path=path,
        plugin_dirs=["./plugins"],
        acquisition_requires_contact_line=False,
        overlay_config=None,
        splitter_sizes=None,
        main_window_state_b64=None,
        main_window_geom_b64=None,
        save=lambda: None,
    )


def _patch_startup(monkeypatch, image_path):
    monkeypatch.setattr(
        main_window_module.AppSettings,
        "load",
        staticmethod(lambda: _settings(image_path)),
    )
    monkeypatch.setattr(
        results_panel_module,
        "get_results_history",
        lambda: FakeHistory(),
    )


def _write_image(path):
    image = QImage(24, 24, QImage.Format_RGB32)
    image.fill(Qt.white)
    assert image.save(str(path))


def test_startup_loads_remembered_single_file(monkeypatch, qtbot, tmp_path):
    image_path = tmp_path / "drop.png"
    _write_image(image_path)
    _patch_startup(monkeypatch, str(image_path))

    window = main_window_module.MainWindow()
    qtbot.addWidget(window)

    assert window.preview_panel.image_view.scene().items()
    assert all(button.isEnabled() for button in window.preview_panel._overlay_buttons)


def test_startup_missing_remembered_file_is_silent(monkeypatch, qtbot, tmp_path):
    missing_path = tmp_path / "missing.png"
    _patch_startup(monkeypatch, str(missing_path))
    warnings = []
    monkeypatch.setattr(
        QMessageBox, "warning", lambda *args, **kwargs: warnings.append(args)
    )

    window = main_window_module.MainWindow()
    qtbot.addWidget(window)

    assert warnings == []
    assert not window.preview_panel.image_view.scene().items()
    assert all(
        not button.isEnabled() for button in window.preview_panel._overlay_buttons
    )
