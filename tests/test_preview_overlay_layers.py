from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PySide6.QtCore import QFile
from PySide6.QtUiTools import QUiLoader
from PySide6.QtWidgets import QCheckBox

from menipy.gui.panels.preview_panel import PreviewPanel
from menipy.gui.views.image_view import ImageView


class DummySettings:
    def __init__(self):
        self.overlay_config = {}
        self.marker_config = {}
        self.saved = 0

    def save(self):
        self.saved += 1


def _load_preview_panel():
    path = Path("src/menipy/gui/views/overlay_panel.ui").resolve()
    file = QFile(str(path))
    assert file.open(QFile.ReadOnly)
    loader = QUiLoader()
    loader.registerCustomWidget(ImageView)
    try:
        return loader.load(file)
    finally:
        file.close()


def _item(view: ImageView, tag: str):
    return view._overlay_items_by_tag[tag]


def test_preview_overlay_layers_toggle_without_deleting(qtbot):
    widget = _load_preview_panel()
    qtbot.addWidget(widget)
    settings = DummySettings()
    panel = PreviewPanel(widget, ImageView, settings)
    panel.display(np.zeros((80, 100, 3), dtype=np.uint8))

    panel.render_overlay_commands(
        [
            {
                "type": "polyline",
                "points": [[10, 10], [40, 20], [20, 40]],
                "closed": True,
                "tag": "result_contour",
                "layer": "contour",
            },
            {
                "type": "polyline",
                "points": [[50, 10], [60, 30]],
                "closed": False,
                "tag": "pendant_fit",
                "layer": "fit",
            },
            {
                "type": "line",
                "p1": [30, 0],
                "p2": [30, 70],
                "tag": "axis",
                "layer": "axes",
            },
        ]
    )

    contour = _item(panel.image_view, "result_contour")
    fit = _item(panel.image_view, "pendant_fit")
    axis = _item(panel.image_view, "axis")
    assert contour.isVisible()
    assert fit.isVisible()
    assert axis.isVisible()

    widget.findChild(QCheckBox, "showContourCheck").setChecked(False)
    assert not contour.isVisible()
    assert "result_contour" in panel.image_view._overlay_items_by_tag
    assert settings.overlay_config["contour_visible"] is False
    assert settings.saved >= 1

    widget.findChild(QCheckBox, "showFitCheck").setChecked(False)
    assert not fit.isVisible()
    assert axis.isVisible()


def test_preview_fit_polyline_is_open(qtbot):
    widget = _load_preview_panel()
    qtbot.addWidget(widget)
    panel = PreviewPanel(widget, ImageView, DummySettings())
    panel.display(np.zeros((80, 100, 3), dtype=np.uint8))

    panel.render_overlay_commands(
        [
            {
                "type": "polyline",
                "points": [[10, 10], [20, 20], [30, 10]],
                "closed": False,
                "tag": "pendant_fit",
                "layer": "fit",
            }
        ]
    )

    path = _item(panel.image_view, "pendant_fit").path()
    assert path.elementCount() == 3


def test_display_context_prefers_base_image_and_rendered_commands(qtbot):
    widget = _load_preview_panel()
    qtbot.addWidget(widget)
    panel = PreviewPanel(widget, ImageView, DummySettings())
    ctx = SimpleNamespace(
        image=np.zeros((80, 100, 3), dtype=np.uint8),
        preview=None,
        overlay_commands=[
            {
                "type": "polyline",
                "points": [[10, 10], [40, 20], [20, 40]],
                "tag": "result_contour",
                "layer": "contour",
            }
        ],
    )

    panel.display_context(ctx)

    assert panel.image_view.has_overlay("result_contour")


def test_display_context_falls_back_to_preview(qtbot):
    widget = _load_preview_panel()
    qtbot.addWidget(widget)
    panel = PreviewPanel(widget, ImageView, DummySettings())
    ctx = SimpleNamespace(
        image=None,
        current_frame=None,
        frames=None,
        preview=np.zeros((80, 100, 3), dtype=np.uint8),
        overlay_commands=None,
    )

    panel.display_context(ctx)

    assert panel.image_view.scene().items()


def test_overlay_strokes_are_screen_constant_by_default(qtbot):
    view = ImageView()
    qtbot.addWidget(view)
    view.set_image(np.zeros((50, 50, 3), dtype=np.uint8))

    view.add_marker_line(
        view.scene().sceneRect().topLeft(),
        view.scene().sceneRect().bottomRight(),
        tag="axis",
        layer="axes",
    )
    assert _item(view, "axis").pen().isCosmetic()

    view.set_overlay_stroke_scale_mode("image")
    view.add_marker_line(
        view.scene().sceneRect().bottomLeft(),
        view.scene().sceneRect().topRight(),
        tag="axis_image",
        layer="axes",
    )
    assert not _item(view, "axis_image").pen().isCosmetic()


def test_layer_config_styles_rendered_overlay(qtbot):
    widget = _load_preview_panel()
    qtbot.addWidget(widget)
    settings = DummySettings()
    settings.overlay_config = {
        "fit_color": "#123456",
        "fit_thickness": 5,
        "fit_alpha": 0.5,
    }
    panel = PreviewPanel(widget, ImageView, settings)
    panel.display(np.zeros((80, 100, 3), dtype=np.uint8))

    panel.render_overlay_commands(
        [
            {
                "type": "polyline",
                "points": [[10, 10], [20, 20]],
                "closed": False,
                "tag": "pendant_fit",
                "layer": "fit",
            }
        ]
    )

    pen = _item(panel.image_view, "pendant_fit").pen()
    assert pen.color().name() == "#123456"
    assert pen.widthF() == 5
    assert 0.45 <= pen.color().alphaF() <= 0.55


def test_marker_config_adds_marker_label(qtbot):
    view = ImageView()
    qtbot.addWidget(view)
    view.set_image(np.zeros((50, 50, 3), dtype=np.uint8))
    view.set_marker_config(
        {
            "default": {"visible": True},
            "apex": {
                "visible": True,
                "shape": "cross",
                "color": "#ff0000",
                "radius": 6,
                "label_visible": True,
                "label_text": "apex",
                "font_size": 12,
            },
        }
    )

    view.add_marker_point(
        view.scene().sceneRect().center(),
        tag="apex",
        layer="markers",
    )

    assert view.has_overlay("apex")
    assert view.has_overlay("apex_label")
