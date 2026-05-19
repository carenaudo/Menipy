from __future__ import annotations

from menipy.gui.dialogs.marker_config_dialog import MarkerConfigDialog


def test_marker_config_dialog_roundtrip(qtbot):
    dialog = MarkerConfigDialog(
        {
            "apex": {
                "visible": False,
                "shape": "cross",
                "color": "#123456",
                "radius": 9,
                "label_visible": True,
                "label_text": "A",
                "label_color": "#abcdef",
                "font_size": 14,
            }
        }
    )
    qtbot.addWidget(dialog)

    config = dialog.get_config()

    assert config["apex"]["visible"] is False
    assert config["apex"]["shape"] == "cross"
    assert config["apex"]["color"] == "#123456"
    assert config["apex"]["radius"] == 9
    assert config["apex"]["label_visible"] is True
    assert config["apex"]["label_text"] == "A"
    assert config["apex"]["label_color"] == "#abcdef"
    assert config["apex"]["font_size"] == 14
    assert "roi" in config
