"""Dialog for interactive marker and label display settings."""

from __future__ import annotations

from typing import Any

from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFontComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

MARKER_TYPES = (
    ("default", "Default markers"),
    ("roi", "ROI"),
    ("needle", "Needle"),
    ("contact", "Contact points / line"),
    ("apex", "Apex"),
    ("drop_center", "Drop center"),
    ("background", "Background samples"),
    ("result_text", "Result text"),
)


DEFAULT_MARKER_CONFIG: dict[str, dict[str, Any]] = {
    key: {
        "visible": True,
        "shape": "circle",
        "color": "#00ff00",
        "radius": 5,
        "label_visible": False,
        "label_text": "",
        "label_color": "#ffffff",
        "font_family": "",
        "font_size": 10,
    }
    for key, _ in MARKER_TYPES
}
DEFAULT_MARKER_CONFIG["roi"].update({"shape": "square", "color": "#ffff00"})
DEFAULT_MARKER_CONFIG["needle"].update({"shape": "square", "color": "#0000ff"})
DEFAULT_MARKER_CONFIG["contact"].update({"color": "#ff3333"})
DEFAULT_MARKER_CONFIG["apex"].update({"shape": "cross", "color": "#ff0000"})
DEFAULT_MARKER_CONFIG["background"].update({"color": "#6495ed", "radius": 4})
DEFAULT_MARKER_CONFIG["result_text"].update(
    {"shape": "circle", "visible": True, "label_visible": True, "font_size": 11}
)


class MarkerConfigDialog(QDialog):
    """Configure marker visibility, symbol style, and labels."""

    def __init__(self, config: dict | None = None, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Marker display")
        self.setMinimumWidth(620)
        self._widgets: dict[str, dict[str, Any]] = {}
        self._build_ui()
        self.set_config(config or DEFAULT_MARKER_CONFIG)

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        content = QWidget(scroll)
        content_layout = QVBoxLayout(content)
        for key, title in MARKER_TYPES:
            group = QGroupBox(title, content)
            form = QFormLayout(group)

            visible = QCheckBox(group)
            form.addRow("Show:", visible)

            shape = QComboBox(group)
            shape.addItems(["circle", "square", "cross"])
            form.addRow("Symbol:", shape)

            color_btn = QPushButton("Choose...", group)
            color_preview = QLabel(group)
            color_preview.setFixedSize(34, 18)
            form.addRow("Color:", self._color_row(color_btn, color_preview))

            radius = QSpinBox(group)
            radius.setRange(1, 50)
            form.addRow("Size:", radius)

            label_visible = QCheckBox(group)
            form.addRow("Show label:", label_visible)

            label_text = QLineEdit(group)
            label_text.setPlaceholderText("Leave blank for automatic label")
            form.addRow("Label text:", label_text)

            font = QFontComboBox(group)
            form.addRow("Font:", font)

            font_size = QSpinBox(group)
            font_size.setRange(6, 48)
            form.addRow("Font size:", font_size)

            label_color_btn = QPushButton("Choose...", group)
            label_color_preview = QLabel(group)
            label_color_preview.setFixedSize(34, 18)
            form.addRow(
                "Label color:", self._color_row(label_color_btn, label_color_preview)
            )

            self._widgets[key] = {
                "visible": visible,
                "shape": shape,
                "color_btn": color_btn,
                "color_preview": color_preview,
                "radius": radius,
                "label_visible": label_visible,
                "label_text": label_text,
                "font": font,
                "font_size": font_size,
                "label_color_btn": label_color_btn,
                "label_color_preview": label_color_preview,
                "_color": QColor(DEFAULT_MARKER_CONFIG[key]["color"]),
                "_label_color": QColor(DEFAULT_MARKER_CONFIG[key]["label_color"]),
            }
            color_btn.clicked.connect(
                lambda _=False, k=key: self._choose_color(k, "color")
            )
            label_color_btn.clicked.connect(
                lambda _=False, k=key: self._choose_color(k, "label_color")
            )
            content_layout.addWidget(group)

        content_layout.addStretch(1)
        scroll.setWidget(content)
        layout.addWidget(scroll)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
            | QDialogButtonBox.StandardButton.RestoreDefaults,
            self,
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        reset = buttons.button(QDialogButtonBox.StandardButton.RestoreDefaults)
        if reset:
            reset.clicked.connect(lambda: self.set_config(DEFAULT_MARKER_CONFIG))
        layout.addWidget(buttons)

    def get_config(self) -> dict[str, dict[str, Any]]:
        config: dict[str, dict[str, Any]] = {}
        for key, widgets in self._widgets.items():
            config[key] = {
                "visible": widgets["visible"].isChecked(),
                "shape": widgets["shape"].currentText(),
                "color": widgets["_color"].name(),
                "radius": widgets["radius"].value(),
                "label_visible": widgets["label_visible"].isChecked(),
                "label_text": widgets["label_text"].text(),
                "label_color": widgets["_label_color"].name(),
                "font_family": widgets["font"].currentFont().family(),
                "font_size": widgets["font_size"].value(),
            }
        return config

    def set_config(self, config: dict | None) -> None:
        merged = {
            key: {**DEFAULT_MARKER_CONFIG[key], **((config or {}).get(key, {}) or {})}
            for key, _ in MARKER_TYPES
        }
        for key, values in merged.items():
            widgets = self._widgets[key]
            widgets["visible"].setChecked(bool(values.get("visible", True)))
            widgets["shape"].setCurrentText(str(values.get("shape", "circle")))
            widgets["radius"].setValue(int(values.get("radius", 5)))
            widgets["label_visible"].setChecked(
                bool(values.get("label_visible", False))
            )
            widgets["label_text"].setText(str(values.get("label_text", "")))
            widgets["font_size"].setValue(int(values.get("font_size", 10)))
            font_family = values.get("font_family")
            if font_family:
                widgets["font"].setCurrentFont(QFont(str(font_family)))
                for i in range(widgets["font"].count()):
                    if widgets["font"].itemText(i) == str(font_family):
                        widgets["font"].setCurrentIndex(i)
                        break
            self._set_color(key, QColor(str(values.get("color", "#00ff00"))), "color")
            self._set_color(
                key, QColor(str(values.get("label_color", "#ffffff"))), "label_color"
            )

    def _color_row(self, button: QPushButton, preview: QLabel) -> QWidget:
        row = QWidget(self)
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(button)
        layout.addWidget(preview)
        layout.addStretch(1)
        return row

    def _choose_color(self, marker_key: str, color_key: str) -> None:
        widgets = self._widgets[marker_key]
        initial = (
            widgets["_label_color"] if color_key == "label_color" else widgets["_color"]
        )
        color = QColorDialog.getColor(initial, self)
        if color.isValid():
            self._set_color(marker_key, color, color_key)

    def _set_color(self, marker_key: str, color: QColor, color_key: str) -> None:
        widgets = self._widgets[marker_key]
        store_key = "_label_color" if color_key == "label_color" else "_color"
        preview_key = (
            "label_color_preview" if color_key == "label_color" else "color_preview"
        )
        widgets[store_key] = color
        widgets[preview_key].setStyleSheet(
            f"background-color: {color.name()}; border: 1px solid #222;"
        )
