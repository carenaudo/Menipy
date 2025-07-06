"""Dialog for interactive calibration using a reference line."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from PySide6.QtCore import Qt, QLineF
from PySide6.QtGui import QImage, QPixmap, QPen, QColor
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QGraphicsScene,
    QGraphicsView,
    QLabel,
    QVBoxLayout,
)

from ..utils import calibrate_from_points


@dataclass
class _Point:
    x: float
    y: float


class CalibrationDialog(QDialog):
    """Interactive dialog to set image scale."""

    def __init__(self, image, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Calibration")
        self.points: list[_Point] = []
        self.line_item = None
        self.pixel_length = 0.0
        self._setup_ui(image)

    def _setup_ui(self, image) -> None:
        layout = QVBoxLayout(self)

        self.view = QGraphicsView()
        self.scene = QGraphicsScene(self)
        self.view.setScene(self.scene)
        layout.addWidget(self.view)

        if image.ndim == 3:
            qimg = QImage(
                image.data,
                image.shape[1],
                image.shape[0],
                image.strides[0],
                QImage.Format_BGR888,
            )
        else:
            qimg = QImage(
                image.data,
                image.shape[1],
                image.shape[0],
                image.strides[0],
                QImage.Format_Grayscale8,
            )
        pixmap = QPixmap.fromImage(qimg)
        self.scene.addPixmap(pixmap)

        self.info_label = QLabel("Click two points to measure reference length.")
        layout.addWidget(self.info_label)

        self.mm_input = QDoubleSpinBox()
        self.mm_input.setSuffix(" mm")
        self.mm_input.setDecimals(3)
        self.mm_input.setValue(1.0)
        layout.addWidget(self.mm_input)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.view.mousePressEvent = self._on_mouse_press  # type: ignore[assignment]

    def _on_mouse_press(self, event):
        pos = self.view.mapToScene(event.pos())
        self.points.append(_Point(pos.x(), pos.y()))
        if len(self.points) == 2:
            if self.line_item:
                self.scene.removeItem(self.line_item)
            line = QLineF(self.points[0].x, self.points[0].y, self.points[1].x, self.points[1].y)
            pen = QPen(QColor("red"))
            pen.setWidth(2)
            self.line_item = self.scene.addLine(line, pen)
            self.pixel_length = line.length()
            self.info_label.setText(f"Pixel length: {self.pixel_length:.1f} px")

    def accept(self) -> None:  # type: ignore[override]
        if len(self.points) == 2 and self.mm_input.value() > 0:
            calibrate_from_points(
                (self.points[0].x, self.points[0].y),
                (self.points[1].x, self.points[1].y),
                self.mm_input.value(),
            )
        super().accept()
