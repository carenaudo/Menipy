"""Interactive timeline and compact temporal plots for dynamic sessile results."""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import QPointF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)


class TemporalPlotWidget(QWidget):
    """Dependency-free four-band plot synchronized to the selected frame."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.frames: list[Any] = []
        self.index = 0
        self.setMinimumHeight(150)

    def set_data(self, frames: list[Any], index: int = 0) -> None:
        self.frames = frames
        self.index = index
        self.update()

    def set_index(self, index: int) -> None:
        self.index = index
        self.update()

    def _series(self, key: str) -> list[float | None]:
        return [getattr(frame, key, None) for frame in self.frames]

    def paintEvent(self, _event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#ffffff"))
        if len(self.frames) < 2:
            painter.drawText(self.rect(), Qt.AlignCenter, "No temporal series")
            return
        bands = [
            ("Angles", [self._series("theta_left_deg"), self._series("theta_right_deg")], [QColor("#0969DA"), QColor("#8250DF")]),
            ("Contacts", [self._series("half_width_mm")], [QColor("#1A7F37")]),
            ("Velocity", [self._series("contact_velocity_mm_s")], [QColor("#BF8700")]),
        ]
        margin, band_height = 52, max(28, (self.height() - 8) // 4)
        for band_index, (label, series_values, colors) in enumerate(bands):
            top = 4 + band_index * band_height
            bottom = top + band_height - 5
            painter.setPen(QColor("#D0D7DE"))
            painter.drawLine(margin, bottom, self.width() - 4, bottom)
            painter.setPen(QColor("#57606A"))
            painter.drawText(4, top + 14, label)
            finite = [float(value) for values in series_values for value in values if value is not None]
            if not finite:
                continue
            low, high = min(finite), max(finite)
            if high <= low:
                high = low + 1.0
            for values, color in zip(series_values, colors):
                painter.setPen(QPen(color, 1.5))
                previous: QPointF | None = None
                for index, value in enumerate(values):
                    if value is None:
                        previous = None
                        continue
                    x = margin + index * (self.width() - margin - 5) / (len(values) - 1)
                    y = bottom - (float(value) - low) * (band_height - 12) / (high - low)
                    point = QPointF(x, y)
                    if previous is not None:
                        painter.drawLine(previous, point)
                    previous = point
        state_top = 4 + 3 * band_height
        state_colors = {"advancing": QColor("#2DA44E"), "receding": QColor("#CF222E"), "pinned": QColor("#6E7781"), "invalid": QColor("#D0D7DE")}
        painter.drawText(4, state_top + 14, "State")
        cell_width = (self.width() - margin - 5) / len(self.frames)
        for index, frame in enumerate(self.frames):
            painter.fillRect(int(margin + index * cell_width), state_top + 3, max(1, int(cell_width + 1)), 12, state_colors.get(frame.state, QColor("#D0D7DE")))
        cursor_x = margin + self.index * (self.width() - margin - 5) / (len(self.frames) - 1)
        painter.setPen(QPen(QColor("#24292F"), 1, Qt.DashLine))
        painter.drawLine(int(cursor_x), 2, int(cursor_x), self.height() - 2)


class DynamicTimelineWidget(QWidget):
    """Playback controls and synchronized Phase-D diagnostic plots."""

    frame_changed = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.frames: list[Any] = []
        self.fps = 30.0
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 0)
        controls = QHBoxLayout()
        self.play_button = QPushButton("Play")
        self.slider = QSlider(Qt.Horizontal)
        self.label = QLabel("0 / 0")
        controls.addWidget(self.play_button)
        controls.addWidget(self.slider, 1)
        controls.addWidget(self.label)
        layout.addLayout(controls)
        self.plot = TemporalPlotWidget(self)
        layout.addWidget(self.plot)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._advance)
        self.play_button.clicked.connect(self._toggle)
        self.slider.valueChanged.connect(self._select)

    def set_sequence(self, frames: list[Any], fps: float) -> None:
        self.frames = list(frames)
        self.fps = max(1.0, float(fps))
        self.slider.setRange(0, max(0, len(frames) - 1))
        self.slider.setValue(0)
        self.plot.set_data(self.frames)
        self._select(0)

    def _toggle(self) -> None:
        if self.timer.isActive():
            self.timer.stop()
            self.play_button.setText("Play")
        else:
            self.timer.start(max(10, int(round(1000.0 / self.fps))))
            self.play_button.setText("Pause")

    def _advance(self) -> None:
        if not self.frames:
            return
        self.slider.setValue((self.slider.value() + 1) % len(self.frames))

    def _select(self, index: int) -> None:
        total = len(self.frames)
        self.label.setText(f"{index + 1 if total else 0} / {total}")
        self.plot.set_index(index)
        if total:
            self.frame_changed.emit(index)
