from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QWidget, QSlider, QVBoxLayout, QLabel


class ZoomControl(QWidget):
    """Widget with a slider to control image zoom."""

    zoomChanged = Signal(float)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        self.label = QLabel("Zoom:")
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(10, 400)  # percent
        self.slider.setValue(100)
        self.slider.valueChanged.connect(self._on_value_changed)
        layout.addWidget(self.label)
        layout.addWidget(self.slider)

    def _on_value_changed(self, value: int) -> None:
        self.zoomChanged.emit(value / 100.0)

    def set_zoom(self, factor: float) -> None:
        self.slider.setValue(int(factor * 100))
