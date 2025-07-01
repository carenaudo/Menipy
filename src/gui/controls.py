from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget,
    QSlider,
    QVBoxLayout,
    QLabel,
    QFormLayout,
    QDoubleSpinBox,
)


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


class ParameterPanel(QWidget):
    """Widget for entering physical parameters."""

    parametersChanged = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QFormLayout(self)
        self.air_density = QDoubleSpinBox()
        self.air_density.setRange(0.0, 10.0)
        self.air_density.setValue(1.2)
        self.air_density.setSuffix(" kg/m³")
        self.air_density.valueChanged.connect(self.parametersChanged.emit)
        layout.addRow("Air density", self.air_density)

        self.liquid_density = QDoubleSpinBox()
        self.liquid_density.setRange(0.0, 5000.0)
        self.liquid_density.setValue(1000.0)
        self.liquid_density.setSuffix(" kg/m³")
        self.liquid_density.valueChanged.connect(self.parametersChanged.emit)
        layout.addRow("Liquid density", self.liquid_density)

        self.surface_tension = QDoubleSpinBox()
        self.surface_tension.setRange(0.0, 500.0)
        self.surface_tension.setValue(72.8)
        self.surface_tension.setSuffix(" mN/m")
        self.surface_tension.valueChanged.connect(self.parametersChanged.emit)
        layout.addRow("Surface tension", self.surface_tension)

    def values(self) -> dict[str, float]:
        return {
            "air_density": self.air_density.value(),
            "liquid_density": self.liquid_density.value(),
            "surface_tension": self.surface_tension.value(),
        }
