from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget,
    QSlider,
    QVBoxLayout,
    QLabel,
    QFormLayout,
    QDoubleSpinBox,
    QCheckBox,
    QPushButton,
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

        self.calibration_mode = QCheckBox("Calibration Mode")
        layout.addRow(self.calibration_mode)

        self.roi_mode = QCheckBox("ROI Mode")
        layout.addRow(self.roi_mode)

        self.ref_length = QDoubleSpinBox()
        self.ref_length.setRange(0.1, 100.0)
        self.ref_length.setValue(1.0)
        self.ref_length.setSuffix(" mm")
        layout.addRow("Ref length", self.ref_length)

        self.manual_toggle = QCheckBox("Manual Calibration")
        self.manual_toggle.setChecked(True)
        layout.addRow(self.manual_toggle)

        self.calibrate_button = QPushButton("Calibrate")
        layout.addRow(self.calibrate_button)

        self.scale_label = QLabel("1.0")
        layout.addRow("Scale (px/mm)", self.scale_label)

    def values(self) -> dict[str, float]:
        return {
            "air_density": self.air_density.value(),
            "liquid_density": self.liquid_density.value(),
            "surface_tension": self.surface_tension.value(),
        }

    def is_calibration_enabled(self) -> bool:
        """Return True if calibration mode is checked."""
        return self.calibration_mode.isChecked()

    def is_roi_enabled(self) -> bool:
        """Return True if ROI mode is checked."""
        return self.roi_mode.isChecked()

    def calibration_method(self) -> str:
        return "manual" if self.manual_toggle.isChecked() else "automatic"

    def calibration_length(self) -> float:
        return self.ref_length.value()

    def set_scale_display(self, px_per_mm: float) -> None:
        self.scale_label.setText(f"{px_per_mm:.2f}")


class MetricsPanel(QWidget):
    """Display calculated droplet metrics."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QFormLayout(self)

        self.ift_label = QLabel("0.0")
        layout.addRow("IFT (mN/m)", self.ift_label)

        self.wo_label = QLabel("0.0")
        layout.addRow("Wo number", self.wo_label)

        self.volume_label = QLabel("0.0")
        layout.addRow("Volume (\u00b5L)", self.volume_label)

        self.angle_label = QLabel("0.0")
        layout.addRow("Contact angle (\u00b0)", self.angle_label)

        self.height_label = QLabel("0.0")
        layout.addRow("Height", self.height_label)

        self.diameter_label = QLabel("0.0")
        layout.addRow("Diameter", self.diameter_label)

    def set_metrics(
        self,
        *,
        ift: float | None = None,
        wo: float | None = None,
        volume: float | None = None,
        contact_angle: float | None = None,
        height: float | None = None,
        diameter: float | None = None,
    ) -> None:
        """Update displayed metric values."""
        if ift is not None:
            self.ift_label.setText(f"{ift:.2f}")
        if wo is not None:
            self.wo_label.setText(f"{wo:.2f}")
        if volume is not None:
            self.volume_label.setText(f"{volume:.2f}")
        if contact_angle is not None:
            self.angle_label.setText(f"{contact_angle:.2f}")
        if height is not None:
            self.height_label.setText(f"{height:.2f}")
        if diameter is not None:
            self.diameter_label.setText(f"{diameter:.2f}")
