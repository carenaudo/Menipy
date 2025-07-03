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
    QComboBox,
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

        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["pendant", "sessile"])
        layout.addRow("Detection Mode", self.mode_combo)

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

    def detection_mode(self) -> str:
        """Return the selected droplet detection mode."""
        return self.mode_combo.currentText()

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

        self.mode_label = QLabel("unknown")
        layout.addRow("Mode", self.mode_label)

    def set_metrics(
        self,
        *,
        ift: float | None = None,
        wo: float | None = None,
        volume: float | None = None,
        contact_angle: float | None = None,
        height: float | None = None,
        diameter: float | None = None,
        mode: str | None = None,
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
        if mode is not None:
            self.mode_label.setText(mode)

    def values(self) -> dict[str, float]:
        """Return the currently displayed metric values."""
        def _to_float(label: QLabel) -> float:
            try:
                return float(label.text())
            except ValueError:
                return 0.0

        return {
            "ift": _to_float(self.ift_label),
            "wo": _to_float(self.wo_label),
            "volume": _to_float(self.volume_label),
            "contact_angle": _to_float(self.angle_label),
            "height": _to_float(self.height_label),
            "diameter": _to_float(self.diameter_label),
            "mode": self.mode_label.text(),
        }


class DropAnalysisPanel(QWidget):
    """Controls for drop analysis workflow."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QFormLayout(self)

        self.method_combo = QComboBox()
        self.method_combo.addItems(["pendant", "contact-angle"])
        layout.addRow("Method", self.method_combo)

        self.needle_region_button = QPushButton("Needle Region")
        layout.addRow(self.needle_region_button)
        self.detect_needle_button = QPushButton("Detect Needle")
        layout.addRow(self.detect_needle_button)

        self.needle_length = QDoubleSpinBox()
        self.needle_length.setRange(0.1, 1000.0)
        self.needle_length.setValue(1.0)
        self.needle_length.setSuffix(" mm")
        layout.addRow("Needle length", self.needle_length)

        self.drop_region_button = QPushButton("Drop Region")
        layout.addRow(self.drop_region_button)
        self.analyze_button = QPushButton("Analyze Image")
        layout.addRow(self.analyze_button)

        self.needle_coords = QLabel("")
        layout.addRow("Needle ROI", self.needle_coords)
        self.drop_coords = QLabel("")
        layout.addRow("Drop ROI", self.drop_coords)

        self.scale_label = QLabel("0.0")
        layout.addRow("Scale (px/mm)", self.scale_label)
        self.height_label = QLabel("0.0")
        layout.addRow("Height (mm)", self.height_label)
        self.diameter_label = QLabel("0.0")
        layout.addRow("Diameter (mm)", self.diameter_label)
        self.volume_label = QLabel("0.0")
        layout.addRow("Volume (\u00b5L)", self.volume_label)
        self.angle_label = QLabel("0.0")
        layout.addRow("Contact angle (\u00b0)", self.angle_label)
        self.ift_label = QLabel("0.0")
        layout.addRow("IFT (mN/m)", self.ift_label)
        self.wo_label = QLabel("0.0")
        layout.addRow("Wo number", self.wo_label)

    def set_metrics(
        self,
        *,
        scale: float | None = None,
        height: float | None = None,
        diameter: float | None = None,
        volume: float | None = None,
        angle: float | None = None,
        ift: float | None = None,
        wo: float | None = None,
    ) -> None:
        """Update displayed metric values."""
        if scale is not None:
            self.scale_label.setText(f"{scale:.2f}")
        if height is not None:
            self.height_label.setText(f"{height:.2f}")
        if diameter is not None:
            self.diameter_label.setText(f"{diameter:.2f}")
        if volume is not None:
            self.volume_label.setText(f"{volume:.2f}")
        if angle is not None:
            self.angle_label.setText(f"{angle:.2f}")
        if ift is not None:
            self.ift_label.setText(f"{ift:.2f}")
        if wo is not None:
            self.wo_label.setText(f"{wo:.2f}")

    def metrics(self) -> dict[str, str]:
        return {
            "scale": self.scale_label.text(),
            "height": self.height_label.text(),
            "diameter": self.diameter_label.text(),
            "volume": self.volume_label.text(),
            "angle": self.angle_label.text(),
            "ift": self.ift_label.text(),
            "wo": self.wo_label.text(),
        }

    def set_regions(
        self, *, needle: tuple[float, float, float, float] | None = None, drop: tuple[float, float, float, float] | None = None
    ) -> None:
        """Display saved ROI coordinates."""
        if needle is not None:
            n_str = ",".join(f"{int(v)}" for v in needle)
            self.needle_coords.setText(n_str)
        if drop is not None:
            d_str = ",".join(f"{int(v)}" for v in drop)
            self.drop_coords.setText(d_str)

    def regions(self) -> dict[str, str]:
        """Return stored ROI coordinates as text."""
        return {
            "needle": self.needle_coords.text(),
            "drop": self.drop_coords.text(),
        }
