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
    QFrame,
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
        if radius is not None:
            self.radius_apex_label.setText(f"{radius:.2f}")
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
        self.radius_apex_label = QLabel("0.0")
        layout.addRow("Radius–Apex (mm)", self.radius_apex_label)
        self.volume_label = QLabel("0.0")
        layout.addRow("Volume (\u00b5L)", self.volume_label)
        self.angle_label = QLabel("0.0")
        layout.addRow("Contact angle (\u00b0)", self.angle_label)
        self.gamma_label = QLabel("0.0")
        layout.addRow("Surface tension (mN/m)", self.gamma_label)
        self.beta_label = QLabel("0.0")
        layout.addRow("Beta", self.beta_label)
        self.s1_label = QLabel("0.0")
        layout.addRow("s1", self.s1_label)
        self.bo_label = QLabel("0.0")
        layout.addRow("Bond number", self.bo_label)
        self.wo_label = QLabel("0.0")
        layout.addRow("Wo number", self.wo_label)
        self.vmax_label = QLabel("0.0")
        layout.addRow("Vmax (\u00b5L)", self.vmax_label)
        self.kappa0_label = QLabel("0.0")
        layout.addRow("\u03ba₀ (1/m)", self.kappa0_label)
        self.aproj_label = QLabel("0.0")
        layout.addRow("A_proj (mm²)", self.aproj_label)
        self.asurf_label = QLabel("0.0")
        layout.addRow("A_surf (mm²)", self.asurf_label)
        self.wapp_label = QLabel("0.0")
        layout.addRow("W_app (mN)", self.wapp_label)

    def set_metrics(
        self,
        *,
        scale: float | None = None,
        height: float | None = None,
        diameter: float | None = None,
        volume: float | None = None,
        angle: float | None = None,
        gamma: float | None = None,
        beta: float | None = None,
        s1: float | None = None,
        bo: float | None = None,
        wo: float | None = None,
        vmax: float | None = None,
        kappa0: float | None = None,
        aproj: float | None = None,
        asurf: float | None = None,
        wapp: float | None = None,
        radius: float | None = None,
    ) -> None:
        """Update displayed metric values."""
        if scale is not None:
            self.scale_label.setText(f"{scale:.2f}")
        if height is not None:
            self.height_label.setText(f"{height:.2f}")
        if diameter is not None:
            self.diameter_label.setText(f"{diameter:.2f}")
        if radius is not None:
            self.radius_apex_label.setText(f"{radius:.2f}")
        if volume is not None:
            self.volume_label.setText(f"{volume:.2f}")
        if angle is not None:
            self.angle_label.setText(f"{angle:.2f}")
        if gamma is not None:
            self.gamma_label.setText(f"{gamma:.2f}")
        if beta is not None:
            self.beta_label.setText(f"{beta:.2f}")
        if s1 is not None:
            self.s1_label.setText(f"{s1:.2f}")
        if bo is not None:
            self.bo_label.setText(f"{bo:.2f}")
        if wo is not None:
            color = "orange" if wo > 0.9 else "black"
            self.wo_label.setText(f"<span style='color:{color}'>{wo:.2f}</span>")
        if vmax is not None:
            self.vmax_label.setText(f"{vmax:.2f}")
        if kappa0 is not None:
            self.kappa0_label.setText(f"{kappa0:.2e}")
        if aproj is not None:
            self.aproj_label.setText(f"{aproj:.2f}")
        if asurf is not None:
            self.asurf_label.setText(f"{asurf:.2f}")
        if wapp is not None:
            self.wapp_label.setText(f"{wapp:.2f}")

    def metrics(self) -> dict[str, str]:
        return {
            "scale": self.scale_label.text(),
            "height": self.height_label.text(),
            "diameter": self.diameter_label.text(),
            "radius": self.radius_apex_label.text(),
            "volume": self.volume_label.text(),
            "angle": self.angle_label.text(),
            "gamma": self.gamma_label.text(),
            "beta": self.beta_label.text(),
            "s1": self.s1_label.text(),
            "bo": self.bo_label.text(),
            "wo": self.wo_label.text(),
            "vmax": self.vmax_label.text(),
            "kappa0": self.kappa0_label.text(),
            "aproj": self.aproj_label.text(),
            "asurf": self.asurf_label.text(),
            "wapp": self.wapp_label.text(),
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


class CalibrationTab(QWidget):
    """Inputs and ROI controls for calibration and fluid properties."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QFormLayout(self)

        self.needle_length = QDoubleSpinBox()
        self.needle_length.setRange(0.1, 1000.0)
        self.needle_length.setValue(1.0)
        self.needle_length.setSuffix(" mm")
        layout.addRow("Needle length", self.needle_length)

        self.needle_region_button = QPushButton("Needle Region")
        layout.addRow(self.needle_region_button)
        self.needle_coords = QLabel("")
        layout.addRow("Needle ROI", self.needle_coords)

        self.detect_needle_button = QPushButton("Detect Needle")
        layout.addRow(self.detect_needle_button)

        self.scale_label = QLabel("0.0")
        layout.addRow("Scale (px/mm)", self.scale_label)

        self.drop_region_button = QPushButton("Region of Interest")
        layout.addRow(self.drop_region_button)
        self.drop_coords = QLabel("")
        layout.addRow("Drop ROI", self.drop_coords)

        self.air_density = QDoubleSpinBox()
        self.air_density.setRange(0.0, 5000.0)
        self.air_density.setValue(1.2)
        self.air_density.setSuffix(" kg/m³")
        layout.addRow("Continuous phase (kg/m³)", self.air_density)

        self.liquid_density = QDoubleSpinBox()
        self.liquid_density.setRange(0.0, 5000.0)
        self.liquid_density.setValue(1000.0)
        self.liquid_density.setSuffix(" kg/m³")
        layout.addRow("Drop phase (kg/m³)", self.liquid_density)

    def set_metrics(self, *, scale: float | None = None) -> None:
        if scale is not None:
            self.scale_label.setText(f"{scale:.2f}")

    def set_regions(
        self,
        *,
        needle: tuple[float, float, float, float] | None = None,
        drop: tuple[float, float, float, float] | None = None,
    ) -> None:
        if needle is not None:
            n_str = ",".join(f"{int(v)}" for v in needle)
            self.needle_coords.setText(n_str)
        if drop is not None:
            d_str = ",".join(f"{int(v)}" for v in drop)
            self.drop_coords.setText(d_str)

    def regions(self) -> dict[str, str]:
        return {
            "needle": self.needle_coords.text(),
            "drop": self.drop_coords.text(),
        }


class AnalysisTab(QWidget):
    """Analysis results and controls for a drop method."""

    def __init__(self, show_contact_angle: bool = False, parent=None) -> None:
        super().__init__(parent)
        self.show_contact_angle = show_contact_angle
        layout = QFormLayout(self)

        if show_contact_angle:
            self.substrate_button = QPushButton("Draw Substrate Line")
            layout.addRow(self.substrate_button)
        else:
            self.substrate_button = None

        self.analyze_button = QPushButton("Analyze")
        layout.addRow(self.analyze_button)

        sep1 = QFrame()
        sep1.setFrameShape(QFrame.HLine)
        layout.addRow(sep1)

        self.height_label = QLabel("0.0")
        layout.addRow("Height (mm)", self.height_label)
        self.diameter_label = QLabel("0.0")
        layout.addRow("Max diameter (mm)", self.diameter_label)
        self.apex_label = QLabel("(0,0)")
        layout.addRow("Apex (x,y)", self.apex_label)
        self.radius_apex_label = QLabel("0.0")
        layout.addRow("Apex radius (mm)", self.radius_apex_label)

        sep2 = QFrame()
        sep2.setFrameShape(QFrame.HLine)
        layout.addRow(sep2)

        if show_contact_angle:
            self.angle_label = QLabel("0.0")
            layout.addRow("Contact angle (º)", self.angle_label)
            self.width_label = QLabel("0.0")
            layout.addRow("Base width (mm)", self.width_label)
            self.rb_label = QLabel("0.0")
            layout.addRow("Base radius (mm)", self.rb_label)
            self.h_label = QLabel("0.0")
            layout.addRow("Apex height (mm)", self.h_label)
        else:
            self.angle_label = None
            self.width_label = None
            self.rb_label = None
            self.h_label = None

        self.volume_label = QLabel("0.0")
        layout.addRow("Volume (uL)", self.volume_label)
        self.asurf_label = QLabel("0.0")
        layout.addRow("Surface Area (mm²)", self.asurf_label)
        self.gamma_label = QLabel("0.0")
        layout.addRow("Surface Tension (mN/m)", self.gamma_label)
        self.wo_label = QLabel("0.0")
        layout.addRow("Wo Number", self.wo_label)
        self.s1_label = QLabel("0.0")
        layout.addRow("S1", self.s1_label)
        self.bo_label = QLabel("0.0")
        layout.addRow("Bond Number", self.bo_label)
        self.aproj_label = QLabel("0.0")
        layout.addRow("Aproj (mm²)", self.aproj_label)
        self.vmax_label = QLabel("0.0")
        layout.addRow("Vmax (uL)", self.vmax_label)
        self.wapp_label = QLabel("0.0")
        layout.addRow("W_app (mN)", self.wapp_label)
        self.kappa0_label = QLabel("0.0")
        layout.addRow("ko (1/m)", self.kappa0_label)

    def set_metrics(
        self,
        *,
        height: float | None = None,
        diameter: float | None = None,
        apex: tuple[int, int] | None = None,
        radius: float | None = None,
        volume: float | None = None,
        angle: float | None = None,
        gamma: float | None = None,
        s1: float | None = None,
        bo: float | None = None,
        wo: float | None = None,
        aproj: float | None = None,
        asurf: float | None = None,
        vmax: float | None = None,
        wapp: float | None = None,
        kappa0: float | None = None,
        width: float | None = None,
        rbase: float | None = None,
        height_line: float | None = None,
    ) -> None:
        if height is not None:
            self.height_label.setText(f"{height:.2f}")
        if diameter is not None:
            self.diameter_label.setText(f"{diameter:.2f}")
        if apex is not None:
            self.apex_label.setText(f"({apex[0]},{apex[1]})")
        if radius is not None:
            self.radius_apex_label.setText(f"{radius:.2f}")
        if volume is not None:
            self.volume_label.setText(f"{volume:.2f}")
        if angle is not None and self.angle_label is not None:
            self.angle_label.setText(f"{angle:.2f}")
        if gamma is not None:
            self.gamma_label.setText(f"{gamma:.2f}")
        if s1 is not None:
            self.s1_label.setText(f"{s1:.2f}")
        if bo is not None:
            self.bo_label.setText(f"{bo:.2f}")
        if wo is not None:
            self.wo_label.setText(f"{wo:.2f}")
        if aproj is not None:
            self.aproj_label.setText(f"{aproj:.2f}")
        if asurf is not None:
            self.asurf_label.setText(f"{asurf:.2f}")
        if vmax is not None:
            self.vmax_label.setText(f"{vmax:.2f}")
        if wapp is not None:
            self.wapp_label.setText(f"{wapp:.2f}")
        if kappa0 is not None:
            self.kappa0_label.setText(f"{kappa0:.2f}")
        if width is not None and self.width_label is not None:
            self.width_label.setText(f"{width:.2f}")
        if rbase is not None and self.rb_label is not None:
            self.rb_label.setText(f"{rbase:.2f}")
        if height_line is not None and self.h_label is not None:
            self.h_label.setText(f"{height_line:.2f}")

    def metrics(self) -> dict[str, str]:
        data = {
            "height": self.height_label.text(),
            "diameter": self.diameter_label.text(),
            "apex": self.apex_label.text(),
            "radius": self.radius_apex_label.text(),
            "volume": self.volume_label.text(),
            "gamma": self.gamma_label.text(),
            "s1": self.s1_label.text(),
            "bo": self.bo_label.text(),
            "wo": self.wo_label.text(),
            "aproj": self.aproj_label.text(),
            "asurf": self.asurf_label.text(),
            "vmax": self.vmax_label.text(),
            "wapp": self.wapp_label.text(),
            "kappa0": self.kappa0_label.text(),
        }
        if self.angle_label is not None:
            data["angle"] = self.angle_label.text()
            data["width"] = self.width_label.text()
            data["rbase"] = self.rb_label.text()
            data["height_line"] = self.h_label.text()
        return data

