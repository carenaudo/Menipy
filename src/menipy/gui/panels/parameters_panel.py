"""
Parameters Panel

Panel for configuring measurement parameters for drop analysis.
Includes density inputs, advanced options, and material database integration.
"""
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame,
    QPushButton, QDoubleSpinBox, QComboBox, QCheckBox,
    QSlider, QGroupBox
)

from menipy.gui import theme


class ParametersPanel(QFrame):
    """
    Panel for configuring measurement parameters.
    
    For sessile drop analysis:
    - Liquid density
    - Surrounding fluid density
    - Surface tension (optional)
    - Contact angle method
    - Baseline detection method
    - Edge detection sensitivity
    
    Signals:
        parameters_changed: Emitted when any parameter changes.
        material_database_requested: User wants to open material database.
    """
    
    parameters_changed = Signal(dict)
    material_database_requested = Signal(str)  # field name
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("parametersPanel")
        self._setup_ui()
    
    def _setup_ui(self):
        """Set up the panel UI."""
        self.setStyleSheet(f"""
            QFrame#parametersPanel {{
                background-color: {theme.BG_SECONDARY};
                border: 1px solid {theme.BORDER_DEFAULT};
                border-radius: 8px;
            }}
        """)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)
        
        # Header
        header = QLabel("Measurement Parameters")
        header.setStyleSheet(f"""
            font-size: {theme.FONT_SIZE_LARGE}px;
            font-weight: bold;
            color: {theme.TEXT_PRIMARY};
        """)
        layout.addWidget(header)
        
        # Liquid Density
        layout.addWidget(self._create_density_input(
            "Liquid Density (kg/m³)",
            "liquid_density",
            default=1000.0
        ))
        
        # Surrounding Density
        layout.addWidget(self._create_density_input(
            "Surrounding Density (kg/m³)",
            "surrounding_density",
            default=1.2
        ))
        
        # Surface Tension (optional)
        st_container = QWidget()
        st_layout = QVBoxLayout(st_container)
        st_layout.setContentsMargins(0, 0, 0, 0)
        st_layout.setSpacing(4)
        
        st_label_row = QHBoxLayout()
        st_label = QLabel("Surface Tension (mN/m)")
        st_label.setStyleSheet(f"color: {theme.TEXT_SECONDARY};")
        st_label_row.addWidget(st_label)
        st_label_row.addWidget(QLabel("(optional)"))
        st_label_row.addStretch()
        st_layout.addLayout(st_label_row)
        
        self._surface_tension_spin = QDoubleSpinBox()
        self._surface_tension_spin.setRange(0, 1000)
        self._surface_tension_spin.setDecimals(1)
        self._surface_tension_spin.setValue(72.8)
        self._surface_tension_spin.setSpecialValueText("Auto")
        self._surface_tension_spin.valueChanged.connect(self._on_parameter_changed)
        st_layout.addWidget(self._surface_tension_spin)
        
        layout.addWidget(st_container)
        
        # Advanced Options (collapsible)
        self._advanced_checkbox = QCheckBox("▼ Advanced Options")
        self._advanced_checkbox.setStyleSheet(f"""
            QCheckBox {{
                color: {theme.TEXT_PRIMARY};
                font-weight: bold;
            }}
        """)
        self._advanced_checkbox.toggled.connect(self._toggle_advanced)
        layout.addWidget(self._advanced_checkbox)
        
        # Advanced options container
        self._advanced_container = QWidget()
        self._advanced_container.hide()
        advanced_layout = QVBoxLayout(self._advanced_container)
        advanced_layout.setContentsMargins(8, 8, 8, 8)
        advanced_layout.setSpacing(12)
        
        # Contact Angle Method
        method_container = QWidget()
        method_layout = QVBoxLayout(method_container)
        method_layout.setContentsMargins(0, 0, 0, 0)
        method_layout.setSpacing(4)
        
        method_label = QLabel("Contact Angle Method")
        method_label.setStyleSheet(f"color: {theme.TEXT_SECONDARY};")
        method_layout.addWidget(method_label)
        
        self._method_combo = QComboBox()
        self._method_combo.addItems([
            "Tangent Method",
            "Polynomial Fit",
            "Young-Laplace Fit"
        ])
        self._method_combo.currentTextChanged.connect(self._on_parameter_changed)
        method_layout.addWidget(self._method_combo)
        
        advanced_layout.addWidget(method_container)
        
        # Baseline Detection
        baseline_container = QWidget()
        baseline_layout = QVBoxLayout(baseline_container)
        baseline_layout.setContentsMargins(0, 0, 0, 0)
        baseline_layout.setSpacing(4)
        
        baseline_label = QLabel("Baseline Detection")
        baseline_label.setStyleSheet(f"color: {theme.TEXT_SECONDARY};")
        baseline_layout.addWidget(baseline_label)
        
        self._baseline_combo = QComboBox()
        self._baseline_combo.addItems([
            "Automatic",
            "Manual",
            "Three-point"
        ])
        self._baseline_combo.currentTextChanged.connect(self._on_parameter_changed)
        baseline_layout.addWidget(self._baseline_combo)
        
        advanced_layout.addWidget(baseline_container)
        
        # Edge Detection Sensitivity
        edge_container = QWidget()
        edge_layout = QVBoxLayout(edge_container)
        edge_layout.setContentsMargins(0, 0, 0, 0)
        edge_layout.setSpacing(4)
        
        edge_label = QLabel("Edge Detection Sensitivity")
        edge_label.setStyleSheet(f"color: {theme.TEXT_SECONDARY};")
        edge_layout.addWidget(edge_label)
        
        slider_layout = QHBoxLayout()
        slider_layout.addWidget(QLabel("Low"))
        
        self._edge_sensitivity_slider = QSlider(Qt.Orientation.Horizontal)
        self._edge_sensitivity_slider.setRange(1, 100)
        self._edge_sensitivity_slider.setValue(50)
        self._edge_sensitivity_slider.valueChanged.connect(self._on_parameter_changed)
        slider_layout.addWidget(self._edge_sensitivity_slider)
        
        slider_layout.addWidget(QLabel("High"))
        edge_layout.addLayout(slider_layout)
        
        advanced_layout.addWidget(edge_container)
        
        # Correction options
        self._gravity_correction = QCheckBox("Apply gravity correction")
        self._gravity_correction.setChecked(True)
        self._gravity_correction.stateChanged.connect(self._on_parameter_changed)
        advanced_layout.addWidget(self._gravity_correction)
        
        self._symmetrize_profile = QCheckBox("Symmetrize drop profile")
        self._symmetrize_profile.setChecked(True)
        self._symmetrize_profile.stateChanged.connect(self._on_parameter_changed)
        advanced_layout.addWidget(self._symmetrize_profile)
        
        layout.addWidget(self._advanced_container)
        
        layout.addStretch()
    
    def _create_density_input(
        self,
        label_text: str,
        field_name: str,
        default: float = 1000.0
    ) -> QWidget:
        """Create a density input with material database button."""
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        
        label = QLabel(label_text)
        label.setStyleSheet(f"color: {theme.TEXT_SECONDARY};")
        layout.addWidget(label)
        
        row = QHBoxLayout()
        row.setSpacing(8)
        
        spinbox = QDoubleSpinBox()
        spinbox.setRange(0, 50000)
        spinbox.setDecimals(1)
        spinbox.setValue(default)
        spinbox.setSuffix("")
        spinbox.valueChanged.connect(self._on_parameter_changed)
        row.addWidget(spinbox, stretch=1)
        
        # Store reference
        setattr(self, f"_{field_name}_spin", spinbox)
        
        db_button = QPushButton("📚 Database")
        db_button.setProperty("secondary", True)
        db_button.setMaximumWidth(100)
        db_button.clicked.connect(lambda: self.material_database_requested.emit(field_name))
        row.addWidget(db_button)
        
        layout.addLayout(row)
        return container
    
    def _toggle_advanced(self, checked: bool):
        """Toggle advanced options visibility."""
        self._advanced_container.setVisible(checked)
        self._advanced_checkbox.setText("▲ Advanced Options" if checked else "▼ Advanced Options")
    
    def _on_parameter_changed(self):
        """Emit parameters changed signal."""
        params = self.get_parameters()
        self.parameters_changed.emit(params)
    
    # -------------------------------------------------------------------------
    # Public Methods
    # -------------------------------------------------------------------------
    
    def get_parameters(self) -> dict:
        """
        Get all current parameter values.
        
        Returns:
            Dictionary with all parameter values.
        """
        return {
            "liquid_density": self._liquid_density_spin.value(),
            "surrounding_density": self._surrounding_density_spin.value(),
            "surface_tension": self._surface_tension_spin.value(),
            "contact_angle_method": self._method_combo.currentText(),
            "baseline_detection": self._baseline_combo.currentText(),
            "edge_sensitivity": self._edge_sensitivity_slider.value(),
            "gravity_correction": self._gravity_correction.isChecked(),
            "symmetrize_profile": self._symmetrize_profile.isChecked(),
        }
    
    def set_parameters(self, params: dict):
        """
        Set parameter values from a dictionary.
        
        Args:
            params: Dictionary with parameter values.
        """
        if "liquid_density" in params:
            self._liquid_density_spin.setValue(params["liquid_density"])
        if "surrounding_density" in params:
            self._surrounding_density_spin.setValue(params["surrounding_density"])
        if "surface_tension" in params:
            self._surface_tension_spin.setValue(params["surface_tension"])
        if "contact_angle_method" in params:
            self._method_combo.setCurrentText(params["contact_angle_method"])
        if "baseline_detection" in params:
            self._baseline_combo.setCurrentText(params["baseline_detection"])
        if "edge_sensitivity" in params:
            self._edge_sensitivity_slider.setValue(params["edge_sensitivity"])
        if "gravity_correction" in params:
            self._gravity_correction.setChecked(params["gravity_correction"])
        if "symmetrize_profile" in params:
            self._symmetrize_profile.setChecked(params["symmetrize_profile"])
    
    def set_density(self, field_name: str, value: float):
        """
        Set a density value (from material database).
        
        Args:
            field_name: "liquid_density" or "surrounding_density"
            value: Density value in kg/m³
        """
        spinbox = getattr(self, f"_{field_name}_spin", None)
        if spinbox:
            spinbox.setValue(value)
