"""
Needle Calibration Panel

Panel for calibrating needle diameter for pendant drop experiments.
"""
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame,
    QPushButton, QDoubleSpinBox, QComboBox
)

from menipy.gui import theme


class NeedleCalibrationPanel(QFrame):
    """
    Panel for needle calibration in pendant drop experiments.
    
    Shows:
        - Needle diameter input (with common presets)
    - Detection status
    - Calibration method selection
    
    Signals:
        needle_diameter_changed: Emitted when needle diameter changes.
        auto_detect_requested: User wants to auto-detect needle.
    """
    
    needle_diameter_changed = Signal(float)
    auto_detect_requested = Signal()
    database_requested = Signal()
    
    # Common needle outer diameters (mm)
    NEEDLE_PRESETS = [
        ("Custom", 0.0),
        ("18G (1.27mm)", 1.27),
        ("19G (1.07mm)", 1.07),
        ("20G (0.91mm)", 0.91),
        ("21G (0.82mm)", 0.82),
        ("22G (0.72mm)", 0.72),
        ("23G (0.64mm)", 0.64),
        ("25G (0.51mm)", 0.51),
        ("27G (0.41mm)", 0.41),
        ("30G (0.31mm)", 0.31),
    ]
    
    def __init__(self, parent=None):
        """Initialize.

        Parameters
        ----------
        parent : type
        Description.
        """
        super().__init__(parent)
        self.setObjectName("needleCalibrationPanel")
        
        self._is_detected = False
        self._setup_ui()
    
    def _setup_ui(self):
        """Set up the panel UI."""
        self.setStyleSheet(f"""
            QFrame#needleCalibrationPanel {{
                background-color: {theme.BG_SECONDARY};
                border: 1px solid {theme.BORDER_DEFAULT};
                border-radius: 8px;
            }}
        """)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)
        
        # Header
        header = QLabel("Needle Calibration")
        header.setStyleSheet(f"""
            font-size: {theme.FONT_SIZE_LARGE}px;
            font-weight: bold;
            color: {theme.TEXT_PRIMARY};
        """)
        layout.addWidget(header)
        
        # Status indicator
        self._status_label = QLabel("Status: ⚠️ Not Calibrated")
        self._status_label.setStyleSheet(f"color: {theme.WARNING_ORANGE}; font-weight: bold;")
        layout.addWidget(self._status_label)
        
        # Needle preset dropdown
        preset_container = QWidget()
        preset_layout = QVBoxLayout(preset_container)
        preset_layout.setContentsMargins(0, 0, 0, 0)
        preset_layout.setSpacing(4)
        
        preset_header = QHBoxLayout()
        preset_label = QLabel("Needle Gauge")
        preset_label.setStyleSheet(f"color: {theme.TEXT_SECONDARY};")
        preset_header.addWidget(preset_label)
        
        db_btn = QPushButton("📚 DB")
        db_btn.setProperty("secondary", True)
        db_btn.setFixedSize(50, 24)
        db_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        db_btn.clicked.connect(self.database_requested.emit)
        preset_header.addWidget(db_btn)
        
        preset_layout.addLayout(preset_header)
        
        self._preset_combo = QComboBox()
        for name, _ in self.NEEDLE_PRESETS:
            self._preset_combo.addItem(name)
        self._preset_combo.currentIndexChanged.connect(self._on_preset_changed)
        preset_layout.addWidget(self._preset_combo)
        
        layout.addWidget(preset_container)
        
        # Diameter input
        diameter_container = QWidget()
        diameter_layout = QVBoxLayout(diameter_container)
        diameter_layout.setContentsMargins(0, 0, 0, 0)
        diameter_layout.setSpacing(4)
        
        diameter_label = QLabel("Outer Diameter (mm)")
        diameter_label.setStyleSheet(f"color: {theme.TEXT_SECONDARY};")
        diameter_layout.addWidget(diameter_label)
        
        self._diameter_spin = QDoubleSpinBox()
        self._diameter_spin.setRange(0.1, 5.0)
        self._diameter_spin.setDecimals(3)
        self._diameter_spin.setValue(0.91)  # 20G default
        self._diameter_spin.setSuffix(" mm")
        self._diameter_spin.valueChanged.connect(self._on_diameter_changed)
        diameter_layout.addWidget(self._diameter_spin)
        
        layout.addWidget(diameter_container)
        
        # Detection info (shown when auto-detected)
        self._detection_info = QFrame()
        self._detection_info.setStyleSheet(f"""
            background-color: {theme.BG_TERTIARY};
            border-radius: 4px;
            padding: 8px;
        """)
        detection_layout = QVBoxLayout(self._detection_info)
        detection_layout.setContentsMargins(8, 8, 8, 8)
        detection_layout.setSpacing(4)
        
        self._detected_width_label = QLabel("Detected Width: -- px")
        self._detected_width_label.setStyleSheet(f"color: {theme.TEXT_SECONDARY};")
        detection_layout.addWidget(self._detected_width_label)
        
        self._scale_factor_label = QLabel("Scale Factor: -- px/mm")
        self._scale_factor_label.setStyleSheet(f"color: {theme.TEXT_SECONDARY};")
        detection_layout.addWidget(self._scale_factor_label)
        
        self._detection_info.hide()
        layout.addWidget(self._detection_info)
        
        # Auto-detect button
        self._detect_button = QPushButton("🔍 Auto-Detect Needle")
        self._detect_button.clicked.connect(self.auto_detect_requested.emit)
        layout.addWidget(self._detect_button)
        
        # Apply button
        self._apply_button = QPushButton("✓ Apply Calibration")
        self._apply_button.clicked.connect(self._on_apply)
        layout.addWidget(self._apply_button)
    
    def _on_preset_changed(self, index: int):
        """Handle needle preset selection."""
        if index > 0:  # Skip "Custom"
            _, diameter = self.NEEDLE_PRESETS[index]
            self._diameter_spin.setValue(diameter)
    
    def _on_diameter_changed(self, value: float):
        """Handle diameter value change."""
        # Check if it matches a preset
        self._preset_combo.blockSignals(True)
        matching_preset = 0  # Default to "Custom"
        for i, (_, dia) in enumerate(self.NEEDLE_PRESETS):
            if abs(dia - value) < 0.001:
                matching_preset = i
                break
        self._preset_combo.setCurrentIndex(matching_preset)
        self._preset_combo.blockSignals(False)
    
    def _on_database_requested(self):
        """Handle database button click."""
        from menipy.gui.dialogs.material_dialog import MaterialDialog
        
        dialog = MaterialDialog(self, selection_mode=True, table_type="needles")
        dialog.item_selected.connect(self._on_needle_selected)
        dialog.exec()
        
    def _on_needle_selected(self, data: dict):
        """Handle needle selection from database."""
        if diameter := data.get("outer_diameter"):
             self._diameter_spin.setValue(float(diameter))
             # Keep detection status valid if only diameter changed? 
             # No, if diameter changes, scale factor changes, so it remains "detected" but result updates.
             self._update_status(self._is_detected, float(diameter))

    def _on_apply(self):
        """Handle apply button click."""
        diameter = self._diameter_spin.value()
        self.needle_diameter_changed.emit(diameter)
        self._update_status(True, diameter)
    
    def _update_status(self, calibrated: bool, diameter: float = 0.0):
        """Update the calibration status display."""
        self._is_detected = calibrated
        if calibrated:
            self._status_label.setText(f"Status: ✓ Calibrated ({diameter:.2f}mm)")
            self._status_label.setStyleSheet(f"color: {theme.SUCCESS_GREEN}; font-weight: bold;")
        else:
            self._status_label.setText("Status: ⚠️ Not Calibrated")
            self._status_label.setStyleSheet(f"color: {theme.WARNING_ORANGE}; font-weight: bold;")
    
    # -------------------------------------------------------------------------
    # Public Methods
    # -------------------------------------------------------------------------
    
    def set_detected_needle(self, width_px: float, scale_factor: float):
        """
        Set the auto-detected needle information.
        
        Args:
            width_px: Detected needle width in pixels.
            scale_factor: Calculated scale factor (px/mm).
        """
        self._detected_width_label.setText(f"Detected Width: {width_px:.1f} px")
        self._scale_factor_label.setText(f"Scale Factor: {scale_factor:.1f} px/mm")
        self._detection_info.show()
        self._update_status(True, self._diameter_spin.value())
    
    def get_diameter(self) -> float:
        """Get the current needle diameter value."""
        return self._diameter_spin.value()
    
    def set_diameter(self, value: float):
        """Set the needle diameter value."""
        self._diameter_spin.setValue(value)
    
    def is_calibrated(self) -> bool:
        """Check if needle is calibrated."""
        return self._is_detected
