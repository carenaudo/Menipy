"""
Calibration Panel

Panel for managing image calibration (scale factor in px/mm).
Shows calibration status and provides access to the calibration wizard.
"""
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame, QPushButton
)

from menipy.gui import theme


class CalibrationPanel(QFrame):
    """
    Panel displaying calibration status and controls.
    
    Shows:
        - Calibration status (calibrated/not calibrated)
    - Scale factor (px/mm)
    - Reference file info
    - Calibration date
    
    Signals:
        calibration_requested: User wants to start calibration wizard.
        recalibration_requested: User wants to recalibrate.
        details_requested: User wants to view calibration details.
    """
    
    calibration_requested = Signal()
    recalibration_requested = Signal()
    details_requested = Signal()
    
    def __init__(self, parent=None):
        """Initialize.

        Parameters
        ----------
        parent : type
        Description.
        """
        super().__init__(parent)
        self.setObjectName("calibrationPanel")
        
        self._is_calibrated = False
        self._scale_factor = 0.0
        self._reference_file: str | None = None
        self._calibration_date: str | None = None
        
        self._setup_ui()
        self._update_display()
    
    def _setup_ui(self):
        """Set up the panel UI."""
        self.setStyleSheet(f"""
            QFrame#calibrationPanel {{
                background-color: {theme.BG_SECONDARY};
                border: 1px solid {theme.BORDER_DEFAULT};
                border-radius: 8px;
            }}
        """)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)
        
        # Header
        header = QLabel("Calibration")
        header.setStyleSheet(f"""
            font-size: {theme.FONT_SIZE_LARGE}px;
            font-weight: bold;
            color: {theme.TEXT_PRIMARY};
        """)
        layout.addWidget(header)
        
        # Status indicator
        self._status_label = QLabel()
        layout.addWidget(self._status_label)
        
        # Scale factor
        self._scale_label = QLabel()
        self._scale_label.setStyleSheet(f"color: {theme.TEXT_SECONDARY};")
        layout.addWidget(self._scale_label)
        
        # Reference file
        self._reference_label = QLabel()
        self._reference_label.setStyleSheet(f"color: {theme.TEXT_SECONDARY};")
        self._reference_label.setWordWrap(True)
        layout.addWidget(self._reference_label)
        
        # Date
        self._date_label = QLabel()
        self._date_label.setStyleSheet(f"color: {theme.TEXT_SECONDARY};")
        layout.addWidget(self._date_label)
        
        # Buttons container (changes based on calibration state)
        self._buttons_container = QWidget()
        buttons_layout = QHBoxLayout(self._buttons_container)
        buttons_layout.setContentsMargins(0, 8, 0, 0)
        buttons_layout.setSpacing(8)
        
        # Calibration button (shown when not calibrated)
        self._calibrate_button = QPushButton("🎯 Start Calibration Wizard")
        self._calibrate_button.clicked.connect(self.calibration_requested.emit)
        buttons_layout.addWidget(self._calibrate_button)
        
        # Recalibrate button (shown when calibrated)
        self._recalibrate_button = QPushButton("🎯 Recalibrate…")
        self._recalibrate_button.setProperty("secondary", True)
        self._recalibrate_button.clicked.connect(self.recalibration_requested.emit)
        buttons_layout.addWidget(self._recalibrate_button)
        
        # Details button (shown when calibrated)
        self._details_button = QPushButton("ℹ️ Details")
        self._details_button.setProperty("secondary", True)
        self._details_button.clicked.connect(self.details_requested.emit)
        buttons_layout.addWidget(self._details_button)
        
        layout.addWidget(self._buttons_container)
    
    def _update_display(self):
        """Update the display based on current calibration state."""
        if self._is_calibrated:
            self._status_label.setText("Status: ✓ Calibrated")
            self._status_label.setStyleSheet(f"color: {theme.SUCCESS_GREEN}; font-weight: bold;")
            
            self._scale_label.setText(f"Scale Factor: {self._scale_factor:.1f} px/mm")
            self._scale_label.show()
            
            if self._reference_file:
                self._reference_label.setText(f"Reference: {self._reference_file}")
                self._reference_label.show()
            else:
                self._reference_label.hide()
            
            if self._calibration_date:
                self._date_label.setText(f"Date: {self._calibration_date}")
                self._date_label.show()
            else:
                self._date_label.hide()
            
            self._calibrate_button.hide()
            self._recalibrate_button.show()
            self._details_button.show()
        
        else:
            self._status_label.setText("Status: ⚠️ Not Calibrated")
            self._status_label.setStyleSheet(f"color: {theme.WARNING_ORANGE}; font-weight: bold;")
            
            self._scale_label.setText("Calibration is required before\naccurate measurements can be made.")
            self._scale_label.show()
            
            self._reference_label.hide()
            self._date_label.hide()
            
            self._calibrate_button.show()
            self._recalibrate_button.hide()
            self._details_button.hide()
    
    # -------------------------------------------------------------------------
    # Public Methods
    # -------------------------------------------------------------------------
    
    def set_calibration(
        self,
        scale_factor: float,
        reference_file: str | None = None,
        date: str | None = None
    ):
        """
        Set the calibration data.
        
        Args:
            scale_factor: Scale factor in pixels per millimeter.
            reference_file: Name of the reference file used for calibration.
            date: Date string of when calibration was performed.
        """
        self._is_calibrated = True
        self._scale_factor = scale_factor
        self._reference_file = reference_file
        self._calibration_date = date
        self._update_display()
    
    def clear_calibration(self):
        """Clear the current calibration."""
        self._is_calibrated = False
        self._scale_factor = 0.0
        self._reference_file = None
        self._calibration_date = None
        self._update_display()
    
    def is_calibrated(self) -> bool:
        """Check if calibration is set."""
        return self._is_calibrated
    
    def get_scale_factor(self) -> float:
        """Get the current scale factor (px/mm)."""
        return self._scale_factor
