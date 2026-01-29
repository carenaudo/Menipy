"""
Tilted Sessile Results Widget

Widget displaying tilted sessile specific results including
advancing and receding contact angles, hysteresis, and roll-off angle.
"""
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame, QGridLayout
)

from menipy.gui import theme


class TiltedSessileResultsWidget(QFrame):
    """
    Widget showing tilted sessile measurement results.
    
    Displays:
    - Advancing contact angle (θA)
    - Receding contact angle (θR)
    - Contact angle hysteresis (θA - θR)
    - Roll-off angle
    - Tilt angle
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("tiltedSessileResultsWidget")
        self._setup_ui()
        self._reset_display()
    
    def _setup_ui(self):
        """Set up the widget UI."""
        self.setStyleSheet(f"""
            QFrame#tiltedSessileResultsWidget {{
                background-color: {theme.BG_SECONDARY};
                border: 1px solid {theme.BORDER_DEFAULT};
                border-radius: 8px;
            }}
        """)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)
        
        # Header
        header = QLabel("Measurement Results")
        header.setStyleSheet(f"""
            font-size: {theme.FONT_SIZE_LARGE}px;
            font-weight: bold;
            color: {theme.TEXT_PRIMARY};
        """)
        layout.addWidget(header)
        
        # Current tilt angle
        tilt_display = QFrame()
        tilt_display.setStyleSheet(f"""
            background-color: {theme.BG_TERTIARY};
            border-radius: 8px;
        """)
        tilt_layout = QVBoxLayout(tilt_display)
        tilt_layout.setContentsMargins(12, 8, 12, 8)
        tilt_layout.setSpacing(2)
        
        tilt_title = QLabel("Current Tilt Angle")
        tilt_title.setStyleSheet(f"color: {theme.TEXT_SECONDARY};")
        tilt_layout.addWidget(tilt_title)
        
        self._tilt_angle_label = QLabel("0.0°")
        self._tilt_angle_label.setStyleSheet(f"""
            color: {theme.TEXT_PRIMARY};
            font-size: 24px;
            font-weight: bold;
        """)
        tilt_layout.addWidget(self._tilt_angle_label)
        
        layout.addWidget(tilt_display)
        
        # Advancing angle section
        advancing_section = QFrame()
        advancing_section.setStyleSheet(f"""
            QFrame {{
                background-color: {theme.OVERLAY_ADVANCING};
                border-radius: 8px;
                padding: 8px;
            }}
        """)
        adv_layout = QVBoxLayout(advancing_section)
        adv_layout.setContentsMargins(12, 8, 12, 8)
        adv_layout.setSpacing(4)
        
        adv_title = QLabel("Advancing Angle (θA)")
        adv_title.setStyleSheet("color: rgba(0,0,0,0.7); font-weight: bold;")
        adv_layout.addWidget(adv_title)
        
        self._advancing_label = QLabel("--")
        self._advancing_label.setStyleSheet("""
            color: rgba(0,0,0,0.9);
            font-size: 20px;
            font-weight: bold;
        """)
        adv_layout.addWidget(self._advancing_label)
        
        layout.addWidget(advancing_section)
        
        # Receding angle section
        receding_section = QFrame()
        receding_section.setStyleSheet(f"""
            QFrame {{
                background-color: {theme.OVERLAY_RECEDING};
                border-radius: 8px;
                padding: 8px;
            }}
        """)
        rec_layout = QVBoxLayout(receding_section)
        rec_layout.setContentsMargins(12, 8, 12, 8)
        rec_layout.setSpacing(4)
        
        rec_title = QLabel("Receding Angle (θR)")
        rec_title.setStyleSheet("color: rgba(0,0,0,0.7); font-weight: bold;")
        rec_layout.addWidget(rec_title)
        
        self._receding_label = QLabel("--")
        self._receding_label.setStyleSheet("""
            color: rgba(0,0,0,0.9);
            font-size: 20px;
            font-weight: bold;
        """)
        rec_layout.addWidget(self._receding_label)
        
        layout.addWidget(receding_section)
        
        # Hysteresis section
        hyst_section = self._create_section("📊 Hysteresis & Roll-off")
        hyst_grid = QGridLayout()
        hyst_grid.setSpacing(8)
        
        self._hysteresis_label = self._create_value_label()
        self._rolloff_label = self._create_value_label()
        
        hyst_grid.addWidget(QLabel("Hysteresis (θA - θR):"), 0, 0)
        hyst_grid.addWidget(self._hysteresis_label, 0, 1)
        hyst_grid.addWidget(QLabel("Roll-off Angle:"), 1, 0)
        hyst_grid.addWidget(self._rolloff_label, 1, 1)
        
        hyst_section.layout().addLayout(hyst_grid)
        layout.addWidget(hyst_section)
        
        # Drop properties section
        props_section = self._create_section("💧 Drop Properties")
        props_grid = QGridLayout()
        props_grid.setSpacing(8)
        
        self._volume_label = self._create_value_label()
        self._base_left_label = self._create_value_label()
        self._base_right_label = self._create_value_label()
        
        props_grid.addWidget(QLabel("Volume:"), 0, 0)
        props_grid.addWidget(self._volume_label, 0, 1)
        props_grid.addWidget(QLabel("Base (left):"), 1, 0)
        props_grid.addWidget(self._base_left_label, 1, 1)
        props_grid.addWidget(QLabel("Base (right):"), 2, 0)
        props_grid.addWidget(self._base_right_label, 2, 1)
        
        props_section.layout().addLayout(props_grid)
        layout.addWidget(props_section)
        
        # Confidence
        self._confidence_label = QLabel()
        self._confidence_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._confidence_label)
        
        layout.addStretch()
    
    def _create_section(self, title: str) -> QFrame:
        """Create a section with a title."""
        section = QFrame()
        section.setStyleSheet(f"""
            QFrame {{
                background-color: {theme.BG_TERTIARY};
                border-radius: 4px;
                padding: 8px;
            }}
        """)
        
        layout = QVBoxLayout(section)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        
        title_label = QLabel(title)
        title_label.setStyleSheet(f"""
            color: {theme.TEXT_PRIMARY};
            font-weight: bold;
        """)
        layout.addWidget(title_label)
        
        return section
    
    def _create_value_label(self) -> QLabel:
        """Create a styled value label."""
        label = QLabel("--")
        label.setStyleSheet(f"color: {theme.TEXT_PRIMARY};")
        label.setAlignment(Qt.AlignmentFlag.AlignRight)
        return label
    
    def _reset_display(self):
        """Reset all values to default."""
        self._tilt_angle_label.setText("0.0°")
        self._advancing_label.setText("--")
        self._receding_label.setText("--")
        self._hysteresis_label.setText("--")
        self._rolloff_label.setText("--")
        self._volume_label.setText("--")
        self._base_left_label.setText("--")
        self._base_right_label.setText("--")
        self._set_confidence(None)
    
    def _set_confidence(self, confidence: float | None):
        """Set the confidence indicator."""
        if confidence is None:
            self._confidence_label.setText("")
            return
        
        if confidence >= 90:
            color = theme.SUCCESS_GREEN
            icon = "✓"
            level = "High"
        elif confidence >= 70:
            color = theme.WARNING_ORANGE
            icon = "⚠️"
            level = "Medium"
        else:
            color = theme.ERROR_RED
            icon = "❌"
            level = "Low"
        
        self._confidence_label.setText(f"{icon} Confidence: {level} ({confidence:.0f}%)")
        self._confidence_label.setStyleSheet(f"color: {color}; font-weight: bold;")
    
    # -------------------------------------------------------------------------
    # Public Methods
    # -------------------------------------------------------------------------
    
    def set_tilt_angle(self, angle: float):
        """Set the current tilt angle display."""
        self._tilt_angle_label.setText(f"{angle:.1f}°")
    
    def set_results(self, results: dict):
        """
        Set measurement results.
        
        Args:
            results: Dictionary with measurement values:
                - tilt_angle: Current tilt angle (degrees)
                - advancing_angle: Advancing contact angle (degrees)
                - receding_angle: Receding contact angle (degrees)
                - hysteresis: θA - θR (degrees)
                - rolloff_angle: Roll-off angle (degrees)
                - volume: Drop volume (μL)
                - base_left: Left base length (mm)
                - base_right: Right base length (mm)
                - confidence: Confidence score (0-100)
        """
        if "tilt_angle" in results:
            self.set_tilt_angle(results["tilt_angle"])
        
        if "advancing_angle" in results:
            unc = results.get("advancing_uncertainty", 0.5)
            self._advancing_label.setText(f"{results['advancing_angle']:.1f}° ± {unc:.1f}°")
        
        if "receding_angle" in results:
            unc = results.get("receding_uncertainty", 0.5)
            self._receding_label.setText(f"{results['receding_angle']:.1f}° ± {unc:.1f}°")
        
        if "hysteresis" in results:
            self._hysteresis_label.setText(f"{results['hysteresis']:.1f}°")
        
        if "rolloff_angle" in results:
            self._rolloff_label.setText(f"{results['rolloff_angle']:.1f}°")
        elif results.get("advancing_angle") and results.get("receding_angle"):
            # Calculate hysteresis if not provided
            hyst = results["advancing_angle"] - results["receding_angle"]
            self._hysteresis_label.setText(f"{hyst:.1f}°")
        
        if "volume" in results:
            self._volume_label.setText(f"{results['volume']:.2f} μL")
        
        if "base_left" in results:
            self._base_left_label.setText(f"{results['base_left']:.2f} mm")
        
        if "base_right" in results:
            self._base_right_label.setText(f"{results['base_right']:.2f} mm")
        
        self._set_confidence(results.get("confidence"))
    
    def clear(self):
        """Clear all displayed results."""
        self._reset_display()
