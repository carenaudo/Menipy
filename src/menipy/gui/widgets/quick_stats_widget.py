"""
Quick Stats Widget

Card-style widget displaying the current measurement summary.
Shows contact angles, drop properties, surface tension, and confidence.
"""
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame, QGridLayout
)

from menipy.gui import theme


class QuickStatsWidget(QFrame):
    """
    Card widget showing current measurement results.
    
    Displays:
        - Contact angles (left, right, mean with uncertainty)
    - Drop properties (volume, diameter, height, base width)
    - Surface tension
    - Confidence indicator
    """
    
    def __init__(self, parent=None):
        """Initialize.

        Parameters
        ----------
        parent : type
        Description.
        """
        super().__init__(parent)
        self.setObjectName("quickStatsWidget")
        self._setup_ui()
        self._reset_display()
    
    def _setup_ui(self):
        """Set up the widget UI."""
        self.setStyleSheet(f"""
            QFrame#quickStatsWidget {{
                background-color: {theme.BG_SECONDARY};
                border: 1px solid {theme.BORDER_DEFAULT};
                border-radius: 8px;
            }}
        """)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)
        
        # Header
        header = QLabel("Current Measurement")
        header.setStyleSheet(f"""
            font-size: {theme.FONT_SIZE_LARGE}px;
            font-weight: bold;
            color: {theme.TEXT_PRIMARY};
        """)
        layout.addWidget(header)
        
        # Contact Angles section
        angles_section = self._create_section("📐 Contact Angles")
        angles_grid = QGridLayout()
        angles_grid.setSpacing(4)
        
        self._angle_left_label = self._create_value_label()
        self._angle_right_label = self._create_value_label()
        self._angle_mean_label = self._create_value_label()
        
        angles_grid.addWidget(QLabel("Left:"), 0, 0)
        angles_grid.addWidget(self._angle_left_label, 0, 1)
        angles_grid.addWidget(QLabel("Right:"), 1, 0)
        angles_grid.addWidget(self._angle_right_label, 1, 1)
        angles_grid.addWidget(QLabel("Mean:"), 2, 0)
        angles_grid.addWidget(self._angle_mean_label, 2, 1)
        
        angles_section.layout().addLayout(angles_grid)
        layout.addWidget(angles_section)
        
        # Drop Properties section
        props_section = self._create_section("💧 Drop Properties")
        props_grid = QGridLayout()
        props_grid.setSpacing(4)
        
        self._volume_label = self._create_value_label()
        self._diameter_label = self._create_value_label()
        self._height_label = self._create_value_label()
        self._base_label = self._create_value_label()
        
        props_grid.addWidget(QLabel("Volume:"), 0, 0)
        props_grid.addWidget(self._volume_label, 0, 1)
        props_grid.addWidget(QLabel("Diameter:"), 1, 0)
        props_grid.addWidget(self._diameter_label, 1, 1)
        props_grid.addWidget(QLabel("Height:"), 2, 0)
        props_grid.addWidget(self._height_label, 2, 1)
        props_grid.addWidget(QLabel("Base:"), 3, 0)
        props_grid.addWidget(self._base_label, 3, 1)
        
        props_section.layout().addLayout(props_grid)
        layout.addWidget(props_section)
        
        # Surface Tension section
        st_section = self._create_section("🧪 Surface Tension")
        self._surface_tension_label = self._create_value_label()
        self._surface_tension_label.setStyleSheet(f"""
            color: {theme.TEXT_PRIMARY};
            font-size: {theme.FONT_SIZE_LARGE}px;
            font-weight: bold;
        """)
        st_section.layout().addWidget(self._surface_tension_label)
        layout.addWidget(st_section)
        
        # Confidence indicator
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
        self._angle_left_label.setText("--")
        self._angle_right_label.setText("--")
        self._angle_mean_label.setText("--")
        self._volume_label.setText("--")
        self._diameter_label.setText("--")
        self._height_label.setText("--")
        self._base_label.setText("--")
        self._surface_tension_label.setText("--")
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
    
    def set_results(self, results: dict):
        """
        Set measurement results.
        
        Args:
            results: Dictionary with measurement values:
                - angle_left: Left contact angle (degrees)
                - angle_right: Right contact angle (degrees)
                - angle_mean: Mean contact angle (degrees)
                - angle_uncertainty: Uncertainty (degrees)
                - volume: Drop volume (μL)
                - diameter: Drop diameter (mm)
                - height: Drop height (mm)
                - base_width: Base width (mm)
                - surface_tension: Surface tension (mN/m)
                - confidence: Confidence score (0-100)
        """
        # Contact angles
        angle_left = results.get("angle_left")
        angle_right = results.get("angle_right")
        angle_mean = results.get("angle_mean")
        uncertainty = results.get("angle_uncertainty", 0.5)
        
        if angle_left is not None:
            self._angle_left_label.setText(f"{angle_left:.1f}° ± {uncertainty:.1f}°")
        if angle_right is not None:
            self._angle_right_label.setText(f"{angle_right:.1f}° ± {uncertainty:.1f}°")
        if angle_mean is not None:
            self._angle_mean_label.setText(f"{angle_mean:.1f}° ± {uncertainty:.1f}°")
        
        # Drop properties
        if "volume" in results:
            self._volume_label.setText(f"{results['volume']:.2f} μL")
        if "diameter" in results:
            self._diameter_label.setText(f"{results['diameter']:.2f} mm")
        if "height" in results:
            self._height_label.setText(f"{results['height']:.2f} mm")
        if "base_width" in results:
            self._base_label.setText(f"{results['base_width']:.2f} mm")
        
        # Surface tension
        if "surface_tension" in results:
            self._surface_tension_label.setText(f"γ: {results['surface_tension']:.1f} mN/m")
        
        # Confidence
        self._set_confidence(results.get("confidence"))
    
    def clear(self):
        """Clear all displayed results."""
        self._reset_display()
