"""
Pendant Results Widget

Widget displaying pendant drop specific results including surface tension,
Bond number, and drop shape parameters.
"""
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame, QGridLayout
)

from menipy.gui import theme


class PendantResultsWidget(QFrame):
    """
    Widget showing pendant drop measurement results.
    
    Displays:
    - Surface/Interfacial tension (primary result)
    - Bond number
    - Drop shape parameters (De, Ds, apex radius)
    - Confidence indicator
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("pendantResultsWidget")
        self._setup_ui()
        self._reset_display()
    
    def _setup_ui(self):
        """Set up the widget UI."""
        self.setStyleSheet(f"""
            QFrame#pendantResultsWidget {{
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
        
        # Primary result - Surface Tension
        st_section = QFrame()
        st_section.setObjectName("surfaceTensionSection")
        st_section.setStyleSheet(f"""
            QFrame#surfaceTensionSection {{
                background-color: {theme.ACCENT_BLUE};
                border-radius: 8px;
                padding: 12px;
            }}
        """)
        st_layout = QVBoxLayout(st_section)
        st_layout.setContentsMargins(12, 12, 12, 12)
        st_layout.setSpacing(4)
        
        st_title = QLabel("Surface Tension")
        # Use simple hex with opacity if needed, or just semi-transparent white
        st_title.setStyleSheet("color: #E6FFFFFF; font-weight: bold;")
        st_layout.addWidget(st_title)
        
        self._surface_tension_label = QLabel("--")
        self._surface_tension_label.setStyleSheet(f"""
            color: white;
            font-size: 28px;
            font-weight: bold;
        """)
        st_layout.addWidget(self._surface_tension_label)
        
        self._st_uncertainty_label = QLabel("")
        self._st_uncertainty_label.setStyleSheet("color: #B3FFFFFF;")
        st_layout.addWidget(self._st_uncertainty_label)
        
        layout.addWidget(st_section)
        
        # Shape Parameters section
        shape_section = self._create_section("📐 Shape Parameters")
        shape_grid = QGridLayout()
        shape_grid.setSpacing(8)
        
        self._de_label = self._create_value_label()
        self._ds_label = self._create_value_label()
        self._apex_label = self._create_value_label()
        self._bond_label = self._create_value_label()
        
        shape_grid.addWidget(QLabel("De (equator):"), 0, 0)
        shape_grid.addWidget(self._de_label, 0, 1)
        shape_grid.addWidget(QLabel("Ds (at De):"), 1, 0)
        shape_grid.addWidget(self._ds_label, 1, 1)
        shape_grid.addWidget(QLabel("Apex Radius:"), 2, 0)
        shape_grid.addWidget(self._apex_label, 2, 1)
        shape_grid.addWidget(QLabel("Bond Number:"), 3, 0)
        shape_grid.addWidget(self._bond_label, 3, 1)
        
        shape_section.layout().addLayout(shape_grid)
        layout.addWidget(shape_section)
        
        # Physical Properties section
        props_section = self._create_section("🧪 Physical Properties")
        props_grid = QGridLayout()
        props_grid.setSpacing(8)
        
        self._volume_label = self._create_value_label()
        self._area_label = self._create_value_label()
        self._density_diff_label = self._create_value_label()
        
        props_grid.addWidget(QLabel("Drop Volume:"), 0, 0)
        props_grid.addWidget(self._volume_label, 0, 1)
        props_grid.addWidget(QLabel("Surface Area:"), 1, 0)
        props_grid.addWidget(self._area_label, 1, 1)
        props_grid.addWidget(QLabel("Δρ:"), 2, 0)
        props_grid.addWidget(self._density_diff_label, 2, 1)
        
        props_section.layout().addLayout(props_grid)
        layout.addWidget(props_section)
        
        # Fit Quality section
        fit_section = self._create_section("📊 Fit Quality")
        fit_grid = QGridLayout()
        fit_grid.setSpacing(8)
        
        self._rmse_label = self._create_value_label()
        self._iterations_label = self._create_value_label()
        
        fit_grid.addWidget(QLabel("RMSE:"), 0, 0)
        fit_grid.addWidget(self._rmse_label, 0, 1)
        fit_grid.addWidget(QLabel("Iterations:"), 1, 0)
        fit_grid.addWidget(self._iterations_label, 1, 1)
        
        fit_section.layout().addLayout(fit_grid)
        layout.addWidget(fit_section)
        
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
        self._surface_tension_label.setText("--")
        self._st_uncertainty_label.setText("")
        self._de_label.setText("--")
        self._ds_label.setText("--")
        self._apex_label.setText("--")
        self._bond_label.setText("--")
        self._volume_label.setText("--")
        self._area_label.setText("--")
        self._density_diff_label.setText("--")
        self._rmse_label.setText("--")
        self._iterations_label.setText("--")
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
        
        self._confidence_label.setText(f"{icon} Fit Confidence: {level} ({confidence:.0f}%)")
        self._confidence_label.setStyleSheet(f"color: {color}; font-weight: bold;")
    
    # -------------------------------------------------------------------------
    # Public Methods
    # -------------------------------------------------------------------------
    
    def set_results(self, results: dict):
        """
        Set measurement results.
        
        Args:
            results: Dictionary with measurement values:
                - surface_tension: Surface tension (mN/m)
                - surface_tension_uncertainty: Uncertainty (mN/m)
                - de: Equatorial diameter (mm)
                - ds: Diameter at de from apex (mm)
                - apex_radius: Apex radius of curvature (mm)
                - bond_number: Bond number (dimensionless)
                - volume: Drop volume (μL)
                - surface_area: Surface area (mm²)
                - density_diff: Density difference (kg/m³)
                - rmse: Root mean square error of fit
                - iterations: Number of fit iterations
                - confidence: Confidence score (0-100)
        """
        # Surface tension
        if "surface_tension" in results:
            st = results["surface_tension"]
            self._surface_tension_label.setText(f"γ = {st:.2f} mN/m")
            
            if "surface_tension_uncertainty" in results:
                unc = results["surface_tension_uncertainty"]
                self._st_uncertainty_label.setText(f"± {unc:.2f} mN/m")
        
        # Shape parameters
        if "de" in results:
            self._de_label.setText(f"{results['de']:.3f} mm")
        if "ds" in results:
            self._ds_label.setText(f"{results['ds']:.3f} mm")
        if "apex_radius" in results:
            self._apex_label.setText(f"{results['apex_radius']:.3f} mm")
        if "bond_number" in results:
            self._bond_label.setText(f"{results['bond_number']:.4f}")
        
        # Physical properties
        if "volume" in results:
            self._volume_label.setText(f"{results['volume']:.2f} μL")
        if "surface_area" in results:
            self._area_label.setText(f"{results['surface_area']:.2f} mm²")
        if "density_diff" in results:
            self._density_diff_label.setText(f"{results['density_diff']:.1f} kg/m³")
        
        # Fit quality
        if "rmse" in results:
            self._rmse_label.setText(f"{results['rmse']:.4f}")
        if "iterations" in results:
            self._iterations_label.setText(f"{results['iterations']}")
        
        # Confidence
        self._set_confidence(results.get("confidence"))
    
    def clear(self):
        """Clear all displayed results."""
        self._reset_display()
