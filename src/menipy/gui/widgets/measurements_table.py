"""
Measurements Table Widget

A table to display analysis results.
"""
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QTableWidget, QTableWidgetItem, QHeaderView
)

from menipy.gui import theme


class MeasurementsTableWidget(QWidget):
    """Table widget for displaying measurement results."""
    
    def __init__(self, parent=None):
        """Initialize.

        Parameters
        ----------
        parent : type
        Description.
        """
        super().__init__(parent)
        self._setup_ui()
        
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        self._table = QTableWidget()
        self._table.setColumnCount(9)
        self._table.setHorizontalHeaderLabels([
            "ID", "θ_L (°)", "θ_R (°)", "θ_M (°)",
            "Vol (µL)", "Area (mm²)", "Radius (mm)", "Height (mm)", "Fit Error"
        ])
        
        # Styling
        self._table.setStyleSheet(f"""
            QTableWidget {{
                background-color: {theme.BG_TERTIARY};
                gridline-color: {theme.BORDER_DEFAULT};
                border: 1px solid {theme.BORDER_DEFAULT};
                color: {theme.TEXT_PRIMARY};
            }}
            QHeaderView::section {{
                background-color: {theme.BG_SECONDARY};
                color: {theme.TEXT_PRIMARY};
                padding: 4px;
                border: 1px solid {theme.BORDER_DEFAULT};
            }}
            QTableWidget::item {{
                padding: 4px;
            }}
            QTableWidget::item:selected {{
                background-color: {theme.ACCENT_BLUE};
                color: white;
            }}
        """)
        
        # Adjust headers
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setStretchLastSection(True)
        
        self._table.verticalHeader().setVisible(False)
        self._table.setAlternatingRowColors(True)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        
        layout.addWidget(self._table)
        
    def add_result(self, data: dict):
        """Add a result row to the table."""
        row_idx = self._table.rowCount()
        self._table.insertRow(row_idx)
        
        # Extract values with safe defaults
        # Assuming data keys from simulate_analysis or real pipeline
        
        # ID/Frame
        frame_id = str(data.get("frame_index", row_idx + 1))
        
        # Angles
        th_l = f"{data.get('angle_left', 0):.1f}"
        th_r = f"{data.get('angle_right', 0):.1f}"
        th_m = f"{data.get('angle_mean', 0):.1f}"
        
        # Geometry
        vol = f"{data.get('volume', 0):.3f}"
        area = f"{data.get('surface_area', 0):.2f}"
        radius = f"{data.get('contact_radius', 0):.3f}"
        height = f"{data.get('height', 0):.3f}"
        
        # Error
        error = f"{data.get('fit_error', 0):.4f}"
        
        items = [frame_id, th_l, th_r, th_m, vol, area, radius, height, error]
        
        for col_idx, text in enumerate(items):
            item = QTableWidgetItem(text)
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self._table.setItem(row_idx, col_idx, item)
            
        self._table.scrollToBottom()
        
    def clear(self):
        """Clear table."""
        self._table.setRowCount(0)
