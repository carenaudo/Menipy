"""Results panel helper for Menipy GUI."""
from __future__ import annotations

from typing import Any, Mapping, Optional, List
import csv
from pathlib import Path

from PySide6.QtWidgets import QWidget, QTableWidget, QTableWidgetItem, QVBoxLayout, QHBoxLayout, QPushButton, QComboBox, QLabel
from PySide6.QtCore import Qt

from menipy.models.results import get_results_history, MeasurementResult
from menipy.gui.controllers.pipeline_ui_manager import PipelineUIManager

LABEL_MAP = {
    "diameter_mm": "Diameter (mm)",
    "height_mm": "Height (mm)",
    "volume_uL": "Volume (µL)",
    "surface_tension_mN_m": "Surface Tension (mN/m)",
    "beta": "β (Shape Factor)",
    "s1": "s₁ (Shape Factor)",
    "r0_mm": "Apex Radius (mm)",
    "needle_surface_mm2": "Needle Surface (mm²)",
    "drop_surface_mm2": "Drop Surface (mm²)",
    "contact_angle_deg": "Contact Angle (°)",
    "contact_surface_mm2": "Contact Surface (mm²)",
    "theta_left_deg": "Left Contact Angle (°)",
    "theta_right_deg": "Right Contact Angle (°)",
    "baseline_tilt_deg": "Baseline Tilt (°)",
    "method": "Method",
    "uncertainty_deg": "Angle Uncertainty (°)",
    "uncertainty_left_deg": "Left Angle Uncertainty (°)",
    "uncertainty_right_deg": "Right Angle Uncertainty (°)",
}


class ResultsPanel:
    """Provides a helper wrapper around the results table widget with measurement history support."""

    def __init__(self, panel: QWidget) -> None:
        self.panel = panel
        self.table: Optional[QTableWidget] = panel.findChild(QTableWidget, "resultsTable")
        self.history = get_results_history()
        self.current_pipeline_filter = None
        self.pipeline_ui_manager = PipelineUIManager()

        # Add controls for history management
        self._setup_controls()

        if self.table:
            # Start with empty table - will be populated by update_history()
            self.table.setColumnCount(0)
            self.table.setRowCount(0)

        # Load and display existing history
        self.update_history()

    def _setup_controls(self) -> None:
        """Setup control buttons and filters."""
        # Find existing layout or create new one
        if hasattr(self.panel, 'layout') and self.panel.layout():
            layout = self.panel.layout()
        else:
            layout = QVBoxLayout(self.panel)

        # Create controls layout
        controls_layout = QHBoxLayout()

        # Pipeline filter
        self.pipeline_combo = QComboBox()
        self.pipeline_combo.addItem("All Pipelines", None)
        self.pipeline_combo.addItem("Sessile", "sessile")
        self.pipeline_combo.addItem("Pendant", "pendant")
        self.pipeline_combo.addItem("Oscillating", "oscillating")
        self.pipeline_combo.addItem("Capillary Rise", "capillary_rise")
        self.pipeline_combo.currentIndexChanged.connect(self._on_pipeline_filter_changed)
        controls_layout.addWidget(QLabel("Filter:"))
        controls_layout.addWidget(self.pipeline_combo)

        # Measurement count label
        self.count_label = QLabel("0 measurements")
        controls_layout.addWidget(self.count_label)

        controls_layout.addStretch()

        # Action buttons
        self.clear_button = QPushButton("Clear History")
        self.clear_button.clicked.connect(self._clear_history)
        controls_layout.addWidget(self.clear_button)

        self.export_button = QPushButton("Export CSV")
        self.export_button.clicked.connect(self._export_csv)
        controls_layout.addWidget(self.export_button)

        # Add controls to main layout
        layout.insertLayout(0, controls_layout)

    def _on_pipeline_filter_changed(self) -> None:
        """Handle pipeline filter change."""
        current_data = self.pipeline_combo.currentData()
        self.current_pipeline_filter = current_data
        self.update_history()

    def _clear_history(self) -> None:
        """Clear measurement history."""
        self.history.clear_history()
        self.update_history()

    def _export_csv(self) -> None:
        """Export current table view to CSV."""
        if not self.table or self.table.rowCount() == 0:
            return

        from PySide6.QtWidgets import QFileDialog
        file_path, _ = QFileDialog.getSaveFileName(
            self.panel, "Export Results", "", "CSV Files (*.csv)"
        )
        if not file_path:
            return

        try:
            with open(file_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)

                # Write headers
                headers = []
                for col in range(self.table.columnCount()):
                    header_item = self.table.horizontalHeaderItem(col)
                    headers.append(header_item.text() if header_item else f"Column {col}")
                writer.writerow(headers)

                # Write data
                for row in range(self.table.rowCount()):
                    row_data = []
                    for col in range(self.table.columnCount()):
                        item = self.table.item(row, col)
                        row_data.append(item.text() if item else "")
                    writer.writerow(row_data)

        except Exception as e:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(self.panel, "Export Error", f"Failed to export CSV: {e}")

    def update(self, results: Mapping[str, Any] | None) -> None:
        """Legacy method for single measurement display - now adds to history."""
        if results:
            # Create a measurement result and add to history
            from datetime import datetime
            import uuid

            measurement = MeasurementResult(
                id=f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}",
                timestamp=datetime.now(),
                pipeline="unknown",  # Will be set by pipeline controller
                results=dict(results)
            )
            self.history.add_measurement(measurement)
            self.update_history()

    def update_history(self) -> None:
        """Update table with measurement history."""
        if not self.table:
            return

        headers, rows = self.history.get_table_data(self.current_pipeline_filter)

        # Apply pipeline-specific column prioritization if filtering by pipeline
        if self.current_pipeline_filter:
            headers, rows = self._prioritize_columns_for_pipeline(headers, rows, self.current_pipeline_filter)

        # Update table
        self.table.setColumnCount(len(headers))
        self.table.setRowCount(len(rows))
        self.table.setHorizontalHeaderLabels(headers)

        # Populate data with pipeline-aware styling
        for row_idx, row_data in enumerate(rows):
            for col_idx, cell_value in enumerate(row_data):
                item = QTableWidgetItem(str(cell_value))
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)  # Make read-only

                # Add pipeline-specific styling
                if self.current_pipeline_filter:
                    self._apply_pipeline_styling(item, self.current_pipeline_filter, headers[col_idx])

                self.table.setItem(row_idx, col_idx, item)

        # Update count label
        measurement_count = len(rows)
        filter_text = f" ({self.pipeline_combo.currentText()})" if self.current_pipeline_filter else ""
        self.count_label.setText(f"{measurement_count} measurements{filter_text}")

        self.table.resizeColumnsToContents()

    def _prioritize_columns_for_pipeline(self, headers: List[str], rows: List[List[Any]], pipeline_name: str) -> tuple[List[str], List[List[Any]]]:
        """Reorder columns to prioritize pipeline-specific metrics."""
        if not pipeline_name:
            return headers, rows

        # Get primary metrics for this pipeline
        primary_metrics = self.pipeline_ui_manager.get_primary_metrics(pipeline_name)

        # Define column priority order based on pipeline
        priority_columns = []

        # Always include timestamp and file info first
        if "Timestamp" in headers:
            priority_columns.append("Timestamp")
        if "File" in headers:
            priority_columns.append("File")
        if "Pipeline" in headers:
            priority_columns.append("Pipeline")

        # Add primary metrics for this pipeline
        for metric in primary_metrics:
            label = LABEL_MAP.get(metric, metric.replace("_", " ").title())
            if label in headers:
                priority_columns.append(label)

        # Add remaining columns
        for header in headers:
            if header not in priority_columns:
                priority_columns.append(header)

        # Reorder headers and data
        if priority_columns != headers:
            header_indices = [headers.index(col) for col in priority_columns if col in headers]
            reordered_headers = [headers[i] for i in header_indices]
            reordered_rows = []
            for row in rows:
                reordered_row = [row[i] for i in header_indices]
                reordered_rows.append(reordered_row)
            return reordered_headers, reordered_rows

        return headers, rows

    def _apply_pipeline_styling(self, item: QTableWidgetItem, pipeline_name: str, column_name: str) -> None:
        """Apply pipeline-specific styling to table cells."""
        # Get pipeline display info
        display_info = self.pipeline_ui_manager.get_display_info(pipeline_name)

        # Style primary metrics differently
        primary_metrics = self.pipeline_ui_manager.get_primary_metrics(pipeline_name)
        for metric in primary_metrics:
            label = LABEL_MAP.get(metric, metric.replace("_", " ").title())
            if column_name == label:
                # Make primary metrics bold
                font = item.font()
                font.setBold(True)
                item.setFont(font)
                break

    def set_pipeline_filter(self, pipeline_name: str) -> None:
        """Set the pipeline filter programmatically."""
        self.current_pipeline_filter = pipeline_name
        # Update combo box selection
        for i in range(self.pipeline_combo.count()):
            if self.pipeline_combo.itemData(i) == pipeline_name:
                self.pipeline_combo.setCurrentIndex(i)
                break
        self.update_history()

    def add_measurement(self, measurement: MeasurementResult) -> None:
        """Add a new measurement to history and update display."""
        self.history.add_measurement(measurement)
        # Update the pipeline filter to match the new measurement if no filter is set
        if self.current_pipeline_filter is None and measurement.pipeline != "unknown":
            # Auto-select the pipeline filter if it's the first measurement of that type
            pipeline_measurements = [m for m in self.history.measurements if m.pipeline == measurement.pipeline]
            if len(pipeline_measurements) == 1:  # This is the first measurement of this pipeline
                self.current_pipeline_filter = measurement.pipeline
                # Update the combo box selection
                for i in range(self.pipeline_combo.count()):
                    if self.pipeline_combo.itemData(i) == measurement.pipeline:
                        self.pipeline_combo.setCurrentIndex(i)
                        break
        self.update_history()

    def update_single_measurement(self, results: Mapping[str, Any], pipeline_name: str = "unknown", file_name: Optional[str] = None) -> None:
        """Update display for a single measurement (legacy compatibility)."""
        if results:
            # Create a measurement result and add to history
            from datetime import datetime
            import uuid

            measurement = MeasurementResult(
                id=f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}",
                timestamp=datetime.now(),
                pipeline=pipeline_name,
                file_name=file_name,
                results=dict(results)
            )
            self.history.add_measurement(measurement)
            self.update_history()
