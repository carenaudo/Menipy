"""Results panel helper for Menipy GUI."""

from __future__ import annotations

import csv
from collections.abc import Mapping
from typing import Any, List, Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLayout,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from menipy.common.units import convert_from_si
from menipy.gui.controllers.pipeline_ui_manager import PipelineUIManager
from menipy.gui.helpers.icon_loader import load_icon, set_button_icon
from menipy.gui.services.settings_service import AppSettings
from menipy.models.results import MeasurementResult, get_results_history


def get_label_with_unit(key: str, system: str) -> str:
    """Return the label with localized unit suffix."""
    labels_si = {
        "density": "kg/m³",
        "surface_tension": "mN/m",
        "volume": "µL",
        "length": "mm",
        "contact_angle": "°",
        "surface_area": "mm²",
    }
    labels_cgs = {
        "density": "g/cm³",
        "surface_tension": "dyn/cm",
        "volume": "cm³",
        "length": "cm",
        "contact_angle": "°",
        "surface_area": "cm²",
    }
    units = labels_si if system == "SI" else labels_cgs

    base_labels = {
        "diameter_mm": ("Diameter", "length"),
        "height_mm": ("Height", "length"),
        "volume_uL": ("Volume", "volume"),
        "surface_tension_mN_m": ("Surface Tension", "surface_tension"),
        "r0_mm": ("Apex Radius", "length"),
        "needle_surface_mm2": ("Needle Surface", "surface_area"),
        "drop_surface_mm2": ("Drop Surface", "surface_area"),
        "contact_angle_deg": ("Contact Angle", "contact_angle"),
        "contact_surface_mm2": ("Contact Surface", "surface_area"),
        "theta_left_deg": ("Left Contact Angle", "contact_angle"),
        "theta_right_deg": ("Right Contact Angle", "contact_angle"),
        "baseline_tilt_deg": ("Baseline Tilt", "contact_angle"),
        "uncertainty_deg": ("Angle Uncertainty", "contact_angle"),
        "uncertainty_left_deg": ("Left Angle Uncertainty", "contact_angle"),
        "uncertainty_right_deg": ("Right Angle Uncertainty", "contact_angle"),
    }

    if key in base_labels:
        name, qtype = base_labels[key]
        return f"{name} ({units[qtype]})"

    return key.replace("_", " ").title()


VALID_PIPELINES = {
    "sessile",
    "pendant",
    "oscillating",
    "capillary_rise",
    "captive_bubble",
}
VALID_PIPELINES_FILTER = "__valid__"
LEGACY_PIPELINES_FILTER = "__legacy__"

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
    "surface_tension_method": "Surface Tension Method",
    "bond_number": "Bond Number",
    "worthington_number": "Worthington Number",
    "vmax_uL": "Vmax (µL)",
    "approx_selected_plane_surface_tension_mN_m": "Selected Plane IFT (mN/m)",
    "approx_multi_selected_plane_surface_tension_mN_m": "Multi-Plane IFT (mN/m)",
    "approx_volume_apex_surface_tension_mN_m": "Volume+Apex IFT (mN/m)",
    "uncertainty_deg": "Angle Uncertainty (°)",
    "uncertainty_left_deg": "Left Angle Uncertainty (°)",
    "uncertainty_right_deg": "Right Angle Uncertainty (°)",
}


class ResultsPanel:
    """Provides a helper wrapper around the results table widget with measurement history support."""

    def __init__(
        self,
        panel: QWidget,
        metric_host: QWidget | None = None,
        residuals_host: QWidget | QLayout | None = None,
        residuals_table: QTableWidget | None = None,
    ) -> None:
        """Initialize.

        Parameters
        ----------
        panel : type
        Description.
        """
        self.panel = panel
        self.metric_host = metric_host
        self.residuals_table: QTableWidget | None = None
        self.table: QTableWidget | None = panel.findChild(QTableWidget, "resultsTable")
        self.summary_label: QLabel | None = panel.findChild(QLabel, "summaryLabel")
        self.history = get_results_history()
        self.settings = AppSettings.load()
        self.unit_system = getattr(self.settings, "unit_system", "SI")
        self.current_pipeline_filter = VALID_PIPELINES_FILTER
        self.pipeline_ui_manager = PipelineUIManager()
        self._raw_headers: list[str] = []
        self.metric_value_labels: dict[str, QLabel] = {}
        self.recent_results_list: QListWidget | None = None
        self.key_results_count_label: QLabel | None = None
        self._syncing_selection = False

        # Add controls for history management
        self._setup_controls()
        self._setup_metric_cards()
        self.install_residuals_table(residuals_host, residuals_table)

        if self.table:
            # Start with empty table - will be populated by update_history()
            self.table.setColumnCount(0)
            self.table.setRowCount(0)
            self.table.itemSelectionChanged.connect(self._on_table_selection_changed)

        # Load and display existing history
        self.update_history()

    def install_residuals_table(
        self,
        residuals_host: QWidget | QLayout | None = None,
        residuals_table: QTableWidget | None = None,
    ) -> QTableWidget | None:
        """Attach a residuals table to the selected measurement."""
        if residuals_table is not None:
            self.residuals_table = residuals_table
        elif residuals_host is not None:
            table = QTableWidget()
            table.setObjectName("residualsTable")
            if isinstance(residuals_host, QLayout):
                layout = residuals_host
            else:
                layout = residuals_host.layout() or QVBoxLayout(residuals_host)
            layout.addWidget(table)
            self.residuals_table = table

        if self.residuals_table is not None:
            self.residuals_table.setAlternatingRowColors(True)
            self._update_residuals_for_current_selection()
        return self.residuals_table

    def _setup_controls(self) -> None:
        """Setup control buttons and filters."""
        # Find existing layout or create new one
        if hasattr(self.panel, "layout") and self.panel.layout():
            layout = self.panel.layout()
        else:
            layout = QVBoxLayout(self.panel)

        # Create controls layout
        controls_layout = QHBoxLayout()

        # Pipeline filter
        self.pipeline_combo = QComboBox()
        self.pipeline_combo.addItem("All Valid Pipelines", VALID_PIPELINES_FILTER)
        self.pipeline_combo.addItem(load_icon("sessile"), "Sessile", "sessile")
        self.pipeline_combo.addItem(load_icon("pendant"), "Pendant", "pendant")
        self.pipeline_combo.addItem(
            load_icon("oscillating"), "Oscillating", "oscillating"
        )
        self.pipeline_combo.addItem(
            load_icon("capillary_rise"), "Capillary Rise", "capillary_rise"
        )
        self.pipeline_combo.addItem(
            load_icon("captive_bubble"), "Captive Bubble", "captive_bubble"
        )
        self.pipeline_combo.addItem("Legacy / Unknown", LEGACY_PIPELINES_FILTER)
        self.pipeline_combo.currentIndexChanged.connect(
            self._on_pipeline_filter_changed
        )
        controls_layout.addWidget(QLabel("Filter:"))
        controls_layout.addWidget(self.pipeline_combo)

        # Measurement count label
        self.count_label = QLabel("0 measurements")
        controls_layout.addWidget(self.count_label)

        controls_layout.addStretch()

        # Action buttons
        self.key_results_button = QPushButton("Key Results")
        set_button_icon(self.key_results_button, "info", size=14)
        self.key_results_button.clicked.connect(self.show_key_results)
        controls_layout.addWidget(self.key_results_button)

        self.compare_button = QPushButton("Compare")
        set_button_icon(self.compare_button, "layout-table", size=14)
        self.compare_button.setCheckable(True)
        self.compare_button.setChecked(
            bool(getattr(self.settings, "compare_methods_visible", False))
        )
        self.compare_button.toggled.connect(self.set_compare_methods_visible)
        controls_layout.addWidget(self.compare_button)

        self.diagnostics_button = QPushButton("Diagnostics")
        set_button_icon(self.diagnostics_button, "list", size=14)
        self.diagnostics_button.setCheckable(True)
        self.diagnostics_button.setChecked(
            bool(getattr(self.settings, "diagnostics_visible", False))
        )
        self.diagnostics_button.toggled.connect(self.set_diagnostics_visible)
        controls_layout.addWidget(self.diagnostics_button)

        self.clear_button = QPushButton("Clear")
        set_button_icon(self.clear_button, "x", size=14)
        self.clear_button.clicked.connect(self._clear_history)
        controls_layout.addWidget(self.clear_button)

        self.export_button = QPushButton("Export")
        set_button_icon(self.export_button, "download", size=14)
        self.export_button.clicked.connect(self._export_csv)
        controls_layout.addWidget(self.export_button)

        self.columns_button = QPushButton("Columns")
        set_button_icon(self.columns_button, "layout-table", size=14)
        self.columns_menu = QMenu(self.columns_button)
        self.columns_button.setMenu(self.columns_menu)
        controls_layout.addWidget(self.columns_button)

        # Add controls to main layout (keep summary label above controls if present)
        insert_index = 0
        if self.summary_label and layout.indexOf(self.summary_label) != -1:
            insert_index = 1
        layout.insertLayout(insert_index, controls_layout)

    def _setup_metric_cards(self) -> None:
        """Create compact key metric cards above the full results table."""
        host = self.metric_host or self.panel
        if hasattr(host, "layout") and host.layout():
            layout = host.layout()
        else:
            layout = QVBoxLayout(host)

        frame = QFrame(host)
        frame.setObjectName("resultsMetricFrame")
        frame.setStyleSheet("""
            QFrame#resultsMetricFrame {
                background-color: transparent;
                border: none;
            }
            QFrame[resultMetricCard="true"] {
                background-color: #FFFFFF;
                border: 1px solid #D0D7DE;
                border-radius: 6px;
            }
            QLabel[resultMetricLabel="true"] {
                color: #57606A;
                font-size: 10px;
                font-weight: 700;
                text-transform: uppercase;
            }
            QLabel[resultMetricValue="true"] {
                color: #24292F;
                font-size: 18px;
                font-weight: 700;
            }
        """)
        grid = QGridLayout(frame)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(8)

        metric_defs = (
            ("ift", "IFT"),
            ("diameter", "Diameter"),
            ("volume", "Volume"),
            ("angle", "Angle"),
        )
        columns = 2 if self.metric_host else 4
        for index, (key, label) in enumerate(metric_defs):
            card = QFrame(frame)
            card.setProperty("resultMetricCard", True)
            card_layout = QVBoxLayout(card)
            card_layout.setContentsMargins(10, 8, 10, 8)
            card_layout.setSpacing(3)

            label_widget = QLabel(label, card)
            label_widget.setProperty("resultMetricLabel", True)
            value_widget = QLabel("--", card)
            value_widget.setObjectName(f"metricValue_{key}")
            value_widget.setProperty("resultMetricValue", True)

            card_layout.addWidget(label_widget)
            card_layout.addWidget(value_widget)
            grid.addWidget(card, index // columns, index % columns)
            self.metric_value_labels[key] = value_widget

        if self.metric_host:
            header = QLabel("Key Results", host)
            header.setObjectName("keyResultsHeader")
            header.setStyleSheet("""
                QLabel#keyResultsHeader {
                    color: #57606A;
                    font-size: 10px;
                    font-weight: 700;
                    letter-spacing: 1px;
                    text-transform: uppercase;
                    padding: 4px 2px;
                }
            """)
            self.key_results_count_label = QLabel("0 measurements", host)
            self.key_results_count_label.setObjectName("keyResultsCountLabel")
            self.key_results_count_label.setStyleSheet("""
                QLabel#keyResultsCountLabel {
                    color: #57606A;
                    font-size: 12px;
                    font-weight: 600;
                    padding: 0 2px 8px 2px;
                }
            """)
            recent_header = QLabel("Recent History", host)
            recent_header.setObjectName("recentHistoryHeader")
            recent_header.setStyleSheet("""
                QLabel#recentHistoryHeader {
                    color: #57606A;
                    font-size: 10px;
                    font-weight: 700;
                    letter-spacing: 1px;
                    text-transform: uppercase;
                    padding: 10px 2px 4px 2px;
                    border-top: 1px solid #D0D7DE;
                }
            """)
            self.recent_results_list = QListWidget(host)
            self.recent_results_list.setObjectName("recentResultsList")
            self.recent_results_list.setAlternatingRowColors(False)
            self.recent_results_list.setStyleSheet("""
                QListWidget#recentResultsList {
                    background: transparent;
                    border: none;
                    outline: none;
                }
                QListWidget#recentResultsList::item {
                    background: #FFFFFF;
                    border: 1px solid #D0D7DE;
                    border-radius: 6px;
                    padding: 8px;
                    margin: 3px 0;
                    color: #24292F;
                }
                QListWidget#recentResultsList::item:selected {
                    background: #DDF4FF;
                    border-color: #0969DA;
                    color: #0969DA;
                }
            """)
            self.recent_results_list.currentRowChanged.connect(
                self._on_recent_result_selected
            )
            layout.addWidget(header)
            layout.addWidget(self.key_results_count_label)
            layout.addWidget(frame)
            layout.addWidget(recent_header)
            layout.addWidget(self.recent_results_list, 1)
            return

        insert_index = 0
        if self.table and layout.indexOf(self.table) != -1:
            insert_index = layout.indexOf(self.table)
        layout.insertWidget(insert_index, frame)

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
            with open(file_path, "w", newline="") as csvfile:
                writer = csv.writer(csvfile)

                # Write headers
                headers = []
                for col in range(self.table.columnCount()):
                    if self.table.isColumnHidden(col):
                        continue
                    header_item = self.table.horizontalHeaderItem(col)
                    headers.append(
                        header_item.text() if header_item else f"Column {col}"
                    )
                writer.writerow(headers)

                # Write data
                for row in range(self.table.rowCount()):
                    row_data = []
                    for col in range(self.table.columnCount()):
                        if self.table.isColumnHidden(col):
                            continue
                        item = self.table.item(row, col)
                        row_data.append(item.text() if item else "")
                    writer.writerow(row_data)

        except Exception as e:
            from PySide6.QtWidgets import QMessageBox

            QMessageBox.warning(
                self.panel, "Export Error", f"Failed to export CSV: {e}"
            )

    def export_csv(self) -> None:
        """Public wrapper for exporting the current table view."""
        self._export_csv()

    def update(self, results: Mapping[str, Any] | None) -> None:
        """Refresh the history-backed table after a legacy update signal."""
        if results:
            self.update_history()

    def update_history(self) -> None:
        """Update table with measurement history."""
        if not self.table:
            return

        unit_system = self.unit_system
        headers, rows = self._get_table_data()
        raw_headers = list(headers)

        # Apply pipeline-specific column prioritization if filtering by pipeline
        if self.current_pipeline_filter in VALID_PIPELINES:
            headers, rows = self._prioritize_columns_for_pipeline(
                headers, rows, self.current_pipeline_filter
            )
            raw_headers = list(headers)

        # Update table
        display_headers = [get_label_with_unit(h, unit_system) for h in headers]
        self.table.setColumnCount(len(display_headers))
        self.table.setRowCount(len(rows))
        self.table.setHorizontalHeaderLabels(display_headers)
        self._raw_headers = list(raw_headers)

        # Populate data with pipeline-aware styling
        for row_idx, row_data in enumerate(rows):
            for col_idx, cell_value in enumerate(row_data):
                # Convert value if needed
                transformed_value = cell_value
                try:
                    # Map header key back to quantity type
                    quantity_map = {
                        "diameter_mm": "length",
                        "height_mm": "length",
                        "r0_mm": "length",
                        "volume_uL": "volume",
                        "surface_tension_mN_m": "surface_tension",
                    }
                    if raw_headers[col_idx] in quantity_map and cell_value != "":
                        val_float = float(cell_value)
                        transformed_float = convert_from_si(
                            val_float,
                            quantity_map[raw_headers[col_idx]],
                            unit_system,
                        )
                        transformed_value = f"{transformed_float:.3g}"
                except (ValueError, TypeError):
                    pass

                item = QTableWidgetItem(str(transformed_value))
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)  # Make read-only
                if raw_headers[col_idx] == "pipeline":
                    icon = self._pipeline_icon(str(transformed_value))
                    if not icon.isNull():
                        item.setIcon(icon)

                # Add pipeline-specific styling
                if self.current_pipeline_filter in VALID_PIPELINES:
                    self._apply_pipeline_styling(
                        item, self.current_pipeline_filter, raw_headers[col_idx]
                    )

                self.table.setItem(row_idx, col_idx, item)

        # Update count label
        measurement_count = len(rows)
        filter_text = (
            f" ({self.pipeline_combo.currentText()})"
            if self.current_pipeline_filter != VALID_PIPELINES_FILTER
            else ""
        )
        self.count_label.setText(f"{measurement_count} measurements{filter_text}")
        if self.key_results_count_label:
            self.key_results_count_label.setText(
                f"{measurement_count} measurements{filter_text}"
            )

        self._rebuild_recent_results(measurements=self._filtered_measurements())
        self._update_summary()
        self._rebuild_columns_menu()
        self._apply_column_visibility()
        self.table.resizeColumnsToContents()
        if rows and self.table.currentRow() < 0:
            self.table.selectRow(0)
        self._update_residuals_for_current_selection()

    def _rebuild_recent_results(
        self, measurements: list[MeasurementResult] | None = None
    ) -> None:
        if not self.recent_results_list:
            return
        measurements = (
            measurements if measurements is not None else self._filtered_measurements()
        )
        self.recent_results_list.blockSignals(True)
        self.recent_results_list.clear()
        for measurement in measurements[:20]:
            item = QListWidgetItem(self._recent_result_text(measurement))
            icon = self._pipeline_icon(measurement.pipeline)
            if not icon.isNull():
                item.setIcon(icon)
            item.setData(Qt.UserRole, measurement.id)
            self.recent_results_list.addItem(item)
        if self.recent_results_list.count() > 0:
            row = self.table.currentRow() if self.table else 0
            if row < 0:
                row = 0
            self.recent_results_list.setCurrentRow(
                min(row, self.recent_results_list.count() - 1)
            )
        self.recent_results_list.blockSignals(False)

    def _recent_result_text(self, measurement: MeasurementResult) -> str:
        name = (
            measurement.file_name
            or measurement.file_path
            or f"Measurement {measurement.id.split('_')[-1]}"
        )
        time_text = measurement.timestamp.strftime("%H:%M:%S")
        results = measurement.results or {}
        metrics: list[str] = []
        for key, label, unit in (
            ("surface_tension_mN_m", "IFT", ""),
            ("diameter_mm", "Diameter", " mm"),
            ("volume_uL", "Volume", " µL"),
            ("contact_angle_deg", "Angle", "°"),
        ):
            value = results.get(key)
            if value in (None, ""):
                continue
            try:
                value_text = f"{float(value):.3g}{unit}"
            except (TypeError, ValueError):
                value_text = str(value)
            metrics.append(f"{label} {value_text}")
        metric_line = " · ".join(metrics) if metrics else measurement.pipeline.title()
        return f"{name}    {time_text}\n{metric_line}"

    def _on_recent_result_selected(self, row: int) -> None:
        if self._syncing_selection or row < 0 or not self.table:
            return
        if row >= self.table.rowCount():
            return
        self._syncing_selection = True
        try:
            self.table.selectRow(row)
            measurement = self._measurement_for_row(row)
            if measurement:
                self._update_metric_cards(measurement.results or {})
            self._update_residuals_table(measurement)
        finally:
            self._syncing_selection = False

    def _on_table_selection_changed(self) -> None:
        if self._syncing_selection or not self.table:
            return
        row = self.table.currentRow()
        if row < 0:
            return
        self._syncing_selection = True
        try:
            if self.recent_results_list and row < self.recent_results_list.count():
                self.recent_results_list.setCurrentRow(row)
            measurement = self._measurement_for_row(row)
            if measurement:
                self._update_metric_cards(measurement.results or {})
            self._update_residuals_table(measurement)
        finally:
            self._syncing_selection = False

    def _update_residuals_for_current_selection(self) -> None:
        if not self.table:
            self._update_residuals_table(None)
            return
        self._update_residuals_table(self._measurement_for_row(self.table.currentRow()))

    def has_residuals_for_current_selection(self) -> bool:
        if not self.table:
            return False
        measurement = self._measurement_for_row(self.table.currentRow())
        if measurement is None:
            return False
        return bool((measurement.results or {}).get("residuals"))

    def _update_residuals_table(self, measurement: MeasurementResult | None) -> None:
        table = self.residuals_table
        if table is None:
            return
        residuals = None
        if measurement is not None:
            residuals = (measurement.results or {}).get("residuals")
        self._render_residuals(residuals)

    def _render_residuals(self, residuals: Any) -> None:
        table = self.residuals_table
        if table is None:
            return
        table.clear()
        table.setRowCount(0)

        if not residuals:
            table.setColumnCount(1)
            table.setHorizontalHeaderLabels(["Residuals"])
            table.setRowCount(1)
            item = QTableWidgetItem("No residuals for selected measurement")
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)
            table.setItem(0, 0, item)
            table.resizeColumnsToContents()
            return

        if isinstance(residuals, Mapping):
            vector = self._first_residual_vector(residuals)
            if vector is not None:
                self._render_residual_vector(vector, residuals.get("units"))
                return
            table.setColumnCount(2)
            table.setHorizontalHeaderLabels(["Metric", "Value"])
            rows = [
                (str(key), self._format_residual_value(value))
                for key, value in residuals.items()
            ]
            table.setRowCount(len(rows))
            for row, (key, value) in enumerate(rows):
                for col, text in enumerate((key, value)):
                    item = QTableWidgetItem(text)
                    item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                    table.setItem(row, col, item)
            table.resizeColumnsToContents()
            return

        if isinstance(residuals, (list, tuple)):
            self._render_residual_vector(residuals, None)
            return

        table.setColumnCount(2)
        table.setHorizontalHeaderLabels(["Metric", "Value"])
        table.setRowCount(1)
        residual_text = self._format_residual_value(residuals)
        for col, text in enumerate(("residuals", residual_text)):
            item = QTableWidgetItem(text)
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)
            table.setItem(0, col, item)
        table.resizeColumnsToContents()

    def _first_residual_vector(self, residuals: Mapping[str, Any]) -> Any:
        for key in ("values", "vector", "residuals", "data"):
            value = residuals.get(key)
            if isinstance(value, (list, tuple)):
                return value
        return None

    def _render_residual_vector(self, values: Any, units: Any) -> None:
        table = self.residuals_table
        if table is None:
            return
        show_units = units not in (None, "")
        headers = ["Index", "Residual"] + (["Unit"] if show_units else [])
        table.setColumnCount(len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.setRowCount(len(values))
        for row, value in enumerate(values):
            cells = [str(row), self._format_residual_value(value)]
            if show_units:
                cells.append(str(units))
            for col, text in enumerate(cells):
                item = QTableWidgetItem(text)
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                table.setItem(row, col, item)
        table.resizeColumnsToContents()

    def _format_residual_value(self, value: Any) -> str:
        if isinstance(value, float):
            return f"{value:.6g}"
        if isinstance(value, (list, tuple)):
            return ", ".join(self._format_residual_value(item) for item in value)
        return str(value)

    def _measurement_for_row(self, row: int) -> MeasurementResult | None:
        measurements = self._filtered_measurements()
        if 0 <= row < len(measurements):
            return measurements[row]
        return None

    def _column_settings_key(self) -> str:
        return str(self.current_pipeline_filter or VALID_PIPELINES_FILTER)

    def _hidden_columns(self) -> set[str]:
        hidden = getattr(self.settings, "results_hidden_columns", {}) or {}
        key = self._column_settings_key()
        if key not in hidden:
            return self._default_hidden_columns()
        values = hidden.get(key, [])
        return {str(value) for value in values}

    def _save_hidden_columns(self, hidden: set[str]) -> None:
        settings = self.settings
        data = dict(getattr(settings, "results_hidden_columns", {}) or {})
        data[self._column_settings_key()] = sorted(hidden)
        settings.results_hidden_columns = data
        settings.save()
        self.settings = settings

    def _apply_column_visibility(self) -> None:
        if not self.table:
            return
        hidden = self._hidden_columns()
        for col, key in enumerate(self._raw_headers):
            self.table.setColumnHidden(col, key in hidden)

    def _rebuild_columns_menu(self) -> None:
        self.columns_menu.clear()
        show_all = QAction("Show All", self.columns_menu)
        show_all.triggered.connect(self.show_all_columns)
        self.columns_menu.addAction(show_all)

        hide_diagnostics = QAction("Hide Diagnostics", self.columns_menu)
        hide_diagnostics.triggered.connect(self.hide_diagnostics_columns)
        self.columns_menu.addAction(hide_diagnostics)

        reset = QAction("Reset Defaults", self.columns_menu)
        reset.triggered.connect(self.reset_column_defaults)
        self.columns_menu.addAction(reset)
        self.columns_menu.addSeparator()

        hidden = self._hidden_columns()
        for key in self._raw_headers:
            action = QAction(self._display_header(key), self.columns_menu)
            action.setCheckable(True)
            action.setChecked(key not in hidden)
            action.toggled.connect(
                lambda checked, column_key=key: self.set_column_visible(
                    column_key, checked
                )
            )
            self.columns_menu.addAction(action)

    def set_column_visible(self, column_key: str, visible: bool) -> None:
        hidden = self._hidden_columns()
        if visible:
            hidden.discard(column_key)
        else:
            hidden.add(column_key)
        self._save_hidden_columns(hidden)
        self._apply_column_visibility()

    def show_all_columns(self) -> None:
        self._save_hidden_columns(set())
        self._rebuild_columns_menu()
        self._apply_column_visibility()

    def show_key_results(self) -> None:
        self.set_compare_methods_visible(False, refresh=False)
        self.set_diagnostics_visible(False, refresh=False)
        self._save_hidden_columns(self._default_hidden_columns())
        self._sync_helper_buttons()
        self._rebuild_columns_menu()
        self._apply_column_visibility()

    def set_compare_methods_visible(
        self, visible: bool, *, refresh: bool = True
    ) -> None:
        self.settings.compare_methods_visible = bool(visible)
        self.settings.results_hidden_columns = dict(
            getattr(self.settings, "results_hidden_columns", {}) or {}
        )
        self.settings.results_hidden_columns.pop(self._column_settings_key(), None)
        self.settings.save()
        self._sync_helper_buttons()
        if refresh:
            self.update_history()

    def set_diagnostics_visible(self, visible: bool, *, refresh: bool = True) -> None:
        self.settings.diagnostics_visible = bool(visible)
        self.settings.results_hidden_columns = dict(
            getattr(self.settings, "results_hidden_columns", {}) or {}
        )
        self.settings.results_hidden_columns.pop(self._column_settings_key(), None)
        self.settings.save()
        self._sync_helper_buttons()
        if refresh:
            self.update_history()

    def hide_diagnostics_columns(self) -> None:
        diagnostic_prefixes = ("strict_", "fit_", "geometric_", "approx_")
        diagnostic_names = {"diameter_line", "model_profile_px"}
        hidden = {
            key
            for key in self._raw_headers
            if key.startswith(diagnostic_prefixes) or key in diagnostic_names
        }
        self._save_hidden_columns(hidden)
        self._rebuild_columns_menu()
        self._apply_column_visibility()

    def reset_column_defaults(self) -> None:
        self.show_key_results()

    def _sync_helper_buttons(self) -> None:
        for button, value in (
            (
                getattr(self, "compare_button", None),
                bool(getattr(self.settings, "compare_methods_visible", False)),
            ),
            (
                getattr(self, "diagnostics_button", None),
                bool(getattr(self.settings, "diagnostics_visible", False)),
            ),
        ):
            if button:
                button.blockSignals(True)
                button.setChecked(value)
                button.blockSignals(False)

    def _default_hidden_columns(self) -> set[str]:
        compare = bool(getattr(self.settings, "compare_methods_visible", False))
        diagnostics = bool(getattr(self.settings, "diagnostics_visible", False))
        hidden: set[str] = set()
        for key in self._raw_headers:
            if key.startswith("approx_") and not compare:
                hidden.add(key)
            if self._is_diagnostic_column(key) and not diagnostics:
                hidden.add(key)
        return hidden

    def _is_diagnostic_column(self, key: str) -> bool:
        diagnostic_prefixes = ("strict_", "fit_", "geometric_")
        diagnostic_names = {
            "diameter_line",
            "model_profile_px",
            "contact_angle_fit_rmse_px",
        }
        return key.startswith(diagnostic_prefixes) or key in diagnostic_names

    def _filtered_measurements(self) -> list[MeasurementResult]:
        measurements = list(self.history.measurements)
        pipeline_filter = self.current_pipeline_filter
        if pipeline_filter == VALID_PIPELINES_FILTER:
            return [m for m in measurements if m.pipeline in VALID_PIPELINES]
        if pipeline_filter == LEGACY_PIPELINES_FILTER:
            return [m for m in measurements if m.pipeline not in VALID_PIPELINES]
        if pipeline_filter:
            return [m for m in measurements if m.pipeline == pipeline_filter]
        return measurements

    def _get_table_data(self) -> tuple[list[str], list[list[Any]]]:
        measurements = self._filtered_measurements()
        if not measurements:
            return ["file_name", "timestamp", "pipeline"], []

        all_keys: set[str] = set()
        for measurement in measurements:
            all_keys.update(measurement.results.keys())
        all_keys.discard("residuals")

        priority_columns = [
            "file_name",
            "timestamp",
            "pipeline",
            "diameter_mm",
            "height_mm",
            "volume_uL",
            "surface_tension_mN_m",
            "surface_tension_method",
            "contact_angle_deg",
            "theta_left_deg",
            "theta_right_deg",
            "contact_surface_mm2",
            "drop_surface_mm2",
            "bond_number",
            "worthington_number",
            "baseline_tilt_deg",
            "beta",
            "s1",
            "r0_mm",
            "needle_surface_mm2",
            "R0_mm",
            "f0_Hz",
            "r0_eq_px",
        ]
        headers = priority_columns + sorted(all_keys - set(priority_columns))

        rows: list[list[Any]] = []
        for measurement in measurements:
            row: list[Any] = []
            for col in headers:
                if col == "file_name":
                    value = (
                        measurement.file_name
                        or measurement.file_path
                        or f"Measurement {measurement.id.split('_')[-1]}"
                    )
                elif col == "timestamp":
                    value = measurement.timestamp.strftime("%H:%M:%S")
                elif col == "pipeline":
                    value = measurement.pipeline.title()
                else:
                    value = measurement.results.get(col)
                    if isinstance(value, (int, float)):
                        if col.endswith("_deg") or "angle" in col.lower():
                            value = f"{value:.1f}"
                        else:
                            value = f"{value:.3g}"
                    elif value is None:
                        value = ""
                    else:
                        value = str(value)
                row.append(value)
            rows.append(row)
        return headers, rows

    def _prioritize_columns_for_pipeline(
        self, headers: list[str], rows: list[list[Any]], pipeline_name: str
    ) -> tuple[list[str], list[list[Any]]]:
        """Reorder columns to prioritize pipeline-specific metrics."""
        if not pipeline_name:
            return headers, rows

        # Get primary metrics for this pipeline
        primary_metrics = self.pipeline_ui_manager.get_primary_metrics(pipeline_name)

        # Define column priority order based on pipeline
        priority_columns = []

        # Always include timestamp and file info first
        for base in ("file_name", "timestamp", "pipeline"):
            if base in headers:
                priority_columns.append(base)

        # Add primary metrics for this pipeline
        for metric in primary_metrics:
            if metric in headers:
                priority_columns.append(metric)

        # Add remaining columns
        for header in headers:
            if header not in priority_columns:
                priority_columns.append(header)

        # Reorder headers and data
        if priority_columns != headers:
            header_indices = [
                headers.index(col) for col in priority_columns if col in headers
            ]
            reordered_headers = [headers[i] for i in header_indices]
            reordered_rows = []
            for row in rows:
                reordered_row = [row[i] for i in header_indices]
                reordered_rows.append(reordered_row)
            return reordered_headers, reordered_rows

        return headers, rows

    def _apply_pipeline_styling(
        self, item: QTableWidgetItem, pipeline_name: str, column_name: str
    ) -> None:
        """Apply pipeline-specific styling to table cells."""
        # Style primary metrics differently
        primary_metrics = self.pipeline_ui_manager.get_primary_metrics(pipeline_name)
        for metric in primary_metrics:
            if column_name == metric:
                # Make primary metrics bold
                font = item.font()
                font.setBold(True)
                item.setFont(font)
                break

    def _pipeline_icon(self, pipeline_name: str):
        key = pipeline_name.lower().replace(" ", "_")
        if key == "capillary_rise":
            return load_icon("capillary_rise")
        if key in VALID_PIPELINES:
            return load_icon(key)
        return load_icon("info")

    def _display_header(self, header: str) -> str:
        if header == "file_name":
            return "File"
        if header == "timestamp":
            return "Time"
        if header == "pipeline":
            return "Pipeline"
        return LABEL_MAP.get(header, header.replace("_", " ").title())

    def _update_summary(self) -> None:
        measurements = self._filtered_measurements()
        count = len(measurements)
        last_run = (
            measurements[0].timestamp.strftime("%H:%M:%S") if measurements else "n/a"
        )
        parts = [f"{count} measurements", f"Last: {last_run}"]
        latest = measurements[0].results if measurements else {}
        if measurements:
            for key, label, suffix in (
                ("surface_tension_mN_m", "IFT", " mN/m"),
                ("contact_angle_deg", "CA", "°"),
                ("diameter_mm", "Dia", " mm"),
                ("volume_uL", "Vol", " µL"),
            ):
                value = latest.get(key)
                if value is None:
                    continue
                try:
                    text = f"{float(value):.3g}{suffix}"
                except (TypeError, ValueError):
                    text = str(value)
                parts.append(f"{label}: {text}")
        if self.summary_label:
            self.summary_label.setText(" | ".join(parts))
        self._update_metric_cards(latest)

    def _update_metric_cards(self, results: Mapping[str, Any]) -> None:
        metrics = {
            "ift": (("surface_tension_mN_m", "surface_tension"), "mN/m"),
            "diameter": (("diameter_mm", "diameter"), "mm"),
            "volume": (("volume_uL", "volume"), "µL"),
            "angle": (
                (
                    "contact_angle_deg",
                    "theta_left_deg",
                    "angle_mean",
                    "angle_left",
                ),
                "°",
            ),
        }
        for name, (keys, unit) in metrics.items():
            label = self.metric_value_labels.get(name)
            if not label:
                continue
            value = None
            for key in keys:
                if key in results and results.get(key) not in (None, ""):
                    value = results.get(key)
                    break
            label.setText(self._format_metric_value(value, unit))

    def _format_metric_value(self, value: Any, unit: str) -> str:
        if value in (None, ""):
            return "--"
        try:
            number = float(value)
        except (TypeError, ValueError):
            return str(value)
        return f"{number:.3g} {unit}"

    def set_pipeline_filter(self, pipeline_name: str) -> None:
        """Set the pipeline filter programmatically."""
        if not pipeline_name:
            pipeline_name = VALID_PIPELINES_FILTER
        elif pipeline_name == "unknown":
            pipeline_name = LEGACY_PIPELINES_FILTER
        self.current_pipeline_filter = pipeline_name
        # Update combo box selection
        for i in range(self.pipeline_combo.count()):
            if self.pipeline_combo.itemData(i) == pipeline_name:
                self.pipeline_combo.setCurrentIndex(i)
                break
        self.update_history()

    def add_measurement(self, measurement: MeasurementResult) -> None:
        """Add a new measurement to history and update display."""
        if measurement is None:
            return
        self.history.add_measurement(measurement)
        self.update_history()

    def update_single_measurement(
        self,
        results: Mapping[str, Any],
        pipeline_name: str = "unknown",
        file_name: str | None = None,
    ) -> None:
        """Update display for a single measurement (legacy compatibility)."""
        if results:
            # Create a measurement result and add to history
            import uuid
            from datetime import datetime

            measurement = MeasurementResult(
                id=f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}",
                timestamp=datetime.now(),
                pipeline=pipeline_name,
                file_name=file_name,
                results=dict(results),
            )
            self.history.add_measurement(measurement)
            self.update_history()

    def update_history_table(
        self, history=None, unit_system: str | None = None
    ) -> None:
        """Refresh the history table using the current global unit setting."""
        if unit_system in ("SI", "CGS"):
            self.unit_system = unit_system
        self.update_history()
