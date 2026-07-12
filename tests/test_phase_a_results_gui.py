"""GUI and export behavior for scientifically rejected Phase-A records."""

from __future__ import annotations

import csv
import json
from datetime import datetime

from PySide6.QtWidgets import QLabel, QTableWidget, QVBoxLayout, QWidget

from menipy.gui.views import results_panel as results_panel_module
from menipy.gui.views.results_panel import ResultsPanel
from menipy.models.results import MeasurementResult, ResultsHistory


class _History:
    def __init__(self, measurement):
        self.measurements = [measurement]

    def add_measurement(self, measurement):
        self.measurements.insert(0, measurement)

    def clear_history(self):
        self.measurements.clear()


def _rejected_measurement() -> MeasurementResult:
    return MeasurementResult(
        id="rejected_001",
        timestamp=datetime(2026, 7, 10, 12, 0, 0),
        pipeline="sessile",
        results={},
        accepted=False,
        rejection_reasons=["insufficient_contour_points"],
        diagnostics={
            "validity": {"ok": False},
            "detectors": {"drop": {"accepted": False}},
        },
    )


def test_rejected_measurement_has_status_empty_cards_and_diagnostics(
    monkeypatch, qtbot
):
    measurement = _rejected_measurement()
    monkeypatch.setattr(
        results_panel_module, "get_results_history", lambda: _History(measurement)
    )
    panel = QWidget()
    layout = QVBoxLayout(panel)
    summary = QLabel(objectName="summaryLabel")
    table = QTableWidget(objectName="resultsTable")
    diagnostics = QTableWidget()
    layout.addWidget(summary)
    layout.addWidget(table)
    qtbot.addWidget(panel)
    qtbot.addWidget(diagnostics)

    ResultsPanel(panel, residuals_table=diagnostics)

    status_col = next(
        col
        for col in range(table.columnCount())
        if table.horizontalHeaderItem(col).text() == "Status"
    )
    assert table.item(0, status_col).text() == "Rejected"
    assert panel.findChild(QLabel, "metricValue_ift").text() == "--"
    values = [
        diagnostics.item(row, 1).text()
        for row in range(diagnostics.rowCount())
        if diagnostics.columnCount() > 1 and diagnostics.item(row, 1)
    ]
    assert any("insufficient_contour_points" in value for value in values)


def test_history_csv_and_json_persist_rejection_metadata(tmp_path):
    history = ResultsHistory()
    history._data_dir = tmp_path
    history._history_file = tmp_path / "history.json"
    history.measurements = [_rejected_measurement()]
    csv_path = tmp_path / "results.csv"

    assert history.export_csv(csv_path)
    with csv_path.open(encoding="utf-8", newline="") as stream:
        row = next(csv.DictReader(stream))
    assert row["status"] == "Rejected"
    assert row["rejection_reasons"] == "insufficient_contour_points"
    assert json.loads(row["diagnostics_json"])["validity"]["ok"] is False

    history._save_history()
    saved = json.loads(history._history_file.read_text(encoding="utf-8"))
    assert saved["measurements"][0]["accepted"] is False
    assert saved["measurements"][0]["results"] == {}
