from __future__ import annotations

from datetime import datetime

from PySide6.QtWidgets import QLabel, QTableWidget, QVBoxLayout, QWidget

from menipy.gui.panels import results_panel as results_panel_module
from menipy.gui.panels.results_panel import (
    LEGACY_PIPELINES_FILTER,
    ResultsPanel,
    VALID_PIPELINES_FILTER,
)
from menipy.models.results import MeasurementResult


class FakeHistory:
    def __init__(self, measurements=None):
        self.measurements = list(measurements or [])

    def add_measurement(self, measurement):
        self.measurements.insert(0, measurement)

    def clear_history(self):
        self.measurements.clear()


def _panel_widget():
    panel = QWidget()
    layout = QVBoxLayout(panel)
    summary = QLabel()
    summary.setObjectName("summaryLabel")
    table = QTableWidget()
    table.setObjectName("resultsTable")
    layout.addWidget(summary)
    layout.addWidget(table)
    return panel, table


def _measurement(pipeline: str, value: float = 1.0) -> MeasurementResult:
    return MeasurementResult(
        id=f"{pipeline}_001",
        timestamp=datetime(2026, 4, 27, 12, 0, 0),
        pipeline=pipeline,
        file_name=f"{pipeline}.png",
        results={"diameter_mm": value},
    )


def test_results_panel_hides_unknown_rows_by_default(monkeypatch, qtbot):
    history = FakeHistory([_measurement("unknown"), _measurement("sessile")])
    monkeypatch.setattr(results_panel_module, "get_results_history", lambda: history)
    panel, table = _panel_widget()
    qtbot.addWidget(panel)

    controller = ResultsPanel(panel)

    assert controller.current_pipeline_filter == VALID_PIPELINES_FILTER
    assert table.rowCount() == 1
    assert table.item(0, 2).text() == "Sessile"


def test_results_panel_exposes_unknown_rows_as_legacy(monkeypatch, qtbot):
    history = FakeHistory([_measurement("unknown"), _measurement("sessile")])
    monkeypatch.setattr(results_panel_module, "get_results_history", lambda: history)
    panel, table = _panel_widget()
    qtbot.addWidget(panel)
    controller = ResultsPanel(panel)

    controller.set_pipeline_filter(LEGACY_PIPELINES_FILTER)

    assert table.rowCount() == 1
    assert table.item(0, 2).text() == "Unknown"


def test_results_update_does_not_add_duplicate_unknown(monkeypatch, qtbot):
    history = FakeHistory()
    monkeypatch.setattr(results_panel_module, "get_results_history", lambda: history)
    panel, _table = _panel_widget()
    qtbot.addWidget(panel)
    controller = ResultsPanel(panel)

    controller.add_measurement(_measurement("sessile"))
    controller.update({"diameter_mm": 2.0})

    assert len(history.measurements) == 1
    assert history.measurements[0].pipeline == "sessile"
