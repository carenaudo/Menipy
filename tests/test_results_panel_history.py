from __future__ import annotations

from datetime import datetime

from PySide6.QtWidgets import QLabel, QListWidget, QTableWidget, QVBoxLayout, QWidget

from menipy.gui.services.settings_service import AppSettings
from menipy.gui.views import results_panel as results_panel_module
from menipy.gui.views.results_panel import (
    LEGACY_PIPELINES_FILTER,
    VALID_PIPELINES_FILTER,
    ResultsPanel,
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


def _rich_measurement(pipeline: str = "pendant") -> MeasurementResult:
    return MeasurementResult(
        id=f"{pipeline}_rich",
        timestamp=datetime(2026, 4, 27, 12, 0, 0),
        pipeline=pipeline,
        file_name=f"{pipeline}.png",
        results={
            "diameter_mm": 2.0,
            "surface_tension_mN_m": 29.0,
            "strict_rmse_mm": 0.01,
            "approx_volume_apex_surface_tension_mN_m": 30.0,
            "residuals": {"rmse": 0.01, "units": "mm"},
        },
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


def test_results_panel_column_visibility_persists(monkeypatch, qtbot, tmp_path):
    history = FakeHistory([_rich_measurement()])
    settings_path = tmp_path / "settings.json"
    original_load = AppSettings.load
    monkeypatch.setattr(results_panel_module, "get_results_history", lambda: history)
    monkeypatch.setattr(
        results_panel_module.AppSettings,
        "load",
        classmethod(lambda cls: original_load(settings_path)),
    )
    panel, table = _panel_widget()
    qtbot.addWidget(panel)
    controller = ResultsPanel(panel)
    controller.set_pipeline_filter("pendant")

    controller.set_column_visible("diameter_mm", False)
    diameter_col = controller._raw_headers.index("diameter_mm")
    assert table.isColumnHidden(diameter_col)

    panel2, table2 = _panel_widget()
    qtbot.addWidget(panel2)
    controller2 = ResultsPanel(panel2)
    controller2.set_pipeline_filter("pendant")
    diameter_col2 = controller2._raw_headers.index("diameter_mm")
    assert table2.isColumnHidden(diameter_col2)


def test_results_panel_exports_visible_columns(monkeypatch, qtbot, tmp_path):
    history = FakeHistory([_rich_measurement()])
    settings_path = tmp_path / "settings.json"
    export_path = tmp_path / "visible.csv"
    original_load = AppSettings.load
    monkeypatch.setattr(results_panel_module, "get_results_history", lambda: history)
    monkeypatch.setattr(
        results_panel_module.AppSettings,
        "load",
        classmethod(lambda cls: original_load(settings_path)),
    )
    monkeypatch.setattr(
        "PySide6.QtWidgets.QFileDialog.getSaveFileName",
        lambda *_args, **_kwargs: (str(export_path), "CSV Files (*.csv)"),
    )
    panel, _table = _panel_widget()
    qtbot.addWidget(panel)
    controller = ResultsPanel(panel)
    controller.set_pipeline_filter("pendant")
    controller.set_column_visible("strict_rmse_mm", False)

    controller.export_csv()

    header = export_path.read_text(encoding="utf-8").splitlines()[0]
    assert "Strict Rmse Mm" not in header
    assert "Surface Tension" in header


def test_results_panel_keeps_residuals_out_of_results_columns(
    monkeypatch, qtbot, tmp_path
):
    history = FakeHistory([_rich_measurement()])
    settings_path = tmp_path / "settings.json"
    export_path = tmp_path / "visible.csv"
    original_load = AppSettings.load
    monkeypatch.setattr(results_panel_module, "get_results_history", lambda: history)
    monkeypatch.setattr(
        results_panel_module.AppSettings,
        "load",
        classmethod(lambda cls: original_load(settings_path)),
    )
    monkeypatch.setattr(
        "PySide6.QtWidgets.QFileDialog.getSaveFileName",
        lambda *_args, **_kwargs: (str(export_path), "CSV Files (*.csv)"),
    )
    panel, _table = _panel_widget()
    qtbot.addWidget(panel)
    controller = ResultsPanel(panel)
    controller.set_pipeline_filter("pendant")

    assert "residuals" not in controller._raw_headers
    column_labels = [action.text() for action in controller.columns_menu.actions()]
    assert "Residuals" not in column_labels

    controller.export_csv()

    header = export_path.read_text(encoding="utf-8").splitlines()[0]
    assert "residual" not in header.lower()


def test_results_panel_guided_defaults_hide_comparison_and_diagnostics(
    monkeypatch, qtbot, tmp_path
):
    history = FakeHistory([_rich_measurement()])
    settings_path = tmp_path / "settings.json"
    original_load = AppSettings.load
    monkeypatch.setattr(results_panel_module, "get_results_history", lambda: history)
    monkeypatch.setattr(
        results_panel_module.AppSettings,
        "load",
        classmethod(lambda cls: original_load(settings_path)),
    )
    panel, table = _panel_widget()
    qtbot.addWidget(panel)

    controller = ResultsPanel(panel)
    controller.set_pipeline_filter("pendant")

    strict_col = controller._raw_headers.index("strict_rmse_mm")
    approx_col = controller._raw_headers.index(
        "approx_volume_apex_surface_tension_mN_m"
    )
    assert table.isColumnHidden(strict_col)
    assert table.isColumnHidden(approx_col)


def test_results_panel_metric_cards_show_latest_key_values(monkeypatch, qtbot):
    history = FakeHistory([_rich_measurement()])
    monkeypatch.setattr(results_panel_module, "get_results_history", lambda: history)
    panel, _table = _panel_widget()
    qtbot.addWidget(panel)

    ResultsPanel(panel)

    assert panel.findChild(QLabel, "metricValue_ift").text() == "29 mN/m"
    assert panel.findChild(QLabel, "metricValue_diameter").text() == "2 mm"


def test_results_panel_renders_dict_residuals(monkeypatch, qtbot):
    history = FakeHistory([_rich_measurement()])
    monkeypatch.setattr(results_panel_module, "get_results_history", lambda: history)
    panel, _table = _panel_widget()
    residuals_table = QTableWidget()
    qtbot.addWidget(panel)
    qtbot.addWidget(residuals_table)

    ResultsPanel(panel, residuals_table=residuals_table)

    assert residuals_table.horizontalHeaderItem(0).text() == "Metric"
    assert residuals_table.item(0, 0).text() == "rmse"
    assert residuals_table.item(0, 1).text() == "0.01"


def test_results_panel_renders_vector_residuals_on_selection(monkeypatch, qtbot):
    vector_measurement = MeasurementResult(
        id="pendant_vector",
        timestamp=datetime(2026, 4, 27, 12, 0, 1),
        pipeline="pendant",
        file_name="vector.png",
        results={
            "diameter_mm": 3.0,
            "residuals": {"values": [0.1, -0.2], "units": "px"},
        },
    )
    history = FakeHistory([_rich_measurement(), vector_measurement])
    monkeypatch.setattr(results_panel_module, "get_results_history", lambda: history)
    panel, table = _panel_widget()
    residuals_table = QTableWidget()
    qtbot.addWidget(panel)
    qtbot.addWidget(residuals_table)
    ResultsPanel(panel, residuals_table=residuals_table)

    table.selectRow(1)

    assert residuals_table.horizontalHeaderItem(0).text() == "Index"
    assert residuals_table.horizontalHeaderItem(1).text() == "Residual"
    assert residuals_table.horizontalHeaderItem(2).text() == "Unit"
    assert residuals_table.item(0, 0).text() == "0"
    assert residuals_table.item(0, 1).text() == "0.1"
    assert residuals_table.item(0, 2).text() == "px"


def test_results_panel_residuals_empty_state(monkeypatch, qtbot):
    history = FakeHistory([_measurement("sessile")])
    monkeypatch.setattr(results_panel_module, "get_results_history", lambda: history)
    panel, _table = _panel_widget()
    residuals_table = QTableWidget()
    qtbot.addWidget(panel)
    qtbot.addWidget(residuals_table)

    ResultsPanel(panel, residuals_table=residuals_table)

    assert residuals_table.horizontalHeaderItem(0).text() == "Residuals"
    assert residuals_table.item(0, 0).text() == "No residuals for selected measurement"


def test_results_panel_can_render_metric_cards_in_lateral_host(monkeypatch, qtbot):
    history = FakeHistory([_rich_measurement()])
    monkeypatch.setattr(results_panel_module, "get_results_history", lambda: history)
    panel, table = _panel_widget()
    metric_host = QWidget()
    metric_host.setLayout(QVBoxLayout())
    qtbot.addWidget(panel)
    qtbot.addWidget(metric_host)

    ResultsPanel(panel, metric_host=metric_host)

    assert metric_host.findChild(QLabel, "keyResultsHeader").text() == "Key Results"
    assert metric_host.findChild(QLabel, "keyResultsCountLabel").text() == (
        "1 measurements"
    )
    assert metric_host.findChild(QLabel, "metricValue_ift").text() == "29 mN/m"
    assert metric_host.findChild(QListWidget, "recentResultsList").count() == 1
    assert panel.findChild(QLabel, "metricValue_ift") is None
    assert table.rowCount() == 1


def test_lateral_recent_history_selection_syncs_table(monkeypatch, qtbot):
    history = FakeHistory(
        [_rich_measurement("pendant"), _measurement("sessile", value=3.0)]
    )
    monkeypatch.setattr(results_panel_module, "get_results_history", lambda: history)
    panel, table = _panel_widget()
    metric_host = QWidget()
    metric_host.setLayout(QVBoxLayout())
    qtbot.addWidget(panel)
    qtbot.addWidget(metric_host)

    ResultsPanel(panel, metric_host=metric_host)
    recent = metric_host.findChild(QListWidget, "recentResultsList")

    recent.setCurrentRow(1)

    assert table.currentRow() == 1
    assert metric_host.findChild(QLabel, "metricValue_diameter").text() == "3 mm"


def test_results_panel_compare_and_diagnostics_helpers(monkeypatch, qtbot, tmp_path):
    history = FakeHistory([_rich_measurement()])
    settings_path = tmp_path / "settings.json"
    original_load = AppSettings.load
    monkeypatch.setattr(results_panel_module, "get_results_history", lambda: history)
    monkeypatch.setattr(
        results_panel_module.AppSettings,
        "load",
        classmethod(lambda cls: original_load(settings_path)),
    )
    panel, table = _panel_widget()
    qtbot.addWidget(panel)

    controller = ResultsPanel(panel)
    controller.set_pipeline_filter("pendant")
    controller.set_compare_methods_visible(True)
    controller.set_diagnostics_visible(True)

    strict_col = controller._raw_headers.index("strict_rmse_mm")
    approx_col = controller._raw_headers.index(
        "approx_volume_apex_surface_tension_mN_m"
    )
    assert not table.isColumnHidden(strict_col)
    assert not table.isColumnHidden(approx_col)

    controller.show_key_results()
    assert table.isColumnHidden(strict_col)
    assert table.isColumnHidden(approx_col)
