"""GUI persistence coverage for pipeline-specific settings widgets."""

from __future__ import annotations

from menipy.gui.dialogs.analysis_settings.pendant_settings import (
    PipelineSettingsWidget as PendantSettings,
)
from menipy.gui.dialogs.analysis_settings.sessile_settings import (
    PipelineSettingsWidget as SessileSettings,
)


def test_onnx_proposals_are_off_by_default(qtbot) -> None:
    for widget_type in (SessileSettings, PendantSettings):
        widget = widget_type(settings={})
        qtbot.addWidget(widget)
        settings = widget.get_settings()
        assert settings["onnx_proposal_mode"] == "off"
        assert settings["segmentation_provider"] == "mobilesam"


def test_onnx_shadow_settings_roundtrip(qtbot) -> None:
    for widget_type in (SessileSettings, PendantSettings):
        widget = widget_type(
            settings={
                "onnx_proposal_mode": "shadow",
                "segmentation_provider": "mobilesam",
            }
        )
        qtbot.addWidget(widget)
        settings = widget.get_settings()
        assert settings["onnx_proposal_mode"] == "shadow"
        assert settings["segmentation_provider"] == "mobilesam"


def test_method_tabs_only_offer_settings_that_reach_the_run(qtbot) -> None:
    from PySide6.QtWidgets import QLabel

    from menipy.gui.controllers.pipeline_controller import FORWARDED_PIPELINE_SETTINGS
    from menipy.gui.dialogs.analysis_settings import PHASE_PROPERTIES_NOTE

    # Keys older versions saved from inputs that never reached a run.
    old = {
        "rho1": 500.0,
        "rho2": 2.0,
        "g": 1.0,
        "solver": "x",
        "preprocessor": "x",
        "edge_detector": "x",
        "overlay_alpha": 0.2,
        "overlay_visible": False,
        "notes": "old",
    }
    for widget_type in (SessileSettings, PendantSettings):
        widget = widget_type(settings=old)
        qtbot.addWidget(widget)
        assert set(widget.get_settings()) <= set(FORWARDED_PIPELINE_SETTINGS)
        assert PHASE_PROPERTIES_NOTE in [label.text() for label in widget.findChildren(QLabel)]


def test_analysis_dialog_drops_dead_settings_but_keeps_legacy_notes(qtbot) -> None:
    from menipy.gui.dialogs.analysis_settings_dialog import AnalysisSettingsDialog

    dialog = AnalysisSettingsDialog(
        "sessile",
        pipeline_settings={
            "contact_angle_method": "circle_fit",
            "physics": {"g": 1.0},
            "solver": "x",
            "notes": "old tab note",
        },
    )
    qtbot.addWidget(dialog)
    settings = dialog.pipeline_settings()
    assert settings["contact_angle_method"] == "circle_fit"
    assert "physics" not in settings and "solver" not in settings
    assert settings["notes"] == "old tab note"  # until a preset takes it over
    assert "enabled_stages" not in settings
    assert "Physics" not in [dialog._tabs.tabText(i) for i in range(dialog._tabs.count())]


def test_analysis_dialog_has_no_steps_tab_and_configures_real_stages(qtbot) -> None:
    from menipy.gui.dialogs.analysis_settings_dialog import AnalysisSettingsDialog

    dialog = AnalysisSettingsDialog(
        "pendant", pipeline_settings={"enabled_stages": ["acquisition"]}
    )
    qtbot.addWidget(dialog)
    titles = [dialog._tabs.tabText(i) for i in range(dialog._tabs.count())]
    assert "Steps" not in titles
    # Pendant's UI metadata listed only 4 stages, which hid these tabs before.
    for title in ("Preprocessing", "Edge Detection", "Geometry", "Overlay"):
        assert title in titles
        assert dialog._tabs.isTabVisible(titles.index(title))


def test_sessile_clothoid_spline_method_is_restored(qtbot) -> None:
    widget = SessileSettings(settings={"contact_angle_method": "clothoid_spline"})
    qtbot.addWidget(widget)
    assert widget.get_settings()["contact_angle_method"] == "clothoid_spline"


def test_physics_menu_points_to_phase_properties(qtbot, monkeypatch) -> None:
    from types import SimpleNamespace

    from PySide6.QtWidgets import QDoubleSpinBox, QMainWindow, QMessageBox

    from menipy.gui.controllers.dialog_coordinator import DialogCoordinator

    window = QMainWindow()
    qtbot.addWidget(window)
    spin = QDoubleSpinBox()
    window.setCentralWidget(spin)
    window.setup_panel_ctrl = SimpleNamespace(dropDensitySpin=spin)
    shown = []
    monkeypatch.setattr(QMessageBox, "information", lambda *args: shown.append(args))
    coordinator = DialogCoordinator(window, settings=SimpleNamespace())

    window.show()
    qtbot.waitExposed(window)
    coordinator.show_dialog_for_stage("physics")
    assert "Phase Properties" in window.statusBar().currentMessage()
    assert not shown

    window.hide()
    coordinator.show_dialog_for_stage("physics")
    assert "Phase Properties" in shown[0][2]


def test_dialog_preview_cleanup_is_idempotent(qtbot) -> None:
    from types import SimpleNamespace

    from PySide6.QtCore import Signal
    from PySide6.QtWidgets import QDialog, QMainWindow

    from menipy.gui.controllers.dialog_coordinator import DialogCoordinator

    class PreviewDialog(QDialog):
        previewRequested = Signal(object)

    window = QMainWindow()
    qtbot.addWidget(window)
    dialog = PreviewDialog(parent=window)
    qtbot.addWidget(dialog)
    coordinator = DialogCoordinator(window, settings=SimpleNamespace())

    coordinator._connect_dialog_preview(dialog, coordinator._on_edge_detection_preview)
    coordinator._disconnect_dialog_preview(dialog)
    # A second close/cleanup path must not ask PySide to disconnect a missing slot.
    coordinator._disconnect_dialog_preview(dialog)

    assert not coordinator._dialog_preview_connections


def test_pendant_approximators_default_and_opt_in(qtbot) -> None:
    from menipy.pipelines.pendant.stages import DEFAULT_PENDANT_APPROXIMATION_METHODS

    widget = PendantSettings(settings={})
    qtbot.addWidget(widget)
    assert not widget._approx_checks["clothoid_zones"].isChecked()
    assert (
        widget.get_settings()["pendant_approximation_methods"]
        == DEFAULT_PENDANT_APPROXIMATION_METHODS
    )

    widget._approx_checks["minimize_adsa"].setChecked(False)
    widget._approx_checks["clothoid_zones"].setChecked(True)
    methods = widget.get_settings()["pendant_approximation_methods"]
    assert "minimize_adsa" not in methods
    assert "clothoid_zones" in methods

    restored = PendantSettings(settings={"pendant_approximation_methods": methods})
    qtbot.addWidget(restored)
    assert restored.get_settings()["pendant_approximation_methods"] == methods


def test_pendant_approximators_listed_from_registry(qtbot, monkeypatch) -> None:
    from menipy.common import registry

    def external(ctx, profile_mm, physics):
        """External plugin estimate."""
        return {}

    monkeypatch.setitem(registry.PENDANT_APPROXIMATORS._items, "external", external)
    widget = PendantSettings(
        settings={"pendant_approximation_methods": ["selected_plane", "uninstalled"]}
    )
    qtbot.addWidget(widget)

    external_box = widget._approx_checks["external"]
    assert not external_box.isChecked()
    assert "External plugin estimate." in external_box.toolTip()
    missing = widget._approx_checks["uninstalled"]
    assert missing.isChecked()
    assert "not registered" in missing.text()
    # Saving without touching the dialog keeps the unregistered selection.
    assert widget.get_settings()["pendant_approximation_methods"] == [
        "selected_plane",
        "uninstalled",
    ]


def test_minimize_adsa_settings_configured_from_dialog(qtbot, monkeypatch) -> None:
    from PySide6.QtWidgets import QDialog

    from menipy.gui.dialogs.plugin_config_dialog import PluginConfigDialog

    widget = PendantSettings(
        settings={"pendant_approximator_settings": {"minimize_adsa": {"maxiter": 7}}}
    )
    qtbot.addWidget(widget)
    assert widget._approx_status["minimize_adsa"].text() == "1 custom"
    assert "clothoid_zones" not in widget._approx_status  # no settings model

    shown = {}

    def accept(dialog):
        shown["maxiter"] = dialog._inputs["maxiter"].value()
        shown["ftol"] = dialog._inputs["ftol"].value()
        dialog._inputs["maxiter"].setValue(300)  # back to the default
        dialog._inputs["robust_clip_enabled"].setChecked(False)
        return QDialog.DialogCode.Accepted

    monkeypatch.setattr(PluginConfigDialog, "exec", accept)
    widget._configure_approximator("minimize_adsa")

    assert shown == {"maxiter": 7, "ftol": 1e-9}
    assert widget.get_settings()["pendant_approximator_settings"] == {
        "minimize_adsa": {"robust_clip_enabled": False}
    }
    widget.set_approximator_settings("minimize_adsa", {"robust_clip_enabled": True})
    assert widget.get_settings()["pendant_approximator_settings"] == {}
    assert widget._approx_status["minimize_adsa"].text() == "defaults"


def test_plugin_config_dialog_keeps_small_floats_and_bounds(qtbot) -> None:
    from pydantic import BaseModel, Field

    from menipy.gui.dialogs.plugin_config_dialog import PluginConfigDialog

    class Model(BaseModel):
        tol: float = Field(1e-9, ge=0.0, description="Stop tolerance")
        fraction: float = Field(0.1, ge=0.01, le=0.5)
        points: int = Field(120, ge=20)

    dialog = PluginConfigDialog(Model, {"tol": 2.5e-12}, title="Model settings")
    qtbot.addWidget(dialog)
    assert dialog.windowTitle() == "Model settings"
    assert dialog.get_settings()["tol"] == 2.5e-12
    assert dialog._inputs["tol"].text() == "2.5e-12"
    assert dialog._inputs["tol"].toolTip() == "Stop tolerance"
    assert dialog._inputs["fraction"].maximum() == 0.5
    assert dialog._inputs["points"].minimum() == 20

    dialog._inputs["tol"].lineEdit().setText("3e-8")
    dialog._inputs["tol"].interpretText()
    assert dialog.get_settings()["tol"] == 3e-8
    dialog._inputs["fraction"].lineEdit().setText("0,25")  # decimal-comma locales
    dialog._inputs["fraction"].interpretText()
    assert dialog.get_settings()["fraction"] == 0.25

    dialog.restore_defaults()
    assert dialog.get_settings() == Model().model_dump()

