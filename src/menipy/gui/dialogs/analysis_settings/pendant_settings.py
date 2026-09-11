"""Pipeline-specific settings panel for the pendant drop pipeline."""

from __future__ import annotations

from typing import Any

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from menipy.common import registry
from menipy.common.plugin_settings import get_detector_settings_model
from menipy.gui.dialogs.analysis_settings import phase_properties_note
from menipy.gui.dialogs.plugin_config_dialog import PluginConfigDialog
from menipy.pipelines.pendant.stages import (
    DEFAULT_PENDANT_APPROXIMATION_METHODS,
    OPTIONAL_PENDANT_APPROXIMATION_METHODS,
)
from menipy.pipelines.pendant.zone_spline import CONTOUR_MODELS

APPROXIMATORS_EXPLANATION = (
    "Computed after the Young-Laplace fit, each from its own method, and reported "
    "as approx_* result columns for cross-checking. They do not seed the fit "
    "(see Pendant initializer). If the fit is rejected, the first valid estimate "
    "(Minimize ADSA, then Multiple planes, Selected plane, Volume/apex lookup) "
    "becomes the reported surface tension."
)

_APPROXIMATOR_LABELS = {
    "minimize_adsa": "Minimize ADSA",
    "selected_plane": "Selected plane",
    "multi_selected_plane": "Multiple selected planes",
    "volume_apex_lookup": "Volume/apex lookup",
    "clothoid_zones": "Clothoid zones",
}


def approximator_choices(enabled: list[str]) -> list[str]:
    """Approximators to offer, in display order.

    Built-in defaults first, then the built-in opt-in methods, then any other
    registered approximator (external plugins), then enabled names that no
    plugin provides, so saving the dialog never drops a selection.

    Parameters
    ----------
    enabled : list of str
        Currently selected approximator names.

    Returns
    -------
    list of str
        Unique approximator names.
    """
    known = DEFAULT_PENDANT_APPROXIMATION_METHODS + OPTIONAL_PENDANT_APPROXIMATION_METHODS
    registered = sorted(
        name for name in registry.PENDANT_APPROXIMATORS.keys() if name not in known
    )
    return list(dict.fromkeys(known + registered + [str(name) for name in enabled]))


def approximator_label(method: str) -> str:
    """Human-readable checkbox label for an approximator name.

    Parameters
    ----------
    method : str
        Registered approximator name.

    Returns
    -------
    str
        Display label; unregistered names are flagged.
    """
    label = _APPROXIMATOR_LABELS.get(method, method.replace("_", " ").capitalize())
    if method not in registry.PENDANT_APPROXIMATORS:
        return f"{label} (not registered)"
    return label


def _approximator_tooltip(method: str) -> str:
    fn = registry.PENDANT_APPROXIMATORS.get(method)
    if fn is None:
        return (
            f"'{method}' is saved in these settings but no loaded plugin provides it; "
            "the run reports it as plugin_not_registered. Uncheck to remove it."
        )
    summary = ((fn.__doc__ or "").strip().splitlines() or [method])[0]
    when = (
        "Runs by default."
        if method in DEFAULT_PENDANT_APPROXIMATION_METHODS
        else "Opt-in."
    )
    return f"{summary}\n{when} Registry name: {method}"


def _model_defaults(model: Any) -> dict[str, Any]:
    try:
        return model().model_dump()
    except Exception:
        return {}


class PipelineSettingsWidget(QWidget):
    """Method and stage selections for pendant drop."""

    def __init__(self, parent=None, settings: dict | None = None):
        super().__init__(parent)
        settings = settings or {}
        layout = QVBoxLayout(self)
        form = QFormLayout()
        form.setContentsMargins(12, 12, 12, 12)
        form.setSpacing(8)
        form.addRow(phase_properties_note())

        self._experimental_mode = QComboBox()
        self._experimental_mode.addItems(["off", "shadow"])
        self._experimental_mode.setCurrentText(settings.get("experimental_geometry_mode", "off"))
        form.addRow("Experimental geometry", self._experimental_mode)

        self._needle_geometry = QComboBox()
        self._needle_geometry.addItems(["legacy", "bilateral_robust"])
        self._needle_geometry.setCurrentText(settings.get("needle_geometry_method", "legacy"))
        form.addRow("Needle geometry", self._needle_geometry)

        self._pendant_initializer = QComboBox()
        self._pendant_initializer.addItems(["legacy", "robust_axis"])
        self._pendant_initializer.setCurrentText(settings.get("pendant_initializer", "legacy"))
        self._pendant_initializer.setToolTip(
            "Symmetry axis and apex the Young-Laplace fit starts from. legacy: "
            "vertical axis through the contour median, lowest point as apex; "
            "robust_axis: axis fitted to the contour mid-points (tolerates tilt).\n"
            "The R0/beta seeds come from the apex radius and the Jennings-Pallas "
            "geometric estimate (or from the spline when Contour model is "
            "clothoid_zones)."
        )
        form.addRow("Pendant initializer", self._pendant_initializer)

        self._contour_model = QComboBox()
        self._contour_model.addItems(list(CONTOUR_MODELS))
        self._contour_model.setCurrentText(settings.get("pendant_contour_model", "raw"))
        self._contour_model.setToolTip(
            "clothoid_zones: fit the strict Young-Laplace model to the two-zone clothoid "
            "spline (denoised contour, physics seeds) and report its surface tension and "
            "needle angle"
        )
        form.addRow("Contour model", self._contour_model)

        self._onnx_mode = QComboBox()
        self._onnx_mode.addItems(["off", "shadow"])
        self._onnx_mode.setCurrentText(settings.get("onnx_proposal_mode", "off"))
        form.addRow("ONNX proposals", self._onnx_mode)

        self._segmentation_provider = QComboBox()
        self._segmentation_provider.addItems(["mobilesam"])
        self._segmentation_provider.setCurrentText(
            settings.get("segmentation_provider", "mobilesam")
        )
        form.addRow("Segmentation provider", self._segmentation_provider)

        enabled_approximations = settings.get("pendant_approximation_methods")
        if enabled_approximations is None:
            enabled_approximations = DEFAULT_PENDANT_APPROXIMATION_METHODS
        saved_approx_settings = settings.get("pendant_approximator_settings") or {}
        self._approx_settings: dict[str, dict] = {
            str(name): dict(values)
            for name, values in saved_approx_settings.items()
            if isinstance(values, dict) and values
        }
        form.addRow(self._build_approximator_group(list(enabled_approximations)))

        layout.addLayout(form)
        layout.addStretch(1)

    # ------------------------------------------------------------ approximators
    def _build_approximator_group(self, enabled: list[str]) -> QGroupBox:
        """One row per approximator: enable checkbox, settings status, Configure."""
        group = QGroupBox("Alternative surface-tension estimates (approximators)")
        grid = QGridLayout(group)
        grid.setColumnStretch(0, 1)
        explanation = QLabel(APPROXIMATORS_EXPLANATION)
        explanation.setWordWrap(True)
        explanation.setStyleSheet("color: #7f8c8d;")
        grid.addWidget(explanation, 0, 0, 1, 3)
        self._approx_checks: dict[str, QCheckBox] = {}
        self._approx_status: dict[str, QLabel] = {}
        for row, method in enumerate(approximator_choices(enabled), start=1):
            checkbox = QCheckBox(approximator_label(method))
            checkbox.setChecked(method in enabled)
            checkbox.setToolTip(_approximator_tooltip(method))
            self._approx_checks[method] = checkbox
            grid.addWidget(checkbox, row, 0)
            if get_detector_settings_model(method) is None:
                continue
            status = QLabel()
            status.setStyleSheet("color: #7f8c8d;")
            self._approx_status[method] = status
            button = QPushButton("Configure…")
            button.setAccessibleName(f"Configure {approximator_label(method)}")
            button.clicked.connect(
                lambda _checked=False, m=method: self._configure_approximator(m)
            )
            grid.addWidget(status, row, 1)
            grid.addWidget(button, row, 2)
            self._update_approximator_status(method)
        return group

    def _configure_approximator(self, method: str) -> None:
        """Open the generated settings form for ``method``."""
        model = get_detector_settings_model(method)
        if model is None:
            return
        current = {**_model_defaults(model), **self._approx_settings.get(method, {})}
        dialog = PluginConfigDialog(
            model,
            current,
            parent=self,
            title=f"{approximator_label(method)} settings",
        )
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.set_approximator_settings(method, dialog.get_settings())

    def set_approximator_settings(self, method: str, values: dict) -> None:
        """Store the settings of one approximator.

        Only values that differ from the settings model's defaults are kept, so
        saved settings follow future default changes.

        Parameters
        ----------
        method : str
            Registered approximator name.
        values : dict
            Complete or partial settings for ``method``.
        """
        model = get_detector_settings_model(method)
        defaults = _model_defaults(model) if model is not None else {}
        overrides = {
            key: value
            for key, value in values.items()
            if key not in defaults or defaults[key] != value
        }
        if overrides:
            self._approx_settings[method] = overrides
        else:
            self._approx_settings.pop(method, None)
        if method in self._approx_status:
            self._update_approximator_status(method)

    def _update_approximator_status(self, method: str) -> None:
        overrides = self._approx_settings.get(method, {})
        label = self._approx_status[method]
        if not overrides:
            label.setText("defaults")
            label.setToolTip("")
            return
        label.setText(f"{len(overrides)} custom")
        label.setToolTip(
            "\n".join(f"{key} = {value}" for key, value in sorted(overrides.items()))
        )

    def get_settings(self) -> dict:
        """Collect the pendant pipeline settings from the widgets.

        Returns
        -------
        dict
            Flat pipeline settings; ``pendant_approximation_methods`` lists the
            checked approximators (an empty list disables all of them) and
            ``pendant_approximator_settings`` maps an approximator name to the
            values that differ from its defaults.
        """
        return {
            "experimental_geometry_mode": self._experimental_mode.currentText(),
            "needle_geometry_method": self._needle_geometry.currentText(),
            "pendant_initializer": self._pendant_initializer.currentText(),
            "pendant_contour_model": self._contour_model.currentText(),
            "onnx_proposal_mode": self._onnx_mode.currentText(),
            "segmentation_provider": self._segmentation_provider.currentText(),
            "pendant_approximation_methods": [
                method
                for method, checkbox in self._approx_checks.items()
                if checkbox.isChecked()
            ],
            "pendant_approximator_settings": {
                method: dict(values) for method, values in self._approx_settings.items()
            },
        }
