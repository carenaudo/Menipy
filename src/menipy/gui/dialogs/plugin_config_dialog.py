"""Dialog for configuring plugin settings based on Pydantic models."""

import math
from enum import Enum
from typing import Any, get_args, get_origin

from pydantic import BaseModel
from pydantic.fields import FieldInfo
from PySide6.QtGui import QValidator
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

_INT_LIMIT = 999999
_FLOAT_LIMIT = 999999.0


class ScientificDoubleSpinBox(QDoubleSpinBox):
    """Double spin box that keeps full precision and accepts ``1e-9`` notation.

    A plain ``QDoubleSpinBox`` rounds to a fixed number of decimals, which
    silently turns tolerances such as ``1e-9`` into ``0``.
    """

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setDecimals(323)  # Qt's maximum: the stored value is not rounded.

    def textFromValue(self, value: float) -> str:  # noqa: N802 - Qt override
        """Format ``value`` compactly (``0.1``, ``1e-09``)."""
        return f"{value:.10g}"

    def valueFromText(self, text: str) -> float:  # noqa: N802 - Qt override
        """Parse plain or scientific notation (a decimal comma is accepted)."""
        try:
            return float(text.strip().replace(",", "."))
        except ValueError:
            return self.value()

    def validate(self, text: str, pos: int) -> object:
        """Accept any float literal and its partial prefixes while typing."""
        stripped = text.strip().replace(",", ".")
        if stripped.lower() in {"", "-", "+", ".", "-.", "+."} or stripped.lower().endswith(
            ("e", "e-", "e+")
        ):
            return QValidator.State.Intermediate, text, pos
        try:
            value = float(stripped)
        except ValueError:
            return QValidator.State.Invalid, text, pos
        if not math.isfinite(value):
            return QValidator.State.Invalid, text, pos
        if value < self.minimum() or value > self.maximum():
            return QValidator.State.Intermediate, text, pos
        return QValidator.State.Acceptable, text, pos

    def fixup(self, text: str) -> str:
        """Fall back to the current value when editing ends on invalid text."""
        return self.textFromValue(self.value())


def _field_bounds(field: FieldInfo | None) -> tuple[float | None, float | None, bool, bool]:
    """Return ``(lower, upper, lower_exclusive, upper_exclusive)`` from constraints."""
    lower = upper = None
    lower_open = upper_open = False
    for item in getattr(field, "metadata", None) or ():
        if getattr(item, "ge", None) is not None:
            lower, lower_open = float(item.ge), False
        if getattr(item, "gt", None) is not None:
            lower, lower_open = float(item.gt), True
        if getattr(item, "le", None) is not None:
            upper, upper_open = float(item.le), False
        if getattr(item, "lt", None) is not None:
            upper, upper_open = float(item.lt), True
    return lower, upper, lower_open, upper_open


def _field_default(field: FieldInfo) -> Any:
    """Return the field's default value, or ``None`` when it is required."""
    if field.default_factory is not None:
        try:
            return field.default_factory()
        except Exception:
            return None
    return None if field.is_required() else field.default


class PluginConfigDialog(QDialog):
    """Form generated from a Pydantic settings model.

    Parameters
    ----------
    model_class : type[BaseModel]
        Settings model; one input is created per supported field. Numeric
        ``ge``/``gt``/``le``/``lt`` constraints limit the inputs and field
        descriptions become tooltips.
    current_values : dict
        Values to show; missing fields show the model defaults.
    parent : QWidget, optional
        Parent widget.
    title : str, optional
        Window title; defaults to ``"Configure <ModelName>"``.
    """

    def __init__(
        self,
        model_class: type[BaseModel],
        current_values: dict[str, Any],
        parent=None,
        *,
        title: str | None = None,
    ):
        super().__init__(parent)
        self.setWindowTitle(title or f"Configure {model_class.__name__}")
        self.model_class = model_class
        self._inputs = {}

        main_layout = QVBoxLayout(self)

        # Scroll area for many settings
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        form = QFormLayout(container)

        # Description
        description = model_class.__doc__
        if description:
            lbl = QLabel(description.strip())
            lbl.setWordWrap(True)
            lbl.setStyleSheet("color: gray; font-style: italic; margin-bottom: 10px;")
            form.addRow(lbl)

        # Generate fields
        for name, field in model_class.model_fields.items():
            value = current_values.get(name, _field_default(field))
            widget = self._create_widget(field.annotation, value, field=field)
            if widget:
                if field.description:
                    widget.setToolTip(field.description)
                self._inputs[name] = widget
                label = name.replace("_", " ").title()
                form.addRow(label, widget)

        scroll.setWidget(container)
        main_layout.addWidget(scroll)

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok
            | QDialogButtonBox.Cancel
            | QDialogButtonBox.RestoreDefaults
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        buttons.button(QDialogButtonBox.RestoreDefaults).clicked.connect(
            self.restore_defaults
        )
        main_layout.addWidget(buttons)

        self.resize(500, 400)

    def _create_widget(
        self, type_annotation, value, *, field: FieldInfo | None = None
    ) -> QWidget | None:
        # Unwrap Optional[T] -> T
        origin = get_origin(type_annotation)
        if origin is not None:
            args = get_args(type_annotation)
            if type(None) in args:
                # Find the non-None type
                for a in args:
                    if a is not type(None):
                        type_annotation = a
                        break

        # Check for Enum
        if isinstance(type_annotation, type) and issubclass(type_annotation, Enum):
            w = QComboBox()
            # Populate with Enum members
            for member in type_annotation:
                w.addItem(member.name, member.value)

            # Set current
            if isinstance(value, Enum):
                w.setCurrentText(value.name)
            elif value is not None:
                # Try to matches value or name
                idx = w.findData(value)
                if idx >= 0:
                    w.setCurrentIndex(idx)
                else:
                    idx = w.findText(str(value))
                    if idx >= 0:
                        w.setCurrentIndex(idx)
            return w

        lower, upper, lower_open, upper_open = _field_bounds(field)

        # Basic types
        if type_annotation is int:
            w = QSpinBox()
            low = -_INT_LIMIT if lower is None else math.ceil(lower + lower_open)
            high = _INT_LIMIT if upper is None else math.floor(upper - upper_open)
            w.setRange(int(low), int(high))
            w.setValue(int(value) if value is not None else 0)
            return w
        elif type_annotation is float:
            w = ScientificDoubleSpinBox()
            w.setRange(
                -_FLOAT_LIMIT if lower is None else lower,
                _FLOAT_LIMIT if upper is None else upper,
            )
            current = float(value) if value is not None else 0.0
            w.setValue(current)
            # Step in the value's own decade so 1e-9 does not jump by 0.1.
            magnitude = abs(current)
            w.setSingleStep(
                10.0 ** math.floor(math.log10(magnitude)) if magnitude > 0 else 0.1
            )
            return w
        elif type_annotation is bool:
            w = QCheckBox()
            w.setChecked(bool(value) if value is not None else False)
            return w
        elif type_annotation is str:
            w = QLineEdit()
            w.setText(str(value) if value is not None else "")
            return w

        # Fallback for complex types (e.g. lists/tuples) -> Text Edit?
        # For now, simplistic approach: use QLineEdit and try to eval/repr?
        # Or just skip
        return None

    def restore_defaults(self) -> None:
        """Reset every input to the model's default value."""
        for name, widget in self._inputs.items():
            default = _field_default(self.model_class.model_fields[name])
            if default is None:
                continue
            if isinstance(widget, QDoubleSpinBox):
                widget.setValue(float(default))
            elif isinstance(widget, QSpinBox):
                widget.setValue(int(default))
            elif isinstance(widget, QCheckBox):
                widget.setChecked(bool(default))
            elif isinstance(widget, QLineEdit):
                widget.setText(str(default))
            elif isinstance(widget, QComboBox):
                key = default.value if isinstance(default, Enum) else default
                index = widget.findData(key)
                if index >= 0:
                    widget.setCurrentIndex(index)

    def get_settings(self) -> dict[str, Any]:
        """Collect the values currently shown in the form.

        Returns
        -------
        dict
            Field name to value for every generated input.
        """
        data = {}
        for name, widget in self._inputs.items():
            if isinstance(widget, QSpinBox):
                data[name] = widget.value()
            elif isinstance(widget, QDoubleSpinBox):
                data[name] = widget.value()
            elif isinstance(widget, QCheckBox):
                data[name] = widget.isChecked()
            elif isinstance(widget, QLineEdit):
                data[name] = widget.text()
            elif isinstance(widget, QComboBox):
                # Retrieve Enum value
                data[name] = widget.currentData()
        return data
