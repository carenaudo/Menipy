"""
Dialog for configuring plugin settings based on Pydantic models.
"""
from typing import Type, Any, Dict, get_origin, get_args
from enum import Enum
from pydantic import BaseModel
from PySide6.QtWidgets import (
    QDialog, QFormLayout, QDialogButtonBox, QVBoxLayout,
    QSpinBox, QDoubleSpinBox, QCheckBox, QLineEdit, QComboBox,
    QLabel, QWidget, QScrollArea
)
from PySide6.QtCore import Qt

class PluginConfigDialog(QDialog):
    def __init__(self, model_class: Type[BaseModel], current_values: Dict[str, Any], parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Configure {model_class.__name__}")
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
            value = current_values.get(name, field.default)
            # Handle PydanticUndefined or factory
            if value is None and field.default_factory:
                 try:
                    value = field.default_factory()
                 except:
                    pass
            
            # If still PydanticUndefined (required field missing), set sensible default for type
            # (Though passed current_values should ideally have it or we rely on type defaults)

            widget = self._create_widget(field.annotation, value)
            if widget:
                self._inputs[name] = widget
                label = name.replace("_", " ").title()
                form.addRow(label, widget)

        scroll.setWidget(container)
        main_layout.addWidget(scroll)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        main_layout.addWidget(buttons)
        
        self.resize(500, 400)

    def _create_widget(self, type_annotation, value) -> QWidget | None:
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

        # Basic types
        if type_annotation is int:
            w = QSpinBox()
            w.setRange(-999999, 999999)
            w.setValue(int(value) if value is not None else 0)
            return w
        elif type_annotation is float:
            w = QDoubleSpinBox()
            w.setRange(-999999.0, 999999.0)
            w.setSingleStep(0.1)
            w.setDecimals(4)
            w.setValue(float(value) if value is not None else 0.0)
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

    def get_settings(self) -> Dict[str, Any]:
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
