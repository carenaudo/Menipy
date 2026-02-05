"""
Widget for embedding plugin configuration directly in the main UI.
Reuses logic from PluginConfigDialog but as a dockable/embeddable widget.
"""
from __future__ import annotations
import json
from typing import Dict, Any, Type, Optional

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QFormLayout, QScrollArea, 
    QLabel, QSpinBox, QDoubleSpinBox, QCheckBox, 
    QLineEdit, QComboBox, QGroupBox, QPushButton, QHBoxLayout
)
from PySide6.QtCore import Signal, Qt

from pydantic import BaseModel
from menipy.common.plugin_settings import get_detector_settings_model, resolve_plugin_settings
from menipy.gui.dialogs.plugin_config_dialog import PluginConfigDialog # reusing _create_widget logic if possible, or duplicating for independence

class PluginSettingsWidget(QWidget):
    """
    A widget that displays configuration fields for a specific plugin method.
    Emits settingsChanged signal when values are modified.
    """
    settingsChanged = Signal(dict)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._current_method: Optional[str] = None
        self._current_model: Optional[Type[BaseModel]] = None
        self._inputs: Dict[str, QWidget] = {}
        self._block_signals = False
        
        # Layouts
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        
        # Header / Title (Optional)
        self.header_label = QLabel("Settings")
        self.header_label.setStyleSheet("font-weight: bold; color: #555;")
        self.main_layout.addWidget(self.header_label)
        
        # Scroll Area for Form
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QScrollArea.NoFrame)
        
        self.form_container = QWidget()
        self.form_layout = QFormLayout(self.form_container)
        self.form_layout.setContentsMargins(0, 5, 0, 5)
        
        self.scroll.setWidget(self.form_container)
        self.main_layout.addWidget(self.scroll)
        
        # Bottom Actions (Reset / Apply if needed? Or just live?)
        # For now, let's keep it live, but adding a "Reset to Defaults" is nice
        self.btn_layout = QHBoxLayout()
        self.btn_reset = QPushButton("Reset Defaults")
        self.btn_reset.clicked.connect(self._reset_to_defaults)
        self.btn_reset.setStyleSheet("text-align: left; padding: 4px;")
        self.btn_reset.setFlat(True)
        self.btn_reset.setCursor(Qt.PointingHandCursor)
        
        self.btn_layout.addWidget(self.btn_reset)
        self.btn_layout.addStretch()
        self.main_layout.addLayout(self.btn_layout)
        
        # Initial State
        self.set_empty_state()

    def set_empty_state(self):
        """Clear form and show placeholder."""
        self._clear_form()
        lbl = QLabel("No configurable settings.")
        lbl.setStyleSheet("color: gray; font-style: italic;")
        self.form_layout.addRow(lbl)
        self.btn_reset.hide()
        self.header_label.setText("Settings")

    def _clear_form(self):
        """Remove all rows from form layout."""
        while self.form_layout.count():
            item = self.form_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
        self._inputs.clear()

    def set_plugin(self, method_name: str, current_settings: Dict[str, Any] = None):
        """
        Build the settings form for the given plugin method.
        """
        self._current_method = method_name
        self._current_model = get_detector_settings_model(method_name)
        
        self._clear_form()
        self.header_label.setText(f"{method_name.replace('_', ' ').title()} Settings")
        
        if not self._current_model:
            self.set_empty_state()
            return

        self.btn_reset.show()

        # If no settings provided, use defaults
        vals = current_settings or {}
        
        # Generate fields based on Pydantic model
        # We can reuse logic from PluginConfigDialog, but for now I'll duplicate the loop 
        # to keep this widget self-contained and tweakable for side-panel use.
        
        model = self._current_model
        
        # Docstring as tooltip or description
        if model.__doc__:
            self.setToolTip(model.__doc__.strip())
        
        for name, field in model.model_fields.items():
            # Get current value or default
            val = vals.get(name, field.default)
            if val is None and field.default_factory:
                try:
                    val = field.default_factory()
                except:
                    pass

            # Create Widget
            widget = self._create_input_widget(field.annotation, val)
            
            if widget:
                self._inputs[name] = widget
                # Connect change signal
                self._connect_change_signal(widget)
                
                label_text = name.replace("_", " ").title()
                self.form_layout.addRow(label_text, widget)
                
                # Add tooltip from field description
                if field.description:
                    widget.setToolTip(field.description)
        
    def _create_input_widget(self, type_annotation, value) -> QWidget | None:
        """Create appropriate widget for type."""
        # Reuse logic similar to PluginConfigDialog
        from menipy.gui.dialogs.plugin_config_dialog import PluginConfigDialog
        # We can actually just instantiate a temp dialog to use its method if we made it static?
        # Or just copy-paste for now to avoid refactoring the other file yet.
        # Let's verify PluginConfigDialog._create_widget is instance method. It is.
        # I'll implement a simplified version here.
        
        from typing import get_origin, get_args
        from enum import Enum
        
        # Unwrap Optional
        origin = get_origin(type_annotation)
        if origin is not None:
             args = get_args(type_annotation)
             if type(None) in args:
                 for a in args:
                     if a is not type(None):
                         type_annotation = a
                         break
                         
        if isinstance(type_annotation, type) and issubclass(type_annotation, Enum):
            w = QComboBox()
            for member in type_annotation:
                w.addItem(member.name, member.value)
            if isinstance(value, Enum):
                w.setCurrentText(value.name)
            elif value is not None:
                idx = w.findData(value)
                if idx >= 0: w.setCurrentIndex(idx)
                else: 
                     idx = w.findText(str(value))
                     if idx >= 0: w.setCurrentIndex(idx)
            return w

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
             # Right align checkbox for form layout
            w.setChecked(bool(value) if value is not None else False)
            return w
            
        elif type_annotation is str:
            w = QLineEdit()
            w.setText(str(value) if value is not None else "")
            return w
            
        return None

    def _connect_change_signal(self, widget: QWidget):
        if isinstance(widget, (QSpinBox, QDoubleSpinBox)):
            widget.valueChanged.connect(self._on_field_changed)
        elif isinstance(widget, QCheckBox):
            widget.toggled.connect(self._on_field_changed)
        elif isinstance(widget, QLineEdit):
            widget.editingFinished.connect(self._on_field_changed)
        elif isinstance(widget, QComboBox):
            widget.currentIndexChanged.connect(self._on_field_changed)

    def _on_field_changed(self):
        if self._block_signals:
            return
        self.settingsChanged.emit(self.get_current_settings())

    def get_current_settings(self) -> Dict[str, Any]:
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
                data[name] = widget.currentData()
        return data

    def _reset_to_defaults(self):
        if not self._current_model:
            return
            
        # Block signals to prevent updates during batch update
        self._block_signals = True
        try:
            for name, widget in self._inputs.items():
                field = self._current_model.model_fields.get(name)
                if not field: continue
                
                # Get default
                val = field.default
                if val is None and field.default_factory:
                    try: val = field.default_factory()
                    except: pass
                
                # Set widget value
                if isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                    if val is not None: widget.setValue(val)
                elif isinstance(widget, QCheckBox):
                    if val is not None: widget.setChecked(bool(val))
                elif isinstance(widget, QLineEdit):
                    if val is not None: widget.setText(str(val))
                elif isinstance(widget, QComboBox):
                    # Handle enum default
                    from enum import Enum
                    if isinstance(val, Enum):
                        widget.setCurrentText(val.name)
                    elif val is not None:
                        idx = widget.findData(val)
                        if idx >= 0: widget.setCurrentIndex(idx)
        finally:
             self._block_signals = False
             
        # Emit one signal at end
        self.settingsChanged.emit(self.get_current_settings())
