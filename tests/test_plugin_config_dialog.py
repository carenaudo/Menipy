import pytest
from pydantic import BaseModel
from menipy.gui.dialogs.plugin_config_dialog import PluginConfigDialog
from PySide6.QtWidgets import QSpinBox, QLineEdit

class DummyModel(BaseModel):
    foo: int = 42
    bar: str = "hello"

def test_plugin_config_dialog_structure(qtbot):
    # qtbot fixture ensures QApplication exists
    current = {"foo": 10, "bar": "world"}
    dlg = PluginConfigDialog(DummyModel, current)
    qtbot.addWidget(dlg)
    
    assert dlg.windowTitle() == "Configure DummyModel"
    assert "foo" in dlg._inputs
    assert "bar" in dlg._inputs
    
    # Check values
    assert isinstance(dlg._inputs["foo"], QSpinBox)
    assert dlg._inputs["foo"].value() == 10
    
    assert isinstance(dlg._inputs["bar"], QLineEdit)
    assert dlg._inputs["bar"].text() == "world"

def test_get_settings(qtbot):
    current = {"foo": 10, "bar": "world"}
    dlg = PluginConfigDialog(DummyModel, current)
    qtbot.addWidget(dlg)
    
    # Modify via UI (simulated)
    dlg._inputs["foo"].setValue(99)
    dlg._inputs["bar"].setText("modified")
    
    settings = dlg.get_settings()
    assert settings["foo"] == 99
    assert settings["bar"] == "modified"
