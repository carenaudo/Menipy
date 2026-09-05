"""Audit-only persistence redirection; does not change Menipy source or wiring."""
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))

def isolate(name):
    target = OUT / 'state' / name
    target.mkdir(parents=True, exist_ok=True)
    home = target / 'home'
    home.mkdir(exist_ok=True)
    Path.home = classmethod(lambda cls: home)
    from PySide6 import QtCore
    original = QtCore.QSettings
    class IsolatedQSettings(original):
        def __init__(self, *args, **kwargs):
            super().__init__(str(target / 'qt-settings.ini'), original.Format.IniFormat)
    QtCore.QSettings = IsolatedQSettings
    from menipy.common.plugin_db import PluginDB, DB_DEFAULT
    plugin_copy = target / 'plugins.sqlite'
    if not plugin_copy.exists() and DB_DEFAULT.exists():
        shutil.copy2(DB_DEFAULT, plugin_copy)
    PluginDB.__init__.__defaults__ = (plugin_copy,)
    from menipy.common.material_db import MaterialDB
    material_copy = target / 'materials.sqlite'
    if not material_copy.exists() and (ROOT / 'menipy_materials.sqlite').exists():
        shutil.copy2(ROOT / 'menipy_materials.sqlite', material_copy)
    MaterialDB.__init__.__defaults__ = (material_copy,)
    return target
