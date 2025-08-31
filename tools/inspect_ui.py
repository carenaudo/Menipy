from importlib import import_module
from PySide6.QtWidgets import QApplication, QWidget
import sys
from pathlib import Path

# Ensure project root is on sys.path so 'src' package imports work when run directly
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

app = QApplication.instance() or QApplication([])
MW = import_module('src.menipy.gui.mainwindow').MainWindow
mw = MW()
print('loaded_ui:', getattr(mw,'_loaded_ui_path',None))
print('central widget type:', type(mw.centralWidget()))

for host_name in ('runHost','overlayHost','resultsHost'):
    host = mw.ui.findChild(QWidget, host_name)
    if host is None:
        print(f"{host_name}: NOT FOUND")
        continue
    children = [w.objectName() for w in host.findChildren(QWidget) if w.parent() is host]
    print(f"{host_name}: {type(host)} children: {children}")

# plugin dock
dock_widget = mw.plugin_dock.widget()
print('plugin_dock widget objectName:', dock_widget.objectName(), 'type:', type(dock_widget))
print('plugin_dock children:', [w.objectName() for w in dock_widget.findChildren(QWidget)])
# Print View menu actions
view_menu = mw.menuBar().findChild(type(mw.menuBar()), 'menu')
actions = [a.text() for a in mw.menuBar().actions()]
print('menu actions:', actions)
print('plugin_dock visible:', mw.plugin_dock.isVisible())

# check top-level centrallayout children names
cw = mw.centralWidget()
print('centralwidget children:', [w.objectName() for w in cw.findChildren(QWidget) if w.parent() is cw])

app.quit()
