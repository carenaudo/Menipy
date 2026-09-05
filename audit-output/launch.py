"""Launch the unchanged native GUI with isolated settings/databases for auditing."""
import json
import sys
import time
from isolation import isolate, OUT

started = time.perf_counter()
isolate('native')
from menipy.gui.app import _configure_qt, _register_qrc
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTimer
_register_qrc()
app = QApplication(sys.argv)
_configure_qt(app)
from menipy.gui.views.main_window import MainWindow
window = MainWindow()
window.setWindowTitle('Menipy — Isolated Improvement Audit')
window.resize(1200, 800)
window.show()

def record_startup():
    payload = {'startup_to_event_loop_s':time.perf_counter()-started,
               'runner':str(window.runner), 'run_vm':str(window.run_vm),
               'sops':str(window.sops), 'settings_type':str(type(window.settings)),
               'window_size':[window.width(),window.height()]}
    (OUT/'native-startup.json').write_text(json.dumps(payload,indent=2))
QTimer.singleShot(0, record_startup)
app.aboutToQuit.connect(window.main_controller.shutdown)
sys.exit(app.exec())
