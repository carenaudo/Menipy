"""Fresh-process startup samples; OS disk caches are not flushed."""
import json
import sys
import time
started=time.perf_counter()
from isolation import isolate, OUT
sample=sys.argv[1]
isolate('startup-'+sample)
from menipy.gui.app import _configure_qt, _register_qrc
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTimer
_register_qrc(); app=QApplication([]); _configure_qt(app)
from menipy.gui.views.main_window import MainWindow
import_done=time.perf_counter()
window=MainWindow(); window.resize(1200,800); window.show()
def finish():
    result={'sample':sample,'startup_to_event_loop_ms':(time.perf_counter()-started)*1000,
       'imports_and_qt_ms':(import_done-started)*1000,'qt_platform':app.platformName(),
       'runner':str(window.runner),'note':'Fresh process, existing OS cache; no uv process overhead.'}
    (OUT/f'startup-{sample}.json').write_text(json.dumps(result,indent=2)); window.close(); app.quit()
QTimer.singleShot(0,finish)
app.exec()
