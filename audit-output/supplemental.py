"""Audit isolation, timing, dynamic-mode visibility, and settings cross-process reads."""
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from isolation import isolate, OUT, ROOT
state=isolate('supplemental')
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTimer
from menipy.gui.app import _register_qrc, _configure_qt
_register_qrc(); app=QApplication([]); _configure_qt(app)
from menipy.gui.views.main_window import MainWindow
from menipy.gui.dialogs.analysis_settings_dialog import AnalysisSettingsDialog
from menipy.gui.services.settings_service import AppSettings
from menipy.common.plugin_db import PluginDB
from menipy.models.config import PreprocessingSettings

mode=sys.argv[1] if len(sys.argv)>1 else 'write'
if mode=='write':
    w=MainWindow(); w.show(); app.processEvents()
    dynamic=w.setup_panel_ctrl.dynamicSessileBtn
    facts={'dynamic_button_exists':dynamic is not None,'dynamic_button_visible':dynamic.isVisible(),
           'dynamic_button_parent_visible':dynamic.parentWidget().isVisible(),
           'settings_runtime':str(type(w.settings))}
    # Current full-run fallback uses direct execution. Inject a deterministic wait,
    # allowing the event loop responsiveness of that call boundary to be measured.
    from menipy.gui.controllers.pipeline_controller import PipelineController
    from menipy.models.context import Context
    class Slow:
        def run(self,**kw): time.sleep(.25); return Context()
    fired=[]; start=time.perf_counter(); QTimer.singleShot(20,lambda:fired.append((time.perf_counter()-start)*1000))
    w.main_controller.pipeline_ctrl._run_pipeline_direct(Slow)
    app.processEvents()
    facts['timer_due_ms']=20; facts['timer_observed_ms']=fired[0] if fired else None
    AppSettings(path=state/'roundtrip.json',unit_system='CGS',marker_config={'size':7},
        overlay_config={'color':'#abcdef'},results_hidden_columns={'sessile':['volume_uL']},
        guided_splitter_sizes=[350,900]).save()
    d=AnalysisSettingsDialog('sessile',preprocessing=PreprocessingSettings())
    d._preproc.resize.target_width=777; d.persist(); d._settings_store.sync(); d.close()
    db=PluginDB(); db.set_setting('audit_profile',json.dumps({'threshold':123}))
    w.settings.unit_system='CGS'; w.settings.save(); w.close()
    (OUT/'supplemental-write.json').write_text(json.dumps(facts,indent=2))
else:
    s=AppSettings.load(state/'roundtrip.json')
    d=AnalysisSettingsDialog('sessile')
    facts={'real_settings_after_process_restart':{'units':s.unit_system,'markers':s.marker_config,
       'overlays':s.overlay_config,'columns':s.results_hidden_columns,'splitters':s.guided_splitter_sizes},
       'analysis_dialog_target_width_after_process_restart':d.preprocessing_settings().resize.target_width,
       'plugin_setting_after_process_restart':PluginDB().get_setting('audit_profile')}
    w=MainWindow(); facts['main_window_units_after_process_restart']=w.settings.unit_system
    d.close(); w.close(); (OUT/'supplemental-read.json').write_text(json.dumps(facts,indent=2))
print(json.dumps(facts,indent=2))
