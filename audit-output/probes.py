"""Reproducible audit probes. Writes only beneath audit-output via isolation.py.

Run with: uv run --no-sync python audit-output/probes.py
These probes characterize existing behavior, not assert desired fixes.
"""
import json
import os
import platform
import statistics
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace, MethodType
from unittest.mock import patch
from isolation import isolate, ROOT, OUT

state = isolate('probes')
import cv2
import numpy as np
from PySide6.QtCore import QTimer, qVersion
from PySide6.QtWidgets import QApplication, QMessageBox
from menipy.gui.app import _configure_qt, _register_qrc
from menipy.gui.views import main_window as mw
from menipy.models.context import Context
from menipy.models.results import MeasurementResult, ResultsHistory, build_persisted_analysis
from menipy.gui.controllers.pipeline_controller import PipelineController
from menipy.gui.controllers.main_controller import MainController
from menipy.gui.views.results_panel import ResultsPanel

_register_qrc()
app = QApplication.instance() or QApplication([])
_configure_qt(app)
dialogs=[]
QMessageBox.warning = staticmethod(lambda *a, **k: dialogs.append(str(a[2])) or QMessageBox.Ok)
QMessageBox.critical = staticmethod(lambda *a, **k: dialogs.append(str(a[2])) or QMessageBox.Ok)
report={'environment':{'python':sys.version,'platform':platform.platform(),
    'cpu':platform.processor(),'logical_cpu_count':os.cpu_count(),'qt':qVersion(),
    'opencv':cv2.__version__,'numpy':np.__version__,'qt_platform':app.platformName()},
    'probes':{},'benchmarks':{}}

def record(name, fn):
    try:
        report['probes'][name]=fn()
    except Exception:
        report['probes'][name]={'probe_error':traceback.format_exc()}
    flush()

def flush():
    (OUT/'evidence.json').write_text(json.dumps(report,indent=2,default=str),encoding='utf-8')

def timed(fn,n=3):
    times=[]
    value=None
    for _ in range(n):
        start=time.perf_counter(); value=fn(); times.append((time.perf_counter()-start)*1000)
    return {'first_ms':times[0], 'warm_median_ms':statistics.median(times[1:]) if n>1 else None,
            'samples_ms':times}, value

record('startup_services',lambda:{'settings_module':mw.AppSettings.__module__,
       'runner':str(mw.PipelineRunner),'vm':str(mw.RunViewModel),'sops':str(mw.SopService)})
start=time.perf_counter(); window=mw.MainWindow(); window.resize(1200,800); window.show(); app.processEvents()
report['benchmarks']['window_after_import_ms']=(time.perf_counter()-start)*1000
ctrl=window.main_controller.pipeline_ctrl
setup=window.setup_panel_ctrl
record('workflow_controls',lambda:{'run_enabled_empty':setup.runAllBtn.isEnabled(),
       'pipeline_buttons':{str(k):v.text() if v else None for v,k in setup._pipeline_button_map.items()},
       'dynamic_button_present':getattr(setup,'dynamicSessileBtn',None) is not None,
       'window_size':[window.width(),window.height()]})

def batch_probe():
    folder=state/'batch'; folder.mkdir(exist_ok=True)
    for n in range(3): cv2.imwrite(str(folder/f'frame{n}.png'),np.full((40,60,3),n,dtype=np.uint8))
    setup.set_batch_path(str(folder)); setup._apply_mode('batch',emit=False); setup._populate_batch_selection()
    params=setup.gather_run_params(); calls=[]
    with patch.object(ctrl,'_collect_acquisition_inputs',return_value=(True,{'roi':(0,0,60,40),'needle_rect':(0,0,10,20)})), patch.object(ctrl,'_run_pipeline_direct',side_effect=lambda cls,**kw:calls.append(kw)):
        ctrl.run_full()
    return {'files':3,'params':params,'submitted_count':len(calls),'submitted_sources':[x.get('image') for x in calls]}
record('static_batch',batch_probe)

def calibration_gate():
    setup._apply_mode('single',emit=False)
    warnings_before=len(dialogs)
    ready,payload=ctrl._collect_acquisition_inputs()
    _,_,kwargs,warnings=ctrl._build_pipeline_run_kwargs(auto_calibrate=False)
    return {'normal_run_without_regions_allowed':ready,'messages':dialogs[warnings_before:],
            'advanced_builder_scale_without_regions':kwargs.get('scale'),'warnings':warnings}
record('missing_calibration',calibration_gate)

def worker_probe():
    from menipy.gui.services import pipeline_runner as runner_module
    from menipy.gui.viewmodels.run_vm import RunViewModel
    from menipy.pipelines.base import PipelineBase
    runner=runner_module.PipelineRunner(); runner.pool.setMaxThreadCount(1)
    vm=RunViewModel(runner); rows=[]; events=[]; begin=threading.Event(); worker_ids=[]
    class ProbePipeline(PipelineBase):
        def run(self,**kwargs):
            begin.set(); worker_ids.append(threading.get_ident()); time.sleep(.35)
            return Context(image=np.zeros((8,8,3),dtype=np.uint8), image_path='original-sessile.png',
                results={'diameter_mm':2.0},qa={'ok':False,'rejection_reasons':['audit_rejection']})
    panel=SimpleNamespace(history=SimpleNamespace(add_measurement=rows.append),update_history=lambda:None)
    panel.update_single_measurement=MethodType(ResultsPanel.update_single_measurement,panel)
    pc=PipelineController(window=window,setup_ctrl=SimpleNamespace(gather_run_params=lambda:{'name':'pendant'}),
        preview_panel=SimpleNamespace(),results_panel=panel,preprocessing_ctrl=None,edge_detection_ctrl=None,
        pipeline_map={},sops=None,run_vm=vm,log_view=None)
    vm.results_ready.connect(pc.on_results_ready)
    runner.finished.connect(lambda p:events.append((time.perf_counter(),p)))
    before=time.perf_counter()
    with patch.dict(runner_module.PIPELINE_MAP,{'audit':ProbePipeline}):
        runner.run('audit','original-sessile.png',None)
        begin.wait(2)
        stop_at=time.perf_counter()
        MainController.stop_pipeline(SimpleNamespace(window=SimpleNamespace(runner=runner,statusBar=window.statusBar)))
        deadline=time.perf_counter()+3
        while not events and time.perf_counter()<deadline: app.processEvents(); time.sleep(.005)
    runner.pool.waitForDone(1000)
    return {'submitted_pipeline':'audit (synthetic sessile context)', 'selected_at_completion':'pendant',
        'input_accepted':False,'completed_after_stop':bool(events),
        'completion_after_stop_ms':(events[0][0]-stop_at)*1000 if events else None,
        'total_ms':(time.perf_counter()-before)*1000,'worker_is_background':worker_ids[0]!=threading.get_ident(),
        'stored_rows':[r.model_dump(mode='json') for r in rows]}
record('worker_rejection_switch_and_stop',worker_probe)

def persistence_probe():
    h=ResultsHistory(); h.measurements=[]
    m=MeasurementResult(id='audit',timestamp=datetime.now(),pipeline='sessile',results={'diameter_mm':2})
    with patch('builtins.open',side_effect=PermissionError('injected read-only destination')):
        value=h.add_measurement(m)
    return {'call_returned':value,'in_memory_count':len(h.measurements),'error_propagated':False}
record('history_write_failure',persistence_probe)

def preset_probe():
    from menipy.gui.services.sop_service import SopService,Sop
    from menipy.gui.controllers.sop_controller import SopController
    service=SopService('audit-sops.json')
    obj=SimpleNamespace(sops=service,window=window,collect_included_stages=lambda:['preprocessing'],
          pipeline_getter=lambda:'sessile',_refresh_sop_combo=lambda **kw:None,_apply_selected_sop=lambda:None)
    with patch('menipy.gui.controllers.sop_controller.QInputDialog.getText',return_value=('audit-preset',True)):
        SopController.on_add_sop(obj)
    created=SopService('audit-sops.json').get('sessile','audit-preset')
    service.upsert('sessile',Sop('configured',['preprocessing'],{'preprocessing':{'sentinel':123}}))
    states=[]; obj._step_widgets=[SimpleNamespace(step_name='preprocessing',set_included=states.append)]
    obj._selected_sop_key=lambda:'configured'
    SopController._apply_selected_sop(obj)
    return {'created_params_after_reload':created.params,'included_stages':created.include_stages,
            'configured_params_retained_by_service':service.get('sessile','configured').params,
            'apply_calls':states,'gui_sop_service':str(window.sops)}
record('sop_roundtrip',preset_probe)

def settings_probe():
    from menipy.gui.services.settings_service import AppSettings
    p=state/'roundtrip.json'
    s=AppSettings(path=p,unit_system='CGS',marker_config={'radius':9},overlay_config={'contour_color':'#abcdef'},
         results_hidden_columns={'sessile':['volume_uL']},guided_splitter_sizes=[350,900])
    s.save(); reread=AppSettings.load(p)
    fallback=mw.AppSettings.load(); fallback.unit_system='CGS'; fallback.save()
    fresh=mw.AppSettings.load()
    return {'real_service_roundtrip':{'units':reread.unit_system,'markers':reread.marker_config,
       'overlay':reread.overlay_config,'columns':reread.results_hidden_columns,'layout':reread.guided_splitter_sizes},
       'main_window_fallback_after_save_and_reload':fresh.unit_system}
record('settings_roundtrip',settings_probe)

from menipy.common.auto_calibrator import run_auto_calibration
from menipy.pipelines.discover import PIPELINE_MAP
for mode,filename in [('sessile','sessile_needle_reference.png'),('pendant','pendant_water_reference.png')]:
    path=ROOT/'data'/'samples'/filename; image=cv2.imread(str(path))
    bench={'input':filename,'shape':list(image.shape)}
    bench['preview'],_=timed(lambda:window.preview_panel.load_path(str(path)))
    bench['calibration'],cal=timed(lambda:run_auto_calibration(image,mode))
    kwargs={'image':str(path),'scale':{'px_per_mm':100.0},'needle_diameter_mm':1.0,
            'roi':cal.roi_rect,'needle_rect':cal.needle_rect,'substrate_line':cal.substrate_line,
            'physics':{'rho1':1000.0,'rho2':1.2,'g':9.80665}}
    try:
        bench['pipeline'],ctx=timed(lambda:PIPELINE_MAP[mode]().run(**kwargs))
        bench['last_stage_timings_ms']=ctx.timings_ms
        bench['last_qa']=ctx.qa
        bench['calibration_note']='Explicit synthetic 100 px/mm: timings only, not physical accuracy validation.'
    except Exception: bench['pipeline_error']=traceback.format_exc()
    report['benchmarks'][mode]=bench; flush()

history=window.results_panel_ctrl.history
for count in (10,100,1000):
    history.measurements=[MeasurementResult(id=f'audit-{i}',timestamp=datetime.now(),pipeline='sessile',
        file_name=f'image{i}.png',results={'diameter_mm':2.0,'height_mm':1.0,'volume_uL':3.0,
        'theta_left_deg':90,'theta_right_deg':90}) for i in range(count)]
    bench={}
    bench['save'],_=timed(history._save_history)
    bench['table_refresh'],_=timed(window.results_panel_ctrl.update_history)
    bench['history_bytes']=history._history_file.stat().st_size
    bench['default_limit']=history.max_history
    report['benchmarks'][f'history_{count}']=bench; flush()
history.measurements=[]
window.close(); app.processEvents()
report['dialogs']=dialogs
flush()
print(json.dumps({'complete':str(OUT/'evidence.json'),'probe_names':list(report['probes'])},indent=2))
