"""Run nearest existing tests with isolated persistence and fresh test temp dirs."""
import os
import sys
import uuid
from isolation import isolate, OUT
isolate('tests')
os.environ['QT_QPA_PLATFORM']='offscreen'
import pytest
sys.exit(pytest.main(['-q','-p','no:cacheprovider','--basetemp',str(OUT/'state'/('pytest-'+uuid.uuid4().hex)),
    '--junitxml',str(OUT/'tests.xml'),
    'tests/test_gui_startup_preview.py','tests/test_guided_ui_simplification.py',
    'tests/test_pipeline_runner.py','tests/test_phase_d_dynamic_sessile.py',
    'tests/test_results_panel_history.py','tests/test_phase_a_results_gui.py',
    'tests/test_smoke_controller_flows.py']))
