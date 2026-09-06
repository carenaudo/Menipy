"""Run GUI execution regressions without touching a user's settings or databases.

Usage: uv run --extra test python tools/check_gui_execution.py [pytest arguments]
"""

import os
import shutil
import sys
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def isolate(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    home = root / "home"
    home.mkdir(exist_ok=True)
    Path.home = classmethod(lambda cls: home)
    from PySide6 import QtCore

    original = QtCore.QSettings

    class IsolatedSettings(original):
        def __init__(self, *args, **kwargs):
            super().__init__(str(root / "qt.ini"), original.Format.IniFormat)

    QtCore.QSettings = IsolatedSettings
    from menipy.common.material_db import MaterialDB
    from menipy.common.plugin_db import DB_DEFAULT, PluginDB

    for source, target in (
        (DB_DEFAULT, root / "plugins.sqlite"),
        (ROOT / "menipy_materials.sqlite", root / "materials.sqlite"),
    ):
        if source.exists() and not target.exists():
            shutil.copy2(source, target)
    PluginDB.__init__.__defaults__ = (root / "plugins.sqlite",)
    MaterialDB.__init__.__defaults__ = (root / "materials.sqlite",)
    return root


def main():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    output = isolate(ROOT / ".cache" / "execution-checks" / uuid4().hex)
    import pytest

    defaults = [
        "tests/test_execution_lifecycle.py",
        "tests/test_gui_execution_integration.py",
        "tests/test_gui_startup_preview.py",
        "tests/test_smoke_controller_flows.py",
        "tests/test_setup_panel.py",
        "tests/test_guided_ui_simplification.py",
        "tests/test_pipeline_runner.py",
        "tests/test_results_panel_history.py",
        "tests/test_phase_a_results_gui.py",
        "tests/test_calibration_wizard_dialog.py",
        "tests/test_auto_calibrator.py",
        "tests/test_phase_d_dynamic_sessile.py",
        "tests/test_pendant_pipeline.py",
        "tests/test_adsa_geometry_phase_b.py",
        "tests/test_phase_a_diagnostics.py",
    ]
    print(f"Isolated test evidence: {output}", flush=True)
    return pytest.main(
        [
            "-q",
            "-p",
            "no:cacheprovider",
            "--basetemp",
            str(output / "tmp"),
            "--junitxml",
            str(output / "tests.xml"),
            *(sys.argv[1:] or defaults),
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
