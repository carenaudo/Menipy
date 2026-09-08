"""Native layout evidence at a caller-supplied QT_SCALE_FACTOR, with isolated data."""

import json
import os
from datetime import datetime
from uuid import uuid4

import numpy as np
from check_gui_execution import ROOT, isolate


def main():
    output = isolate(ROOT / ".cache" / "presets-layout-smoke" / uuid4().hex)
    os.environ["QT_QPA_PLATFORM"] = "windows"
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication

    from menipy.common.auto_calibrator import CalibrationResult
    from menipy.gui.app import _configure_qt, _register_qrc
    from menipy.gui.dialogs.calibration_wizard_dialog import CalibrationWizardDialog
    from menipy.gui.views.main_window import MainWindow
    from menipy.models.results import MeasurementResult

    _register_qrc()
    app = QApplication([])
    _configure_qt(app)
    window = MainWindow()
    window.results_panel_ctrl.add_measurement(
        MeasurementResult(
            id="layout",
            timestamp=datetime.now(),
            pipeline="sessile",
            file_name="synthetic.png",
            results={"volume_uL": 2},
        )
    )
    window.resize(1200, 800)
    window.show()
    report = {
        "scale_factor": os.environ.get("QT_SCALE_FACTOR", "1"),
        "device_pixel_ratio": window.devicePixelRatioF(),
        "layouts": [],
    }
    dialog = CalibrationWizardDialog(
        np.zeros((1800, 1200, 3), dtype=np.uint8), parent=window
    )
    dialog.result = CalibrationResult(
        substrate_line=((0, 1500), (1200, 1500)),
        confidence_scores={"substrate": 0.25, "overall": 0.81},
    )
    dialog._update_region_statuses()
    sizes = [(1200, 800), (1366, 768)]

    def capture(index):
        table = window.results_panel_ctrl.table
        report["layouts"].append(
            {
                "requested": sizes[index],
                "actual": [window.width(), window.height()],
                "minimums": {
                    name: getattr(window, name).minimumSizeHint().height()
                    for name in (
                        "workflowBar",
                        "previewHost",
                        "inspectTabs",
                        "keyResultsHost",
                    )
                },
                "table_viewport_height": table.viewport().height(),
                "row_height": table.rowHeight(0),
            }
        )
        window.grab().save(str(output / f"window-{index}.png"))
        dialog.resize(*sizes[index])
        dialog.show()
        QTimer.singleShot(150, lambda: capture_dialog(index))

    def capture_dialog(index):
        dialog.grab().save(str(output / f"calibration-{index}.png"))
        dialog.hide()
        if index == 0:
            window.resize(*sizes[1])
            QTimer.singleShot(150, lambda: capture(1))
        else:
            (output / "report.json").write_text(
                json.dumps(report, indent=2), encoding="utf-8"
            )
            print(output)
            dialog.close()
            window.close()
            app.quit()

    QTimer.singleShot(350, lambda: capture(0))
    app.exec()
    for captured in report["layouts"]:
        assert captured["table_viewport_height"] >= captured["row_height"], captured
        if captured["actual"] != list(captured["requested"]):
            print(
                "Native OS constrained requested size:",
                captured["requested"],
                "to",
                captured["actual"],
            )


if __name__ == "__main__":
    main()
