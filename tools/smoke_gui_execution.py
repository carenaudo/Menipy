"""Native Qt smoke check with synthetic jobs and isolated persistence."""

import json
import os
import platform
import time
from uuid import uuid4

import numpy as np
from check_gui_execution import ROOT, isolate


def main():
    os.environ["QT_QPA_PLATFORM"] = "windows"
    output = isolate(ROOT / ".cache" / "native-execution-smoke" / uuid4().hex)
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication

    from menipy.gui.app import _configure_qt, _register_qrc
    from menipy.gui.services.pipeline_runner import PIPELINE_MAP, RunRequest
    from menipy.gui.views.main_window import MainWindow
    from menipy.models.context import Context
    from menipy.models.results import get_results_history

    _register_qrc()
    app = QApplication([])
    _configure_qt(app)
    window = MainWindow()
    image = np.full((240, 320, 3), 235, dtype=np.uint8)
    image[30:150, 120:200] = 35
    window.preview_panel.display(image)
    window.show()
    beats, states, failures = [], [], []
    report = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "image_shape": list(image.shape),
        "worker_delay_ms": 500,
    }
    heartbeat = QTimer()
    heartbeat.setInterval(20)
    heartbeat.timeout.connect(lambda: beats.append(time.perf_counter()))
    heartbeat.start()
    window.runner.state_changed.connect(lambda _, state: states.append(state))

    class SyntheticPipeline:
        def __init__(self, **kwargs):
            pass

        def run(self, **kwargs):
            time.sleep(0.5)
            return Context(
                image=image.copy(),
                results={"diameter_mm": 123},
                qa={"ok": False, "rejection_reasons": ["synthetic_rejection"]},
            )

    PIPELINE_MAP["sessile"] = SyntheticPipeline
    phase = [0]
    stop_time = [None]
    analysis_start = [None]

    def start():
        capture_start = time.perf_counter()
        window.grab().save(str(output / "startup.png"))
        report["startup_capture_ms"] = (time.perf_counter() - capture_start) * 1000
        request = RunRequest.create(
            "sessile",
            {"image": "synthetic.png"},
            revision=window.pipeline_ctrl._revision(),
        )
        analysis_start[0] = time.perf_counter()
        window.runner.submit(request)
        QTimer.singleShot(60, window.setup_panel_ctrl.pendantBtn.click)

    def finish(completion):
        try:
            if phase[0] == 0:
                end = time.perf_counter()
                active_beats = [
                    beat for beat in beats if analysis_start[0] <= beat <= end
                ]
                report["analysis_heartbeat_count"] = len(active_beats)
                report["analysis_max_heartbeat_gap_ms"] = (
                    max(np.diff([analysis_start[0], *active_beats, end])) * 1000
                )
                assert len(active_beats) >= 5
                records = get_results_history().measurements
                record = next(m for m in records if m.id == completion.request.job_id)
                assert record.pipeline == "sessile" and not record.accepted
                assert record.file_path == "synthetic.png" and record.results == {}
                window.grab().save(str(output / "rejected-history.png"))
                report["rejection_and_identity"] = "passed"
                phase[0] = 1
                window.runner.submit(
                    RunRequest.create("sessile", {}, operation="calibration"),
                    lambda params, token: time.sleep(0.35),
                )

                def stop():
                    stop_time[0] = time.perf_counter()
                    window.runner.cancel()
                    window.grab().save(str(output / "stopping.png"))

                QTimer.singleShot(50, stop)
            elif phase[0] == 1:
                assert completion.state == "cancelled" and completion.value is None
                report["native_call_cancellation_ms"] = (
                    time.perf_counter() - stop_time[0]
                ) * 1000
                phase[0] = 2

                def cooperative(params, token):
                    while True:
                        token.check()
                        time.sleep(0.005)

                window.runner.submit(
                    RunRequest.create("sessile", {}, operation="calibration"),
                    cooperative,
                )

                def stop_cooperative():
                    stop_time[0] = time.perf_counter()
                    window.runner.cancel()

                QTimer.singleShot(50, stop_cooperative)
            else:
                assert completion.state == "cancelled"
                report["cooperative_cancellation_ms"] = (
                    time.perf_counter() - stop_time[0]
                ) * 1000
                report["heartbeat_count"] = len(beats)
                report["max_heartbeat_gap_ms"] = max(np.diff(beats), default=0) * 1000
                report["states"] = states
                assert len(beats) >= 10
                (output / "report.json").write_text(
                    json.dumps(report, indent=2), encoding="utf-8"
                )
                window.close()
        except Exception as exc:
            failures.append(str(exc))
            window.close()

    window.runner.finished.connect(finish)
    QTimer.singleShot(200, start)
    QTimer.singleShot(15000, window.close)
    app.exec()
    if failures or phase[0] != 2 or "states" not in report:
        raise RuntimeError(f"Native smoke failed: {failures}")
    print(json.dumps(report, indent=2))
    print(f"Native screenshots and timings: {output}")


if __name__ == "__main__":
    main()
