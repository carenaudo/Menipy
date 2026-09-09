"""Measure active-history refresh/save before proposing a storage migration."""

import json
import os
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from check_gui_execution import isolate


def main():
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    root = (
        Path(__file__).resolve().parents[1] / ".cache" / "history-profile" / uuid4().hex
    )
    isolate(root)
    from PySide6.QtWidgets import QApplication

    from menipy.gui.views.main_window import MainWindow
    from menipy.models.results import MeasurementResult

    app = QApplication([])
    window = MainWindow()
    window.show()
    app.processEvents()
    panel = window.results_panel_ctrl
    report = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cases": [],
    }
    for count in (100, 1000):
        panel.history.measurements = [
            MeasurementResult(
                id=str(i),
                timestamp=datetime.now(timezone.utc),
                pipeline="sessile",
                file_path=f"image-{i}.png",
                results={
                    "diameter_mm": 2.0,
                    "height_mm": 1.0,
                    "volume_uL": 3.0,
                    "theta_left_deg": 90,
                    "theta_right_deg": 91,
                },
            )
            for i in range(count)
        ]
        times = []
        for _ in range(4):
            start = time.perf_counter()
            panel.update_history()
            app.processEvents()
            refresh = time.perf_counter() - start
            start = time.perf_counter()
            panel.history.retry_save()
            times.append({"refresh_s": refresh, "save_s": time.perf_counter() - start})
        report["cases"].append({"count": count, "runs_first_then_warm": times})
    (root / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    window.close()
    print(root / "report.json")


if __name__ == "__main__":
    main()
