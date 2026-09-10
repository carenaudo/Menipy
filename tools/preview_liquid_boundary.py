"""Render a reproducible calibration-boundary regression example offscreen."""

import json
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np


def main():
    from check_gui_execution import isolate

    root = (
        Path(__file__).resolve().parents[1] / ".cache" / "liquid-boundary" / uuid4().hex
    )
    isolate(root / "state")
    from PySide6.QtWidgets import QApplication

    from menipy.common.auto_calibrator import CalibrationResult
    from menipy.common.liquid_boundary import update_calibration_boundary
    from menipy.common.sessile_detection import detect_sessile_drop_contour
    from menipy.gui.dialogs.calibration_wizard_dialog import CalibrationWizardDialog

    app = QApplication.instance() or QApplication([])
    image = np.full((300, 400, 3), 230, np.uint8)
    cv2.ellipse(image, (200, 220), (90, 65), 0, 180, 360, (20, 20, 20), -1)
    cv2.rectangle(image, (0, 220), (399, 299), (100, 100, 100), -1)
    cv2.ellipse(image, (200, 220), (22, 12), 0, 180, 360, (210, 210, 210), -1)
    detected = detect_sessile_drop_contour(image, substrate_y=220)
    result = CalibrationResult(
        drop_contour=detected.contour,
        contact_points=detected.contact_points,
        substrate_line=((50, 220), (350, 220)),
    )
    dialog = CalibrationWizardDialog(image, pipeline_name="sessile")
    dialog.result = result
    before = dialog._draw_overlays()
    update_calibration_boundary(result, "sessile")
    after = dialog._draw_overlays()
    cv2.putText(
        before, "Before", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 30, 30), 2
    )
    cv2.putText(
        after, "After", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 30, 30), 2
    )
    cv2.imwrite(str(root / "comparison.png"), np.hstack([before, after]))
    report = {
        "measured_points_preserved": len(result.drop_contour),
        "display_boundary_points": len(result.liquid_boundary),
    }
    (root / "report.json").write_text(json.dumps(report, indent=2))
    dialog.close()
    app.processEvents()
    print(root)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
