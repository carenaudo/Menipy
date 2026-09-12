from __future__ import annotations

import cv2
import numpy as np

from menipy.common.auto_calibrator import run_auto_calibration
from menipy.gui.dialogs.calibration_wizard_dialog import CalibrationWizardDialog


def _pendant_image() -> np.ndarray:
    image = np.full((480, 640, 3), 200, dtype=np.uint8)
    cv2.rectangle(image, (300, 0), (340, 100), (30, 30, 30), -1)
    cv2.ellipse(image, (320, 200), (80, 120), 0, 0, 360, (30, 30, 30), -1)
    return image


def test_calibration_wizard_falls_back_to_supported_detector_for_pendant(qtbot):
    image = _pendant_image()
    dialog = CalibrationWizardDialog(image, pipeline_name="captive_bubble")
    qtbot.addWidget(dialog)

    result = dialog._run_best_auto_calibration(run_auto_calibration)

    assert result.drop_contour is not None
    assert result.needle_rect is not None
    assert result.confidence_scores["detector_pipeline"] == "pendant"


def test_worker_calibration_preserves_manual_regions(monkeypatch):
    from menipy.common import auto_calibrator
    from menipy.common.auto_calibrator import CalibrationResult
    from menipy.gui.services.calibration_service import CalibrationComputation

    manual = CalibrationResult(
        roi_rect=(2, 3, 8, 9),
        needle_rect=(5, 0, 4, 10),
        substrate_line=((0, 15), (20, 15)),
        confidence_scores={"roi": 1.0, "needle": 1.0, "substrate": 1.0},
    )
    detected = CalibrationResult(roi_rect=(0, 0, 1, 1), needle_rect=(0, 0, 1, 1))
    monkeypatch.setattr(auto_calibrator, "run_auto_calibration", lambda *_: detected)
    monkeypatch.setattr(
        auto_calibrator.AutoCalibrator, "_segment_image_adaptive", lambda _: None
    )
    received_roi = []

    def detect_drop(calibrator):
        received_roi.append(getattr(calibrator, "_roi_rect", None))
        return None, None, 0

    monkeypatch.setattr(auto_calibrator.AutoCalibrator, "_detect_drop_sessile", detect_drop)
    _, result = CalibrationComputation(
        np.zeros((32, 32, 3), dtype=np.uint8), "sessile", manual
    ).run()
    assert result.roi_rect == manual.roi_rect
    assert result.needle_rect == manual.needle_rect
    assert result.substrate_line == manual.substrate_line
    assert result.confidence_scores["substrate"] == 1.0
    assert received_roi == [manual.roi_rect]
