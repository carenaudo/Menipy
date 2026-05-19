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
