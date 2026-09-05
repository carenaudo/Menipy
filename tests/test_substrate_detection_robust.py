"""Tests for robust sessile substrate baseline detection, tilt recovery, and interactive GUI arc drawing."""

from __future__ import annotations

import numpy as np
import pytest
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor, QImage
from PySide6.QtWidgets import QWidget

from menipy.common.sessile_detection import detect_sessile_substrate_robust
from menipy.gui.views.image_view import DRAW_ARC, ImageView
from menipy.gui.views.preview_panel import PreviewPanel


class TestDetectSessileSubstrateRobust:
    """Test robust bilateral sector substrate detector."""

    def test_clean_horizontal_baseline(self):
        # Create a synthetic image with a clear horizontal baseline at y=150
        img = np.full((200, 300), 220, dtype=np.uint8)
        img[150:, :] = 40  # Dark substrate below y=150

        line, conf, diag, prof = detect_sessile_substrate_robust(img)
        assert line is not None
        assert abs(line[0][1] - 150) <= 2
        assert abs(line[1][1] - 150) <= 2
        assert conf >= 0.75
        assert diag["status"] == "confident"
        assert diag["warning"] is False
        assert prof is not None
        assert prof.type == "line"

    def test_tilted_baseline_recovery(self):
        # Substrate tilted: y = 145 on left (x=0), y = 155 on right (x=300)
        # tilt angle = arctan(10 / 300) = 1.91 deg
        img = np.full((200, 300), 220, dtype=np.uint8)
        for x in range(300):
            sub_y = int(round(145 + (10.0 / 300.0) * x))
            img[sub_y:, x] = 40

        line, conf, diag, prof = detect_sessile_substrate_robust(img)
        assert line is not None
        assert abs(diag["tilt_deg"]) == pytest.approx(1.91, abs=1.0)
        assert line[0][1] < line[1][1]  # Left is higher (lower y) than right
        assert conf >= 0.70

    def test_low_contrast_triggers_doubtful_warning(self):
        # Substrate with very weak gradient difference (e.g. 130 vs 125)
        img = np.full((200, 300), 130, dtype=np.uint8)
        img[140:, :] = 125
        # Add random noise
        rng = np.random.default_rng(42)
        noise = rng.integers(-3, 4, size=img.shape, dtype=np.int16)
        noisy_img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        line, conf, diag, prof = detect_sessile_substrate_robust(noisy_img)
        # Should flag a warning or doubtful status due to low contrast
        assert diag["warning"] is True
        assert diag["status"] in ("doubtful", "failed")

    def test_blank_image_fails_safely(self):
        # Completely flat blank image -> no gradient
        img = np.full((200, 300), 128, dtype=np.uint8)
        line, conf, diag, prof = detect_sessile_substrate_robust(img)
        assert diag["status"] == "failed"
        assert diag["warning"] is True
        assert conf <= 0.35


class TestImageViewInteractiveArcDrawing:
    """Test interactive 3-point curved arc drawing on ImageView."""

    def test_three_click_arc_drawing(self, qtbot):
        view = ImageView()
        qtbot.addWidget(view)

        # Set image
        img = QImage(400, 300, QImage.Format_RGB32)
        img.fill(Qt.white)
        view.set_image(img)

        # Connect arc_drawn signal
        captured_arcs = []
        view.arc_drawn.connect(lambda pts: captured_arcs.append(pts))

        # Enter DRAW_ARC mode
        view.set_draw_mode(DRAW_ARC, color=QColor(255, 0, 0), tag="contact_arc")
        assert view._draw_mode == DRAW_ARC

        # Click 1: Left anchor (50, 150)
        p1 = QPointF(50.0, 150.0)
        view.mousePressEvent(
            _create_mouse_event(view, Qt.MouseButton.LeftButton, p1)
        )
        assert len(view._arc_points) == 1

        # Click 2: Right anchor (250, 150)
        p2 = QPointF(250.0, 150.0)
        view.mousePressEvent(
            _create_mouse_event(view, Qt.MouseButton.LeftButton, p2)
        )
        assert len(view._arc_points) == 2

        # Click 3: Crest point (150, 100) -> Completes arc
        p3 = QPointF(150.0, 100.0)
        view.mousePressEvent(
            _create_mouse_event(view, Qt.MouseButton.LeftButton, p3)
        )

        # Should emit arc_drawn with (p1, p2, p3)
        assert len(captured_arcs) == 1
        arc = captured_arcs[0]
        assert arc[0] == pytest.approx((50.0, 150.0))
        assert arc[1] == pytest.approx((250.0, 150.0))
        assert arc[2] == pytest.approx((150.0, 100.0))
        assert len(view._arc_points) == 0


class TestPreviewPanelBaselineWarningBanner:
    """Test warning banner in PreviewPanel."""

    def test_warning_banner_show_and_hide(self, qtbot):
        panel_widget = QWidget()
        qtbot.addWidget(panel_widget)
        preview = PreviewPanel(panel_widget, ImageView)
        panel_widget.show()

        # Initially hidden
        assert preview._baseline_warning_banner.isHidden() is True

        # Show warning on doubtful confidence
        preview.show_baseline_warning(confidence=0.48, status="doubtful")
        assert preview._baseline_warning_banner.isHidden() is False
        assert "48%" in preview._warning_label.text()

        # Dismiss warning
        preview.hide_baseline_warning()
        assert preview._baseline_warning_banner.isHidden() is True


def _create_mouse_event(view: ImageView, button: Qt.MouseButton, scene_pos: QPointF):
    from PySide6.QtCore import QEvent
    from PySide6.QtGui import QMouseEvent

    view_pos = view.mapFromScene(scene_pos)
    return QMouseEvent(
        QEvent.Type.MouseButtonPress,
        QPointF(view_pos),
        scene_pos,
        button,
        button,
        Qt.KeyboardModifier.NoModifier,
    )
