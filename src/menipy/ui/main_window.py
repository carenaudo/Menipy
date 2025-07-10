from __future__ import annotations

import numpy as np
import cv2
from PySide6.QtCore import QLineF, Qt
from PySide6.QtGui import QColor, QPen, QAction
from PySide6.QtWidgets import (
    QMessageBox,
    QDialog,
    QVBoxLayout,
    QCheckBox,
    QPushButton,
)
from pathlib import Path

from .. import plugins
from ..gui.base_window import BaseMainWindow
from ..gui.items import SubstrateLineItem
from ..pipelines import (
    analyze_pendant,
    analyze_sessile,
    analyze_sessile_alt,
    draw_pendant_overlays,
    draw_sessile_overlays,
    draw_sessile_overlays_alt,
)
from ..pipelines.pendant import HelperBundle as PendantHelpers
from ..pipelines.sessile import HelperBundle as SessileHelpers
from ..detection.needle import detect_vertical_edges
from ..detection.substrate import detect_substrate_line
from ..analysis.commons import extract_external_contour
from ..physics.contact_geom import geom_metrics


class PluginDialog(QDialog):
    """Dialog listing available plugins with check boxes."""

    def __init__(self, parent, plugins_map: dict[str, object], active: set[str]):
        super().__init__(parent)
        self.setWindowTitle("Plugins")
        layout = QVBoxLayout(self)
        self.checks: dict[str, QCheckBox] = {}
        for name in plugins_map:
            cb = QCheckBox(name)
            cb.setChecked(name in active)
            layout.addWidget(cb)
            self.checks[name] = cb
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn)


class MainWindow(BaseMainWindow):
    """Main window using refactored detection and analysis modules."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        plugins.load_plugins()
        self.active_plugins: set[str] = set()

        plugin_menu = self.menuBar().addMenu("Plugins")
        manage_action = QAction("Manage Plugins", self)
        manage_action.triggered.connect(self.open_plugin_manager)
        plugin_menu.addAction(manage_action)

    def open_plugin_manager(self) -> None:
        dlg = PluginDialog(self, plugins.PLUGINS, self.active_plugins)
        if dlg.exec():
            self.active_plugins = {
                name for name, cb in dlg.checks.items() if cb.isChecked()
            }
            if getattr(self, "original_image", None) is not None:
                self._apply_plugins()

    def _apply_plugins(self) -> None:
        img = self.original_image.copy()
        for name in list(self.active_plugins):
            func = plugins.PLUGINS.get(name)
            if callable(func):
                try:
                    img = func(img)
                except Exception as exc:  # pragma: no cover - runtime safety
                    QMessageBox.warning(self, "Plugin Error", str(exc))
            else:
                self.active_plugins.discard(name)
        self.image = img
        self._display_image(img)

    def load_image(self, path: Path) -> None:  # type: ignore[override]
        super().load_image(path)
        if self.active_plugins:
            self._apply_plugins()


    def detect_needle(self) -> None:  # type: ignore[override]
        if getattr(self, "image", None) is None or self.needle_rect is None:
            return
        x1, y1, x2, y2 = map(int, self.needle_rect)
        roi = self.image[y1:y2, x1:x2]
        try:
            top, bottom, length_px = detect_vertical_edges(roi)
        except ValueError as exc:
            QMessageBox.warning(self, "Needle Detection", str(exc))
            return
        self.px_per_mm_drop = length_px / max(self.calibration_tab.needle_length.value(), 1e-6)
        self.calibration_tab.set_metrics(scale=self.px_per_mm_drop)
        axis_x = top[0] + x1
        y_top = top[1] + y1
        y_bottom = bottom[1] + y1
        half_width = length_px / 2.0
        left_x = int(round(axis_x - half_width))
        right_x = int(round(axis_x + half_width))

        if self.needle_axis_item is not None:
            self.graphics_scene.removeItem(self.needle_axis_item)
            self.needle_axis_item = None
        for item in self.needle_edge_items:
            self.graphics_scene.removeItem(item)
        self.needle_edge_items.clear()

        pen = QPen(QColor("yellow"))
        pen.setWidth(2)
        left_item = self.graphics_scene.addLine(left_x, y_top, left_x, y_bottom, pen)
        right_item = self.graphics_scene.addLine(right_x, y_top, right_x, y_bottom, pen)
        self.needle_edge_items = [left_item, right_item]

    def _detect_substrate_line(self) -> None:
        if getattr(self, "image", None) is None or self.drop_rect is None:
            return
        x1, y1, x2, y2 = map(int, self.drop_rect)
        roi = self.image[y1:y2, x1:x2]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi
        mask = np.zeros_like(gray)
        try:
            p1, p2 = detect_substrate_line(gray, mask, "sessile")
        except Exception as exc:  # SubstrateNotFoundError or ValueError
            QMessageBox.warning(self, "Substrate Detection", str(exc))
            return
        p1 += np.array([x1, y1], float)
        p2 += np.array([x1, y1], float)
        linef = QLineF(p1[0], p1[1], p2[0], p2[1])
        if self.substrate_line_item is not None:
            self.graphics_scene.removeItem(self.substrate_line_item)
        self.substrate_line_item = SubstrateLineItem(linef)
        self.graphics_scene.addItem(self.substrate_line_item)
        self.substrate_line = (linef.x1(), linef.y1(), linef.x2(), linef.y2())
        self._keep_above = None

    def analyze_drop_image(self) -> None:  # type: ignore[override]
        if (
            getattr(self, "image", None) is None
            or self.drop_rect is None
            or self.px_per_mm_drop <= 0
        ):
            return
        x1, y1, x2, y2 = map(int, self.drop_rect)
        roi = self.image[y1:y2, x1:x2]
        try:
            contour = extract_external_contour(roi)
        except ValueError as exc:
            QMessageBox.warning(self, "Drop Analysis", str(exc))
            return
        contour += np.array([x1, y1])
        mode = getattr(self, "analysis_method", "pendant")

        if mode == "pendant":
            helpers = PendantHelpers(
                px_per_mm=self.px_per_mm_drop,
                needle_diam_mm=self.calibration_tab.needle_length.value(),
            )
            pm = analyze_pendant(roi, helpers)
            pm.contour += np.array([x1, y1])
            pm.apex = (pm.apex[0] + x1, pm.apex[1] + y1)
            pm.diameter_line = (
                (pm.diameter_line[0][0] + x1, pm.diameter_line[0][1] + y1),
                (pm.diameter_line[1][0] + x1, pm.diameter_line[1][1] + y1),
            )
            if pm.diameter_center is not None:
                pm.diameter_center = (
                    pm.diameter_center[0] + x1,
                    pm.diameter_center[1] + y1,
                )
            if pm.contact_line is not None:
                pm.contact_line = (
                    (pm.contact_line[0][0] + x1, pm.contact_line[0][1] + y1),
                    (pm.contact_line[1][0] + x1, pm.contact_line[1][1] + y1),
                )
            metrics = pm.derived
            panel = self.pendant_tab
            overlay = draw_pendant_overlays(self.image, pm)
        elif mode == "contact-angle":
            helpers = SessileHelpers(px_per_mm=self.px_per_mm_drop)
            if self.substrate_line_item is not None:
                line = self.substrate_line_item.line()
                substrate = (
                    (line.x1() - x1, line.y1() - y1),
                    (line.x2() - x1, line.y2() - y1),
                )
            else:
                substrate = ((0, 0), (1, 0))
            sm = analyze_sessile(roi, helpers, substrate)
            sm.contour += np.array([x1, y1])
            sm.apex = (sm.apex[0] + x1, sm.apex[1] + y1)
            sm.diameter_line = (
                (sm.diameter_line[0][0] + x1, sm.diameter_line[0][1] + y1),
                (sm.diameter_line[1][0] + x1, sm.diameter_line[1][1] + y1),
            )
            sm.p1 = (sm.p1[0] + x1, sm.p1[1] + y1)
            sm.p2 = (sm.p2[0] + x1, sm.p2[1] + y1)
            sm.substrate_line = (
                (sm.substrate_line[0][0] + x1, sm.substrate_line[0][1] + y1),
                (sm.substrate_line[1][0] + x1, sm.substrate_line[1][1] + y1),
            )
            metrics = sm.derived
            panel = self.contact_tab
            overlay = draw_sessile_overlays(self.image, sm)
        else:
            helpers = SessileHelpers(px_per_mm=self.px_per_mm_drop)
            if self.substrate_line_item is not None:
                line = self.substrate_line_item.line()
                substrate = (
                    (line.x1() - x1, line.y1() - y1),
                    (line.x2() - x1, line.y2() - y1),
                )
            else:
                substrate = ((0, 0), (1, 0))
            sm = analyze_sessile_alt(roi, helpers, substrate)
            sm.contour += np.array([x1, y1])
            sm.apex = (sm.apex[0] + x1, sm.apex[1] + y1)
            sm.diameter_line = (
                (sm.diameter_line[0][0] + x1, sm.diameter_line[0][1] + y1),
                (sm.diameter_line[1][0] + x1, sm.diameter_line[1][1] + y1),
            )
            sm.p1 = (sm.p1[0] + x1, sm.p1[1] + y1)
            sm.p2 = (sm.p2[0] + x1, sm.p2[1] + y1)
            sm.substrate_line = (
                (sm.substrate_line[0][0] + x1, sm.substrate_line[0][1] + y1),
                (sm.substrate_line[1][0] + x1, sm.substrate_line[1][1] + y1),
            )
            metrics = sm.derived
            panel = self.contact_tab_alt
            overlay = draw_sessile_overlays_alt(self.image, sm)
        panel.set_metrics(
            height=metrics["height_mm"],
            diameter=metrics["diameter_mm"],
            volume=metrics["volume_uL"] if metrics["volume_uL"] is not None else 0.0,
            angle=metrics["contact_angle_deg"],
            gamma=metrics["gamma_mN_m"],
            s1=metrics["s1"],
            bo=metrics["Bo"],
            wo=metrics["wo"],
            vmax=metrics["vmax_uL"],
            kappa0=metrics["kappa0_inv_m"],
            aproj=metrics["A_proj_mm2"],
            aproj_left=metrics["A_proj_left_mm2"],
            aproj_right=metrics["A_proj_right_mm2"],
            asurf=metrics["A_surf_mm2"],
            asurf_mean=metrics["A_surf_mean_mm2"],
            asurf_left=metrics["A_surf_left_mm2"],
            asurf_right=metrics["A_surf_right_mm2"],
            wapp=metrics["W_app_mN"],
            radius=metrics["radius_apex_mm"],
            width=metrics.get("w_mm"),
            rbase=metrics.get("rb_mm"),
            height_line=metrics.get("h_mm"),
            apex_to_diam=metrics.get("apex_to_diam_mm"),
            contact_to_diam=metrics.get("contact_to_diam_mm"),
            angle_p1=metrics.get("theta_slope_p1"),
            angle_p2=metrics.get("theta_slope_p2"),
        )
        if self.drop_contour_item is not None:
            self.graphics_scene.removeItem(self.drop_contour_item)
        if self.diameter_item is not None:
            self.graphics_scene.removeItem(self.diameter_item)
        if self.drop_axis_item is not None:
            self.graphics_scene.removeItem(self.drop_axis_item)
        if self.apex_dot_item is not None:
            self.graphics_scene.removeItem(self.apex_dot_item)
        self.drop_contour_item = self.graphics_scene.addPixmap(overlay)
        if (
            mode == "pendant"
            and getattr(self.pendant_tab, "save_profiles_checkbox", None) is not None
            and self.pendant_tab.save_profiles_checkbox.isChecked()
        ):
            from pathlib import Path
            from ..analysis import find_apex_index, save_contour_side_profiles

            apex_idx = find_apex_index(pm.contour.astype(float), "pendant")
            save_contour_side_profiles(pm.contour.astype(float), apex_idx, str(Path("plot")))
