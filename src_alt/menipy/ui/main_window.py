from __future__ import annotations

import numpy as np
import cv2
from PySide6.QtCore import QLineF, Qt
from PySide6.QtGui import QColor, QPen
from PySide6.QtWidgets import QMessageBox

from src.gui.main_window import MainWindow as LegacyMainWindow
from src.gui.items import SubstrateLineItem
from src.gui.overlay import draw_drop_overlay
from menipy.detection.needle import detect_vertical_edges
from menipy.detection.substrate import detect_substrate_line
from menipy.analysis.commons import (
    extract_external_contour,
    compute_drop_metrics,
)
from src.physics.contact_geom import geom_metrics


class MainWindow(LegacyMainWindow):
    """Main window using refactored detection and analysis modules."""

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
        apex_idx = int(np.argmin(contour[:, 1])) if mode == "contact-angle" else int(np.argmax(contour[:, 1]))
        metrics = compute_drop_metrics(
            contour.astype(float),
            self.px_per_mm_drop,
            mode,
            needle_diam_mm=self.calibration_tab.needle_length.value(),
            substrate_line=(
                self.substrate_line_item.line().p1().toTuple(),
                self.substrate_line_item.line().p2().toTuple(),
            )
            if self.substrate_line_item is not None
            else None,
        )
        extra = {}

        if mode == "contact-angle" and self.substrate_line_item is not None:
            p1 = self.substrate_line_item.line().p1()
            p2 = self.substrate_line_item.line().p2()
            extra = geom_metrics(
                p1.toTuple(),
                p2.toTuple(),
                contour.astype(float),
                apex_idx,
                self.px_per_mm_drop,
            )
            droplet_poly = extra.pop("droplet_poly")
            metrics = compute_drop_metrics(
                droplet_poly.astype(float),
                self.px_per_mm_drop,
                mode,
                needle_diam_mm=self.calibration_tab.needle_length.value(),
                substrate_line=(
                    self.substrate_line_item.line().p1().toTuple(),
                    self.substrate_line_item.line().p2().toTuple(),
                ),
            )
            metrics.update(extra)
            contour = droplet_poly
        panel = self.pendant_tab if mode == "pendant" else self.contact_tab
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
            asurf=metrics["A_surf_mm2"],
            wapp=metrics["W_app_mN"],
            radius=metrics["radius_apex_mm"],
            width=metrics.get("w_mm"),
            rbase=metrics.get("rb_mm"),
            height_line=metrics.get("h_mm"),
        )
        diameter_line = (
            metrics["diameter_line"][0],
            metrics["diameter_line"][1],
        )
        if self.substrate_line_item is not None:
            line_dir = np.array(
                self.substrate_line_item.line().p2().toTuple(), float
            ) - np.array(self.substrate_line_item.line().p1().toTuple(), float)
            p1 = np.array(diameter_line[0], float)
            p2 = np.array(diameter_line[1], float)
        else:
            line_dir = np.array([1.0, 0.0])
            p1 = np.array(diameter_line[0], float)
            p2 = np.array(diameter_line[1], float)

        apex_pt = np.array(metrics["apex"], dtype=float)
        t = np.dot(apex_pt - p1, line_dir) / np.dot(line_dir, line_dir)
        t = np.clip(t, 0.0, 1.0)
        foot_pt = p1 + t * line_dir
        axis_line = (
            tuple(np.round(foot_pt).astype(int)),
            tuple(np.round(apex_pt).astype(int)),
        )
        if self.drop_contour_item is not None:
            self.graphics_scene.removeItem(self.drop_contour_item)
        if self.diameter_item is not None:
            self.graphics_scene.removeItem(self.diameter_item)
        if self.drop_axis_item is not None:
            self.graphics_scene.removeItem(self.drop_axis_item)
        if self.apex_dot_item is not None:
            self.graphics_scene.removeItem(self.apex_dot_item)

        overlay = draw_drop_overlay(
            self.image,
            contour,
            diameter_line=diameter_line,
            axis_line=axis_line,
            contact_line=None,
            apex=metrics["apex"],
            contact_pts=metrics.get("contact_line"),
        )
        self.drop_contour_item = self.graphics_scene.addPixmap(overlay)

