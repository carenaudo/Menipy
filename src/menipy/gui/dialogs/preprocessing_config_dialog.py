"""
Dialog for preprocessing settings configuration.
"""

from __future__ import annotations

"""Configuration dialog for preprocessing pipeline settings."""

from typing import Optional

import numpy as np
from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)
from PySide6.QtGui import QImage, QPixmap, QPainter, QPen, QColor

from menipy.models.config import PreprocessingSettings


class PreprocessingConfigDialog(QDialog):
    """Allows users to tweak preprocessing pipeline parameters."""

    previewRequested = Signal(object)

    def __init__(
        self, settings: PreprocessingSettings, parent: Optional[QDialog] = None
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Preprocessing Configuration")
        self._settings = settings.model_copy(deep=True)

        self._build_ui()
        self._load_settings()
        self._wire_signals()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        self.setMinimumSize(680, 440)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(12, 12, 12, 12)
        outer.setSpacing(12)

        content_layout = QHBoxLayout()
        content_layout.setSpacing(16)
        outer.addLayout(content_layout, 1)

        nav_frame = QFrame(self)
        nav_frame.setObjectName("navFrame")
        nav_frame.setFrameShape(QFrame.StyledPanel)
        nav_frame.setFrameShadow(QFrame.Raised)
        nav_layout = QVBoxLayout(nav_frame)
        nav_layout.setContentsMargins(0, 0, 0, 0)
        nav_layout.setSpacing(4)

        self.stage_list = QListWidget(nav_frame)
        for label in (
            "Auto Detect",
            "Crop",
            "Resize",
            "Filter",
            "Background",
            "Normalize",
            "Contact Line",
            "Fill Holes",
        ):
            QListWidgetItem(label, self.stage_list)
        self.stage_list.setCurrentRow(0)
        self.stage_list.setFixedWidth(180)
        self.stage_list.setAlternatingRowColors(True)
        self.stage_list.setSpacing(2)
        self.stage_list.setFocusPolicy(Qt.NoFocus)
        self.stage_list.setSelectionMode(QListWidget.SingleSelection)
        self.stage_list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.stage_list.setUniformItemSizes(True)
        nav_layout.addWidget(self.stage_list)
        nav_layout.addStretch(1)
        content_layout.addWidget(nav_frame, 0)

        # --- Start of changes ---
        right_panel_layout = QVBoxLayout()
        right_panel_layout.setContentsMargins(0, 0, 0, 0)
        right_panel_layout.setSpacing(12)

        self.pages = QStackedWidget(self)
        self.pages.setMinimumWidth(360)
        right_panel_layout.addWidget(self.pages)

        self.preview_label = QLabel("No preview available", self)
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumSize(360, 240)
        self.preview_label.setStyleSheet("background-color: #333; color: #CCC;")
        self.preview_label.setScaledContents(True)
        right_panel_layout.addWidget(self.preview_label)

        content_layout.addLayout(right_panel_layout, 1)
        # --- End of changes ---

        self.setStyleSheet(
            """
            QFrame#navFrame, QFrame#pagesFrame {
                background-color: palette(base);
                border: 1px solid palette(mid);
                border-radius: 6px;
            }
            QListWidget {
                border: none;
                padding: 4px;
            }
            QListWidget::item {
                margin: 2px 0;
            }
            QListWidget::item:selected {
                background-color: palette(highlight);
                color: palette(highlighted-text);
            }
            QPushButton {
                min-width: 110px;
            }
            """
        )

        self._build_auto_detect_page()
        self._build_crop_page()
        self._build_resize_page()
        self._build_filter_page()
        self._build_background_page()
        self._build_normalize_page()
        self._build_contact_page()
        self._build_fill_holes_page()

        button_bar = QHBoxLayout()
        button_bar.setContentsMargins(0, 0, 0, 0)
        button_bar.setSpacing(10)
        outer.addLayout(button_bar)

        self._preview_btn = QPushButton("Preview", self)
        self._preview_btn.setAutoDefault(False)
        self._reset_btn = QPushButton("Restore Defaults", self)
        self._reset_btn.setAutoDefault(False)
        button_bar.addWidget(self._preview_btn)
        button_bar.addWidget(self._reset_btn)
        button_bar.addStretch(1)

        self._button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self
        )
        button_bar.addWidget(self._button_box)
        ok_button = self._button_box.button(QDialogButtonBox.Ok)
        if ok_button is not None:
            ok_button.setDefault(True)

            ok_button.setDefault(True)

    def _build_auto_detect_page(self) -> None:
        widget = QWidget(self)
        layout = QVBoxLayout(widget)
        
        self.auto_enable = QCheckBox("Enable Automatic Feature Detection", widget)
        layout.addWidget(self.auto_enable)
        
        group = QFrame(widget)
        group.setFrameShape(QFrame.StyledPanel)
        group_layout = QVBoxLayout(group)
        
        self.auto_detect_roi = QCheckBox("Detect Region of Interest (ROI)", group)
        self.auto_detect_needle = QCheckBox("Detect Needle/Nozzle", group)
        self.auto_detect_substrate = QCheckBox("Detect Substrate/Baseline", group)
        
        group_layout.addWidget(self.auto_detect_roi)
        group_layout.addWidget(self.auto_detect_needle)
        group_layout.addWidget(self.auto_detect_substrate)
        
        layout.addWidget(group)
        
        self.btn_run_auto = QPushButton("Run Auto Detect", widget)
        self.btn_run_auto.clicked.connect(self._on_preview) # Reuse preview logic which runs pipeline
        layout.addWidget(self.btn_run_auto)
        
        layout.addStretch(1)
        self.pages.addWidget(widget)

    def _build_crop_page(self) -> None:
        widget = QWidget(self)
        layout = QVBoxLayout(widget)
        self.crop_checkbox = QCheckBox("Crop to ROI", widget)
        self.grayscale_checkbox = QCheckBox("Convert to Grayscale", widget)
        self.mask_checkbox = QCheckBox("Process only inside ROI mask", widget)
        layout.addWidget(self.crop_checkbox)
        layout.addWidget(self.grayscale_checkbox)
        layout.addWidget(self.mask_checkbox)
        layout.addStretch(1)
        self.pages.addWidget(widget)

    def _build_resize_page(self) -> None:
        widget = QWidget(self)
        form = QFormLayout(widget)

        self.resize_enable = QCheckBox("Enable resizing", widget)
        form.addRow(self.resize_enable)

        self.resize_width = QSpinBox(widget)
        self.resize_width.setRange(1, 8192)
        self.resize_width.setSpecialValueText("Auto")
        self.resize_width.setValue(0)
        form.addRow("Target width", self.resize_width)

        self.resize_height = QSpinBox(widget)
        self.resize_height.setRange(1, 8192)
        self.resize_height.setSpecialValueText("Auto")
        self.resize_height.setValue(0)
        form.addRow("Target height", self.resize_height)

        self.resize_preserve = QCheckBox("Preserve aspect ratio", widget)
        form.addRow(self.resize_preserve)

        self.resize_mode = QComboBox(widget)
        for label in ("nearest", "linear", "cubic", "area", "lanczos"):
            self.resize_mode.addItem(label)
        form.addRow("Interpolation", self.resize_mode)

        form.addRow(QLabel("Leave width/height at Auto to scale proportionally."))
        self.pages.addWidget(widget)

    def _build_filter_page(self) -> None:
        widget = QWidget(self)
        form = QFormLayout(widget)

        self.filter_enable = QCheckBox("Enable filtering", widget)
        form.addRow(self.filter_enable)

        self.filter_method = QComboBox(widget)
        for label in ("none", "gaussian", "median", "bilateral"):
            self.filter_method.addItem(label)
        form.addRow("Method", self.filter_method)

        self.filter_kernel = QSpinBox(widget)
        self.filter_kernel.setRange(1, 99)
        self.filter_kernel.setSingleStep(2)
        self.filter_kernel.setValue(3)
        form.addRow("Kernel size", self.filter_kernel)

        self.filter_sigma = QDoubleSpinBox(widget)
        self.filter_sigma.setRange(0.0, 25.0)
        self.filter_sigma.setSingleStep(0.1)
        self.filter_sigma.setValue(1.0)
        form.addRow("Sigma", self.filter_sigma)

        self.filter_sigma_color = QDoubleSpinBox(widget)
        self.filter_sigma_color.setRange(0.0, 255.0)
        self.filter_sigma_color.setSingleStep(1.0)
        self.filter_sigma_color.setValue(75.0)
        form.addRow("Sigma color", self.filter_sigma_color)

        self.filter_sigma_space = QDoubleSpinBox(widget)
        self.filter_sigma_space.setRange(0.0, 255.0)
        self.filter_sigma_space.setSingleStep(1.0)
        self.filter_sigma_space.setValue(75.0)
        form.addRow("Sigma space", self.filter_sigma_space)

        info = QLabel(
            "Gaussian uses kernel/sigma. Median ignores sigma. Bilateral uses all parameters."
        )
        info.setWordWrap(True)
        form.addRow(info)
        self.pages.addWidget(widget)

    def _build_background_page(self) -> None:
        widget = QWidget(self)
        form = QFormLayout(widget)

        self.bg_enable = QCheckBox("Enable background subtraction", widget)
        form.addRow(self.bg_enable)

        self.bg_mode = QComboBox(widget)
        for label in ("flat", "rolling_ball"):
            self.bg_mode.addItem(label)
        form.addRow("Mode", self.bg_mode)

        self.bg_strength = QDoubleSpinBox(widget)
        self.bg_strength.setRange(0.0, 1.0)
        self.bg_strength.setSingleStep(0.05)
        self.bg_strength.setValue(0.8)
        form.addRow("Strength", self.bg_strength)

        self.bg_radius = QSpinBox(widget)
        self.bg_radius.setRange(1, 256)
        self.bg_radius.setValue(15)
        form.addRow("Rolling radius", self.bg_radius)

        self.pages.addWidget(widget)

    def _build_normalize_page(self) -> None:
        widget = QWidget(self)
        form = QFormLayout(widget)

        self.norm_enable = QCheckBox("Enable normalization", widget)
        form.addRow(self.norm_enable)

        self.norm_method = QComboBox(widget)
        for label in ("clahe", "histogram", "otsu"):
            self.norm_method.addItem(label)
        form.addRow("Method", self.norm_method)

        self.norm_clip = QDoubleSpinBox(widget)
        self.norm_clip.setRange(0.0, 40.0)
        self.norm_clip.setSingleStep(0.1)
        self.norm_clip.setValue(2.0)
        form.addRow("CLAHE clip limit", self.norm_clip)

        self.norm_grid = QSpinBox(widget)
        self.norm_grid.setRange(1, 64)
        self.norm_grid.setValue(8)
        form.addRow("Grid size", self.norm_grid)

        self.pages.addWidget(widget)

    def _build_contact_page(self) -> None:
        widget = QWidget(self)
        form = QFormLayout(widget)

        self.contact_strategy = QComboBox(widget)
        for label in ("preserve", "attenuate", "mask"):
            self.contact_strategy.addItem(label)
        form.addRow("Strategy", self.contact_strategy)

        self.contact_threshold = QDoubleSpinBox(widget)
        self.contact_threshold.setRange(0.0, 1.0)
        self.contact_threshold.setSingleStep(0.05)
        self.contact_threshold.setValue(0.15)
        form.addRow("Mask threshold", self.contact_threshold)

        self.contact_dilation = QSpinBox(widget)
        self.contact_dilation.setRange(1, 99)
        self.contact_dilation.setValue(3)
        form.addRow("Dilation (px)", self.contact_dilation)

        info = QLabel(
            "Double-click anchors define the contact line. Strategy controls how pixels are treated."
        )
        info.setWordWrap(True)
        form.addRow(info)

        self.pages.addWidget(widget)

    def _build_fill_holes_page(self) -> None:
        widget = QWidget(self)
        form = QFormLayout(widget)

        self.fill_enable = QCheckBox("Enable fill holes / remove spurious", widget)
        form.addRow(self.fill_enable)

        self.fill_max_area = QSpinBox(widget)
        self.fill_max_area.setRange(0, 10000000)
        self.fill_max_area.setValue(500)
        form.addRow("Max hole area (px)", self.fill_max_area)

        self.fill_remove_spurious = QCheckBox(
            "Remove small objects near contact line", widget
        )
        form.addRow(self.fill_remove_spurious)

        self.fill_proximity = QSpinBox(widget)
        self.fill_proximity.setRange(0, 200)
        self.fill_proximity.setValue(5)
        form.addRow("Proximity to contact (px)", self.fill_proximity)

        info = QLabel(
            "Fills small interior holes and optionally removes small spurious objects near the contact line."
        )
        info.setWordWrap(True)
        form.addRow(info)

        self.pages.addWidget(widget)

    # ------------------------------------------------------------------
    # Data binding
    # ------------------------------------------------------------------
    def _wire_signals(self) -> None:
        self.stage_list.currentRowChanged.connect(self.pages.setCurrentIndex)
        self._button_box.accepted.connect(self._on_accept)
        self._button_box.rejected.connect(self.reject)
        self._preview_btn.clicked.connect(self._on_preview)
        self._reset_btn.clicked.connect(self._on_reset)

        self.auto_enable.toggled.connect(self._update_auto_detect_controls) # Wire signal

        self.filter_method.currentTextChanged.connect(self._update_filter_controls)
        self.norm_method.currentTextChanged.connect(self._update_norm_controls)
        self.resize_enable.toggled.connect(self._update_resize_controls)
        self.filter_enable.toggled.connect(self._update_filter_controls)
        self.bg_enable.toggled.connect(self._update_background_controls)
        self.norm_enable.toggled.connect(self._update_norm_controls)
        # Fill holes controls (widget may not exist in older settings)
        if hasattr(self, "fill_enable"):
            self.fill_enable.toggled.connect(self._update_fill_controls)

    def _load_settings(self) -> None:
        s = self._settings
        
        if hasattr(s, "auto_detect"):
            ad = s.auto_detect
            self.auto_enable.setChecked(ad.enabled)
            self.auto_detect_roi.setChecked(ad.detect_roi)
            self.auto_detect_needle.setChecked(ad.detect_needle)
            self.auto_detect_substrate.setChecked(ad.detect_substrate)
            
        self.crop_checkbox.setChecked(s.crop_to_roi)
        self.grayscale_checkbox.setChecked(s.convert_to_grayscale)
        self.mask_checkbox.setChecked(s.work_on_roi_mask)

        resize = s.resize
        self.resize_enable.setChecked(resize.enabled)
        self.resize_width.setValue(resize.target_width or 0)
        self.resize_height.setValue(resize.target_height or 0)
        self.resize_preserve.setChecked(resize.preserve_aspect)
        self.resize_mode.setCurrentText(resize.interpolation)

        filtering = s.filtering
        self.filter_enable.setChecked(filtering.enabled)
        self.filter_method.setCurrentText(filtering.method)
        self.filter_kernel.setValue(filtering.kernel_size)
        self.filter_sigma.setValue(filtering.sigma)
        self.filter_sigma_color.setValue(filtering.sigma_color)
        self.filter_sigma_space.setValue(filtering.sigma_space)

        background = s.background
        self.bg_enable.setChecked(background.enabled)
        self.bg_mode.setCurrentText(background.mode)
        self.bg_strength.setValue(background.strength)
        self.bg_radius.setValue(background.rolling_radius)

        norm = s.normalization
        self.norm_enable.setChecked(norm.enabled)
        self.norm_method.setCurrentText(norm.method)
        self.norm_clip.setValue(norm.clip_limit)
        self.norm_grid.setValue(norm.grid_size)

        contact = s.contact_line
        self.contact_strategy.setCurrentText(contact.strategy)
        self.contact_threshold.setValue(contact.threshold)
        self.contact_dilation.setValue(contact.dilation)

        # Load fill_holes settings if available
        if hasattr(s, "fill_holes"):
            fill = s.fill_holes
            try:
                self.fill_enable.setChecked(getattr(fill, "enabled", False))
                self.fill_max_area.setValue(getattr(fill, "max_hole_area", 500) or 0)
                self.fill_remove_spurious.setChecked(
                    getattr(fill, "remove_spurious_near_contact", True)
                )
                self.fill_proximity.setValue(getattr(fill, "proximity_px", 5) or 0)
            except Exception:
                # defensively restore defaults
                self.fill_enable.setChecked(False)
                self.fill_max_area.setValue(500)
                self.fill_remove_spurious.setChecked(True)
                self.fill_proximity.setValue(5)

        self._update_resize_controls()
        self._update_filter_controls()
        self._update_background_controls()
        self._update_norm_controls()

    def _collect_settings(self) -> PreprocessingSettings:
        s = self._settings.model_copy(deep=True)
        
        if hasattr(s, "auto_detect"):
            ad = s.auto_detect
            ad.enabled = self.auto_enable.setChecked(self.auto_enable.isChecked()) # Fix side effect
            ad.enabled = self.auto_enable.isChecked()
            ad.detect_roi = self.auto_detect_roi.isChecked()
            ad.detect_needle = self.auto_detect_needle.isChecked()
            ad.detect_substrate = self.auto_detect_substrate.isChecked()
            
        s.crop_to_roi = self.crop_checkbox.isChecked()
        s.convert_to_grayscale = self.grayscale_checkbox.isChecked()
        s.work_on_roi_mask = self.mask_checkbox.isChecked()

        resize = s.resize
        resize.enabled = self.resize_enable.isChecked()
        resize.target_width = (
            self.resize_width.value() if self.resize_width.value() > 0 else None
        )
        resize.target_height = (
            self.resize_height.value() if self.resize_height.value() > 0 else None
        )
        resize.preserve_aspect = self.resize_preserve.isChecked()
        resize.interpolation = self.resize_mode.currentText()

        filtering = s.filtering
        filtering.enabled = self.filter_enable.isChecked()
        filtering.method = self.filter_method.currentText()
        filtering.kernel_size = max(1, self.filter_kernel.value() | 1)
        filtering.sigma = self.filter_sigma.value()
        filtering.sigma_color = self.filter_sigma_color.value()
        filtering.sigma_space = self.filter_sigma_space.value()

        background = s.background
        background.enabled = self.bg_enable.isChecked()
        background.mode = self.bg_mode.currentText()
        background.strength = self.bg_strength.value()
        background.rolling_radius = self.bg_radius.value()

        norm = s.normalization
        norm.enabled = self.norm_enable.isChecked()
        norm.method = self.norm_method.currentText()
        norm.clip_limit = self.norm_clip.value()
        norm.grid_size = self.norm_grid.value()

        contact = s.contact_line
        contact.strategy = self.contact_strategy.currentText()
        contact.threshold = self.contact_threshold.value()
        contact.dilation = self.contact_dilation.value()

        # Collect fill_holes settings
        if hasattr(s, "fill_holes"):
            fill = s.fill_holes
            fill.enabled = self.fill_enable.isChecked()
            fill.max_hole_area = self.fill_max_area.value()
            fill.remove_spurious_near_contact = self.fill_remove_spurious.isChecked()
            fill.proximity_px = self.fill_proximity.value()

        self._settings = s
        return s

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------
    def _on_preview(self) -> None:
        settings = self._collect_settings()
        self.previewRequested.emit(settings)

    def _on_reset(self) -> None:
        self._settings = PreprocessingSettings()
        self._load_settings()

    def _on_accept(self) -> None:
        self._collect_settings()
        self.accept()

    # Control enablement helpers
    # ------------------------------------------------------------------
    def _update_auto_detect_controls(self) -> None:
        enabled = self.auto_enable.isChecked()
        self.auto_detect_roi.setEnabled(enabled)
        self.auto_detect_needle.setEnabled(enabled)
        self.auto_detect_substrate.setEnabled(enabled)
        self.btn_run_auto.setEnabled(enabled)

    def _update_resize_controls(self) -> None:
        enabled = self.resize_enable.isChecked()
        for widget in (
            self.resize_width,
            self.resize_height,
            self.resize_preserve,
            self.resize_mode,
        ):
            widget.setEnabled(enabled)

    def _update_filter_controls(self) -> None:
        enabled = self.filter_enable.isChecked()
        method = self.filter_method.currentText()
        self.filter_method.setEnabled(enabled)
        self.filter_kernel.setEnabled(enabled and method != "none")
        sigma_enabled = enabled and method in {"gaussian", "bilateral"}
        bilateral_enabled = enabled and method == "bilateral"
        self.filter_sigma.setEnabled(sigma_enabled)
        self.filter_sigma_color.setEnabled(bilateral_enabled)
        self.filter_sigma_space.setEnabled(bilateral_enabled)

    def _update_background_controls(self) -> None:
        enabled = self.bg_enable.isChecked()
        self.bg_mode.setEnabled(enabled)
        self.bg_strength.setEnabled(enabled)
        self.bg_radius.setEnabled(
            enabled and self.bg_mode.currentText() == "rolling_ball"
        )

    def _update_norm_controls(self) -> None:
        enabled = self.norm_enable.isChecked()
        self.norm_method.setEnabled(enabled)
        clahe = enabled and self.norm_method.currentText() == "clahe"
        self.norm_clip.setEnabled(clahe)
        self.norm_grid.setEnabled(enabled)

    def _update_fill_controls(self) -> None:
        enabled = getattr(self, "fill_enable", None) and self.fill_enable.isChecked()
        for widget in (
            self.fill_max_area,
            self.fill_remove_spurious,
            self.fill_proximity,
        ):
            widget.setEnabled(enabled)

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------
    def settings(self) -> PreprocessingSettings:
        return self._settings

    @Slot(object, dict)
    def _on_preview_image_ready(self, image: np.ndarray, metadata: dict) -> None:
        if image is None:
            self.preview_label.setText("No preview available")
            self.preview_label.clear()
            return

        h, w = image.shape[:2]
        bytes_per_line = 3 * w
        q_image: QImage

        if image.ndim == 2:  # Grayscale
            q_image = QImage(image.data, w, h, w, QImage.Format.Format_Grayscale8)
        elif image.ndim == 3 and image.shape[2] == 3:  # BGR
            q_image = QImage(
                image.data, w, h, bytes_per_line, QImage.Format.Format_BGR888
            )
        elif image.ndim == 3 and image.shape[2] == 4:  # BGRA
            q_image = QImage(
                image.data, w, h, bytes_per_line, QImage.Format.Format_ARGB32
            )  # Assuming BGRA is ARGB32
        else:
            self.preview_label.setText("Unsupported image format")
            self.preview_label.clear()
            return

        pixmap = QPixmap.fromImage(q_image)

        # Draw overlays if metadata present
        if metadata and (metadata.get("roi") or metadata.get("needle_rect") or metadata.get("substrate_line")):
            painter = QPainter(pixmap)
            painter.setRenderHint(QPainter.Antialiasing)
            
            # Determine offset if cropped
            offset_x, offset_y = 0, 0
            # If cropped, the image represents the ROI, so valid coordinates are shifted
            # But wait, if crop_to_roi is True, the image IS the ROI.
            # So global coordinates need to be shifted by -ROI.x, -ROI.y
            roi = metadata.get("roi")
            if self._settings.crop_to_roi and roi:
                offset_x, offset_y = roi[0], roi[1]

            def to_local_rect(r):
                if not r: return None
                return (r[0] - offset_x, r[1] - offset_y, r[2], r[3])
                
            def to_local_line(l):
                if not l: return None
                p1, p2 = l
                return ((p1[0] - offset_x, p1[1] - offset_y), (p2[0] - offset_x, p2[1] - offset_y))

            # Draw ROI (if not cropped, or if we want to show it anyway - if cropped it's the border)
            if roi:
                rx, ry, rw, rh = to_local_rect(roi)
                pen = QPen(QColor(255, 255, 0)) # Yellow
                pen.setWidth(2)
                pen.setStyle(Qt.DashLine)
                painter.setPen(pen)
                painter.drawRect(rx, ry, rw, rh)

            # Draw Needle
            needle = metadata.get("needle_rect")
            if needle:
                nx, ny, nw, nh = to_local_rect(needle)
                pen = QPen(QColor(0, 255, 255)) # Cyan
                pen.setWidth(2)
                painter.setPen(pen)
                painter.drawRect(nx, ny, nw, nh)

            # Draw Substrate
            sub = metadata.get("substrate_line")
            if sub:
                (x1, y1), (x2, y2) = to_local_line(sub)
                pen = QPen(QColor(255, 0, 255)) # Magenta
                pen.setWidth(2)
                painter.setPen(pen)
                painter.drawLine(x1, y1, x2, y2)

            painter.end()

        self.preview_label.setPixmap(
            pixmap.scaled(
                self.preview_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
        )
