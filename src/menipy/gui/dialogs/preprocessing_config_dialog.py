
from __future__ import annotations

"""Configuration dialog for preprocessing pipeline settings."""

from typing import Optional

from PySide6.QtCore import Qt, Signal
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

from menipy.models.datatypes import PreprocessingSettings


class PreprocessingConfigDialog(QDialog):
    """Allows users to tweak preprocessing pipeline parameters."""

    previewRequested = Signal(object)

    def __init__(self, settings: PreprocessingSettings, parent: Optional[QDialog] = None) -> None:
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
        for label in ("Crop", "Resize", "Filter", "Background", "Normalize", "Contact Line"):
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

        pages_frame = QFrame(self)
        pages_frame.setObjectName("pagesFrame")
        pages_frame.setFrameShape(QFrame.StyledPanel)
        pages_frame.setFrameShadow(QFrame.Raised)
        pages_layout = QVBoxLayout(pages_frame)
        pages_layout.setContentsMargins(16, 16, 16, 16)
        pages_layout.setSpacing(12)

        self.pages = QStackedWidget(pages_frame)
        self.pages.setMinimumWidth(360)
        pages_layout.addWidget(self.pages)
        content_layout.addWidget(pages_frame, 1)

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

        self._build_crop_page()
        self._build_resize_page()
        self._build_filter_page()
        self._build_background_page()
        self._build_normalize_page()
        self._build_contact_page()

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

        self._button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        button_bar.addWidget(self._button_box)
        ok_button = self._button_box.button(QDialogButtonBox.Ok)
        if ok_button is not None:
            ok_button.setDefault(True)


    def _build_crop_page(self) -> None:
        widget = QWidget(self)
        layout = QVBoxLayout(widget)
        self.crop_checkbox = QCheckBox("Crop to ROI", widget)
        self.mask_checkbox = QCheckBox("Process only inside ROI mask", widget)
        layout.addWidget(self.crop_checkbox)
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

        info = QLabel("Gaussian uses kernel/sigma. Median ignores sigma. Bilateral uses all parameters.")
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
        for label in ("clahe", "histogram"):
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

        info = QLabel("Double-click anchors define the contact line. Strategy controls how pixels are treated.")
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

        self.filter_method.currentTextChanged.connect(self._update_filter_controls)
        self.norm_method.currentTextChanged.connect(self._update_norm_controls)
        self.resize_enable.toggled.connect(self._update_resize_controls)
        self.filter_enable.toggled.connect(self._update_filter_controls)
        self.bg_enable.toggled.connect(self._update_background_controls)
        self.norm_enable.toggled.connect(self._update_norm_controls)

    def _load_settings(self) -> None:
        s = self._settings
        self.crop_checkbox.setChecked(s.crop_to_roi)
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

        self._update_resize_controls()
        self._update_filter_controls()
        self._update_background_controls()
        self._update_norm_controls()

    def _collect_settings(self) -> PreprocessingSettings:
        s = self._settings.model_copy(deep=True)
        s.crop_to_roi = self.crop_checkbox.isChecked()
        s.work_on_roi_mask = self.mask_checkbox.isChecked()

        resize = s.resize
        resize.enabled = self.resize_enable.isChecked()
        resize.target_width = self.resize_width.value() or None
        resize.target_height = self.resize_height.value() or None
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

    # ------------------------------------------------------------------
    # Control enablement helpers
    # ------------------------------------------------------------------
    def _update_resize_controls(self) -> None:
        enabled = self.resize_enable.isChecked()
        for widget in (self.resize_width, self.resize_height, self.resize_preserve, self.resize_mode):
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
        self.bg_radius.setEnabled(enabled and self.bg_mode.currentText() == "rolling_ball")

    def _update_norm_controls(self) -> None:
        enabled = self.norm_enable.isChecked()
        self.norm_method.setEnabled(enabled)
        clahe = enabled and self.norm_method.currentText() == "clahe"
        self.norm_clip.setEnabled(clahe)
        self.norm_grid.setEnabled(enabled)

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------
    def settings(self) -> PreprocessingSettings:
        return self._settings






