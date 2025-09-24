from __future__ import annotations

"""Configuration dialog for edge detection pipeline settings."""

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

from menipy.models.datatypes import EdgeDetectionSettings


class EdgeDetectionConfigDialog(QDialog):
    """Allows users to tweak edge detection pipeline parameters."""

    previewRequested = Signal(object)

    def __init__(self, settings: EdgeDetectionSettings, parent: Optional[QDialog] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Edge Detection Configuration")
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
        for label in ("General Settings", "Canny", "Threshold", "Sobel/Scharr/Laplacian", "Active Contour", "Refinement", "Interface Specific"):
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

        self._build_general_page()
        self._build_canny_page()
        self._build_threshold_page()
        self._build_sobel_laplacian_page()
        self._build_active_contour_page()
        self._build_refinement_page()
        self._build_interface_page()

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

    def _build_general_page(self) -> None:
        widget = QWidget(self)
        form = QFormLayout(widget)

        self.enabled_checkbox = QCheckBox("Enable Edge Detection", widget)
        form.addRow(self.enabled_checkbox)

        self.method_combo = QComboBox(widget)
        for label in ("canny", "sobel", "scharr", "laplacian", "threshold", "active_contour"):
            self.method_combo.addItem(label)
        form.addRow("Method", self.method_combo)

        self.gaussian_blur_checkbox = QCheckBox("Apply Gaussian Blur Before", widget)
        form.addRow(self.gaussian_blur_checkbox)

        self.gaussian_kernel_size_spin = QSpinBox(widget)
        self.gaussian_kernel_size_spin.setRange(1, 99)
        self.gaussian_kernel_size_spin.setSingleStep(2)
        self.gaussian_kernel_size_spin.setValue(5)
        form.addRow("Gaussian Kernel Size (odd)", self.gaussian_kernel_size_spin)

        self.gaussian_sigma_x_spin = QDoubleSpinBox(widget)
        self.gaussian_sigma_x_spin.setRange(0.0, 10.0)
        self.gaussian_sigma_x_spin.setSingleStep(0.1)
        self.gaussian_sigma_x_spin.setValue(0.0)
        form.addRow("Gaussian Sigma X", self.gaussian_sigma_x_spin)

        self.pages.addWidget(widget)

    def _build_canny_page(self) -> None:
        widget = QWidget(self)
        form = QFormLayout(widget)

        self.canny_threshold1_spin = QSpinBox(widget)
        self.canny_threshold1_spin.setRange(0, 255)
        self.canny_threshold1_spin.setValue(50)
        form.addRow("Threshold 1", self.canny_threshold1_spin)

        self.canny_threshold2_spin = QSpinBox(widget)
        self.canny_threshold2_spin.setRange(0, 255)
        self.canny_threshold2_spin.setValue(150)
        form.addRow("Threshold 2", self.canny_threshold2_spin)

        self.canny_aperture_size_combo = QComboBox(widget)
        for size in (3, 5, 7):
            self.canny_aperture_size_combo.addItem(str(size))
        form.addRow("Aperture Size", self.canny_aperture_size_combo)

        self.canny_L2_gradient_checkbox = QCheckBox("L2 Gradient", widget)
        form.addRow(self.canny_L2_gradient_checkbox)

        self.pages.addWidget(widget)

    def _build_threshold_page(self) -> None:
        widget = QWidget(self)
        form = QFormLayout(widget)

        self.threshold_value_spin = QSpinBox(widget)
        self.threshold_value_spin.setRange(0, 255)
        self.threshold_value_spin.setValue(128)
        form.addRow("Threshold Value", self.threshold_value_spin)

        self.threshold_max_value_spin = QSpinBox(widget)
        self.threshold_max_value_spin.setRange(0, 255)
        self.threshold_max_value_spin.setValue(255)
        form.addRow("Max Value", self.threshold_max_value_spin)

        self.threshold_type_combo = QComboBox(widget)
        for t in ("binary", "binary_inv", "trunc", "to_zero", "to_zero_inv"):
            self.threshold_type_combo.addItem(t)
        form.addRow("Type", self.threshold_type_combo)

        self.pages.addWidget(widget)

    def _build_sobel_laplacian_page(self) -> None:
        widget = QWidget(self)
        form = QFormLayout(widget)

        self.sobel_kernel_size_spin = QSpinBox(widget)
        self.sobel_kernel_size_spin.setRange(1, 31)
        self.sobel_kernel_size_spin.setSingleStep(2)
        self.sobel_kernel_size_spin.setValue(3)
        form.addRow("Sobel/Scharr Kernel Size (odd)", self.sobel_kernel_size_spin)

        self.laplacian_kernel_size_spin = QSpinBox(widget)
        self.laplacian_kernel_size_spin.setRange(1, 31)
        self.laplacian_kernel_size_spin.setSingleStep(2)
        self.laplacian_kernel_size_spin.setValue(1)
        form.addRow("Laplacian Kernel Size (odd)", self.laplacian_kernel_size_spin)

        self.pages.addWidget(widget)

    def _build_active_contour_page(self) -> None:
        widget = QWidget(self)
        form = QFormLayout(widget)

        self.active_contour_iterations_spin = QSpinBox(widget)
        self.active_contour_iterations_spin.setRange(1, 1000)
        self.active_contour_iterations_spin.setValue(100)
        form.addRow("Iterations", self.active_contour_iterations_spin)

        self.active_contour_alpha_spin = QDoubleSpinBox(widget)
        self.active_contour_alpha_spin.setRange(0.0, 1.0)
        self.active_contour_alpha_spin.setSingleStep(0.01)
        self.active_contour_alpha_spin.setValue(0.01)
        form.addRow("Alpha (Contour Length)", self.active_contour_alpha_spin)

        self.active_contour_beta_spin = QDoubleSpinBox(widget)
        self.active_contour_beta_spin.setRange(0.0, 1.0)
        self.active_contour_beta_spin.setSingleStep(0.01)
        self.active_contour_beta_spin.setValue(0.1)
        form.addRow("Beta (Contour Smoothness)", self.active_contour_beta_spin)

        self.pages.addWidget(widget)

    def _build_refinement_page(self) -> None:
        widget = QWidget(self)
        form = QFormLayout(widget)

        self.min_contour_length_spin = QSpinBox(widget)
        self.min_contour_length_spin.setRange(0, 10000)
        self.min_contour_length_spin.setValue(10)
        form.addRow("Minimum Contour Length", self.min_contour_length_spin)

        self.max_contour_length_spin = QSpinBox(widget)
        self.max_contour_length_spin.setRange(0, 100000)
        self.max_contour_length_spin.setValue(10000)
        form.addRow("Maximum Contour Length", self.max_contour_length_spin)

        self.pages.addWidget(widget)

    def _build_interface_page(self) -> None:
        widget = QWidget(self)
        form = QFormLayout(widget)

        self.detect_fluid_interface_checkbox = QCheckBox("Detect Fluid-Droplet Interface", widget)
        form.addRow(self.detect_fluid_interface_checkbox)

        self.detect_solid_interface_checkbox = QCheckBox("Detect Solid-Droplet Interface", widget)
        form.addRow(self.detect_solid_interface_checkbox)

        self.solid_interface_proximity_spin = QSpinBox(widget)
        self.solid_interface_proximity_spin.setRange(0, 100)
        self.solid_interface_proximity_spin.setValue(10)
        form.addRow("Solid Interface Proximity (px)", self.solid_interface_proximity_spin)

        info = QLabel("Proximity defines search area around contact line for solid interface.")
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

        self.enabled_checkbox.toggled.connect(self._update_controls_enablement)
        self.method_combo.currentTextChanged.connect(self._update_controls_enablement)
        self.gaussian_blur_checkbox.toggled.connect(self._update_controls_enablement)

    def _load_settings(self) -> None:
        s = self._settings
        self.enabled_checkbox.setChecked(s.enabled)
        self.method_combo.setCurrentText(s.method)
        self.gaussian_blur_checkbox.setChecked(s.gaussian_blur_before)
        self.gaussian_kernel_size_spin.setValue(s.gaussian_kernel_size)
        self.gaussian_sigma_x_spin.setValue(s.gaussian_sigma_x)

        self.canny_threshold1_spin.setValue(s.canny_threshold1)
        self.canny_threshold2_spin.setValue(s.canny_threshold2)
        self.canny_aperture_size_combo.setCurrentText(str(s.canny_aperture_size))
        self.canny_L2_gradient_checkbox.setChecked(s.canny_L2_gradient)

        self.threshold_value_spin.setValue(s.threshold_value)
        self.threshold_max_value_spin.setValue(s.threshold_max_value)
        self.threshold_type_combo.setCurrentText(s.threshold_type)

        self.sobel_kernel_size_spin.setValue(s.sobel_kernel_size)
        self.laplacian_kernel_size_spin.setValue(s.laplacian_kernel_size)

        self.active_contour_iterations_spin.setValue(s.active_contour_iterations)
        self.active_contour_alpha_spin.setValue(s.active_contour_alpha)
        self.active_contour_beta_spin.setValue(s.active_contour_beta)

        self.min_contour_length_spin.setValue(s.min_contour_length)
        self.max_contour_length_spin.setValue(s.max_contour_length)

        self.detect_fluid_interface_checkbox.setChecked(s.detect_fluid_interface)
        self.detect_solid_interface_checkbox.setChecked(s.detect_solid_interface)
        self.solid_interface_proximity_spin.setValue(s.solid_interface_proximity)

        self._update_controls_enablement()

    def _collect_settings(self) -> EdgeDetectionSettings:
        s = self._settings.model_copy(deep=True)
        s.enabled = self.enabled_checkbox.isChecked()
        s.method = self.method_combo.currentText()
        s.gaussian_blur_before = self.gaussian_blur_checkbox.isChecked()
        s.gaussian_kernel_size = self.gaussian_kernel_size_spin.value()
        s.gaussian_sigma_x = self.gaussian_sigma_x_spin.value()

        s.canny_threshold1 = self.canny_threshold1_spin.value()
        s.canny_threshold2 = self.canny_threshold2_spin.value()
        s.canny_aperture_size = int(self.canny_aperture_size_combo.currentText())
        s.canny_L2_gradient = self.canny_L2_gradient_checkbox.isChecked()

        s.threshold_value = self.threshold_value_spin.value()
        s.threshold_max_value = self.threshold_max_value_spin.value()
        s.threshold_type = self.threshold_type_combo.currentText()

        s.sobel_kernel_size = self.sobel_kernel_size_spin.value()
        s.laplacian_kernel_size = self.laplacian_kernel_size_spin.value()

        s.active_contour_iterations = self.active_contour_iterations_spin.value()
        s.active_contour_alpha = self.active_contour_alpha_spin.value()
        s.active_contour_beta = self.active_contour_beta_spin.value()

        s.min_contour_length = self.min_contour_length_spin.value()
        s.max_contour_length = self.max_contour_length_spin.value()

        s.detect_fluid_interface = self.detect_fluid_interface_checkbox.isChecked()
        s.detect_solid_interface = self.detect_solid_interface_checkbox.isChecked()
        s.solid_interface_proximity = self.solid_interface_proximity_spin.value()

        self._settings = s
        return s

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------
    def _on_preview(self) -> None:
        settings = self._collect_settings()
        self.previewRequested.emit(settings)

    def _on_reset(self) -> None:
        self._settings = EdgeDetectionSettings()
        self._load_settings()

    def _on_accept(self) -> None:
        self._collect_settings()
        self.accept()

    # ------------------------------------------------------------------
    # Control enablement helpers
    # ------------------------------------------------------------------
    def _update_controls_enablement(self) -> None:
        enabled = self.enabled_checkbox.isChecked()
        method = self.method_combo.currentText()

        # General settings controls
        self.method_combo.setEnabled(enabled)
        self.gaussian_blur_checkbox.setEnabled(enabled)
        self.gaussian_kernel_size_spin.setEnabled(enabled and self.gaussian_blur_checkbox.isChecked())
        self.gaussian_sigma_x_spin.setEnabled(enabled and self.gaussian_blur_checkbox.isChecked())

        # Canny controls
        canny_enabled = enabled and method == "canny"
        self.canny_threshold1_spin.setEnabled(canny_enabled)
        self.canny_threshold2_spin.setEnabled(canny_enabled)
        self.canny_aperture_size_combo.setEnabled(canny_enabled)
        self.canny_L2_gradient_checkbox.setEnabled(canny_enabled)

        # Threshold controls
        threshold_enabled = enabled and method == "threshold"
        self.threshold_value_spin.setEnabled(threshold_enabled)
        self.threshold_max_value_spin.setEnabled(threshold_enabled)
        self.threshold_type_combo.setEnabled(threshold_enabled)

        # Sobel/Scharr/Laplacian controls
        sobel_laplacian_enabled = enabled and method in {"sobel", "scharr", "laplacian"}
        self.sobel_kernel_size_spin.setEnabled(sobel_laplacian_enabled and method in {"sobel", "scharr"})
        self.laplacian_kernel_size_spin.setEnabled(sobel_laplacian_enabled and method == "laplacian")

        # Active Contour controls
        active_contour_enabled = enabled and method == "active_contour"
        self.active_contour_iterations_spin.setEnabled(active_contour_enabled)
        self.active_contour_alpha_spin.setEnabled(active_contour_enabled)
        self.active_contour_beta_spin.setEnabled(active_contour_enabled)

        # Refinement controls (always enabled if edge detection is enabled)
        self.min_contour_length_spin.setEnabled(enabled)
        self.max_contour_length_spin.setEnabled(enabled)

        # Interface specific controls
        self.detect_fluid_interface_checkbox.setEnabled(enabled)
        self.detect_solid_interface_checkbox.setEnabled(enabled)
        self.solid_interface_proximity_spin.setEnabled(enabled and self.detect_solid_interface_checkbox.isChecked())

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------
    def settings(self) -> EdgeDetectionSettings:
        return self._settings
