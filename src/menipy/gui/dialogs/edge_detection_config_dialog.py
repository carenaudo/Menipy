"""Dialog for edge detection method configuration."""

from __future__ import annotations

"""Configuration dialog for edge detection pipeline settings."""

from typing import Optional

from PySide6.QtCore import Qt, Signal
import numpy as np
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
from PySide6.QtGui import QImage, QPixmap

from menipy.models.config import EdgeDetectionSettings
from menipy.common.registry import EDGE_DETECTORS
from menipy.gui.components.plugin_settings_widget import PluginSettingsWidget


class EdgeDetectionConfigDialog(QDialog):
    """Allows users to tweak edge detection pipeline parameters."""

    previewRequested = Signal(object)

    def __init__(
        self,
        settings: EdgeDetectionSettings,
        parent: Optional[QDialog] = None,
        compact_mode: bool = False,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Edge Detection Configuration")
        self._settings = settings.model_copy(deep=True)
        self._compact = compact_mode

        self._build_ui()
        self._load_settings()
        self._wire_signals()

        if self._compact:
            self._apply_compact_mode()

    def _apply_compact_mode(self):
        """Hide navigation and preview for focused editing."""
        # Hide Sidebar
        self.stage_list.parentWidget().hide()
        # Hide Preview
        self.preview_label.hide()
        # Resize
        self.resize(450, 300)
        # Select correct page based on method
        self._select_page_for_method(self._settings.method)

    def _select_page_for_method(self, method: str):
        """Switch stacked widget to the page relevant for the method."""
        # Map method names to list widget items/pages
        # Indices: 0=General, 1=Canny, 2=Threshold, 3=Sobel/etc, 4=Result?, 5=Refinement?, 6=Interface
        # This is brittle relying on order, but _build_ui defines order.
        # General=0, Canny=1, Threshold=2, Sobel=3, Active=4, Refinement=5, Interface=6
        
        if method == "canny":
            self.pages.setCurrentIndex(1)
        elif method == "threshold":
            self.pages.setCurrentIndex(2)
        elif method in ("sobel", "scharr", "laplacian"):
            self.pages.setCurrentIndex(3)
        elif method == "active_contour":
            self.pages.setCurrentIndex(4)
        else:
            # Fallback or plugin page (which is usually added last if at all)
            # If standard core method fallback to General (0)
            self.pages.setCurrentIndex(0)

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
            "General Settings",
            "Canny",
            "Threshold",
            "Sobel/Scharr/Laplacian",
            "Active Contour",
            "Refinement",
            "Interface Specific",
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

        # Preview area (matches Preprocessing dialog pattern)
        self.preview_label = QLabel("No preview available", self)
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumSize(360, 240)
        self.preview_label.setStyleSheet("background-color: #333; color: #CCC;")
        self.preview_label.setScaledContents(True)
        pages_layout.addWidget(self.preview_label)

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
        self._build_active_contour_page()
        self._build_refinement_page()
        self._build_interface_page()
        self._build_plugin_page()

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

    def _build_general_page(self) -> None:
        widget = QWidget(self)
        form = QFormLayout(widget)

        self.enabled_checkbox = QCheckBox("Enable Edge Detection", widget)
        form.addRow(self.enabled_checkbox)

        self.method_combo = QComboBox(widget)
        # Core methods
        core_methods = [
            "canny", "sobel", "scharr", "laplacian", "threshold", "active_contour"
        ]
        for label in core_methods:
            self.method_combo.addItem(label)
            
        # Plugin methods
        for name in EDGE_DETECTORS.keys():
            if name not in core_methods:
                self.method_combo.addItem(name)
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
        """_build_active_contour_page."""
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
        """_build_refinement_page."""
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
        """_build_interface_page."""
        widget = QWidget(self)
        form = QFormLayout(widget)

        self.detect_fluid_interface_checkbox = QCheckBox(
            "Detect Fluid-Droplet Interface", widget
        )
        form.addRow(self.detect_fluid_interface_checkbox)

        self.detect_solid_interface_checkbox = QCheckBox(
            "Detect Solid-Droplet Interface", widget
        )
        form.addRow(self.detect_solid_interface_checkbox)

        self.solid_interface_proximity_spin = QSpinBox(widget)
        self.solid_interface_proximity_spin.setRange(0, 100)
        self.solid_interface_proximity_spin.setValue(10)
        form.addRow(
            "Solid Interface Proximity (px)", self.solid_interface_proximity_spin
        )

        info = QLabel(
            "Proximity defines search area around contact line for solid interface."
        )
        info.setWordWrap(True)
        form.addRow(info)

        self.pages.addWidget(widget)

    def _build_plugin_page(self) -> None:
        """Page for dynamic plugin settings."""
        self.plugin_settings_widget = PluginSettingsWidget(self)
        self.plugin_settings_widget.settingsChanged.connect(self._on_plugin_settings_changed)
        self.pages.addWidget(self.plugin_settings_widget)

    # ------------------------------------------------------------------
    # Data binding
    # ------------------------------------------------------------------
    def _wire_signals(self) -> None:
        """_wire_signals."""
        self.stage_list.currentRowChanged.connect(self.pages.setCurrentIndex)
        self._button_box.accepted.connect(self._on_accept)
        self._button_box.rejected.connect(self.reject)
        self._preview_btn.clicked.connect(self._on_preview)
        self._reset_btn.clicked.connect(self._on_reset)
        # Dialog can receive preview images via a slot with signature (image, metadata)

        self.enabled_checkbox.toggled.connect(self._update_controls_enablement)
        self.method_combo.currentTextChanged.connect(self._update_controls_enablement)
        self.gaussian_blur_checkbox.toggled.connect(self._update_controls_enablement)

    def _load_settings(self) -> None:
        """_load_settings."""
        s = self._settings
        self.enabled_checkbox.setChecked(s.enabled)
        self.method_combo.setCurrentText(s.method)
        self.gaussian_blur_checkbox.setChecked(s.gaussian_blur_before)
        self.gaussian_kernel_size_spin.setValue(s.gaussian_kernel_size)
        self.gaussian_kernel_size_spin.setValue(s.gaussian_kernel_size)
        self.gaussian_sigma_x_spin.setValue(s.gaussian_sigma_x)

        # Load plugin settings into widget if current method is a plugin
        current_method = s.method
        if current_method not in ["canny", "sobel", "scharr", "laplacian", "threshold", "active_contour"]:
            # It's a plugin or unhandled
            plugin_config = s.plugin_settings.get(current_method, {})
            self.plugin_settings_widget.set_plugin(current_method, plugin_config)

        self.canny_threshold1_spin.setValue(s.canny_threshold1)
        self.canny_threshold2_spin.setValue(s.canny_threshold2)
        self.canny_aperture_size_combo.setCurrentText(str(s.canny_aperture_size))
        self.canny_L2_gradient_checkbox.setChecked(s.canny_L2_gradient)

        self.threshold_value_spin.setValue(s.threshold_value)
        self.threshold_max_value_spin.setValue(s.threshold_max_value)
        self.threshold_type_combo.setCurrentText(s.threshold_type)

        self.sobel_kernel_size_spin.setValue(s.sobel_kernel_size)
        self.laplacian_kernel_size_spin.setValue(s.laplacian_kernel_size)

        self.active_contour_iterations_spin.setValue(s.snake_iterations)
        self.active_contour_alpha_spin.setValue(s.snake_alpha)
        self.active_contour_beta_spin.setValue(s.snake_beta)

        self.min_contour_length_spin.setValue(s.min_contour_length)
        self.max_contour_length_spin.setValue(s.max_contour_length)

        self.detect_fluid_interface_checkbox.setChecked(s.detect_fluid_interface)
        self.detect_solid_interface_checkbox.setChecked(s.detect_solid_interface)
        self.solid_interface_proximity_spin.setValue(s.solid_interface_proximity)

        self._update_controls_enablement()

    def _collect_settings(self) -> EdgeDetectionSettings:
        """_collect_settings."""
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

        s.snake_iterations = self.active_contour_iterations_spin.value()
        s.snake_alpha = self.active_contour_alpha_spin.value()
        s.snake_beta = self.active_contour_beta_spin.value()

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
        """_on_preview."""
        settings = self._collect_settings()
        self.previewRequested.emit(settings)

    def _on_reset(self) -> None:
        """_on_reset."""
        self._settings = EdgeDetectionSettings()
        self._load_settings()

    def _on_accept(self) -> None:
        """_on_accept."""
        self._collect_settings()
        self.accept()

    def _on_preview_image_ready(self, image: np.ndarray, metadata: dict) -> None:
        if image is None:
            self.preview_label.setText("No preview available")
            self.preview_label.clear()
            return

        h, w = image.shape[:2]
        bytes_per_line = 3 * w

        if image.ndim == 2:  # Grayscale
            q_image = QImage(image.data, w, h, w, QImage.Format.Format_Grayscale8)
        elif image.ndim == 3 and image.shape[2] == 3:  # BGR
            q_image = QImage(
                image.data, w, h, bytes_per_line, QImage.Format.Format_BGR888
            )
        elif image.ndim == 3 and image.shape[2] == 4:  # BGRA
            q_image = QImage(
                image.data, w, h, bytes_per_line, QImage.Format.Format_ARGB32
            )
        else:
            self.preview_label.setText("Unsupported image format")
            self.preview_label.clear()
            return

        pixmap = QPixmap.fromImage(q_image)
        self.preview_label.setPixmap(
            pixmap.scaled(
                self.preview_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
        )

    # ------------------------------------------------------------------
    # Control enablement helpers
    # ------------------------------------------------------------------
    def _update_controls_enablement(self) -> None:
        """_update_controls_enablement."""
        enabled = self.enabled_checkbox.isChecked()
        method = self.method_combo.currentText()

        # General settings controls
        self.method_combo.setEnabled(enabled)
        self.gaussian_blur_checkbox.setEnabled(enabled)
        self.gaussian_kernel_size_spin.setEnabled(
            enabled and self.gaussian_blur_checkbox.isChecked()
        )
        self.gaussian_sigma_x_spin.setEnabled(
            enabled and self.gaussian_blur_checkbox.isChecked()
        )

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
        self.sobel_kernel_size_spin.setEnabled(
            sobel_laplacian_enabled and method in {"sobel", "scharr"}
        )
        self.laplacian_kernel_size_spin.setEnabled(
            sobel_laplacian_enabled and method == "laplacian"
        )

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
        self.solid_interface_proximity_spin.setEnabled(
            enabled and self.detect_solid_interface_checkbox.isChecked()
        )
        
        # Check if we should switch to plugin page
        core_methods = {"canny", "sobel", "scharr", "laplacian", "threshold", "active_contour"}
        if method not in core_methods and enabled:
             # It is a plugin
             if self.pages.currentWidget() != self.plugin_settings_widget:
                 self.pages.setCurrentWidget(self.plugin_settings_widget)
                 # Also refresh the widget content if needed, though usually handled by load_settings or manual trigger
                 # We trigger a reload of settings for this plugin
                 current_config = self._settings.plugin_settings.get(method, {})
                 self.plugin_settings_widget.set_plugin(method, current_config)
             
             # Disable other pages in list? Or just let stacked widget handle view.
             # Ideally we highlight "Plugin Settings" in list if we added it, but we didn't add it to the list.
             # Instead we are hijacking the view.
             
        elif enabled:
            # If it's a core method, ensure we are not on plugin page (unless user clicked it?)
            # The list widget drives the page, but we want method combo to ALSO drive the page?
            # Existing specific pages (Canny, Threshold) seem to be manually selected by user via list.
            # BUT _update_controls_enablement is called on combo change.
            # Let's auto-switch to the relevant page for core methods too for better UX
            pass

    def _on_plugin_settings_changed(self, new_settings: dict):
        """Update the internal settings model when plugin widget changes."""
        method = self.method_combo.currentText()
        if method:
            # Update the specific plugin's settings in the dict
            # We need to be careful not to overwrite other plugins' settings
            # But here we are just updating the entry for 'method'
            if self._settings.plugin_settings is None:
                self._settings.plugin_settings = {}
            self._settings.plugin_settings[method] = new_settings
            
            # Trigger preview if auto-preview is desired?
            # self._on_preview() 


    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------
    def settings(self) -> EdgeDetectionSettings:
        """settings.

        Returns
        -------
        type
        Description.
        """
        return self._settings
