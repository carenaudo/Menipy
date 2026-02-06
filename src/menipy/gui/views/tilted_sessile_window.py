"""
Tilted Sessile Window

Specialized window for tilted sessile drop measurements.
Measures advancing and receding contact angles during tilting.
"""
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame,
    QScrollArea, QTabWidget, QToolBar, QPushButton
)
from PySide6.QtGui import QAction

from menipy.gui import theme
from menipy.gui.views.base_experiment_window import BaseExperimentWindow
from menipy.gui.panels import ImageSourcePanel, ActionPanel
from menipy.gui.panels.tilt_stage_panel import TiltStagePanel
from menipy.gui.widgets.tilted_sessile_results_widget import TiltedSessileResultsWidget


class TiltedImageViewer(QWidget):
    """Image viewer for tilted sessile with tilt angle indicator."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._pixmap: QPixmap | None = None
        self._zoom = 1.0
        self._tilt_angle = 0.0
        
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Tilt indicator at top
        self._tilt_indicator = QLabel("Tilt: 0.0°")
        self._tilt_indicator.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._tilt_indicator.setStyleSheet(f"""
            background-color: {theme.BG_SECONDARY};
            color: {theme.TEXT_PRIMARY};
            font-size: 16px;
            font-weight: bold;
            padding: 8px;
            border-radius: 4px;
        """)
        self._tilt_indicator.setMaximumHeight(40)
        layout.addWidget(self._tilt_indicator)
        
        # Scroll area
        self._scroll_area = QScrollArea()
        self._scroll_area.setWidgetResizable(True)
        self._scroll_area.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._scroll_area.setStyleSheet(f"""
            QScrollArea {{
                background-color: {theme.BG_TERTIARY};
                border: none;
            }}
        """)
        
        self._image_label = QLabel()
        self._image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._image_label.setStyleSheet(f"color: {theme.TEXT_SECONDARY};")
        self._image_label.setText("Load an image to begin analysis\n\n📐\n\nFor tilted measurements,\nplace drop on tilting platform")
        
        self._scroll_area.setWidget(self._image_label)
        layout.addWidget(self._scroll_area)
    
    def set_image(self, pixmap: QPixmap | None):
        """Set the image to display."""
        self._pixmap = pixmap
        if pixmap and not pixmap.isNull():
            scaled = pixmap.scaled(
                pixmap.size() * self._zoom,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self._image_label.setPixmap(scaled)
        else:
            self._image_label.clear()
            self._image_label.setText("Load an image to begin analysis")
    
    def set_tilt_angle(self, angle: float):
        """Update the tilt angle indicator."""
        self._tilt_angle = angle
        self._tilt_indicator.setText(f"Tilt: {angle:.1f}°")
    
    def set_zoom(self, zoom: float):
        self._zoom = max(0.1, min(10.0, zoom))
        if self._pixmap:
            self.set_image(self._pixmap)
    
    def zoom_in(self):
        self.set_zoom(self._zoom * 1.25)
    
    def zoom_out(self):
        self.set_zoom(self._zoom / 1.25)
    
    def fit_to_view(self):
        if self._pixmap and not self._pixmap.isNull():
            vp_size = self._scroll_area.viewport().size()
            img_size = self._pixmap.size()
            scale_x = vp_size.width() / img_size.width()
            scale_y = vp_size.height() / img_size.height()
            self._zoom = min(scale_x, scale_y) * 0.95
            self.set_image(self._pixmap)
    
    def reset_view(self):
        self.set_zoom(1.0)


class TiltedSessileWindow(BaseExperimentWindow):
    """
    Window for tilted sessile drop contact angle analysis.
    
    Provides controls for:
        - Image loading
    - Tilt stage control
    - Advancing/receding angle measurement
    - Tilt sequence automation
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._connect_signals()
    
    def get_experiment_type(self) -> str:
        return theme.EXPERIMENT_TILTED_SESSILE
    
    def get_experiment_title(self) -> str:
        """Get experiment title.

        Returns
        -------
        type
        Description.
        """
        return "Tilted Sessile"
    
    def _create_left_panel_content(self) -> QWidget:
        """Create the left panel content with setup controls."""
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setStyleSheet("background: transparent;")
        
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setContentsMargins(0, 0, 0, 0)
        scroll_layout.setSpacing(12)
        
        # Image Source panel
        self._image_source_panel = ImageSourcePanel()
        scroll_layout.addWidget(self._image_source_panel)
        
        # Tilt Stage panel
        self._tilt_panel = TiltStagePanel()
        scroll_layout.addWidget(self._tilt_panel)
        
        # Action panel
        self._action_panel = ActionPanel()
        self._action_panel.set_button_text("▶ Measure at Current Tilt")
        scroll_layout.addWidget(self._action_panel)
        
        scroll_layout.addStretch()
        scroll.setWidget(scroll_content)
        layout.addWidget(scroll)
        
        return content
    
    def _create_center_panel_content(self) -> QWidget:
        """Create the center panel content with image viewer."""
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        self._image_viewer = TiltedImageViewer()
        layout.addWidget(self._image_viewer)
        
        return content
    
    def _create_right_panel_content(self) -> QWidget:
        """Create the right panel content with results."""
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)
        
        # Tilted results widget
        self._results_widget = TiltedSessileResultsWidget()
        layout.addWidget(self._results_widget)
        
        # Tabs for sequence history and graph
        tabs = QTabWidget()
        tabs.setStyleSheet(f"""
            QTabWidget::pane {{
                background-color: {theme.BG_TERTIARY};
                border: 1px solid {theme.BORDER_DEFAULT};
                border-radius: 4px;
            }}
            QTabBar::tab {{
                background-color: {theme.BG_SECONDARY};
                color: {theme.TEXT_SECONDARY};
                padding: 8px 16px;
                margin-right: 2px;
            }}
            QTabBar::tab:selected {{
                background-color: {theme.BG_TERTIARY};
                color: {theme.TEXT_PRIMARY};
            }}
        """)
        
        # Graph tab - angle vs tilt
        graph_placeholder = QWidget()
        graph_layout = QVBoxLayout(graph_placeholder)
        graph_label = QLabel("📈 Contact angle vs tilt\nPlot will appear here after\nmultiple measurements")
        graph_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        graph_label.setStyleSheet(f"color: {theme.TEXT_SECONDARY};")
        graph_layout.addWidget(graph_label)
        tabs.addTab(graph_placeholder, "θ vs Tilt")
        
        # History tab
        history_placeholder = QWidget()
        history_layout = QVBoxLayout(history_placeholder)
        history_label = QLabel("📋 Measurement sequence\nhistory will appear here")
        history_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        history_label.setStyleSheet(f"color: {theme.TEXT_SECONDARY};")
        history_layout.addWidget(history_label)
        tabs.addTab(history_placeholder, "History")
        
        layout.addWidget(tabs, stretch=1)
        
        # Export buttons
        export_frame = QFrame()
        export_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {theme.BG_SECONDARY};
                border: 1px solid {theme.BORDER_DEFAULT};
                border-radius: 8px;
            }}
        """)
        export_layout = QHBoxLayout(export_frame)
        export_layout.setContentsMargins(8, 8, 8, 8)
        export_layout.setSpacing(8)
        
        csv_btn = QPushButton("📊 Export CSV")
        csv_btn.setProperty("secondary", True)
        export_layout.addWidget(csv_btn)
        
        report_btn = QPushButton("📄 Report")
        report_btn.setProperty("secondary", True)
        export_layout.addWidget(report_btn)
        
        layout.addWidget(export_frame)
        
        return content
    
    def _add_toolbar_items(self, toolbar: QToolBar):
        """Add tilted-sessile specific toolbar items."""
        # Overlay toggle
        overlay_action = QAction("🔲 Overlay", self)
        overlay_action.setCheckable(True)
        overlay_action.setChecked(True)
        toolbar.addAction(overlay_action)
        
        # Advancing point
        adv_action = QAction("🟢 Advancing", self)
        adv_action.setCheckable(True)
        adv_action.setChecked(True)
        toolbar.addAction(adv_action)
        
        # Receding point
        rec_action = QAction("🟠 Receding", self)
        rec_action.setCheckable(True)
        rec_action.setChecked(True)
        toolbar.addAction(rec_action)
        
        # Baseline
        baseline_action = QAction("📏 Baseline", self)
        baseline_action.setCheckable(True)
        baseline_action.setChecked(True)
        toolbar.addAction(baseline_action)
    
    def _connect_signals(self):
        """Connect internal signals."""
        self._image_source_panel.image_loaded.connect(self._on_image_loaded)
        self._action_panel.analyze_requested.connect(self._on_analyze_requested)
        self._action_panel.cancel_requested.connect(self._on_cancel_requested)
        self._tilt_panel.angle_changed.connect(self._on_tilt_changed)
        self._tilt_panel.sequence_started.connect(self._on_sequence_started)
        self._tilt_panel.sequence_stopped.connect(self._on_sequence_stopped)
    
    # -------------------------------------------------------------------------
    # Signal Handlers
    # -------------------------------------------------------------------------
    
    def _on_image_loaded(self, path: str, pixmap: QPixmap | None):
        """Handle image loaded."""
        if pixmap:
            self._image_viewer.set_image(pixmap)
            self._image_viewer.fit_to_view()
            self.set_status(f"Loaded: {path}")
    
    def _on_analyze_requested(self):
        """Handle analyze button click."""
        self._action_panel.set_state(ActionPanel.STATE_PROCESSING)
        self.set_status("Analyzing tilted sessile drop...")
        
        # Simulate analysis
        self._simulate_analysis()
    
    def _on_cancel_requested(self):
        """Handle cancel button click."""
        self._action_panel.set_state(ActionPanel.STATE_READY)
        self.set_status("Analysis cancelled")
    
    def _on_tilt_changed(self, angle: float):
        """Handle tilt angle change."""
        self._image_viewer.set_tilt_angle(angle)
        self._results_widget.set_tilt_angle(angle)
        self.set_status(f"Tilt angle: {angle:.1f}°")
    
    def _on_sequence_started(self, start: float, end: float, step: float):
        """Handle tilt sequence start."""
        self.set_status(f"Starting sequence: {start}° to {end}° by {step}°")
    
    def _on_sequence_stopped(self):
        """Handle tilt sequence stop."""
        self.set_status("Sequence stopped")
    
    def _simulate_analysis(self):
        """Simulate analysis with mock results."""
        import random
        
        self._action_panel.set_progress(50, "Detecting contact points...")
        
        tilt = self._tilt_panel.get_angle()
        
        # Mock results
        results = {
            "tilt_angle": tilt,
            "advancing_angle": 75.0 + tilt * 0.2 + random.uniform(-1, 1),
            "receding_angle": 55.0 - tilt * 0.1 + random.uniform(-1, 1),
            "advancing_uncertainty": 0.5,
            "receding_uncertainty": 0.5,
            "hysteresis": 20.0 + tilt * 0.3,
            "volume": 5.2 + random.uniform(-0.1, 0.1),
            "base_left": 1.25 + random.uniform(-0.05, 0.05),
            "base_right": 1.18 + random.uniform(-0.05, 0.05),
            "confidence": 95 + random.uniform(-5, 5),
        }
        
        self._results_widget.set_results(results)
        self._action_panel.set_state(ActionPanel.STATE_COMPLETE)
        self.set_status("Analysis complete")
        
        self.analysis_completed.emit(results)
    
    # -------------------------------------------------------------------------
    # Toolbar Actions
    # -------------------------------------------------------------------------
    
    def _on_zoom_in(self):
        self._image_viewer.zoom_in()
    
    def _on_zoom_out(self):
        self._image_viewer.zoom_out()
    
    def _on_fit_view(self):
        self._image_viewer.fit_to_view()
    
    def _on_reset_view(self):
        self._image_viewer.reset_view()
