"""
Sessile Drop Window

Specialized window for sessile drop contact angle measurements.
Provides the full three-panel layout with all necessary controls.
"""
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame,
    QScrollArea, QSplitter, QTabWidget, QToolBar, QPushButton,
    QTableWidget, QTableWidgetItem, QHeaderView
)
from PySide6.QtGui import QAction

from menipy.gui import theme
from menipy.gui.views.base_experiment_window import BaseExperimentWindow
from menipy.gui.panels import (
    ImageSourcePanel,
    CalibrationPanel,
    ParametersPanel,
    ActionPanel
)
from menipy.gui.widgets.quick_stats_widget import QuickStatsWidget
from menipy.gui.widgets.interactive_image_viewer import InteractiveImageViewer
from menipy.pipelines.sessile import SessilePipeline
from menipy.models.context import Context
from menipy.models.config import PreprocessingSettings, EdgeDetectionSettings
from menipy.gui.dialogs.preprocessing_config_dialog import PreprocessingConfigDialog


class HistoryTableWidget(QWidget):
    """Table showing measurement history."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._count = 0
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        self.table = QTableWidget()
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels([
            "ID", "θ_L (°)", "θ_R (°)", "θ_M (°)", "Vol (μL)", "Area (mm²)"
        ])
        
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        
        # Basic styling to match theme
        self.table.setStyleSheet(f"""
            QTableWidget {{
                background-color: {theme.BG_SECONDARY};
                color: {theme.TEXT_PRIMARY};
                border: none;
                gridline-color: {theme.BORDER_DEFAULT};
            }}
            QHeaderView::section {{
                background-color: {theme.BG_TERTIARY};
                color: {theme.TEXT_SECONDARY};
                padding: 4px;
                border: none;
                border-right: 1px solid {theme.BORDER_DEFAULT};
                border-bottom: 1px solid {theme.BORDER_DEFAULT};
            }}
            QTableCornerButton::section {{
                background-color: {theme.BG_TERTIARY};
                border: none;
            }}
        """)
        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(True)
        
        layout.addWidget(self.table)
        
    def add_result(self, results: dict):
        """Add a result row to the table."""
        self._count += 1
        row = self.table.rowCount()
        self.table.insertRow(row)
        
        def item(text):
            it = QTableWidgetItem(text)
            it.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            return it
            
        def fmt(key, fstr="{:.1f}"):
            val = results.get(key)
            if val is None:
                return "-"
            return fstr.format(val)
            
        # Use cleaned keys from ui_results
        al = results.get("angle_left")
        ar = results.get("angle_right")
        am = results.get("angle_mean")
             
        self.table.setItem(row, 0, item(str(self._count)))
        self.table.setItem(row, 1, item(f"{al:.1f}" if al is not None else "-"))
        self.table.setItem(row, 2, item(f"{ar:.1f}" if ar is not None else "-"))
        self.table.setItem(row, 3, item(f"{am:.1f}" if am is not None else "-"))
        self.table.setItem(row, 4, item(fmt("volume", "{:.3f}")))
        self.table.setItem(row, 5, item(fmt("area", "{:.2f}")))
        
        self.table.scrollToBottom()


class SessileDropWindow(BaseExperimentWindow):
    """
    Window for sessile drop contact angle analysis.
    
    Provides controls for:
    - Image loading (single, batch, camera)
    - Calibration setup
    - Measurement parameters
    - Analysis execution
    - Results display
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._show_overlay = True
        self._show_baseline = True
        self._show_contact_points = True
        self._show_contact_points = True
        self._current_image_path = None
        self._last_ctx: Context | None = None
        self._preprocessing_settings = PreprocessingSettings()
        self._edge_settings = EdgeDetectionSettings()
        self._pipeline_settings: dict = {}
        self._connect_signals()
        
    def load_image(self, path: str):
        """Load image via source panel."""
        self._image_source_panel.load_image(path)
    
    def get_experiment_type(self) -> str:
        return theme.EXPERIMENT_SESSILE
    
    def get_experiment_title(self) -> str:
        return "Sessile Drop"
    
    def _create_left_panel_content(self) -> QWidget:
        """Create the left panel content with setup controls."""
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)
        
        # Wrap in scroll area for long content
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
        
        # Calibration panel
        self._calibration_panel = CalibrationPanel()
        scroll_layout.addWidget(self._calibration_panel)
        
        # Parameters panel
        self._parameters_panel = ParametersPanel()
        scroll_layout.addWidget(self._parameters_panel)
        
        # Action panel (Analyze button)
        self._action_panel = ActionPanel()
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
        
        # Image viewer
        self._image_viewer = ImageViewer()
        layout.addWidget(self._image_viewer)
        
        return content
    
    def _create_right_panel_content(self) -> QWidget:
        """Create the right panel content with results."""
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)
        
        # Quick stats widget
        self._quick_stats = QuickStatsWidget()
        layout.addWidget(self._quick_stats)
        
        # Tabs for history and charts
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
        
        # History tab
        self._history_widget = HistoryTableWidget()
        tabs.addTab(self._history_widget, "History")
        
        # Chart tab
        chart_placeholder = QWidget()
        chart_layout = QVBoxLayout(chart_placeholder)
        chart_label = QLabel("📊 Charts will appear here\nafter multiple measurements")
        chart_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        chart_label.setStyleSheet(f"color: {theme.TEXT_SECONDARY};")
        chart_layout.addWidget(chart_label)
        tabs.addTab(chart_placeholder, "Charts")
        
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
        
    def _create_center_panel_content(self) -> QWidget:
        """Create the center panel content (Image Viewer)."""
        self._image_viewer = InteractiveImageViewer()
        self._image_viewer.overlay_painted.connect(self._on_draw_overlay)
        return self._image_viewer
    
    def _create_right_panel_content(self) -> QWidget:
        """Create the right panel content (Results)."""
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)
        
        # Quick Stats
        self._quick_stats = QuickStatsWidget()
        layout.addWidget(self._quick_stats)
        
        # History/Charts placeholder
        from menipy.gui.widgets.measurements_table import MeasurementsTableWidget
        self._history_widget = MeasurementsTableWidget()
        layout.addWidget(self._history_widget, stretch=1)
        
        return container
    def _add_toolbar_items(self, toolbar: QToolBar):
        """Add sessile-drop specific toolbar items."""
        # Overlay toggle
        overlay_action = QAction("🔲 Overlay", self)
        overlay_action.setCheckable(True)
        overlay_action.setChecked(True)
        overlay_action.triggered.connect(self._on_toggle_overlay)
        toolbar.addAction(overlay_action)
        
        # Baseline markers
        baseline_action = QAction("📏 Baseline", self)
        baseline_action.setCheckable(True)
        baseline_action.setChecked(True)
        baseline_action.triggered.connect(self._on_toggle_baseline)
        toolbar.addAction(baseline_action)
        
        # Contact points
        contact_action = QAction("📍 Contact Points", self)
        contact_action.setCheckable(True)
        contact_action.setChecked(True)
        contact_action.triggered.connect(self._on_toggle_contact_points)
        toolbar.addAction(contact_action)
    
    def _connect_signals(self):
        """Connect internal signals."""
        # Image loading
        self._image_source_panel.image_loaded.connect(self._on_image_loaded)
        
        # Analysis
        self._action_panel.analyze_requested.connect(self._on_analyze_requested)
        self._action_panel.settings_requested.connect(self._on_settings_requested)
        self._action_panel.cancel_requested.connect(self._on_cancel_requested)
        
        # Calibration
        # Calibration
        self._calibration_panel.calibration_requested.connect(
            self._on_calibration_requested
        )
        
        # Material Database
        self._parameters_panel.material_database_requested.connect(
            self._on_material_database_requested
        )
    
    # -------------------------------------------------------------------------
    # Signal Handlers
    # -------------------------------------------------------------------------
    
    def _on_material_database_requested(self, field_name: str):
        """Handle material database request."""
        from menipy.gui.dialogs.material_dialog import MaterialDialog
        
        dialog = MaterialDialog(self, selection_mode=True)
        # Connect signal before exec
        dialog.material_selected.connect(
            lambda data: self._on_material_selected(data, field_name)
        )
        dialog.exec()
            
    def _on_material_selected(self, material_data: dict, field_name: str):
        """Handle material selection from dialog."""
        if density := material_data.get("density"):
            self._parameters_panel.set_density(field_name, density)
            self.set_status(f"Selected material: {material_data['name']} ({density} kg/m³)")
    
    def _on_image_loaded(self, path: str, pixmap: QPixmap | None):
        """Handle image loaded."""
        self._current_image_path = path
        if pixmap:
            self._image_viewer.set_image(pixmap)
            self._image_viewer.fit_to_view()
            self.set_status(f"Loaded: {path}")
    
    def _on_analyze_requested(self):
        """Handle analyze button click."""
        self._action_panel.set_state(ActionPanel.STATE_PROCESSING)
        self.set_status("Running analysis...")
        
        # Run actual analysis pipeline
        self._run_analysis()
    
    def _on_cancel_requested(self):
        """Handle cancel button click."""
        self._action_panel.set_state(ActionPanel.STATE_READY)
        self.set_status("Analysis cancelled")
    
    def _on_settings_requested(self):
        """Handle settings button click."""
        # Open preprocessing config dialog
        # Note: We duplicate some logic from DialogCoordinator here because
        # SessileDropWindow is currently acting independently of the main controller
        # in some aspects (like running the pipeline directly).
        dialog = PreprocessingConfigDialog(
            self._preprocessing_settings, parent=self
        )
        
        # We need a controller to handle previews if we want previews in the dialog.
        # Ideally we refactor this to use the DialogCoordinator.
        # For now, we instantiate a temporary controller or just let the dialog run without
        # live preview if too complex, BUT the user wants the auto-detect preview.
        
        from menipy.gui.controllers.preprocessing_controller import PreprocessingPipelineController
        controller = PreprocessingPipelineController(parent=self)
        controller.set_settings(self._preprocessing_settings)
        
        # If we have an image currently loaded, pass it to controller
        if self._image_viewer._pixmap and not self._image_viewer._pixmap.isNull():
             img = self._image_viewer._pixmap.toImage()
             # Convert QImage to numpy
             import numpy as np
             width = img.width()
             height = img.height()
             
             ptr = img.bits()
             # Assuming Format_RGB888 or similar for simple conversion, check format
             # If loaded via QPixmap(path), format depends. 
             # Simpler: read from path if available
             if self._current_image_path:
                 import cv2
                 cv_img = cv2.imread(self._current_image_path)
                 if cv_img is not None:
                      controller.set_source(cv_img)
        
        controller.previewReady.connect(dialog._on_preview_image_ready)
        dialog.previewRequested.connect(lambda s: self._run_preview(controller, s))
        
        if dialog.exec():
            self._preprocessing_settings = dialog.settings()
            self.set_status("Settings updated.")
            
    def _run_preview(self, controller, settings):
        controller.set_settings(settings)
        controller.run()
    
    def _on_calibration_requested(self):
        """Handle calibration wizard request."""
        # Use the main application's CalibrationWizardDialog for full auto-detection
        from menipy.gui.dialogs.calibration_wizard_dialog import CalibrationWizardDialog
        
        # Get raw image data
        image_data = None
        if self._current_image_path:
            import cv2
            image_data = cv2.imread(self._current_image_path)
        
        if image_data is None:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Calibration", "Please load an image first.")
            return
        
        wizard = CalibrationWizardDialog(image_data, pipeline_name="sessile", parent=self)
        wizard.calibration_complete.connect(self._on_calibration_result)
        wizard.exec()
        
    def _on_calibration_result(self, result):
        """Handle calibration result from CalibrationWizardDialog."""
        if result is None:
            return
            
        # Store detected regions for later use in analysis
        self._last_calibration_result = result
        
        # Update status with what was detected
        conf = result.confidence_scores.get("overall", 0.0)
        parts = []
        if result.needle_rect:
            parts.append("needle")
        if result.substrate_line:
            parts.append("substrate")
        if result.roi_rect:
            parts.append("ROI")
        
        detected_str = ", ".join(parts) if parts else "nothing"
        self.set_status(f"Detected: {detected_str} (confidence: {conf*100:.0f}%)")
        
        # If needle was detected, prompt user for physical dimension to compute scale
        if result.needle_rect:
            _, _, width_px, _ = result.needle_rect
            self._detected_needle_width_px = width_px
            
            # Ask user for physical needle diameter
            from PySide6.QtWidgets import QInputDialog
            # getDouble(parent, title, label, value, min, max, decimals)
            diameter_mm, ok = QInputDialog.getDouble(
                self,
                "Enter Needle Diameter",
                f"Detected needle width: {width_px:.1f} px\n\n"
                "Enter the physical needle outer diameter (mm):",
                0.72,   # default value (21 gauge)
                0.1,    # minimum
                5.0,    # maximum
                3       # decimals
            )
            
            if ok and diameter_mm > 0:
                # Compute scale factor: px_per_mm = width_px / diameter_mm
                scale_factor = width_px / diameter_mm
                self._calibration_panel.set_calibration(
                    scale_factor,
                    reference_file="needle",
                    date=None
                )
                self.set_status(f"Calibration set: {scale_factor:.2f} px/mm")
    
    def _on_calibration_completed(self, scale_factor: float):
        """Handle successful calibration (legacy signal)."""
        self._calibration_panel.set_calibration(scale_factor)
        self.set_status(f"Calibration updated: {scale_factor:.2f} px/mm")
    
    def _run_analysis(self):
        """Run the actual analysis pipeline."""
        if not self._current_image_path:
            self.set_status("No image loaded.")
            self._action_panel.set_state(ActionPanel.STATE_READY)
            return

        try:
            # 1. Setup Context / Params
            # Get density directly from panel (assuming method exists or we parse it)
            # For now, default or parsing from UI if possible.
            # (ParametersPanel usually manages this, assumes defaults if not set)
            
            # 2. Run Pipeline
            pipeline = SessilePipeline(
                preprocessing_settings=self._preprocessing_settings,
                edge_detection_settings=self._edge_settings,
            )
            if "solver" in self._pipeline_settings:
                pipeline.solver_name = self._pipeline_settings["solver"]
            if "preprocessor" in self._pipeline_settings:
                pipeline.preprocessor_name = self._pipeline_settings["preprocessor"]
            if "edge_detector" in self._pipeline_settings:
                pipeline.edge_detector_name = self._pipeline_settings["edge_detector"]
            
            # Build kwargs from calibration result if available
            pipeline_kwargs = {
                "image_path": self._current_image_path,
                "preprocessing_settings": self._preprocessing_settings,
            }
            # pipeline-stage choices
            if "contact_angle_method" in self._pipeline_settings:
                pipeline_kwargs["contact_angle_method"] = self._pipeline_settings["contact_angle_method"]
            # physics overrides
            if any(k in self._pipeline_settings for k in ("rho1", "rho2", "g")):
                pipeline_kwargs["physics"] = {
                    "rho1": float(self._pipeline_settings.get("rho1", 1000.0)),
                    "rho2": float(self._pipeline_settings.get("rho2", 1.2)),
                    "g": float(self._pipeline_settings.get("g", 9.80665)),
                }
            
            # Use calibration result if available (from CalibrationWizardDialog)
            calib = getattr(self, "_last_calibration_result", None)
            if calib:
                # Pass pre-detected regions to skip auto-detection
                if calib.needle_rect:
                    pipeline_kwargs["needle_rect"] = calib.needle_rect
                if calib.substrate_line:
                    pipeline_kwargs["substrate_line"] = calib.substrate_line
                if calib.roi_rect:
                    pipeline_kwargs["roi_rect"] = calib.roi_rect
                if calib.drop_contour is not None:
                    pipeline_kwargs["drop_contour"] = calib.drop_contour
                if calib.contact_points:
                    pipeline_kwargs["contact_points"] = calib.contact_points
                    # Derive contact_line from contact_points (left, right)
                    # contact_line is the baseline where drop meets substrate
                    left_pt, right_pt = calib.contact_points
                    pipeline_kwargs["contact_line"] = (left_pt, right_pt)
                # With pre-detected features, we can skip auto-detection
                pipeline_kwargs["auto_detect_features"] = False
            else:
                # No calibration - run auto-detection
                pipeline_kwargs["auto_detect_features"] = True
            
            # Get scale factor from calibration panel (px/mm)
            if self._calibration_panel.is_calibrated():
                scale_factor = self._calibration_panel.get_scale_factor()
                if scale_factor > 0:
                    pipeline_kwargs["px_per_mm"] = scale_factor
            
            ctx = pipeline.run(**pipeline_kwargs)
            
            # 3. Handle Results
            if ctx.error:
                self.set_status(f"Analysis failed: {ctx.error}")
                self._action_panel.set_state(ActionPanel.STATE_READY)
                return
                
            self._last_ctx = ctx
            results = ctx.results or {}
            
            # Map pipeline results to UI widget expected format
            raw_ui_results = {
                "angle_left": results.get("theta_left_deg"),
                "angle_right": results.get("theta_right_deg"),
                "angle_mean": (
                    (results.get("theta_left_deg", 0) + results.get("theta_right_deg", 0)) / 2 
                    if results.get("theta_left_deg") is not None else None
                ),
                "angle_uncertainty": max(
                    results.get("uncertainty_deg", {}).get("left", 0.5),
                    results.get("uncertainty_deg", {}).get("right", 0.5)
                ),
                "volume": results.get("volume_uL"),
                "diameter": results.get("diameter_mm"),
                "height": results.get("height_mm"),
                "base_width": results.get("diameter_mm"), # Base width ~= diameter for sessile
                "surface_tension": results.get("surface_tension"), # If computed
                "area": results.get("contact_surface_mm2"),
                "confidence": results.get("baseline_confidence", 1.0) * 100
            }
            
            # Filter out None values to avoid formatting errors in widget
            ui_results = {k: v for k, v in raw_ui_results.items() if v is not None}
            
            # 4. update UI
            self._quick_stats.set_results(ui_results)
            self._history_widget.add_result(ui_results) # Update valid history table
            
            self._action_panel.set_state(ActionPanel.STATE_COMPLETE)
            self.set_status("Analysis complete")
            
            self.analysis_completed.emit(results)
            self._image_viewer.update()
            
        except Exception as e:
            self.set_status(f"Error: {str(e)}")
            self._action_panel.set_state(ActionPanel.STATE_READY)
            import traceback
            traceback.print_exc()

    def _on_draw_overlay(self, painter, target_rect, zoom):
        """Draw analysis overlay."""
        if not self._show_overlay or not self._last_ctx:
            return
            
        from PySide6.QtGui import QColor, QPen, QPolygonF
        from PySide6.QtCore import QPointF, QRectF
        
        ctx = self._last_ctx
        
        # Map function
        img_w = self._image_viewer._pixmap.width()
        img_h = self._image_viewer._pixmap.height()
        
        def to_view(x, y):
            # x, y are in image pixels
            # map 0..img_w to target_rect.left()..target_rect.right()
            vx = target_rect.left() + (x / img_w) * target_rect.width()
            vy = target_rect.top() + (y / img_h) * target_rect.height()
            return QPointF(vx, vy)
            
        def to_view_rect(rect_obj):
             # rect_obj has x, y, width, height
             p1 = to_view(rect_obj.x, rect_obj.y)
             p2 = to_view(rect_obj.x + rect_obj.width, rect_obj.y + rect_obj.height)
             return QRectF(p1, p2)

        # 1. Draw ROI (Yellow)
        if ctx.roi:
            # ROI is likely a tuple or object with x,y,w,h. 
            # Context.roi is Tuple[int, int, int, int] (x, y, w, h)
            if isinstance(ctx.roi, (list, tuple)):
                rx, ry, rw, rh = ctx.roi
                r_view = to_view_rect(type('R',(),{'x':rx,'y':ry,'width':rw,'height':rh}))
                
                pen = QPen(QColor(255, 255, 0, 200)) # Yellow
                pen.setWidth(1)
                pen.setStyle(Qt.PenStyle.DotLine)
                painter.setPen(pen)
                painter.drawRect(r_view)

        # 2. Draw Needle Rect (Cyan)
        if ctx.needle_rect:
            if isinstance(ctx.needle_rect, (list, tuple)):
                nx, ny, nw, nh = ctx.needle_rect
                n_view = to_view_rect(type('N',(),{'x':nx,'y':ny,'width':nw,'height':nh}))
                
                pen = QPen(QColor(0, 255, 255, 180)) # Cyan
                pen.setWidth(1)
                painter.setPen(pen)
                painter.drawRect(n_view)

        # 3. Draw Contour (Green)
        if ctx.contour and ctx.contour.xy is not None:
            # xy is Nx2 numpy array
            pts = []
            for pt in ctx.contour.xy:
                pts.append(to_view(pt[0], pt[1]))
            
            if pts:
                pen = QPen(QColor(0, 255, 0, 200)) # Green
                pen.setWidth(2)
                painter.setPen(pen)
                painter.drawPolyline(pts)

        # 4. Draw Geometry (Red/Blue)
        if ctx.geometry:
            geom = ctx.geometry
            
            # Baseline (Blue)
            if self._show_baseline and geom.baseline_y is not None:
                y_base = geom.baseline_y
                p1 = to_view(0, y_base)
                p2 = to_view(img_w, y_base)
                
                pen = QPen(QColor(0, 100, 255, 200))
                pen.setWidth(1)
                painter.setPen(pen)
                painter.drawLine(p1, p2)
            
            # Axis (Red dash)
            if geom.axis_x is not None:
                x_axis = geom.axis_x
                p1 = to_view(x_axis, 0)
                p2 = to_view(x_axis, img_h)
                
                pen = QPen(QColor(255, 50, 50, 180))
                pen.setWidth(1)
                pen.setStyle(Qt.PenStyle.DashLine)
                painter.setPen(pen)
                painter.drawLine(p1, p2)
                
            # Apex (Red Point)
            if geom.apex_xy:
                ap = to_view(geom.apex_xy[0], geom.apex_xy[1])
                painter.setBrush(QColor(255, 0, 0))
                painter.setPen(Qt.PenStyle.NoPen)
                painter.drawEllipse(ap, 4, 4)
                
        # 5. Draw Contact Points (Magenta)
        if self._show_contact_points and ctx.contact_line:
            # contact_line is ((x1, y1), (x2, y2))
            (x1, y1), (x2, y2) = ctx.contact_line
            p1 = to_view(x1, y1)
            p2 = to_view(x2, y2)
            
            pen = QPen(QColor(255, 0, 255, 255))
            pen.setWidth(5)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(pen)
            painter.drawPoint(p1)
            painter.drawPoint(p2)

    
    def _on_toggle_overlay(self, checked: bool):
        """Toggle overlay visibility."""
        self._show_overlay = checked
        self._image_viewer.update()
        self.set_status(f"Overlay: {'on' if checked else 'off'}")
    
    def _on_toggle_baseline(self, checked: bool):
        """Toggle baseline visibility."""
        self._show_baseline = checked
        self._image_viewer.update()
        self.set_status(f"Baseline: {'on' if checked else 'off'}")
    
    def _on_toggle_contact_points(self, checked: bool):
        """Toggle contact points visibility."""
        self._show_contact_points = checked
        self._image_viewer.update()
        self.set_status(f"Contact points: {'on' if checked else 'off'}")
    
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

    # ------------------------------------------------------------------
    # Analysis settings hook
    # ------------------------------------------------------------------
    def apply_analysis_settings(
        self,
        pre: PreprocessingSettings,
        edge: EdgeDetectionSettings,
        pipeline_settings: dict | None = None,
    ):
        self._preprocessing_settings = pre
        self._edge_settings = edge
        self._pipeline_settings = pipeline_settings or {}
        # Overlay prefs
        if "baseline_visible" in self._pipeline_settings:
            self._show_baseline = self._pipeline_settings["baseline_visible"]
        if "contact_visible" in self._pipeline_settings:
            self._show_contact_points = self._pipeline_settings["contact_visible"]
