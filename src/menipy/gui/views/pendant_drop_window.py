"""
Pendant Drop Window

Specialized window for pendant drop surface tension measurements.
"""
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame,
    QScrollArea, QTabWidget, QToolBar, QPushButton,
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
from menipy.gui.dialogs.calibration_wizard_dialog import CalibrationWizardDialog
from menipy.gui.widgets.pendant_results_widget import PendantResultsWidget
from menipy.pipelines.pendant import PendantPipeline
from menipy.models.context import Context
import cv2
import numpy as np


class PendantHistoryTableWidget(QWidget):
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
            "ID", "γ (mN/m)", "Vol (μL)", "Beta", "R0 (mm)", "Conf (%)"
        ])
        
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        
        # Basic styling
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
            
        self.table.setItem(row, 0, item(str(self._count)))
        self.table.setItem(row, 1, item(fmt("surface_tension", "{:.2f}")))
        self.table.setItem(row, 2, item(fmt("volume", "{:.3f}")))
        self.table.setItem(row, 3, item(fmt("bond_number", "{:.3f}")))
        self.table.setItem(row, 4, item(fmt("r0_mm", "{:.3f}"))) # Pipeline uses r0_mm
        
        # Confidence
        conf = results.get("confidence")
        self.table.setItem(row, 5, item(f"{conf:.0f}" if conf is not None else "-"))
        
        self.table.scrollToBottom()


class PendantImageViewer(QWidget):
    """
    Central image viewer for pendant drop with zoom and pan.
    Displays the loaded image with analysis overlays.
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._pixmap: QPixmap | None = None
        self._zoom = 1.0
        
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Scroll area for zoom/pan
        self._scroll_area = QScrollArea()
        self._scroll_area.setWidgetResizable(True)
        self._scroll_area.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._scroll_area.setStyleSheet(f"""
            QScrollArea {{
                background-color: {theme.BG_TERTIARY};
                border: none;
            }}
        """)
        
        # Image label
        self._image_label = QLabel()
        self._image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._image_label.setStyleSheet(f"color: {theme.TEXT_SECONDARY};")
        self._image_label.setText("Load an image to begin analysis\n\n💧\n\nPendant drop should hang from\na needle visible in the image")
        
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


class PendantParametersPanel(QFrame):
    """Simplified parameters panel for pendant drop."""
    
    parameters_changed = Signal(dict)
    material_db_requested = Signal(str)  # field name
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("pendantParametersPanel")
        self._setup_ui()
    
    def _setup_ui(self):
        self.setStyleSheet(f"""
            QFrame#pendantParametersPanel {{
                background-color: {theme.BG_SECONDARY};
                border: 1px solid {theme.BORDER_DEFAULT};
                border-radius: 8px;
            }}
        """)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)
        
        header = QLabel("Physical Properties")
        header.setStyleSheet(f"""
            font-size: {theme.FONT_SIZE_LARGE}px;
            font-weight: bold;
            color: {theme.TEXT_PRIMARY};
        """)
        layout.addWidget(header)
        
        # Liquid density
        from PySide6.QtWidgets import QDoubleSpinBox
        
        # Helper for input rows with DB button
        def add_density_input(label_text, field_name, default):
            container = QWidget()
            h_layout = QVBoxLayout(container)
            h_layout.setContentsMargins(0, 0, 0, 0)
            h_layout.setSpacing(4)
            
            label = QLabel(label_text)
            label.setStyleSheet(f"color: {theme.TEXT_SECONDARY};")
            h_layout.addWidget(label)
            
            row = QHBoxLayout()
            row.setSpacing(8)
            
            spin = QDoubleSpinBox()
            spin.setRange(0, 50000)
            spin.setValue(default)
            spin.setDecimals(1)
            row.addWidget(spin, stretch=1)
            
            setattr(self, f"_{field_name}", spin)
            
            db_btn = QPushButton("📚")
            db_btn.setProperty("secondary", True)
            db_btn.setFixedSize(32, 24)
            db_btn.setToolTip("Select from Material Database")
            db_btn.clicked.connect(lambda: self.material_db_requested.emit(field_name))
            row.addWidget(db_btn)
            
            h_layout.addLayout(row)
            layout.addWidget(container)

        add_density_input("Liquid Density (kg/m³)", "liquid_density", 1000.0)
        add_density_input("Surrounding Density (kg/m³)", "surrounding_density", 1.2)
        
        # Gravity
        g_label = QLabel("Gravitational Acceleration (m/s²)")
        g_label.setStyleSheet(f"color: {theme.TEXT_SECONDARY};")
        layout.addWidget(g_label)
        
        self._gravity = QDoubleSpinBox()
        self._gravity.setRange(0, 20)
        self._gravity.setValue(9.81)
        self._gravity.setDecimals(4)
        layout.addWidget(self._gravity)
        
        layout.addStretch()
    
    def get_parameters(self) -> dict:
        return {
            "liquid_density": self._liquid_density.value(),
            "surrounding_density": self._surrounding_density.value(),
            "gravity": self._gravity.value(),
        }
        
    def set_density(self, field_name: str, value: float):
        """Set density value from database selection."""
        spin = getattr(self, f"_{field_name}", None)
        if spin:
            spin.setValue(value)

class PendantDropWindow(BaseExperimentWindow):
    """
    Window for pendant drop surface tension analysis.
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._current_image = None
        self._current_path = None
        self._connect_signals()
    
    def get_experiment_type(self) -> str:
        return theme.EXPERIMENT_PENDANT
    
    def get_experiment_title(self) -> str:
        return "Pendant Drop"
    
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
        
        # Calibration panel
        self._calibration_panel = CalibrationPanel()
        scroll_layout.addWidget(self._calibration_panel)
        
        # Parameters panel
        self._parameters_panel = PendantParametersPanel()
        scroll_layout.addWidget(self._parameters_panel)
        
        # Action panel
        self._action_panel = ActionPanel()
        self._action_panel.set_button_text("▶ Analyze Drop Shape")
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
        
        self._image_viewer = PendantImageViewer()
        layout.addWidget(self._image_viewer)
        
        return content
    
    def _create_right_panel_content(self) -> QWidget:
        """Create the right panel content with results."""
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)
        
        # Pendant results widget
        self._results_widget = PendantResultsWidget()
        layout.addWidget(self._results_widget)
        
        # Tabs for profile and history
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
        
        # Profile tab
        profile_placeholder = QWidget()
        profile_layout = QVBoxLayout(profile_placeholder)
        profile_label = QLabel("📈 Drop profile comparison\nwill appear here after analysis")
        profile_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        profile_label.setStyleSheet(f"color: {theme.TEXT_SECONDARY};")
        profile_layout.addWidget(profile_label)
        tabs.addTab(profile_placeholder, "Profile")
        
        # History tab
        self._history_widget = PendantHistoryTableWidget()
        tabs.addTab(self._history_widget, "History")
        
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
        """Add pendant-drop specific toolbar items."""
        # Overlay toggle
        overlay_action = QAction("🔲 Overlay", self)
        overlay_action.setCheckable(True)
        overlay_action.setChecked(True)
        toolbar.addAction(overlay_action)
        
        # Needle detection
        needle_action = QAction("📍 Needle", self)
        needle_action.setCheckable(True)
        needle_action.setChecked(True)
        toolbar.addAction(needle_action)
        
        # Drop contour
        contour_action = QAction("〰️ Contour", self)
        contour_action.setCheckable(True)
        contour_action.setChecked(True)
        toolbar.addAction(contour_action)
        
        # Fitted profile
        fit_action = QAction("📐 Fitted Profile", self)
        fit_action.setCheckable(True)
        fit_action.setChecked(True)
        toolbar.addAction(fit_action)
    
    def _connect_signals(self):
        """Connect internal signals."""
        self._image_source_panel.image_loaded.connect(self._on_image_loaded)
        self._action_panel.analyze_requested.connect(self._on_analyze_requested)
        self._action_panel.cancel_requested.connect(self._on_cancel_requested)
        
        # Calibration signals
        self._calibration_panel.calibration_requested.connect(self._on_calibration_requested)
        
        self._parameters_panel.material_db_requested.connect(self._on_material_db_requested)
    
    def _on_image_loaded(self, path: str, pixmap: QPixmap | None):
        """Handle image loaded."""
        self._current_path = path
        if path:
            self._current_image = cv2.imread(path)
            
        if pixmap:
            self._image_viewer.set_image(pixmap)
            self._image_viewer.fit_to_view()
            self.set_status(f"Loaded: {path}")
    

    
    
    def _on_calibration_requested(self):
        """Handle calibration request."""
        if self._current_image is None:
            self.set_status("Please load an image first")
            return
            
        wizard = CalibrationWizardDialog(self._current_image, "pendant", self)
        if wizard.exec():
            result = wizard.get_result()
            self._on_calibration_result(result)

    def _on_calibration_result(self, result):
        """Handle calibration result."""
        self._last_calibration_result = result
        
        # Calculate scale
        px_per_mm = 1.0
        needle_width_px = 0
        
        # Try to get needle width from result
        if result.needle_rect:
             _, _, w, _ = result.needle_rect
             needle_width_px = w
             
        # Ask user for needle diameter if we have pixel width
        if needle_width_px > 0:
            from PySide6.QtWidgets import QInputDialog
            diameter, ok = QInputDialog.getDouble(
                self, "Needle Diameter", 
                "Enter needle outer diameter (mm):", 
                0.71, 0.01, 100.0, 3
            )
            known_diameter_mm = diameter if ok else 0.71
        else:
             known_diameter_mm = 0.71

        if needle_width_px > 0 and known_diameter_mm > 0:
            px_per_mm = needle_width_px / known_diameter_mm
            self.set_status(f"Calibrated: {px_per_mm:.1f} px/mm (Needle: {needle_width_px}px, {known_diameter_mm}mm)")
        else:
            self.set_status("Calibration incomplete: Need needle width and diameter")

        # Update panel
        # Use set_calibration instead of set_scale (which doesn't exist)
        import datetime
        date_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        self._calibration_panel.set_calibration(px_per_mm, "Auto-Calibration", date_str)
        
        # Store diameter for analysis
        self._last_known_diameter = known_diameter_mm
        
    def _run_analysis(self):
        """Run the actual pendant drop analysis."""
        if self._current_image is None:
            self._action_panel.set_state(ActionPanel.STATE_READY)
            return

        try:
            # 1. Setup Context
            ctx = Context(image=self._current_image)
            ctx.image_path = self._current_path
            
            # Calibration
            # Use get_scale_factor instead of get_scale
            scale = self._calibration_panel.get_scale_factor()
            ctx.scale = {"px_per_mm": scale}
            
            # If we have calibration result (ROI/Needle), pass it
            calib = getattr(self, "_last_calibration_result", None)
            if calib:
                ctx.needle_rect = calib.needle_rect
                ctx.roi_rect = calib.roi_rect
                # Reuse needle diameter
                ctx.needle_diameter_mm = getattr(self, "_last_known_diameter", 0.71)
                
            # Physics params
            params = self._parameters_panel.get_parameters()
            ctx.physics = {
                "rho1": params.get("liquid_density", 1000.0),
                "rho2": params.get("surrounding_density", 1.2),
                "g": params.get("gravity", 9.81)
            }

            # 2. Run Pipeline
            pipeline = PendantPipeline()
            # In a real app, run in background thread. For now, run directly.
            pipeline.run(ctx)
            
            # 3. Process Results
            results = ctx.results or {}
            
            # Map pipeline results to UI keys
            # Pipeline: surface_tension_mN_m, beta, r0_mm, volume_uL, height_mm, diameter_mm
            # Widget: surface_tension, bond_number, apex_radius, volume, de, ds...
            
            ui_results = {
                "surface_tension": results.get("surface_tension_mN_m"),
                "bond_number": results.get("beta"),
                "apex_radius": results.get("r0_mm"),
                "volume": results.get("volume_uL"),
                "de": results.get("diameter_mm"), # Approx DE = 2*R0 for sphere, close enough for visual?
                "ds": results.get("diameter_mm"), # Placeholder
                "density_diff": params.get("liquid_density", 0) - params.get("surrounding_density", 0),
                "rmse": results.get("residuals", {}).get("optimality", 0.0), # or extract properly
                "iterations": 0, # Pipeline fit doesn't expose iterations cleanly in top dict yet
                "confidence": 100.0 # Placeholder
            }
            
            # Filter None
            ui_results = {k: v for k, v in ui_results.items() if v is not None}
            
            self._results_widget.set_results(ui_results)
            self._history_widget.add_result(results) # History uses pipeline keys (surface_tension_mN_m etc... wait!)
            
            # History widget expects: surface_tension, volume, bond_number, r0_mm
            # My HistoryWidget implementation uses fmt("surface_tension", ...)
            # BUT pipeline output has "surface_tension_mN_m".
            # I must fix HistoryWidget OR flatten results.
            # I'll update HistoryWidget keys in Next Step or map them here?
            # Better: Map them here to a 'clean' dict for history? 
            # Or use `results` but add aliases.
            
            results["surface_tension"] = results.get("surface_tension_mN_m")
            results["volume"] = results.get("volume_uL")
            results["bond_number"] = results.get("beta")
            
            self._history_widget.add_result(results)

            self._action_panel.set_state(ActionPanel.STATE_COMPLETE)
            self.set_status("Analysis complete")
            
            # Update image viewer with overlay
            if ctx.image is not None:
                import cv2
                from PySide6.QtGui import QImage
                 # Ensure contiguous array for QImage
                rgb = cv2.cvtColor(ctx.image, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb.shape
                bytes_per_line = ch * w
                qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
                self._image_viewer.set_image(QPixmap.fromImage(qimg))
            
            self.analysis_completed.emit(results)
            
        except Exception as e:
            self.set_status(f"Analysis failed: {e}")
            self._action_panel.set_state(ActionPanel.STATE_READY)
            import traceback
            traceback.print_exc()

    def _on_analyze_requested(self):
        """Handle analyze button click."""
        self._action_panel.set_state(ActionPanel.STATE_PROCESSING)
        self.set_status("Analyzing pendant drop...")
        
        # Use QTimer to allow UI update before blocking
        from PySide6.QtCore import QTimer
        QTimer.singleShot(100, self._run_analysis)

    def _on_cancel_requested(self):
        """Handle cancel button click."""
        self._action_panel.set_state(ActionPanel.STATE_READY)
        self.set_status("Analysis cancelled")
        
    def _on_material_db_requested(self, field_name: str):
        """Handle material database request."""
        from menipy.gui.dialogs.material_dialog import MaterialDialog
        dialog = MaterialDialog(self, selection_mode=True, table_type="materials")
        dialog.item_selected.connect(
            lambda data: self._on_material_selected(data, field_name)
        )
        dialog.exec()
        
    def _on_material_selected(self, data: dict, field_name: str):
        """Handle material selection."""
        if density := data.get("density"):
            self._parameters_panel.set_density(field_name, density)
            self.set_status(f"Selected material: {data['name']} ({density} kg/m³)")
            
    def _on_needle_db_requested(self):
        # CalibrationPanel might not emit this, but if we add it later:
        pass


    
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
