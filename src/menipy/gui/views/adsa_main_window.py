"""
ADSA Main Window

The primary application window for the ADSA (Automated Drop Shape Analysis) software.
Uses a stacked widget to switch between the experiment selector and experiment-specific windows.
"""
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QAction, QKeySequence, QCloseEvent
from PySide6.QtWidgets import (
    QMainWindow, QStackedWidget, QMenuBar, QMenu, QStatusBar,
    QMessageBox, QFileDialog, QWidget
)

from menipy.gui import theme
from menipy.gui.views.experiment_selector import ExperimentSelectorView
from PySide6.QtCore import QSettings
import json


class ADSAMainWindow(QMainWindow):
    """
    Main application window for ADSA.
    
    Uses a QStackedWidget to switch between:
    - Experiment Selector screen (index 0)
    - Experiment-specific windows (indices 1+)
    
    Signals:
        experiment_changed: Emitted when the user switches experiment types.
    """
    
    experiment_changed = Signal(str)
    
    # Stack indices
    INDEX_SELECTOR = 0
    INDEX_EXPERIMENT = 1
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ADSA - Automated Drop Shape Analysis")
        self.setMinimumSize(theme.MIN_WINDOW_WIDTH, theme.MIN_WINDOW_HEIGHT)
        
        # Apply theme
        self.setStyleSheet(theme.get_stylesheet())
        
        # Track current experiment type and window
        self._current_experiment_type: str | None = None
        self._experiment_windows: dict[str, QWidget] = {}
        self._settings_store = QSettings("Menipy", "ADSA")
        
        self._setup_ui()
        self._setup_menus()
        self._setup_status_bar()
        
        # Connect signals
        self._connect_signals()
        
        # Notification Manager
        from menipy.gui.widgets.notification import NotificationManager
        self._notification_manager = NotificationManager(self)
        
        # Load recent projects (placeholder)
        self._load_recent_projects()
        
    
    def _setup_ui(self):
        """Set up the main UI structure."""
        # Central stacked widget
        self._stack = QStackedWidget()
        self.setCentralWidget(self._stack)
        
        # Experiment selector (always at index 0)
        self._selector = ExperimentSelectorView()
        self._stack.addWidget(self._selector)
        
        # Start showing the selector
        self._stack.setCurrentIndex(self.INDEX_SELECTOR)
    
    def _setup_menus(self):
        """Set up the menu bar."""
        menubar = self.menuBar()
        
        # === File Menu ===
        file_menu = menubar.addMenu("&File")
        
        switch_action = QAction("🏠 Switch Experiment Type", self)
        switch_action.setShortcut(QKeySequence("Ctrl+E"))
        switch_action.triggered.connect(self.show_experiment_selector)
        file_menu.addAction(switch_action)
        
        file_menu.addSeparator()
        
        open_image_action = QAction("📂 Open Image...", self)
        open_image_action.setShortcut(QKeySequence.Open)
        open_image_action.triggered.connect(self._on_open_image)
        file_menu.addAction(open_image_action)
        
        open_folder_action = QAction("📁 Open Folder...", self)
        open_folder_action.setShortcut(QKeySequence("Ctrl+Shift+O"))
        open_folder_action.triggered.connect(self._on_open_folder)
        file_menu.addAction(open_folder_action)
        
        connect_camera_action = QAction("📷 Connect Camera...", self)
        connect_camera_action.triggered.connect(self._on_connect_camera)
        file_menu.addAction(connect_camera_action)
        
        file_menu.addSeparator()
        
        save_action = QAction("💾 Save Project", self)
        save_action.setShortcut(QKeySequence.Save)
        save_action.triggered.connect(self._on_save_project)
        file_menu.addAction(save_action)
        
        save_as_action = QAction("💾 Save Project As...", self)
        save_as_action.setShortcut(QKeySequence("Ctrl+Shift+S"))
        save_as_action.triggered.connect(self._on_save_project_as)
        file_menu.addAction(save_as_action)
        
        file_menu.addSeparator()
        
        export_menu = file_menu.addMenu("📤 Export")
        
        export_csv_action = QAction("Export CSV...", self)
        export_csv_action.triggered.connect(self._on_export_csv)
        export_menu.addAction(export_csv_action)
        
        export_excel_action = QAction("Export Excel...", self)
        export_excel_action.triggered.connect(self._on_export_excel)
        export_menu.addAction(export_excel_action)
        
        export_image_action = QAction("🖼️ Export Image...", self)
        export_image_action.triggered.connect(self._on_export_image)
        export_menu.addAction(export_image_action)
        
        export_report_action = QAction("📄 Generate Report...", self)
        export_report_action.triggered.connect(self._on_generate_report)
        export_menu.addAction(export_report_action)
        
        file_menu.addSeparator()
        
        recent_menu = file_menu.addMenu("Recent Projects")
        self._recent_menu = recent_menu
        
        file_menu.addSeparator()
        
        exit_action = QAction("🚪 Exit", self)
        exit_action.setShortcut(QKeySequence("Alt+F4"))
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # === Edit Menu ===
        edit_menu = menubar.addMenu("&Edit")
        
        undo_action = QAction("Undo", self)
        undo_action.setShortcut(QKeySequence.Undo)
        undo_action.setEnabled(False)  # Placeholder
        edit_menu.addAction(undo_action)
        
        redo_action = QAction("Redo", self)
        redo_action.setShortcut(QKeySequence.Redo)
        redo_action.setEnabled(False)  # Placeholder
        edit_menu.addAction(redo_action)
        
        edit_menu.addSeparator()
        
        preferences_action = QAction("⚙️ Preferences...", self)
        preferences_action.triggered.connect(self._on_preferences)
        edit_menu.addAction(preferences_action)
        
        # === View Menu ===
        view_menu = menubar.addMenu("&View")
        
        zoom_in_action = QAction("🔍+ Zoom In", self)
        zoom_in_action.setShortcut(QKeySequence.ZoomIn)
        zoom_in_action.triggered.connect(self._on_zoom_in)
        view_menu.addAction(zoom_in_action)
        
        zoom_out_action = QAction("🔍- Zoom Out", self)
        zoom_out_action.setShortcut(QKeySequence.ZoomOut)
        zoom_out_action.triggered.connect(self._on_zoom_out)
        view_menu.addAction(zoom_out_action)
        
        fit_action = QAction("↔️ Fit to Window", self)
        fit_action.setShortcut(QKeySequence("Ctrl+0"))
        fit_action.triggered.connect(self._on_fit_view)
        view_menu.addAction(fit_action)
        
        actual_size_action = QAction("Actual Size", self)
        actual_size_action.setShortcut(QKeySequence("Ctrl+1"))
        actual_size_action.triggered.connect(self._on_actual_size)
        view_menu.addAction(actual_size_action)
        
        view_menu.addSeparator()
        
        fullscreen_action = QAction("Toggle Fullscreen", self)
        fullscreen_action.setShortcut(QKeySequence("F11"))
        fullscreen_action.triggered.connect(self._on_toggle_fullscreen)
        view_menu.addAction(fullscreen_action)
        
        # === Tools Menu ===
        tools_menu = menubar.addMenu("&Tools")
        
        calibrate_action = QAction("🎯 Calibrate...", self)
        calibrate_action.setShortcut(QKeySequence("Ctrl+K"))
        calibrate_action.triggered.connect(self._on_calibrate)
        tools_menu.addAction(calibrate_action)
        
        material_db_action = QAction("📚 Material Database...", self)
        material_db_action.triggered.connect(self._on_material_database)
        tools_menu.addAction(material_db_action)

        needle_db_action = QAction("💉 Needle Database...", self)
        needle_db_action.triggered.connect(self._on_needle_database)
        tools_menu.addAction(needle_db_action)

        syringe_db_action = QAction("💊 Syringe Database...", self)
        syringe_db_action.triggered.connect(self._on_syringe_database)
        tools_menu.addAction(syringe_db_action)
        
        # === Analysis Menu ===
        analysis_menu = menubar.addMenu("&Analysis")
        
        run_action = QAction("▶ Run Analysis", self)
        run_action.setShortcut(QKeySequence("Ctrl+R"))
        run_action.triggered.connect(self._on_run_analysis)
        analysis_menu.addAction(run_action)
        
        pause_action = QAction("⏸️ Pause", self)
        pause_action.setShortcut(QKeySequence("Space"))
        pause_action.triggered.connect(self._on_pause_analysis)
        analysis_menu.addAction(pause_action)
        
        stop_action = QAction("⏹️ Stop", self)
        stop_action.setShortcut(QKeySequence("Escape"))
        stop_action.triggered.connect(self._on_stop_analysis)
        analysis_menu.addAction(stop_action)
        
        analysis_menu.addSeparator()
        
        settings_action = QAction("⚙️ Analysis Settings...", self)
        settings_action.triggered.connect(self._on_analysis_settings)
        analysis_menu.addAction(settings_action)
        
        analysis_menu.addSeparator()
        
        batch_action = QAction("🔄 Batch Process Folder...", self)
        batch_action.triggered.connect(self._on_batch_process)
        analysis_menu.addAction(batch_action)
        
        video_action = QAction("⏯️ Process Video...", self)
        video_action.triggered.connect(self._on_process_video)
        analysis_menu.addAction(video_action)
        
        analysis_menu.addSeparator()
        
        clear_action = QAction("🧹 Clear All Results", self)
        clear_action.setShortcut(QKeySequence("Ctrl+Shift+Delete"))
        clear_action.triggered.connect(self._on_clear_results)
        analysis_menu.addAction(clear_action)
        
        # === Utilities Menu ===
        utilities_menu = menubar.addMenu("&Utilities")
        
        utilities_action = QAction("🔧 Image Utilities...", self)
        utilities_action.setShortcut(QKeySequence("Ctrl+U"))
        utilities_action.triggered.connect(self._on_utilities)
        utilities_menu.addAction(utilities_action)
        
        utilities_menu.addSeparator()
        
        image_quality_action = QAction("📊 Image Quality Analysis", self)
        image_quality_action.triggered.connect(self._on_image_quality_analysis)
        utilities_menu.addAction(image_quality_action)
        
        edge_comparison_action = QAction("🔍 Edge Detection Comparison", self)
        edge_comparison_action.triggered.connect(self._on_edge_comparison)
        utilities_menu.addAction(edge_comparison_action)
        
        # === Help Menu ===
        help_menu = menubar.addMenu("&Help")
        
        help_action = QAction("📖 Help", self)
        help_action.setShortcut(QKeySequence.HelpContents)
        help_action.triggered.connect(self._on_help)
        help_menu.addAction(help_action)
        
        help_menu.addSeparator()
        
        about_action = QAction("About ADSA", self)
        about_action.triggered.connect(self._on_about)
        help_menu.addAction(about_action)
    
    def _setup_status_bar(self):
        """Set up the status bar."""
        status = self.statusBar()
        status.showMessage("Ready")
    
    def _connect_signals(self):
        """Connect internal signals."""
        self._selector.experiment_selected.connect(self._on_experiment_selected)
        self._selector.project_opened.connect(self._on_project_opened)
    
    def _load_recent_projects(self):
        """Load and display recent projects from settings."""
        import datetime

        raw = self._settings_store.value("recent_projects")
        projects = []
        if raw:
            try:
                projects = json.loads(raw)
            except Exception:
                projects = []

        self._selector.set_recent_projects(projects)
        self._update_recent_menu()

    def _record_recent(self, experiment_type: str, title: str, path: str | None):
        """Append an entry to recent projects list (max 10)."""
        import datetime

        raw = self._settings_store.value("recent_projects") or "[]"
        try:
            projects = json.loads(raw)
        except Exception:
            projects = []
        
        # Ensure it's a list
        if not isinstance(projects, list):
            projects = []
            
        entry = {
            "filename": title,
            "experiment_type": experiment_type,
            "date_str": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
            "path": path or "",
        }
        # Remove duplicates by filename/type OR by path if provided
        filtered = []
        for p in projects:
            same_path = (path is not None) and (p.get("path") == path)
            same_name = (p.get("filename") == title) and (p.get("experiment_type") == experiment_type)
            if not same_path and not same_name:
                filtered.append(p)
                
        projects = filtered
        projects.insert(0, entry)
        projects = projects[:10]
        
        self._settings_store.setValue("recent_projects", json.dumps(projects))
        self._selector.set_recent_projects(projects)
        self._update_recent_menu()
    
    def _update_recent_menu(self):
        """Populate the recent projects menu."""
        if not hasattr(self, "_recent_menu"):
            return
            
        self._recent_menu.clear()
        
        raw = self._settings_store.value("recent_projects")
        projects = []
        if raw:
            try:
                projects = json.loads(raw)
            except Exception:
                projects = []
                
        if not projects:
            dummy = QAction("(No recent projects)", self)
            dummy.setEnabled(False)
            self._recent_menu.addAction(dummy)
            return
            
        for proj in projects:
            name = proj.get("filename", "Untitled")
            date = proj.get("date_str", "")
            path = proj.get("path", "")
            
            label = f"{name} ({date})"
            action = QAction(label, self)
            if path:
                action.setToolTip(path)
                # Use default argument capture for lambda closure
                action.triggered.connect(lambda checked=False, p=path: self._on_project_opened(p))
            else:
                action.setEnabled(False) # Disable seed items with no path
                
            self._recent_menu.addAction(action)
            
        self._recent_menu.addSeparator()
        clear_action = QAction("Clear Recent Projects", self)
        clear_action.triggered.connect(self._clear_recent_projects)
        self._recent_menu.addAction(clear_action)
        
    def _clear_recent_projects(self):
        """Clear the recent projects list."""
        self._settings_store.setValue("recent_projects", "[]")
        self._selector.set_recent_projects([])
        self._update_recent_menu()
        
    def _open_project_file(self, path: str):
        """
        Open a project file and restore state.
        
        Args:
            path: Absolute path to .adsa.json file.
        """
        import os
        from menipy.models.config import PreprocessingSettings, EdgeDetectionSettings

        if not os.path.exists(path):
            self.statusBar().showMessage(f"Project file not found: {path}", 4000)
            return
            
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                
            exp_type = data.get("experiment_type")
            image_path = data.get("image_path")
            analysis_data = data.get("analysis_settings") or {}
            
            if not exp_type:
                raise ValueError("Missing experiment_type in project file")
                
            # 1. Switch to experiment
            self.show_experiment_window(exp_type)
            window = self.get_current_experiment_window()
            
            if window:
                # 2. Load image
                if image_path and os.path.exists(image_path):
                    if hasattr(window, "load_image"):
                        window.load_image(image_path)
                    elif hasattr(window, "_image_source_panel"):
                        window._image_source_panel.load_image(image_path)
                
                # 3. Apply settings
                # If analysis_settings are in the project file, use them overrides stored defaults
                if analysis_data:
                    pre = PreprocessingSettings(**analysis_data.get("preproc", {}))
                    edge = EdgeDetectionSettings(**analysis_data.get("edge", {}))
                    pipe = analysis_data.get("pipeline", {})
                    
                    if hasattr(window, "apply_analysis_settings"):
                        window.apply_analysis_settings(pre, edge, pipe)
            
            self.statusBar().showMessage(f"Opened project: {path}", 3000)
            
            # Update recents list (move to top)
            title = data.get("filename") or os.path.basename(path).replace(".adsa.json", "").replace(".json", "")
            self._record_recent(exp_type, title, path)
            
        except Exception as e:
            self.statusBar().showMessage(f"Error opening project: {str(e)}", 4000)
            print(f"Project open error: {e}")
    
    # =========================================================================
    # Public Methods
    # =========================================================================
    
    def show_experiment_selector(self):
        """Show the experiment selector screen."""
        self._stack.setCurrentIndex(self.INDEX_SELECTOR)
        self.setWindowTitle("ADSA - Automated Drop Shape Analysis")
    
    def show_experiment_window(self, experiment_type: str):
        """
        Show the window for a specific experiment type.
        
        Args:
            experiment_type: One of the EXPERIMENT_* constants from theme.
        """
        # Get or create the experiment window
        if experiment_type not in self._experiment_windows:
            window = self._create_experiment_window(experiment_type)
            if window:
                self._experiment_windows[experiment_type] = window
                self._stack.addWidget(window)
        
        window = self._experiment_windows.get(experiment_type)
        if window:
            self._current_experiment_type = experiment_type
            index = self._stack.indexOf(window)
            self._stack.setCurrentIndex(index)
            
            # Update window title
            title = self._get_experiment_title(experiment_type)
            self.setWindowTitle(f"ADSA - {title}")
            
            self.experiment_changed.emit(experiment_type)
            self._record_recent(experiment_type, title, path=None)
    
    def get_current_experiment_type(self) -> str | None:
        """Get the currently active experiment type."""
        return self._current_experiment_type
    
    def get_current_experiment_window(self) -> QWidget | None:
        """Get the currently active experiment window."""
        if self._current_experiment_type:
            return self._experiment_windows.get(self._current_experiment_type)
        return None

    def _apply_saved_analysis_settings(self, experiment_type: str, window: QWidget):
        """Load persisted analysis settings and apply to window if supported."""
        from menipy.models.config import PreprocessingSettings, EdgeDetectionSettings
        import json

        key = f"analysis/{experiment_type.lower()}"
        raw = self._settings_store.value(key)
        if not raw:
            return
        try:
            data = json.loads(raw)
            pre = (
                PreprocessingSettings(**data.get("preproc", {}))
                if data.get("preproc")
                else PreprocessingSettings()
            )
            edge = (
                EdgeDetectionSettings(**data.get("edge", {}))
                if data.get("edge")
                else EdgeDetectionSettings()
            )
            pipe = data.get("pipeline") or {}
            if hasattr(window, "apply_analysis_settings"):
                window.apply_analysis_settings(pre, edge, pipe)
            else:
                window._preprocessing_settings = pre
                window._edge_settings = edge
                window._pipeline_settings = pipe
        except Exception:
            return

    def _save_analysis_settings(self, experiment_type: str, pre, edge, pipe):
        import json

        payload = {
            "preproc": pre.model_dump(),
            "edge": edge.model_dump(),
            "pipeline": pipe or {},
        }
        key = f"analysis/{experiment_type.lower()}"
        self._settings_store.setValue(key, json.dumps(payload))
    
    # =========================================================================
    # Private Methods
    # =========================================================================
    
    def _create_experiment_window(self, experiment_type: str) -> QWidget | None:
        """
        Create an experiment window for the given type.
        
        Args:
            experiment_type: Experiment type constant.
            
        Returns:
            The created experiment window, or None if type not supported.
        """
        window = None
        
        # Import here to avoid circular imports
        if experiment_type == theme.EXPERIMENT_SESSILE:
            try:
                from menipy.gui.views.sessile_drop_window import SessileDropWindow
                window = SessileDropWindow()
            except ImportError as e:
                print(f"Error importing SessileDropWindow: {e}")
        
        elif experiment_type == theme.EXPERIMENT_PENDANT:
            try:
                from menipy.gui.views.pendant_drop_window import PendantDropWindow
                window = PendantDropWindow()
            except ImportError:
                pass
        
        elif experiment_type == theme.EXPERIMENT_TILTED_SESSILE:
            try:
                from menipy.gui.views.tilted_sessile_window import TiltedSessileWindow
                window = TiltedSessileWindow()
            except ImportError:
                pass
        
        if window:
            window.switch_experiment_requested.connect(self.show_experiment_selector)
            window.notification_requested.connect(self._on_notification_requested)
            self._apply_saved_analysis_settings(experiment_type, window)
            return window
        else:
            return self._create_placeholder_window(experiment_type)
    
    def _create_placeholder_window(self, experiment_type: str) -> QWidget:
        """Create a placeholder window for not-yet-implemented experiment types."""
        from PySide6.QtWidgets import QLabel, QVBoxLayout, QPushButton
        
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        title = self._get_experiment_title(experiment_type)
        label = QLabel(f"{title}\n\n(Coming Soon)")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setStyleSheet(f"""
            font-size: 24px;
            color: {theme.TEXT_SECONDARY};
        """)
        layout.addWidget(label)
        
        back_btn = QPushButton("🏠 Back to Experiment Selector")
        back_btn.clicked.connect(self.show_experiment_selector)
        layout.addWidget(back_btn, alignment=Qt.AlignmentFlag.AlignCenter)
        
        return widget
    
    def _get_experiment_title(self, experiment_type: str) -> str:
        """Get human-readable title for experiment type."""
        from menipy.gui.widgets.experiment_card import EXPERIMENT_DEFINITIONS
        for defn in EXPERIMENT_DEFINITIONS:
            if defn["type"] == experiment_type:
                return defn["title"]
        return experiment_type.replace("_", " ").title()
    
    # =========================================================================
    # Signal Handlers
    # =========================================================================
    
    def _on_experiment_selected(self, experiment_type: str):
        """Handle experiment type selection."""
        self.show_experiment_window(experiment_type)
    
    def _on_project_opened(self, path: str):
        """Handle project file opening."""
        if not path:
            # User clicked "Open Project..." button
            path, _ = QFileDialog.getOpenFileName(
                self,
                "Open ADSA Project",
                "",
                "ADSA Projects (*.adsa);;All Files (*)"
            )
        
        if path:
            self._open_project_file(path)
    
    # -------------------------------------------------------------------------
    # Menu Action Handlers (Placeholders)
    # -------------------------------------------------------------------------
    
    def _on_open_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Image",
            "",
            "Images (*.png *.jpg *.jpeg *.tiff *.bmp);;All Files (*)"
        )
        if path:
            self.statusBar().showMessage(f"Opened: {path}", 3000)
            # TODO: Pass to current experiment window
    
    def _on_open_folder(self):
        path = QFileDialog.getExistingDirectory(self, "Open Folder")
        if path:
            self.statusBar().showMessage(f"Opened folder: {path}", 3000)
    
    def _on_connect_camera(self):
        self.statusBar().showMessage("Camera connection not implemented yet", 3000)
    
    def _on_save_project(self):
        self._save_project_dialog()
    
    def _on_save_project_as(self):
        self._save_project_dialog()
    
    def _on_export_csv(self):
        self.statusBar().showMessage("Export CSV not implemented yet", 3000)
    
    def _on_export_excel(self):
        self.statusBar().showMessage("Export Excel not implemented yet", 3000)
    
    def _on_export_image(self):
        self.statusBar().showMessage("Export Image not implemented yet", 3000)
    
    def _on_generate_report(self):
        self.statusBar().showMessage("Report generation not implemented yet", 3000)
    
    def _on_preferences(self):
        """Open settings dialog."""
        from menipy.gui.dialogs.settings_dialog import SettingsDialog
        dialog = SettingsDialog(self)
        if dialog.exec():
            # Apply settings (placeholder for now)
            settings = dialog.get_settings()
            self._notification_manager.show("Settings saved (Simulated)", "success")
    
    def _on_zoom_in(self):
        window = self.get_current_experiment_window()
        if hasattr(window, "_on_zoom_in"):
            window._on_zoom_in()
    
    def _on_zoom_out(self):
        window = self.get_current_experiment_window()
        if hasattr(window, "_on_zoom_out"):
            window._on_zoom_out()
    
    def _on_fit_view(self):
        window = self.get_current_experiment_window()
        if hasattr(window, "_on_fit_view"):
            window._on_fit_view()
    
    def _on_actual_size(self):
        self.statusBar().showMessage("Actual size view", 1500)
    
    def _on_toggle_fullscreen(self):
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()
    
    def _on_notification_requested(self, message: str, type_: str):
        """Handle notification request from view."""
        self._notification_manager.show(message, type_)

    def _on_calibrate(self):
        """Open calibration wizard."""
        # Forward to current window if it supports it, 
        # otherwise act global (not supported yet globally)
        window = self.get_current_experiment_window()
        if hasattr(window, "_on_calibration_requested"):
            window._on_calibration_requested()
        else:
            self._notification_manager.show("Calibration wizard not supported in this view.", "warning")
    
    def _on_material_database(self):
        """Open material database manager."""
        from menipy.gui.dialogs.material_dialog import MaterialDialog
        dialog = MaterialDialog(self, selection_mode=False)
        dialog.exec()

    def _on_needle_database(self):
        """Open needle database manager."""
        from menipy.gui.dialogs.material_dialog import MaterialDialog
        dialog = MaterialDialog(self, selection_mode=False, table_type="needles")
        dialog.exec()

    def _on_syringe_database(self):
        """Open syringe database manager."""
        from menipy.gui.dialogs.material_dialog import MaterialDialog
        dialog = MaterialDialog(self, selection_mode=False, table_type="syringes")
        dialog.exec()
    
    def _on_run_analysis(self):
        self.statusBar().showMessage("Running analysis...", 1500)
    
    def _on_pause_analysis(self):
        self.statusBar().showMessage("Analysis paused", 1500)
    
    def _on_stop_analysis(self):
        self.statusBar().showMessage("Analysis stopped", 1500)
    
    def _on_analysis_settings(self):
        from menipy.gui.dialogs.analysis_settings_dialog import AnalysisSettingsDialog
        from menipy.models.config import PreprocessingSettings, EdgeDetectionSettings

        window = self.get_current_experiment_window()
        pipeline_name = None
        if hasattr(window, "get_experiment_type"):
            pipeline_name = window.get_experiment_type().lower()
        pipeline_name = pipeline_name or "generic"

        dlg = AnalysisSettingsDialog(
            pipeline_name,
            preprocessing=getattr(window, "_preprocessing_settings", PreprocessingSettings()),
            edge=getattr(window, "_edge_settings", EdgeDetectionSettings()),
            pipeline_settings=getattr(window, "_pipeline_settings", {}),
            parent=self,
        )
        if dlg.exec():
            pre = dlg.preprocessing_settings()
            edge = dlg.edge_settings()
            pipe = dlg.pipeline_settings()
            if hasattr(dlg, "persist"):
                dlg.persist()
            if hasattr(window, "apply_analysis_settings"):
                window.apply_analysis_settings(pre, edge, pipe)
            else:
                window._preprocessing_settings = pre
                window._edge_settings = edge
                window._pipeline_settings = pipe or {}
            self._save_analysis_settings(pipeline_name, pre, edge, pipe)
            self.statusBar().showMessage("Analysis settings saved", 2000)
    
    def _on_batch_process(self):
        self.statusBar().showMessage("Batch processing not implemented yet", 3000)
    
    def _on_process_video(self):
        self.statusBar().showMessage("Video processing not implemented yet", 3000)
    
    def _on_clear_results(self):
        reply = QMessageBox.question(
            self,
            "Clear All Results",
            "Are you sure you want to clear all results?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.statusBar().showMessage("Results cleared", 1500)
    
    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def _on_utilities(self):
        """Open the utilities dialog."""
        from menipy.gui.dialogs.utilities_dialog import UtilitiesDialog
        
        # Get current image from experiment window if available
        image = None
        window = self.get_current_experiment_window()
        if window and hasattr(window, "_current_image_path"):
            import cv2
            path = window._current_image_path
            if path:
                image = cv2.imread(path)
        
        dialog = UtilitiesDialog(image=image, parent=self)
        dialog.exec()
    
    def _on_image_quality_analysis(self):
        """Run image quality analysis on current image."""
        window = self.get_current_experiment_window()
        if not window or not hasattr(window, "_current_image_path") or not window._current_image_path:
            self._notification_manager.show("Load an image first", "warning")
            return
        
        import cv2
        from menipy.common.registry import UTILITIES
        
        image = cv2.imread(window._current_image_path)
        if image is None:
            self._notification_manager.show("Could not load image", "error")
            return
        
        # Run image quality utility if registered
        if "image_quality" in UTILITIES:
            result = UTILITIES["image_quality"](image)
            msg = f"Quality: {result.get('overall_quality', 'N/A')}"
            self._notification_manager.show(msg, "success")
        else:
            self._notification_manager.show("Image quality utility not registered", "warning")
    
    def _on_edge_comparison(self):
        """Open edge detection comparison tool."""
        window = self.get_current_experiment_window()
        if not window or not hasattr(window, "_current_image_path") or not window._current_image_path:
            self._notification_manager.show("Load an image first", "warning")
            return
        
        import cv2
        from menipy.common.registry import UTILITIES
        
        image = cv2.imread(window._current_image_path)
        if image is None:
            self._notification_manager.show("Could not load image", "error")
            return
        
        # Run edge comparison utility if registered
        if "edge_comparison" in UTILITIES:
            result = UTILITIES["edge_comparison"](image)
            msg = f"Best method: {result.get('recommended_method', 'N/A')}"
            self._notification_manager.show(msg, "success")
        else:
            self._notification_manager.show("Edge comparison utility not registered", "warning")

    # ------------------------------------------------------------------
    # Project persistence
    # ------------------------------------------------------------------
    def _save_project_dialog(self):
        """Prompt for file path and save current project as JSON (.adsa.json)."""
        from pathlib import Path
        exp_type = self.get_current_experiment_type()
        if not exp_type:
            self.statusBar().showMessage("No experiment to save.", 2000)
            return
        default_name = f"{exp_type}_project.adsa.json"
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save ADSA Project",
            default_name,
            "ADSA Project (*.adsa.json);;JSON (*.json)",
        )
        if not path:
            return
        if not path.lower().endswith(".json"):
            path += ".json"
        self._save_project(path)
        self._record_recent(exp_type, Path(path).name, path)
        self.statusBar().showMessage(f"Saved project: {path}", 2000)

    def _save_project(self, path: str):
        """Write minimal project JSON including current analysis settings and image path."""
        import json, datetime
        from pathlib import Path

        exp_type = self.get_current_experiment_type() or "unknown"
        window = self.get_current_experiment_window()
        image_path = None
        for attr in ("_current_path", "_current_image_path"):
            if window and hasattr(window, attr):
                image_path = getattr(window, attr)
                if image_path:
                    break

        analysis_raw = self._settings_store.value(f"analysis/{exp_type.lower()}")
        try:
            analysis = json.loads(analysis_raw) if analysis_raw else None
        except Exception:
            analysis = None

        payload = {
            "experiment_type": exp_type,
            "saved_at": datetime.datetime.now().isoformat(),
            "image_path": image_path,
            "analysis_settings": analysis,
        }
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    def _on_help(self):
        from pathlib import Path
        from menipy.gui.dialogs.help_dialog import HelpDialog

        docs_root = Path("docs")
        dlg = HelpDialog(self, docs_dir=docs_root)
        dlg.exec()
    
    def _on_about(self):
        QMessageBox.about(
            self,
            "About ADSA",
            "<h2>ADSA</h2>"
            "<p>Automated Drop Shape Analysis</p>"
            "<p>Version 1.0.0</p>"
            "<p>A comprehensive tool for surface tension and contact angle measurements.</p>"
        )
    
    def closeEvent(self, event: QCloseEvent):
        """Handle window close."""
        # TODO: Check for unsaved changes
        event.accept()
