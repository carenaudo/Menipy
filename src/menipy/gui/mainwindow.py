# src/menipy/gui/mainwindow.py
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from PySide6.QtCore import QFile, QByteArray, Qt
from PySide6.QtGui import QAction, QCloseEvent
from PySide6.QtUiTools import QUiLoader
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QFileDialog, QMessageBox,
    QLayout, QPlainTextEdit
)

from menipy.gui.views.image_view import DRAW_NONE

# --- compiled split main window UI ---
from menipy.gui.views.ui_main_window import Ui_MainWindow

from menipy.gui.logging_bridge import install_qt_logging
from menipy.gui.plugins_panel import PluginsController
from menipy.gui.panels.setup_panel import SetupPanelController
from menipy.gui.panels.preview_panel import PreviewPanel
from menipy.gui.panels.results_panel import ResultsPanel
from menipy.gui.pipeline_controller import PipelineController

# --- promoted preview widget (registered into QUiLoader) ---
try:
    from .views.image_view import ImageView
except Exception:  # keep app booting even if file missing during refactors
    ImageView = None  # type: ignore

# --- optional step row & SOP service (guarded) ---
try:
    from .views.step_item_widget import StepItemWidget
except Exception:
    StepItemWidget = None  # type: ignore

try:
    from .services.sop_service import SopService, Sop
except Exception:
    SopService = None  # type: ignore
    Sop = None  # type: ignore

# --- app settings (you already have this in your project) ---
try:
    from .services.settings_service import AppSettings
except Exception:
    # tiny fallback so file still runs
    class AppSettings:  # type: ignore
        selected_pipeline: Optional[str] = None
        last_image_path: Optional[str] = None
        plugin_dirs: list[str] = []
        main_window_state_b64: Optional[str] = None
        main_window_geom_b64: Optional[str] = None
        splitter_sizes: Optional[list[int]] = None

        @classmethod
        def load(cls): return cls()
        def save(self): pass

# --- pipelines map (guarded) ---
try:
    from menipy.pipelines.discover import PIPELINE_MAP
except Exception:
    PIPELINE_MAP = {}

# --- optional runner & viewmodel (guarded) ---
try:
    from menipy.gui.viewmodels.run_vm import RunViewModel
    from menipy.gui.services.pipeline_runner import PipelineRunner
except Exception:
    RunViewModel = None  # type: ignore
    PipelineRunner = None  # type: ignore

# default stage order for SOPs / step list
STAGE_ORDER: List[str] = [
    "acquisition", "preprocessing", "edge_detection", "geometry", "scaling",
    "physics", "solver", "optimization", "outputs", "overlay", "validation"
]


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        # ---------- build split main window ----------
        self.setupUi(self)

        # settings
        self.settings = AppSettings.load()

        # restore geometry/state if present (pre-split or split—it’s fine)
        self._restore_window_layout()

        # ---------- loader that knows our custom widgets ----------
        loader = QUiLoader()
        if ImageView:
            loader.registerCustomWidget(ImageView)

        def load_ui(res_path: str, fallback_filename: str) -> QWidget:
            f = QFile(res_path)
            if not f.exists():
                f = QFile(str(Path(__file__).resolve().parent / "views" / fallback_filename))
            f.open(QFile.ReadOnly)
            w = loader.load(f, self)
            f.close()
            return w

        # ---------- load panels into split hosts ----------
        self.setup_panel: QWidget = load_ui(":/views/setup_panel.ui", "setup_panel.ui")
        self.overlay_panel: QWidget = load_ui(":/views/overlay_panel.ui", "overlay_panel.ui")
        self.results_panel: QWidget = load_ui(":/views/results_panel.ui", "results_panel.ui")

        # the split UI provides these host layouts/widgets
        self._embed(self.setup_panel, self.setupHostLayout)
        self._embed(self.overlay_panel, self.previewHostLayout)
        self._embed(self.results_panel, self.resultsHostLayout)

        self.preview_panel = PreviewPanel(self, self.overlay_panel, ImageView)
        self.results_panel_ctrl = ResultsPanel(self.results_panel)

        # add simple log view into the Log tab
        self.logView = QPlainTextEdit(self)
        self.logView.setReadOnly(True)
        self._embed(self.logView, self.logHostLayout)

        # Install Qt logging bridge (only one handler) and connect to logView
        try:
            self._qt_log_bridge, self._qt_log_handler = install_qt_logging(self.logView)
        except Exception:
            pass

        # ---------- plugin dock (optional) ----------
        self.plugins_controller = PluginsController(self, self.settings)

        # ---------- services / VMs ----------
        if PipelineRunner and RunViewModel:
            self.runner = PipelineRunner()
            self.run_vm = RunViewModel(self.runner)
            # signals            # new: status and logs from pipeline Context
            try:
                self.run_vm.status_ready.connect(lambda msg: self.statusBar().showMessage(msg, 5000))
            except Exception:
                pass
        else:
            self.runner = None
            self.run_vm = None

        # SOP service
        self.sops = SopService() if SopService else None

        self.setup_panel_ctrl = SetupPanelController(
            self,
            self.setup_panel,
            self.settings,
            self.sops,
            STAGE_ORDER,
            StepItemWidget,
            list(PIPELINE_MAP.keys()) if PIPELINE_MAP else [],
        )

        self.pipeline_ctrl = PipelineController(
            window=self,
            setup_ctrl=self.setup_panel_ctrl,
            preview_panel=self.preview_panel,
            results_panel=self.results_panel_ctrl,
            pipeline_map=PIPELINE_MAP,
            sops=self.sops,
            run_vm=self.run_vm,
            log_view=self.logView,
        )

        self.setup_panel_ctrl.browse_requested.connect(self._browse_image)
        self.setup_panel_ctrl.preview_requested.connect(self._on_preview_image)
        self.setup_panel_ctrl.draw_mode_requested.connect(self.preview_panel.set_draw_mode)
        self.setup_panel_ctrl.clear_overlays_requested.connect(self.preview_panel.clear_overlays)
        self.setup_panel_ctrl.run_all_requested.connect(self.pipeline_ctrl.run_all)
        self.setup_panel_ctrl.play_stage_requested.connect(self.pipeline_ctrl.run_stage)
        self.setup_panel_ctrl.config_stage_requested.connect(self._on_config_step)

        if self.run_vm:
            self.run_vm.preview_ready.connect(self.pipeline_ctrl.on_preview_ready)
            self.run_vm.results_ready.connect(self.pipeline_ctrl.on_results_ready)
            self.run_vm.error_occurred.connect(self.pipeline_ctrl.on_pipeline_error)
            try:
                self.run_vm.logs_ready.connect(self.pipeline_ctrl.append_logs)
            except Exception:
                pass

        # ---------- wire panels ----------
        if self.preview_panel.has_view():
            try:
                self.preview_panel.set_draw_mode(DRAW_NONE)
            except Exception:
                pass
        # menubar actions from split UI
        self._wire_menu_actions()

        # restore saved splitter sizes (optional)
        if getattr(self.settings, "splitter_sizes", None):
            try:
                self.rootSplitter.setSizes(self.settings.splitter_sizes)  # type: ignore[attr-defined]
            except Exception:
                pass

        # focus
        self.statusBar().showMessage("Ready", 1500)

    # -------------------------- helpers & wiring --------------------------

    def _embed(self, child: QWidget, host_or_layout):
        """Add child into a QWidget host or directly into a QLayout."""
        if isinstance(host_or_layout, QLayout):
            host_or_layout.addWidget(child)
            return
        lay = host_or_layout.layout() or QVBoxLayout(host_or_layout)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(child)

    def _wire_menu_actions(self):
        # Actions declared in main_window_split.ui
        try:
            self.actionOpenImage.triggered.connect(self._browse_image)       # type: ignore[attr-defined]
            self.actionOpenCamera.triggered.connect(lambda: self._select_camera(True))  # type: ignore[attr-defined]
            self.actionQuit.triggered.connect(self.close)                    # type: ignore[attr-defined]
            self.actionRunFull.triggered.connect(self.pipeline_ctrl.run_full)       # type: ignore[attr-defined]
            # Reuse Run Full for now; swap to subset if you add it
            self.actionRunSelected.triggered.connect(self.pipeline_ctrl.run_full)   # type: ignore[attr-defined]
            self.actionStop.triggered.connect(lambda: self.statusBar().showMessage("Stop requested", 1000))  # type: ignore[attr-defined]
            self.actionAbout.triggered.connect(lambda: QMessageBox.information(self, "About", "Menipy ADSA GUI"))  # type: ignore[attr-defined]
        except Exception:
            pass

    def _on_browse_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Choose image", "", "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)")


    def _on_preview_image(self):
        path = self.setup_panel_ctrl.image_path()
        if not path:
            QMessageBox.information(self, "Preview", "Please select an image first.")
            return
        try:
            self.preview_panel.load_path(path)
        except Exception as e:
            QMessageBox.warning(self, "Preview error", f"Could not load image:\n{e}")

    # -------------------------- menu actions --------------------------

    def _browse_image(self):
        initial = self.setup_panel_ctrl.image_path() or self.settings.last_image_path or ""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Image",
            initial,
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)",
        )
        if path:
            self.setup_panel_ctrl.set_image_path(path)
            self.settings.last_image_path = path
            self.settings.save()

    def _select_camera(self, on: bool):
        self.setup_panel_ctrl.set_camera_enabled(on)

    # -------------------------- pipeline ops --------------------------

    def _on_config_step(self, stage_name: str):
        QMessageBox.information(self, "Configure Step", f"Open configuration for: {stage_name}")

    # -------------------------- VM callbacks / preview & results --------------------------

    def _restore_window_layout(self):
        s = self.settings
        try:
            if getattr(s, "main_window_geom_b64", None):
                self.restoreGeometry(QByteArray.fromBase64(s.main_window_geom_b64.encode("ascii")))
            if getattr(s, "main_window_state_b64", None):
                self.restoreState(QByteArray.fromBase64(s.main_window_state_b64.encode("ascii")))
        except Exception:
            pass

    def closeEvent(self, event: QCloseEvent) -> None:
        # save dock/splitter/geom + last selections
        try:
            self.settings.main_window_state_b64 = bytes(self.saveState().toBase64()).decode("ascii")
            self.settings.main_window_geom_b64 = bytes(self.saveGeometry().toBase64()).decode("ascii")
        except Exception:
            pass
        try:
            self.settings.splitter_sizes = getattr(self.rootSplitter, "sizes", lambda: None)()  # type: ignore[attr-defined]
        except Exception:
            pass
        # persist selections
        pipeline = self.setup_panel_ctrl.current_pipeline_name()
        if pipeline:
            self.settings.selected_pipeline = pipeline
        image_path = self.setup_panel_ctrl.image_path()
        if image_path is not None:
            self.settings.last_image_path = image_path or None
        try:
            self.settings.save()
        except Exception:
            pass
        super().closeEvent(event)















