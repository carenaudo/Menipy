# src/menipy/gui/mainwindow.py
from __future__ import annotations

from pathlib import Path
from typing import Sequence, List, Optional

import logging
from PySide6.QtCore import QFile, QByteArray, Qt, QObject, Signal
from PySide6.QtGui import QAction, QCloseEvent, QImage
from PySide6.QtUiTools import QUiLoader
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QFileDialog, QMessageBox, QInputDialog,
    QMenuBar, QLayout, QPlainTextEdit, QMenu, QListWidget, QListWidgetItem,
    QComboBox, QToolButton, QLineEdit, QCheckBox, QSpinBox, QPushButton, QLabel, QTableWidget, QTableWidgetItem
)

from menipy.gui.views.image_view import DRAW_NONE, DRAW_POINT, DRAW_LINE, DRAW_RECT

# --- compiled split main window UI ---
from menipy.gui.views.ui_main_window import Ui_MainWindow


# Small Qt bridge and logging.Handler to stream Python logs into the Log tab.
class QtLogBridge(QObject):
    log = Signal(str)


class QtLogHandler(logging.Handler):
    """Logging handler that emits records via a Qt signal (thread-safe queued emits)."""
    def __init__(self, bridge: QtLogBridge):
        super().__init__()
        self.bridge = bridge

    def emit(self, record: logging.LogRecord) -> None:  # type: ignore[override]
        try:
            msg = self.format(record)
            # Emit via Qt signal; if called from a non-GUI thread this becomes a queued signal
            self.bridge.log.emit(msg)
        except Exception:
            try:
                # fallback to console if signal fails
                print(self.format(record))
            except Exception:
                pass

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

# --- plugins infra (guarded) ---
try:
    from menipy.common.plugins import PluginDB, discover_into_db, load_active_plugins
except Exception:
    PluginDB = None  # type: ignore
    def discover_into_db(*a, **k): pass  # type: ignore
    def load_active_plugins(*a, **k): pass  # type: ignore

# --- pipelines map (guarded) ---
PIPELINE_MAP = {}
try:
    from menipy.pipelines import (
        SessilePipeline, PendantPipeline, OscillatingPipeline,
        CapillaryRisePipeline, CaptiveBubblePipeline,
    )
    PIPELINE_MAP = {
        "sessile": SessilePipeline,
        "pendant": PendantPipeline,
        "oscillating": OscillatingPipeline,
        "capillary_rise": CapillaryRisePipeline,
        "captive_bubble": CaptiveBubblePipeline,
    }
except Exception:
    pass

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

        # add simple log view into the Log tab
        self.logView = QPlainTextEdit(self)
        self.logView.setReadOnly(True)
        self._embed(self.logView, self.logHostLayout)

        # Install Qt logging bridge (only one handler) and connect to logView
        try:
            self._qt_log_bridge = QtLogBridge()
            self._qt_log_bridge.log.connect(lambda m: self.logView.appendPlainText(str(m)))
            root_logger = logging.getLogger()
            # avoid adding multiple handlers in repeated constructions
            if not any(isinstance(h, QtLogHandler) for h in root_logger.handlers):
                h = QtLogHandler(self._qt_log_bridge)
                h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
                root_logger.addHandler(h)
        except Exception:
            pass

        # ---------- plugin dock (optional) ----------
        self._build_plugin_dock()

        # View ▸ Plugins toggle (after dock exists)
        self._add_plugins_toggle_to_view_menu()

        # ---------- services / VMs ----------
        if PipelineRunner and RunViewModel:
            self.runner = PipelineRunner()
            self.run_vm = RunViewModel(self.runner)
            # signals
            self.run_vm.preview_ready.connect(self._on_preview_ready)
            self.run_vm.results_ready.connect(self._on_results_ready)
            self.run_vm.error_occurred.connect(self._on_pipeline_error)
            # new: status and logs from pipeline Context
            try:
                self.run_vm.status_ready.connect(lambda msg: self.statusBar().showMessage(msg, 5000))
                self.run_vm.logs_ready.connect(self._append_logs)
            except Exception:
                pass
        else:
            self.runner = None
            self.run_vm = None

        # SOP service
        self.sops = SopService() if SopService else None

        # ---------- wire panels ----------
        self._wire_setup_panel()
        self._wire_overlay_panel()
        self._wire_results_panel()
        # Start with no drawing on the preview view (if present)
        if getattr(self, "imageView", None) and hasattr(self.imageView, "set_draw_mode"):
            try:
                self.imageView.set_draw_mode(DRAW_NONE)
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

    def _build_plugin_dock(self):
        # Optional plugin system (graceful no-op if not available)
        self.plugin_dock = None
        if PluginDB:
            try:
                from PySide6.QtWidgets import QDockWidget
                self.db = PluginDB()
                self.db.init_schema()
                # discover configured dirs (settings.plugin_dirs is a list[str])
                self._plugins_discover(self.settings.plugin_dirs or [])
                load_active_plugins(self.db)

                self.plugin_dock = QDockWidget("Plugins", self)
                self.plugin_dock.setObjectName("pluginDock")
                self.plugin_dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
                self.plugin_dock.setFeatures(
                    QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable | QDockWidget.DockWidgetClosable
                )
                # lightweight placeholder; you probably already have a plugin panel
                info = QLabel("Plugins loaded.", self.plugin_dock)
                info.setMargin(8)
                self.plugin_dock.setWidget(info)
                self.addDockWidget(Qt.RightDockWidgetArea, self.plugin_dock)
                self.plugin_dock.hide()  # hidden by default (toggle via View menu)
            except Exception as e:
                print("[plugins] disabled:", e)

    def _add_plugins_toggle_to_view_menu(self):
        menu_view: Optional[QMenu] = self.findChild(QMenu, "menuView")
        if not menu_view or not self.plugin_dock:
            return
        self.actionPlugins = QAction("Plugins", self)
        self.actionPlugins.setCheckable(True)
        self.actionPlugins.setChecked(self.plugin_dock.isVisible())
        menu_view.addAction(self.actionPlugins)
        self.actionPlugins.toggled.connect(self.plugin_dock.setVisible)
        self.plugin_dock.visibilityChanged.connect(self.actionPlugins.setChecked)

    def _wire_menu_actions(self):
        # Actions declared in main_window_split.ui
        try:
            self.actionOpenImage.triggered.connect(self._browse_image)       # type: ignore[attr-defined]
            self.actionOpenCamera.triggered.connect(lambda: self._select_camera(True))  # type: ignore[attr-defined]
            self.actionQuit.triggered.connect(self.close)                    # type: ignore[attr-defined]
            self.actionRunFull.triggered.connect(self._on_run_clicked)       # type: ignore[attr-defined]
            # Reuse Run Full for now; swap to subset if you add it
            self.actionRunSelected.triggered.connect(self._on_run_clicked)   # type: ignore[attr-defined]
            self.actionStop.triggered.connect(lambda: self.statusBar().showMessage("Stop requested", 1000))  # type: ignore[attr-defined]
            self.actionAbout.triggered.connect(lambda: QMessageBox.information(self, "About", "Menipy ADSA GUI"))  # type: ignore[attr-defined]
        except Exception:
            pass

    def _wire_setup_panel(self):
        # Grab widgets from setup_panel
        self.testCombo: Optional[QComboBox] = self.setup_panel.findChild(QComboBox, "testCombo")
        self.sopCombo: Optional[QComboBox] = self.setup_panel.findChild(QComboBox, "sopCombo")
        self.addSopBtn: Optional[QToolButton] = self.setup_panel.findChild(QToolButton, "addSopBtn")

        self.imagePathEdit: Optional[QLineEdit] = self.setup_panel.findChild(QLineEdit, "imagePathEdit")
        self.browseBtn: Optional[QToolButton] = self.setup_panel.findChild(QToolButton, "browseBtn")

        # NEW: bind preview & calibration controls from the left panel
        self.previewBtn: Optional[QToolButton] = self.setup_panel.findChild(QToolButton, "previewBtn")
        #from PySide6.QtWidgets import QPushButton  # local import to avoid top clutter
        self.drawPointBtn: Optional[QPushButton] = self.setup_panel.findChild(QPushButton, "drawPointBtn")
        self.drawLineBtn: Optional[QPushButton]  = self.setup_panel.findChild(QPushButton, "drawLineBtn")
        self.drawRectBtn: Optional[QPushButton]  = self.setup_panel.findChild(QPushButton, "drawRectBtn")
        self.clearOverlayBtn: Optional[QPushButton] = self.setup_panel.findChild(QPushButton, "clearOverlayBtn")


        self.cameraCheck: Optional[QCheckBox] = self.setup_panel.findChild(QCheckBox, "cameraCheck")
        self.cameraIdSpin: Optional[QSpinBox] = self.setup_panel.findChild(QSpinBox, "cameraIdSpin")
        self.framesSpin: Optional[QSpinBox] = self.setup_panel.findChild(QSpinBox, "framesSpin")

        self.stepsList: Optional[QListWidget] = self.setup_panel.findChild(QListWidget, "stepsList")
        self.runAllBtn: Optional[QPushButton] = self.setup_panel.findChild(QPushButton, "runAllBtn")

        # Restore saved selections
        if self.imagePathEdit and self.settings.last_image_path:
            self.imagePathEdit.setText(self.settings.last_image_path)
        if self.testCombo and self.settings.selected_pipeline:
            i = self.testCombo.findText(self.settings.selected_pipeline)
            if i != -1:
                self.testCombo.setCurrentIndex(i)

        # Populate steps UI rows (if widget class available)
        self._populate_steps_list()

        # Pipeline change -> rebuild SOP combo and apply default
        if self.testCombo:
            self.testCombo.currentTextChanged.connect(self._on_pipeline_changed)
            self._on_pipeline_changed(self.testCombo.currentText())

        # SOP add (simple “clone current UI selection”)
        if self.addSopBtn and self.sops:
            self.addSopBtn.clicked.connect(self._on_add_sop)

        # Browse image
        if self.browseBtn:
            self.browseBtn.clicked.connect(self._browse_image)
        
        # Preview button: loads the image in the right preview area
        if hasattr(self, "previewBtn") and hasattr(self, "imagePathEdit"):
            self.previewBtn.clicked.connect(self._on_preview_image)

        # Calibration buttons (left panel group)
        # Expecting: drawPointBtn, drawLineBtn, drawRectBtn, clearOverlayBtn
        if hasattr(self, "drawPointBtn"):
            self.drawPointBtn.clicked.connect(lambda: self._set_draw_mode(DRAW_POINT))
        if hasattr(self, "drawLineBtn"):
            self.drawLineBtn.clicked.connect(lambda: self._set_draw_mode(DRAW_LINE))
        if hasattr(self, "drawRectBtn"):
            self.drawRectBtn.clicked.connect(lambda: self._set_draw_mode(DRAW_RECT))
        if hasattr(self, "clearOverlayBtn"):
            self.clearOverlayBtn.clicked.connect(self._clear_overlays)
        
        # Run All
        if self.runAllBtn:
            self.runAllBtn.clicked.connect(self._on_run_all_sop)

    def _wire_overlay_panel(self):
        # promoted ImageView and zoom buttons
        view = self.overlay_panel.findChild(ImageView, "previewView") if ImageView else None
        self.imageView = view
        if self.imageView:
            try:
                self.imageView.set_auto_policy("preserve")
                self.imageView.set_wheel_zoom_requires_ctrl(False)
            except Exception:
                pass

        # Try both toolbutton/pushbutton
        def _btn(name: str):
            b = self.overlay_panel.findChild(QToolButton, name)
            return b or self.overlay_panel.findChild(QPushButton, name)

        z_in = _btn("zoomInBtn")
        z_out = _btn("zoomOutBtn")
        z_100 = _btn("actualBtn")
        z_fit = _btn("fitBtn")
        if self.imageView:
            if z_in:  z_in.clicked.connect(self.imageView.zoom_in)
            if z_out: z_out.clicked.connect(self.imageView.zoom_out)
            if z_100: z_100.clicked.connect(self.imageView.actual_size)
            if z_fit: z_fit.clicked.connect(self.imageView.fit_to_window)

    def _on_browse_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Choose image", "", "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)")
        if path and hasattr(self, "imagePathEdit"):
            self.imagePathEdit.setText(path)

    def _on_preview_image(self):
        if not hasattr(self, "imagePathEdit") or not hasattr(self, "previewImageView"):
            return
        path = self.imagePathEdit.text().strip()
        if not path:
            QMessageBox.information(self, "Preview", "Please select an image first.")
            return
        try:
            # Assuming your ImageView exposes setImage(path) or similar.
            # If not, replace with your existing method to load a pixmap.
            if hasattr(self.previewImageView, "setImage"):
                self.previewImageView.setImage(path)
            elif hasattr(self.previewImageView, "load"):
                self.previewImageView.load(path)
            else:
                # Fallback: you may have a controller method to push images into the view
                raise RuntimeError("Preview ImageView has no loader method")
        except Exception as e:
            QMessageBox.warning(self, "Preview error", f"Could not load image:\n{e}")

    def _set_draw_mode(self, mode):
        if hasattr(self, "previewImageView") and hasattr(self.previewImageView, "set_draw_mode"):
            self.previewImageView.set_draw_mode(mode)

    def _clear_overlays(self):
        if hasattr(self, "previewImageView") and hasattr(self.previewImageView, "clear_overlays"):
            self.previewImageView.clear_overlays()

    def _wire_results_panel(self):
        # Set up simple placeholders; adjust to your UI
        self.resultsTable: Optional[QTableWidget] = self.results_panel.findChild(QTableWidget, "resultsTable")
        if self.resultsTable:
            self.resultsTable.setColumnCount(2)
            self.resultsTable.setHorizontalHeaderLabels(["Parameter", "Value"])

    # -------------------------- menu actions --------------------------

    def _browse_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Image", self.settings.last_image_path or "", "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)")
        if path and self.imagePathEdit:
            self.imagePathEdit.setText(path)
            self.settings.last_image_path = path
            self.settings.save()

    def _select_camera(self, on: bool):
        if self.cameraCheck:
            self.cameraCheck.setChecked(bool(on))

    # -------------------------- pipeline ops --------------------------

    def _current_pipeline_name(self) -> Optional[str]:
        cb = getattr(self, "testCombo", None)
        if cb and hasattr(cb, "currentText"):
            txt = cb.currentText().strip()
            return txt or None
        # legacy fallback
        cb2 = getattr(self, "pipelineCombo", None)
        if cb2 and hasattr(cb2, "currentText"):
            txt = cb2.currentText().strip()
            return txt or None
        return None

    def _gather_run_params(self):
        name = self._current_pipeline_name() or "sessile"
        use_camera = bool(self.cameraCheck.isChecked()) if self.cameraCheck else False
        frames = int(self.framesSpin.value()) if self.framesSpin else 1
        image = None if use_camera else (self.imagePathEdit.text().strip() if self.imagePathEdit else None)
        cam_id = int(self.cameraIdSpin.value()) if (self.cameraIdSpin and use_camera) else None
        return {"name": name, "use_camera": use_camera, "frames": frames, "image": image, "cam_id": cam_id}


    def _on_run_clicked(self):
        p = self._gather_run_params()
        name, image, cam_id, frames = p["name"], p["image"], p["cam_id"], p["frames"]

        pcls = PIPELINE_MAP.get(name)
        if not pcls:
            QMessageBox.warning(self, "Run", f"Unknown pipeline: {name}")
            return

        self.statusBar().showMessage(f"Running {name}…")

        if self.run_vm:
            try:
                # call with keyword 'pipeline' matching RunViewModel signature
                self.run_vm.run(pipeline=name, image=image, camera=cam_id, frames=frames)
                return
            except Exception as e:
                print("[run_vm] fallback to direct run:", e)

        # pass params to the pipeline
        try:
            pipe = pcls()
            ctx = pipe.run(image=image, camera=cam_id, frames=frames)
            if self.imageView and getattr(ctx, "preview", None) is not None:
                self.imageView.set_image(ctx.preview)
            if self.resultsTable and getattr(ctx, "results", None):
                self._fill_results_table(ctx.results)
            self.statusBar().showMessage("Done", 1500)
        except Exception as e:
            self._on_pipeline_error(str(e))

    # “Run All” honoring SOP subset (if SOP service available; else same as Run Full)
    def _on_run_all_sop(self):
        # If no SOP service, just run full
        if not self.sops:
            return self._on_run_clicked()

        # Gather run parameters once
        p = self._gather_run_params()
        name, image, cam_id, frames = p["name"], p["image"], p["cam_id"], p["frames"]

        # Figure out which stages are enabled by the current SOP UI
        stages = self._collect_included_stages_from_ui()
        if not stages:
            QMessageBox.warning(self, "Run All", "No stages enabled in the current SOP.")
            return

        # Prefer ViewModel subset run (it knows how to set up acquisition/context)
        if self.run_vm and hasattr(self.run_vm, "run_subset"):
            try:
                self.statusBar().showMessage(f"Running {name} (SOP) …")
                self.run_vm.run_subset(name, only=stages, image=image, camera=cam_id, frames=frames)
                return
            except Exception as e:
                print("[run_vm subset] falling back to full run:", e)

        # Fallback: use the existing full-run path (ensures source is set properly)
        # This means no per-stage subset when VM subset isn't available, but it won't crash.
        self._on_run_clicked()


    # -------------------------- steps / SOP UI --------------------------

    def _populate_steps_list(self):
        self._step_widgets: List[StepItemWidget] = []
        if not self.stepsList:
            return
        self.stepsList.clear()
        if not StepItemWidget:
            # plain text fallback
            for s in STAGE_ORDER:
                self.stepsList.addItem(s)
            return

        for stage in STAGE_ORDER:
            w = StepItemWidget(stage, self.stepsList)
            it = QListWidgetItem(self.stepsList)
            it.setSizeHint(w.sizeHint())
            self.stepsList.addItem(it)
            self.stepsList.setItemWidget(it, w)
            try:
                w.set_status("pending")
                w.playClicked.connect(self._on_play_step)
                w.configClicked.connect(self._on_config_step)
            except Exception:
                pass
            self._step_widgets.append(w)

    def _on_pipeline_changed(self, _name: str):
        # Ensure default SOP exists and refresh SOP list
        if not self.sops:
            return
        pname = self._current_pipeline_name() or "sessile"
        try:
            self.sops.ensure_default(pname, STAGE_ORDER)
            self._refresh_sop_combo()
            self._apply_selected_sop()
        except Exception as e:
            print("[SOP] pipeline change:", e)



    def _refresh_sop_combo(self, select: Optional[str] = None):
        if not (self.sops and self.sopCombo):
            return
        self.sopCombo.blockSignals(True)
        self.sopCombo.clear()
        DEFAULT_KEY = self.sops.default_name()
        self.sopCombo.addItem("Default (pipeline)", userData=DEFAULT_KEY)
        try:
            pname = self._current_pipeline_name() or "sessile"
            customs = [n for n in self.sops.list(pname) if n != DEFAULT_KEY]
            for name in customs:
                self.sopCombo.addItem(name, userData=name)
        except Exception:
            pass
        # select specific or default
        idx = 0
        if select:
            for i in range(self.sopCombo.count()):
                if self.sopCombo.itemData(i) == select:
                    idx = i; break
        self.sopCombo.setCurrentIndex(idx)
        self.sopCombo.blockSignals(False)
        # apply on change
        self.sopCombo.currentTextChanged.connect(lambda _t: self._apply_selected_sop())

    def _on_add_sop(self):
        """Create a new SOP from the currently enabled steps."""
        # Make sure SOP service exists
        if not getattr(self, "sops", None):
            QMessageBox.warning(self, "SOP", "SOP service is not available.")
            return

        # Ask for a name
        name, ok = QInputDialog.getText(self, "Add SOP", "SOP name:")
        if not ok or not name.strip():
            return
        name = name.strip()

        # Collect included stages from UI
        include = self._collect_included_stages_from_ui()

        # Build and save SOP
        try:
            # If Sop dataclass is available, use it; otherwise store plain dict
            if "Sop" in globals() and Sop is not None:
                sop_obj = Sop(name=name, include_stages=include, params={})
                self.sops.upsert(self._current_pipeline_name() or "sessile", sop_obj)
            else:
                # Fallback: emulate Sop object shape
                self.sops.upsert(self._current_pipeline_name() or "sessile",  # type: ignore[attr-defined]
                                type("SopLike", (), {"name": name, "include_stages": include, "params": {}}))
        except Exception as e:
            QMessageBox.critical(self, "SOP", f"Could not save SOP:\n{e}")
            return

        # Refresh combo and apply it
        self._refresh_sop_combo(select=name)
        self._apply_selected_sop()
        self.statusBar().showMessage(f"SOP '{name}' added", 1500)


    def _selected_sop_key(self) -> str:
        if not self.sops or not self.sopCombo:
            return "__default__"
        return self.sopCombo.currentData() or self.sops.default_name()

    def _apply_selected_sop(self):
        if not (self.sops and self._step_widgets):
            return
        key = self._selected_sop_key()
        pname = self._current_pipeline_name() or "sessile"
        try:
            sop = self.sops.get(pname, key)
            include = set(sop.include_stages if sop else STAGE_ORDER)
        except Exception:
            include = set(STAGE_ORDER)
        for w in self._step_widgets:
            enabled = getattr(w, "step_name", None) in include
            try:
                w.setEnabled(enabled)
                if not enabled:
                    w.set_status("pending")
            except Exception:
                pass

    def _collect_included_stages_from_ui(self) -> List[str]:
        if not self._step_widgets:
            return STAGE_ORDER[:]
        return [w.step_name for w in self._step_widgets if w.isEnabled()]

    def _on_play_step(self, stage_name: str):
        # Prefer VM subset
        if self.run_vm and hasattr(self.run_vm, "run_subset"):
            p = self._gather_run_params()
            try:
                self.run_vm.run_subset(p["name"], only=[stage_name],
                                    image=p["image"], camera=p["cam_id"], frames=p["frames"])
                return
            except Exception as e:
                print("[run_vm single step] falling back to pipeline:", e)

        # direct pipeline with prereqs AND params
        params = self._gather_run_params()
        name = params["name"]
        pcls = PIPELINE_MAP.get(name)
        if not pcls:
            return
        pipe = pcls()
        pipe.run_with_plan(only=[stage_name], include_prereqs=True,
                        image=params["image"], camera=params["cam_id"], frames=params["frames"])



    def _on_config_step(self, stage_name: str):
        QMessageBox.information(self, "Configure Step", f"Open configuration for: {stage_name}")

    # -------------------------- VM callbacks / preview & results --------------------------

    def _on_preview_ready(self, pix_or_img_or_np):
        if getattr(self, "imageView", None):
            try:
                self.imageView.set_image(pix_or_img_or_np)
            except Exception:
                pass
        self.statusBar().showMessage("Preview updated", 1000)

    def _on_results_ready(self, results: dict):
        if self.resultsTable and results:
            self._fill_results_table(results)
        self.statusBar().showMessage("Results ready", 1000)

    def _append_logs(self, lines):
        try:
            if not lines:
                return
            if isinstance(lines, (list, tuple)):
                for ln in lines:
                    self.logView.appendPlainText(str(ln))
            else:
                # single string
                self.logView.appendPlainText(str(lines))
        except Exception:
            pass

    def _fill_results_table(self, results: dict):
        if not self.resultsTable:
            return
        rows = list(results.items())
        self.resultsTable.setRowCount(len(rows))
        for i, (k, v) in enumerate(rows):
            self.resultsTable.setItem(i, 0, QTableWidgetItem(str(k)))
            self.resultsTable.setItem(i, 1, QTableWidgetItem(str(v)))
        self.resultsTable.resizeColumnsToContents()

    def _on_pipeline_error(self, msg: str):
        QMessageBox.critical(self, "Pipeline Error", msg)
        self.statusBar().showMessage("Error", 1500)

    # -------------------------- plugins discovery --------------------------

    def _plugins_discover(self, dirs: Sequence[str | Path]) -> None:
        try:
            if PluginDB:
                discover_into_db(self.db, [Path(d) for d in dirs])  # type: ignore[attr-defined]
        except Exception as e:
            print("[plugins] discover error:", e)

    # -------------------------- window state --------------------------

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
        self.settings.selected_pipeline = self._current_pipeline_name() or self.settings.selected_pipeline
        if getattr(self, "imagePathEdit", None) and hasattr(self.imagePathEdit, "text"):
            txt = self.imagePathEdit.text().strip()
            self.settings.last_image_path = txt or None
        try:
            self.settings.save()
        except Exception:
            pass
        super().closeEvent(event)
