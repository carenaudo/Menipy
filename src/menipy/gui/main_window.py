"""
Main window class for Menipy GUI.
"""

# src/menipy/gui/mainwindow.py
# type: ignore
from __future__ import annotations

from pathlib import Path
import logging
from typing import List, Optional

from PySide6.QtCore import QFile, QByteArray, Qt
from PySide6.QtGui import QCloseEvent
from PySide6.QtUiTools import QUiLoader
from PySide6.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QLayout, QPlainTextEdit

from menipy.gui.views.image_view import DRAW_NONE

from menipy.gui.views.ui_main_window import Ui_MainWindow

from menipy.gui.logging_bridge import install_qt_logging
from menipy.gui.controllers.plugins_controller import PluginsController
from menipy.gui.controllers.setup_panel_controller import SetupPanelController
from menipy.gui.panels.preview_panel import PreviewPanel
from menipy.gui.panels.results_panel import ResultsPanel
from menipy.gui.services.camera_service import CameraController, CameraConfig
from menipy.gui.controllers.pipeline_controller import PipelineController
from menipy.gui.controllers.preprocessing_controller import (
    PreprocessingPipelineController,
)
from menipy.gui.controllers.edge_detection_controller import (
    EdgeDetectionPipelineController,
)
from menipy.gui.helpers.image_marking import ImageMarkerHelper

from menipy.pipelines.discover import PIPELINE_MAP

logger = logging.getLogger(__name__)

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
    from .services.sop_service import SopService
    from .services.settings_service import AppSettings
    from menipy.gui.viewmodels.run_vm import RunViewModel
    from menipy.gui.services.pipeline_runner import PipelineRunner
except Exception:
    SopService = None  # type: ignore

    # tiny fallback so file still runs
    class AppSettings:  # type: ignore
        selected_pipeline: Optional[str] = None
        last_image_path: Optional[str] = None
        plugin_dirs: list[str] = []
        main_window_state_b64: Optional[str] = None
        main_window_geom_b64: Optional[str] = None
        splitter_sizes: Optional[list[int]] = None

        @classmethod
        def load(cls):
            return cls()

        def save(self):
            pass

    RunViewModel = None  # type: ignore
    PipelineRunner = None  # type: ignore

from menipy.gui.main_controller import MainController

# default stage order for SOPs / step list
STAGE_ORDER: List[str] = [
    "acquisition",
    "preprocessing",
    "edge_detection",
    "geometry",
    "scaling",
    "physics",
    "solver",
    "optimization",
    "outputs",
    "overlay",
    "validation",
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
                f = QFile(
                    str(Path(__file__).resolve().parent / "views" / fallback_filename)
                )
            f.open(QFile.ReadOnly)
            w = loader.load(f, self)
            f.close()
            return w

        # ---------- load panels into split hosts ----------
        self.setup_panel: QWidget = load_ui(":/views/setup_panel.ui", "setup_panel.ui")
        self.overlay_panel: QWidget = load_ui(
            ":/views/overlay_panel.ui", "overlay_panel.ui"
        )
        self.results_panel: QWidget = load_ui(
            ":/views/results_panel.ui", "results_panel.ui"
        )

        # the split UI provides these host layouts/widgets
        self._embed(self.setup_panel, self.setupHostLayout)
        self._embed(self.overlay_panel, self.previewHostLayout)
        self._embed(self.results_panel, self.resultsHostLayout)

        self.preview_panel = PreviewPanel(self.overlay_panel, ImageView)
        self.results_panel_ctrl = ResultsPanel(self.results_panel)

        self.preprocessing_ctrl = PreprocessingPipelineController(self)
        self.edge_detection_ctrl = EdgeDetectionPipelineController(self)
        self.marker_helper = (
            ImageMarkerHelper(self.preview_panel, self.preprocessing_ctrl, parent=self)
            if self.preview_panel.has_view()
            else None
        )

        self.camera_ctrl = CameraController(self)

        # add simple log view into the Log tab
        self.logView = QPlainTextEdit(self)
        self.logView.setReadOnly(True)
        self._embed(self.logView, self.logHostLayout)

        # Install Qt logging bridge (only one handler) and connect to logView
        try:
            gui_logger = logging.getLogger("menipy")
            gui_logger.setLevel(logging.INFO)
            self._qt_log_bridge, self._qt_log_handler = install_qt_logging(
                self.logView, logger=gui_logger
            )
        except Exception:
            pass

        # ---------- plugin dock (optional) ----------
        self.plugins_controller = PluginsController(self, self.settings)

        # ---------- services / VMs ----------
        if PipelineRunner and RunViewModel:
            self.runner = PipelineRunner()
            self.run_vm = RunViewModel(self.runner)
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
            preprocessing_ctrl=self.preprocessing_ctrl,
            edge_detection_ctrl=self.edge_detection_ctrl,
            pipeline_map=PIPELINE_MAP,
            sops=self.sops,
            run_vm=self.run_vm,
            log_view=self.logView,
        )

        if self.preview_panel.has_view():
            try:
                self.preview_panel.set_draw_mode(DRAW_NONE)
            except Exception:
                pass
        # restore saved splitter sizes (optional)
        if getattr(self.settings, "splitter_sizes", None):
            try:
                self.rootSplitter.setSizes(self.settings.splitter_sizes)  # type: ignore[attr-defined]
            except Exception:
                pass

        # focus
        self.statusBar().showMessage("Ready", 1500)

        # The MainController now orchestrates everything.
        if MainController:
            self.main_controller = MainController(self)
        else:
            self.main_controller = None  # type: ignore

        # menubar actions from split UI
        self._wire_menu_actions()
        self._wire_layout_controls()
        self._wire_action_bar()

    # -------------------------- helpers & wiring --------------------------

    def _wire_menu_actions(self):
        # Actions declared in main_window_split.ui
        logger.info("Wiring main window menu actions...")
        if not hasattr(self, "main_controller") or not self.main_controller:
            logger.error("MainController not available. Cannot wire menu actions.")
            return

        actions = {
            "actionOpenImage": "browse_image",
            "actionOpenCamera": "select_camera",
            "actionQuit": "close",
            "actionRunFull": "run_full_pipeline",
            "actionRunSelected": "run_full_pipeline",
            "actionStop": "stop_pipeline",
            "actionOverlay": "open_overlay",
            "actionAbout": "show_about_dialog",
            "actionPreview": "on_preview_requested",
            "actionExportCsv": "export_results_csv",
        }

        for action_name, method_name in actions.items():
            try:
                action = getattr(self, action_name)
            except AttributeError:
                # Action missing from the .ui — skip wiring (we expect the UI to provide actions)
                logger.warning(
                    f"Action '{action_name}' not found in UI; skipping wiring."
                )
                continue

            # Prepare handler mapping
            if method_name == "close":
                handler = self.close
            elif method_name == "select_camera":
                handler = lambda: self.main_controller.select_camera(True)  # type: ignore
            elif method_name == "open_overlay":
                # Always call the MainController to open the overlay config
                handler = lambda: self.main_controller.on_config_stage_requested(
                    "overlay"
                )
            else:
                handler = getattr(self.main_controller, method_name)

            try:
                action.triggered.connect(handler)
                logger.debug(f"Connected {action_name} to {method_name}")
            except Exception as e:
                logger.error(f"Failed to connect action '{action_name}': {e}")

            # No runtime icon/tooltip setting here; the .ui defines action properties.

    def _wire_action_bar(self) -> None:
        if not hasattr(self, "main_controller") or not self.main_controller:
            return

        button_action_pairs = [
            ("actionOpenImageBtn", "actionOpenImage"),
            ("actionPreviewBtn", "actionPreview"),
            ("actionRunBtn", "actionRunFull"),
            ("actionStopBtn", "actionStop"),
            ("actionExportCsvBtn", "actionExportCsv"),
        ]
        for button_name, action_name in button_action_pairs:
            button = getattr(self, button_name, None)
            action = getattr(self, action_name, None)
            if not button or not action:
                continue
            try:
                button.clicked.connect(action.trigger)
            except Exception:
                pass

    def _wire_layout_controls(self) -> None:
        self._splitter_sizes_full: Optional[list[int]] = None
        layout_buttons = {
            "layoutAnalysisBtn": "analysis",
            "layoutSetupBtn": "setup",
            "layoutReviewBtn": "review",
        }
        for button_name, mode in layout_buttons.items():
            button = getattr(self, button_name, None)
            if not button:
                continue
            try:
                button.clicked.connect(
                    lambda _checked=False, m=mode: self._apply_layout_mode(m)
                )
            except Exception:
                pass

        setup_toggle = getattr(self, "toggleSetupBtn", None)
        inspect_toggle = getattr(self, "toggleInspectBtn", None)
        if setup_toggle:
            try:
                setup_toggle.toggled.connect(
                    lambda checked=False: self._set_panel_visibility(
                        checked, getattr(inspect_toggle, "isChecked", lambda: True)()
                    )
                )
            except Exception:
                pass
        if inspect_toggle:
            try:
                inspect_toggle.toggled.connect(
                    lambda checked=False: self._set_panel_visibility(
                        getattr(setup_toggle, "isChecked", lambda: True)(), checked
                    )
                )
            except Exception:
                pass

    def _cache_splitter_sizes(self) -> None:
        if not hasattr(self, "rootSplitter"):
            return
        setup_visible = bool(getattr(self, "setupHost", None)) and self.setupHost.isVisible()  # type: ignore[attr-defined]
        inspect_visible = bool(getattr(self, "inspectTabs", None)) and self.inspectTabs.isVisible()  # type: ignore[attr-defined]
        if setup_visible and inspect_visible:
            try:
                self._splitter_sizes_full = list(self.rootSplitter.sizes())  # type: ignore[attr-defined]
            except Exception:
                pass

    def _apply_layout_mode(self, mode: str) -> None:
        if not hasattr(self, "rootSplitter"):
            return
        self._cache_splitter_sizes()
        if mode == "analysis":
            self._set_panel_visibility(False, False)
        elif mode == "setup":
            self._set_panel_visibility(True, False)
        else:
            self._set_panel_visibility(True, True)

    def _set_panel_visibility(self, show_setup: bool, show_inspector: bool) -> None:
        if not hasattr(self, "rootSplitter"):
            return
        setup_host = getattr(self, "setupHost", None)
        inspect_tabs = getattr(self, "inspectTabs", None)
        if setup_host is not None:
            setup_host.setVisible(show_setup)
        if inspect_tabs is not None:
            inspect_tabs.setVisible(show_inspector)

        try:
            sizes = self.rootSplitter.sizes()  # type: ignore[attr-defined]
        except Exception:
            sizes = [240, 600, 280]

        if show_setup and show_inspector:
            target_sizes = self._splitter_sizes_full or sizes
        elif show_setup and not show_inspector:
            total = sizes[0] + sizes[1] + sizes[2]
            target_sizes = [max(200, sizes[0]), max(1, total - sizes[0]), 0]
        elif not show_setup and show_inspector:
            total = sizes[0] + sizes[1] + sizes[2]
            target_sizes = [0, max(1, total - sizes[2]), max(200, sizes[2])]
        else:
            total = max(1, sum(sizes))
            target_sizes = [0, total, 0]

        try:
            self.rootSplitter.setSizes(target_sizes)  # type: ignore[attr-defined]
        except Exception:
            pass

        toggle_setup = getattr(self, "toggleSetupBtn", None)
        toggle_inspect = getattr(self, "toggleInspectBtn", None)
        if toggle_setup:
            toggle_setup.blockSignals(True)
            toggle_setup.setChecked(show_setup)
            toggle_setup.blockSignals(False)
        if toggle_inspect:
            toggle_inspect.blockSignals(True)
            toggle_inspect.setChecked(show_inspector)
            toggle_inspect.blockSignals(False)

        self._sync_layout_buttons(show_setup, show_inspector)

    def _sync_layout_buttons(self, show_setup: bool, show_inspector: bool) -> None:
        layout_analysis = getattr(self, "layoutAnalysisBtn", None)
        layout_setup = getattr(self, "layoutSetupBtn", None)
        layout_review = getattr(self, "layoutReviewBtn", None)
        if not (layout_analysis and layout_setup and layout_review):
            return

        target = None
        if show_setup and show_inspector:
            target = layout_review
        elif show_setup and not show_inspector:
            target = layout_setup
        elif not show_setup and not show_inspector:
            target = layout_analysis

        if target is None:
            return
        for button in (layout_analysis, layout_setup, layout_review):
            try:
                button.blockSignals(True)
                button.setChecked(button is target)
            finally:
                button.blockSignals(False)

    def _embed(self, child: QWidget, host_or_layout):
        """Add child into a QWidget host or directly into a QLayout."""
        if isinstance(host_or_layout, QLayout):
            host_or_layout.addWidget(child)
            return
        lay = host_or_layout.layout() or QVBoxLayout(host_or_layout)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(child)

    def _restore_window_layout(self):
        s = self.settings
        try:
            if getattr(s, "main_window_geom_b64", None):
                self.restoreGeometry(
                    QByteArray.fromBase64(s.main_window_geom_b64.encode("ascii"))
                )
            if getattr(s, "main_window_state_b64", None):
                self.restoreState(
                    QByteArray.fromBase64(s.main_window_state_b64.encode("ascii"))
                )
        except Exception:
            pass

    def closeEvent(self, event: QCloseEvent) -> None:
        """Delegate shutdown logic to the controller."""
        if hasattr(self, "main_controller") and self.main_controller:
            self.main_controller.shutdown()
        super().closeEvent(event)
