"""Main window class for Menipy GUI."""

# src/menipy/gui/mainwindow.py
# type: ignore
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

from PySide6.QtCore import QByteArray, QFile, Qt, QTimer
from PySide6.QtGui import QAction, QCloseEvent
from PySide6.QtUiTools import QUiLoader
from PySide6.QtWidgets import (
    QLayout,
    QMainWindow,
    QPlainTextEdit,
    QVBoxLayout,
    QWidget,
)

from menipy.gui import theme
from menipy.gui.controllers.edge_detection_controller import (
    EdgeDetectionPipelineController,
)
from menipy.gui.controllers.pipeline_controller import PipelineController
from menipy.gui.controllers.plugins_controller import PluginsController
from menipy.gui.controllers.preprocessing_controller import (
    PreprocessingPipelineController,
)
from menipy.gui.controllers.setup_panel_controller import SetupPanelController
from menipy.gui.helpers.image_marking import ImageMarkerHelper
from menipy.gui.logging_bridge import install_qt_logging
from menipy.gui.panels.preview_panel import PreviewPanel
from menipy.gui.panels.results_panel import ResultsPanel
from menipy.gui.services.camera_service import CameraConfig, CameraController
from menipy.gui.views.image_view import DRAW_NONE
from menipy.gui.views.ui_main_window import Ui_MainWindow
from menipy.pipelines.discover import PIPELINE_MAP

logger = logging.getLogger(__name__)


def _workbench_root_sizes(
    available: int, saved: Optional[list[int]] = None
) -> list[int]:
    """Return horizontal sizes for setup rail plus workbench area."""
    total = max(900, int(available or 0))
    if saved:
        values = [max(0, int(v)) for v in saved]
        if len(values) >= 3:
            setup = values[0]
            workbench = values[1] + values[2]
            if workbench > setup:
                return [setup, workbench]
        elif len(values) >= 2 and values[1] > values[0]:
            return values[:2]
    setup = min(340, max(300, int(total * 0.22)))
    return [setup, max(420, total - setup)]


def _workbench_vertical_sizes(
    available: int, saved: Optional[list[int]] = None
) -> list[int]:
    """Return vertical sizes that keep the image preview above the results."""
    total = max(620, int(available or 0))
    if saved and len(saved) >= 2:
        saved2 = [max(0, int(v)) for v in saved[:2]]
        if saved2[0] > saved2[1]:
            return saved2
    results = max(220, int(total * 0.30))
    preview = max(360, total - results)
    return [preview, results]


def _preview_dominant_sizes(
    available: int, saved: Optional[list[int]] = None
) -> list[int]:
    """Compatibility wrapper for older tests/imports."""
    root = _workbench_root_sizes(available, saved)
    vertical = _workbench_vertical_sizes(available, None)
    return [root[0], vertical[0], vertical[1]]


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
    from menipy.gui.services.pipeline_runner import PipelineRunner
    from menipy.gui.viewmodels.run_vm import RunViewModel

    from .services.settings_service import AppSettings
    from .services.sop_service import SopService
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
        guided_splitter_sizes: Optional[list[int]] = None
        guided_vertical_splitter_sizes: Optional[list[int]] = None
        overlay_config: Optional[dict] = None
        marker_config: dict = {}
        unit_system: str = "SI"

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
        self._apply_workbench_polish()

        # settings
        self.settings = AppSettings.load()

        # restore geometry/state if present (pre-split or split—it’s fine)
        self._restore_window_layout()

        # ---------- loader that knows our custom widgets ----------
        loader = QUiLoader()
        if ImageView:
            loader.registerCustomWidget(ImageView)

        def load_ui(res_path: str, fallback_filename: str) -> QWidget:
            """Load a .ui file from a Qt resource path or a filesystem fallback."""
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

        self.preview_panel = PreviewPanel(self.overlay_panel, ImageView, self.settings)
        self.results_panel_ctrl = ResultsPanel(
            self.results_panel, getattr(self, "keyResultsHost", None)
        )

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

        # Restore only preview-dominant saved sizes; migrate older layouts where
        # the image canvas was smaller than setup/results.
        guided_sizes = getattr(self.settings, "guided_splitter_sizes", None)
        guided_vertical_sizes = getattr(
            self.settings, "guided_vertical_splitter_sizes", None
        )
        legacy_sizes = getattr(self.settings, "splitter_sizes", None)
        saved_sizes = guided_sizes or legacy_sizes
        try:
            self.setupHost.setMinimumWidth(300)  # type: ignore[attr-defined]
            self.setupHost.setMaximumWidth(360)  # type: ignore[attr-defined]
            self.workbenchHost.setMinimumWidth(520)  # type: ignore[attr-defined]
            self.previewHost.setMinimumHeight(320)  # type: ignore[attr-defined]
            self.inspectTabs.setMinimumHeight(190)  # type: ignore[attr-defined]
            self.rootSplitter.setStretchFactor(0, 0)  # type: ignore[attr-defined]
            self.rootSplitter.setStretchFactor(1, 1)  # type: ignore[attr-defined]
            self.rootSplitter.setSizes(  # type: ignore[attr-defined]
                _workbench_root_sizes(self.width(), saved_sizes)
            )
            self.workbenchSplitter.setStretchFactor(0, 1)  # type: ignore[attr-defined]
            self.workbenchSplitter.setStretchFactor(1, 0)  # type: ignore[attr-defined]
            self.workbenchSplitter.setSizes(  # type: ignore[attr-defined]
                _workbench_vertical_sizes(self.height(), guided_vertical_sizes)
            )
            if hasattr(self, "previewResultsSplitter"):
                self.previewResultsSplitter.setStretchFactor(0, 1)  # type: ignore[attr-defined]
                self.previewResultsSplitter.setStretchFactor(1, 0)  # type: ignore[attr-defined]
                self.previewResultsSplitter.setSizes([900, 320])  # type: ignore[attr-defined]
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
        self._set_guided_advanced_visible(
            bool(getattr(self.settings, "advanced_ui_visible", False)), save=False
        )
        self._setup_units_menu()
        try:
            self.main_controller.load_startup_preview()
        except Exception:
            logger.debug("Startup preview load failed", exc_info=True)

    def _setup_units_menu(self):
        """Creates a Units menu to toggle between SI and CGS."""
        from PySide6.QtGui import QActionGroup

        menu_bar = self.menuBar()
        units_menu = getattr(self, "menuUnits", None)
        if units_menu is None:
            units_menu = menu_bar.addMenu("&Units")
            units_menu.setObjectName("menuUnits")
            help_menu = getattr(self, "menuHelp", None)
            if help_menu is not None:
                menu_bar.removeAction(units_menu.menuAction())
                menu_bar.insertMenu(help_menu.menuAction(), units_menu)
            self.menuUnits = units_menu
        else:
            units_menu.clear()

        si_action = QAction("SI (mm, kg/m³)", self)
        si_action.setCheckable(True)
        si_action.setData("SI")

        cgs_action = QAction("CGS (cm, g/cm³)", self)
        cgs_action.setCheckable(True)
        cgs_action.setData("CGS")

        group = QActionGroup(self)
        group.addAction(si_action)
        group.addAction(cgs_action)
        group.setExclusive(True)

        units_menu.addAction(si_action)
        units_menu.addAction(cgs_action)

        # Initial check
        current = getattr(self.settings, "unit_system", "SI")
        if current == "CGS":
            cgs_action.setChecked(True)
        else:
            si_action.setChecked(True)

        def on_triggered(action):
            system = action.data()
            if hasattr(self, "main_controller") and self.main_controller:
                self.main_controller.change_unit_system(system)

        group.triggered.connect(on_triggered)

        # Force initial labels refresh
        QTimer.singleShot(0, self.main_controller.refresh_unit_labels)

    def _apply_workbench_polish(self) -> None:
        """Apply proposal-inspired styling to the current generated workbench."""
        self.setStyleSheet(
            theme.get_stylesheet()
            + f"""
            QWidget#centralwidget {{
                background-color: {theme.BG_PRIMARY};
            }}
            QWidget#workflowBar {{
                background-color: {theme.BG_SECONDARY};
                border-bottom: 1px solid {theme.BORDER_DEFAULT};
            }}
            QToolButton {{
                background-color: {theme.BG_PRIMARY};
                border: 1px solid {theme.BORDER_DEFAULT};
                border-radius: 5px;
                color: {theme.TEXT_PRIMARY};
                padding: 5px 12px;
                font-weight: 600;
            }}
            QToolButton:hover {{
                background-color: {theme.BG_HOVER};
            }}
            QToolButton:checked {{
                background-color: #DDF4FF;
                border-color: {theme.ACCENT_BLUE};
                color: {theme.ACCENT_BLUE};
            }}
            QToolButton#actionRunBtn {{
                background-color: {theme.ACCENT_BLUE};
                border-color: {theme.ACCENT_BLUE};
                color: white;
            }}
            QToolButton#actionRunBtn:hover {{
                background-color: {theme.ACCENT_BLUE_HOVER};
            }}
            QToolButton#workflowAdvancedBtn {{
                color: #8250DF;
            }}
            QWidget#setupHost {{
                background-color: {theme.BG_SECONDARY};
                border-right: 1px solid {theme.BORDER_DEFAULT};
            }}
            QWidget#workbenchHost,
            QWidget#previewHost {{
                background-color: {theme.BG_TERTIARY};
            }}
            QWidget#keyResultsHost {{
                background-color: {theme.BG_SECONDARY};
                border-left: 1px solid {theme.BORDER_DEFAULT};
            }}
            QToolButton#toggleSetupBtn,
            QToolButton#toggleInspectBtn,
            QToolButton#toggleKeyResultsBtn {{
                min-width: 56px;
                padding-left: 8px;
                padding-right: 8px;
            }}
            QTabWidget#inspectTabs {{
                background-color: {theme.BG_PRIMARY};
            }}
            """
        )

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
            "actionConfigOverlay": "open_overlay",
            "actionConfigMarkers": "open_marker_config",
            "actionConfigPipeline": "open_pipeline_config",
            "actionConfigPreprocessing": "open_preprocessing_config",
            "actionConfigEdgeDetection": "open_edge_detection_config",
            "actionConfigGeometry": "open_geometry_config",
            "actionConfigPhysics": "open_physics_config",
            "actionConfigAcquisition": "open_acquisition_config",
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
            ("actionRunBtn", "actionRunFull"),
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

        auto_calibrate = getattr(self, "workflowAutoCalibrateBtn", None)
        if auto_calibrate and getattr(self, "setup_panel_ctrl", None):
            try:
                auto_calibrate.clicked.connect(
                    lambda _checked=False: self.setup_panel_ctrl.auto_calibrate_requested.emit()
                )
            except Exception:
                pass

        advanced = getattr(self, "workflowAdvancedBtn", None)
        if advanced:
            try:
                advanced.toggled.connect(self._set_guided_advanced_visible)
            except Exception:
                pass

    def _set_guided_advanced_visible(self, visible: bool, save: bool = True) -> None:
        advanced = getattr(self, "workflowAdvancedBtn", None)
        if advanced:
            advanced.blockSignals(True)
            advanced.setChecked(visible)
            advanced.setText("Advanced -" if visible else "Advanced +")
            advanced.blockSignals(False)

        setup_ctrl = getattr(self, "setup_panel_ctrl", None)
        if setup_ctrl and hasattr(setup_ctrl, "set_advanced_visible"):
            setup_ctrl.set_advanced_visible(visible, save=False)

        tabs = getattr(self, "inspectTabs", None)
        if tabs:
            for tab in (
                getattr(self, "residualsTab", None),
                getattr(self, "timingsTab", None),
                getattr(self, "logTab", None),
            ):
                if tab is None:
                    continue
                index = tabs.indexOf(tab)
                if index >= 0:
                    try:
                        tabs.setTabVisible(index, visible)
                    except AttributeError:
                        tab.setVisible(visible)
            if not visible and tabs.currentWidget() in {
                getattr(self, "residualsTab", None),
                getattr(self, "timingsTab", None),
                getattr(self, "logTab", None),
            }:
                tabs.setCurrentWidget(self.resultsTab)

        if save:
            self.settings.advanced_ui_visible = visible
            try:
                self.settings.save()
            except OSError:
                pass

    def _wire_layout_controls(self) -> None:
        self._splitter_sizes_full: Optional[list[int]] = None
        self._workbench_splitter_sizes_full: Optional[list[int]] = None
        self._preview_results_sizes_full: Optional[list[int]] = None
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
        key_results_toggle = getattr(self, "toggleKeyResultsBtn", None)
        setup_action = getattr(self, "actionToggleSetup", None)
        inspect_action = getattr(self, "actionToggleResultsTable", None)
        key_results_action = getattr(self, "actionToggleKeyResults", None)

        def _connect_checkable(source, target) -> None:
            if not source or not target:
                return
            try:
                source.toggled.connect(target.setChecked)
                target.toggled.connect(source.setChecked)
            except Exception:
                pass

        _connect_checkable(setup_toggle, setup_action)
        _connect_checkable(inspect_toggle, inspect_action)
        _connect_checkable(key_results_toggle, key_results_action)

        def _current_setup_visible() -> bool:
            return bool(getattr(setup_toggle, "isChecked", lambda: True)())

        def _current_inspect_visible() -> bool:
            return bool(getattr(inspect_toggle, "isChecked", lambda: True)())

        def _current_key_results_visible() -> bool:
            return bool(getattr(key_results_toggle, "isChecked", lambda: True)())

        if setup_toggle:
            try:
                setup_toggle.toggled.connect(
                    lambda checked=False: self._set_panel_visibility(
                        checked,
                        _current_inspect_visible(),
                        _current_key_results_visible(),
                    )
                )
            except Exception:
                pass
        if inspect_toggle:
            try:
                inspect_toggle.toggled.connect(
                    lambda checked=False: self._set_panel_visibility(
                        _current_setup_visible(),
                        checked,
                        _current_key_results_visible(),
                    )
                )
            except Exception:
                pass
        if key_results_toggle:
            try:
                key_results_toggle.toggled.connect(
                    lambda checked=False: self._set_panel_visibility(
                        _current_setup_visible(),
                        _current_inspect_visible(),
                        checked,
                    )
                )
            except Exception:
                pass

        reset_action = getattr(self, "actionResetLayout", None)
        if reset_action:
            try:
                reset_action.triggered.connect(self._reset_workbench_layout)
            except Exception:
                pass

        fit_action = getattr(self, "actionFitPreview", None)
        if fit_action:
            try:
                fit_action.triggered.connect(self._fit_preview_to_window)
            except Exception:
                pass

        self._set_panel_visibility(True, True, True)

    def _cache_splitter_sizes(self) -> None:
        if not hasattr(self, "rootSplitter"):
            return
        setup_visible = bool(getattr(self, "setupHost", None)) and self.setupHost.isVisible()  # type: ignore[attr-defined]
        inspect_visible = bool(getattr(self, "inspectTabs", None)) and self.inspectTabs.isVisible()  # type: ignore[attr-defined]
        key_results_visible = bool(getattr(self, "keyResultsHost", None)) and self.keyResultsHost.isVisible()  # type: ignore[attr-defined]
        if setup_visible and inspect_visible and key_results_visible:
            try:
                self._splitter_sizes_full = list(self.rootSplitter.sizes())  # type: ignore[attr-defined]
                self._workbench_splitter_sizes_full = list(
                    self.workbenchSplitter.sizes()  # type: ignore[attr-defined]
                )
                self._preview_results_sizes_full = list(
                    self.previewResultsSplitter.sizes()  # type: ignore[attr-defined]
                )
            except Exception:
                pass

    def _apply_layout_mode(self, mode: str) -> None:
        if not hasattr(self, "rootSplitter"):
            return
        self._cache_splitter_sizes()
        if mode == "analysis":
            self._set_panel_visibility(False, False, True)
        elif mode == "setup":
            self._set_panel_visibility(True, False, True)
        else:
            self._set_panel_visibility(True, True, True)

    def _set_panel_visibility(
        self,
        show_setup: bool,
        show_inspector: bool,
        show_key_results: Optional[bool] = None,
    ) -> None:
        if not hasattr(self, "rootSplitter"):
            return

        setup_host = getattr(self, "setupHost", None)
        inspect_tabs = getattr(self, "inspectTabs", None)
        key_results_host = getattr(self, "keyResultsHost", None)

        if show_key_results is None:
            show_key_results = (
                bool(key_results_host.isVisible()) if key_results_host else True
            )

        try:
            sizes = list(self.rootSplitter.sizes())  # type: ignore[attr-defined]
        except Exception:
            sizes = [260, 740]
        try:
            vertical_sizes = list(self.workbenchSplitter.sizes())  # type: ignore[attr-defined]
        except Exception:
            vertical_sizes = [560, 240]
        try:
            preview_results_sizes = list(self.previewResultsSplitter.sizes())  # type: ignore[attr-defined]
        except Exception:
            preview_results_sizes = [900, 320]

        # Cache non-collapsed sizes before hiding panels, so they can be restored.
        if show_setup and show_inspector:
            if len(sizes) >= 2 and sizes[0] > 0 and sizes[1] > 0:
                self._splitter_sizes_full = sizes
            if len(vertical_sizes) >= 2 and vertical_sizes[0] > 0 and vertical_sizes[1] > 0:
                self._workbench_splitter_sizes_full = vertical_sizes
        if show_key_results and len(preview_results_sizes) >= 2 and preview_results_sizes[1] > 0:
            self._preview_results_sizes_full = preview_results_sizes

        if setup_host is not None:
            setup_host.setVisible(show_setup)
        if inspect_tabs is not None:
            inspect_tabs.setVisible(show_inspector)
        if key_results_host is not None:
            key_results_host.setVisible(show_key_results)

        if show_setup and show_inspector:
            cached_root = getattr(self, "_splitter_sizes_full", None)
            if not cached_root or len(cached_root) < 2 or min(cached_root[:2]) <= 0:
                cached_root = _workbench_root_sizes(self.width(), None)
            target_sizes = [max(220, int(cached_root[0])), max(1, int(cached_root[1]))]

            cached_vertical = getattr(self, "_workbench_splitter_sizes_full", None)
            if (
                not cached_vertical
                or len(cached_vertical) < 2
                or min(cached_vertical[:2]) <= 0
            ):
                cached_vertical = _workbench_vertical_sizes(self.height(), None)
            target_vertical_sizes = [
                max(1, int(cached_vertical[0])),
                max(160, int(cached_vertical[1])),
            ]
        elif show_setup and not show_inspector:
            total = max(1, sum(sizes))
            setup_width = max(220, sizes[0] if sizes and sizes[0] > 0 else 300)
            target_sizes = [setup_width, max(1, total - setup_width)]
            target_vertical_sizes = [max(1, sum(vertical_sizes)), 0]
        elif not show_setup and show_inspector:
            total = max(1, sum(sizes))
            target_sizes = [0, total]
            target_vertical_sizes = [
                max(1, vertical_sizes[0] if vertical_sizes else 560),
                max(160, vertical_sizes[1] if len(vertical_sizes) > 1 else 240),
            ]
        else:
            total = max(1, sum(sizes))
            target_sizes = [0, total]
            target_vertical_sizes = [max(1, sum(vertical_sizes)), 0]

        try:
            self.rootSplitter.setSizes(target_sizes)  # type: ignore[attr-defined]
            self.workbenchSplitter.setSizes(target_vertical_sizes)  # type: ignore[attr-defined]
        except Exception:
            pass

        try:
            if show_key_results:
                cached_preview = getattr(self, "_preview_results_sizes_full", None)
                if (
                    not cached_preview
                    or len(cached_preview) < 2
                    or cached_preview[1] <= 0
                ):
                    total = max(1, sum(preview_results_sizes))
                    cached_preview = [max(1, total - 320), 320]
                target_preview_results = [
                    max(1, int(cached_preview[0])),
                    max(180, int(cached_preview[1])),
                ]
            else:
                target_preview_results = [max(1, sum(preview_results_sizes)), 0]
            self.previewResultsSplitter.setSizes(target_preview_results)  # type: ignore[attr-defined]
        except Exception:
            pass

        toggle_setup = getattr(self, "toggleSetupBtn", None)
        toggle_inspect = getattr(self, "toggleInspectBtn", None)
        toggle_key_results = getattr(self, "toggleKeyResultsBtn", None)
        action_setup = getattr(self, "actionToggleSetup", None)
        action_inspect = getattr(self, "actionToggleResultsTable", None)
        action_key_results = getattr(self, "actionToggleKeyResults", None)

        for control, visible in (
            (toggle_setup, show_setup),
            (toggle_inspect, show_inspector),
            (toggle_key_results, show_key_results),
            (action_setup, show_setup),
            (action_inspect, show_inspector),
            (action_key_results, show_key_results),
        ):
            if control:
                control.blockSignals(True)
                control.setChecked(visible)
                control.blockSignals(False)

        self._sync_layout_buttons(show_setup, show_inspector)

    def _reset_workbench_layout(self) -> None:
        self._splitter_sizes_full = None
        self._workbench_splitter_sizes_full = None
        self._preview_results_sizes_full = None
        self._set_panel_visibility(True, True, True)
        try:
            self.rootSplitter.setSizes(_workbench_root_sizes(self.width(), None))  # type: ignore[attr-defined]
            self.workbenchSplitter.setSizes(  # type: ignore[attr-defined]
                _workbench_vertical_sizes(self.height(), None)
            )
            self.previewResultsSplitter.setSizes([900, 320])  # type: ignore[attr-defined]
        except Exception:
            pass

    def _fit_preview_to_window(self) -> None:
        preview = getattr(self, "preview_panel", None)
        image_view = getattr(preview, "image_view", None)
        fit = getattr(image_view, "fit_to_window", None)
        if callable(fit):
            fit()

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
