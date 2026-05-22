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
    QAbstractButton,
    QHBoxLayout,
    QLabel,
    QLayout,
    QLineEdit,
    QMainWindow,
    QPlainTextEdit,
    QSizePolicy,
    QStackedLayout,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from menipy.gui import theme
from menipy.gui.controllers.edge_detection_controller import (
    EdgeDetectionPipelineController,
)
from menipy.gui.controllers.pipeline_controller import PipelineController
from menipy.gui.controllers.pipeline_step_test_controller import (
    PipelineStepTestController,
)
from menipy.gui.controllers.plugins_controller import PluginsController
from menipy.gui.controllers.preprocessing_controller import (
    PreprocessingPipelineController,
)
from menipy.gui.controllers.setup_panel_controller import SetupPanelController
from menipy.gui.dialogs.advanced_workflow_dialog import AdvancedWorkflowDialog
from menipy.gui.dialogs.camera_settings_dialog import CameraSettingsDialog
from menipy.gui.helpers.icon_loader import set_button_icon
from menipy.gui.helpers.image_marking import ImageMarkerHelper
from menipy.gui.helpers.logging_bridge import install_qt_logging
from menipy.gui.services.camera_service import CameraConfig, CameraController
from menipy.gui.views.image_view import DRAW_NONE
from menipy.gui.views.pipeline_step_test_panel import PipelineStepTestPanel
from menipy.gui.views.preview_panel import PreviewPanel
from menipy.gui.views.results_panel import ResultsPanel
from menipy.gui.views.ui_main_window import Ui_MainWindow
from menipy.pipelines.discover import PIPELINE_MAP

logger = logging.getLogger(__name__)


def _workbench_root_sizes(available: int, saved: list[int] | None = None) -> list[int]:
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
    setup = min(360, max(340, int(total * 0.24)))
    return [setup, max(420, total - setup)]


def _workbench_vertical_sizes(
    available: int, saved: list[int] | None = None
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
    available: int, saved: list[int] | None = None
) -> list[int]:
    """Compatibility wrapper for older tests/imports."""
    root = _workbench_root_sizes(available, saved)
    vertical = _workbench_vertical_sizes(available, None)
    return [root[0], vertical[0], vertical[1]]


# --- promoted preview widget (registered into QUiLoader) ---
try:
    from .image_view import ImageView
except Exception:  # keep app booting even if file missing during refactors
    ImageView = None  # type: ignore

# --- optional step row & SOP service (guarded) ---
try:
    from .step_item_widget import StepItemWidget
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
        selected_pipeline: str | None = None
        last_image_path: str | None = None
        plugin_dirs: list[str] = []
        main_window_state_b64: str | None = None
        main_window_geom_b64: str | None = None
        splitter_sizes: list[int] | None = None
        guided_splitter_sizes: list[int] | None = None
        guided_vertical_splitter_sizes: list[int] | None = None
        overlay_config: dict | None = None
        marker_config: dict = {}
        unit_system: str = "SI"

        @classmethod
        def load(cls):
            return cls()

        def save(self):
            pass

    RunViewModel = None  # type: ignore
    PipelineRunner = None  # type: ignore

from menipy.gui.controllers.main_controller import MainController

# default stage order for SOPs / step list
STAGE_ORDER: list[str] = [
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
        self._apply_workbench_icons()
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
                f = QFile(str(Path(__file__).resolve().parent / fallback_filename))
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
            self.results_panel,
            getattr(self, "keyResultsHost", None),
            getattr(self, "residualsHostLayout", None),
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
        self._install_workflow_setup_controls()
        self.advanced_workflow_dialog = AdvancedWorkflowDialog(
            self,
            self.setup_panel_ctrl.sopGroup,
            self.setup_panel_ctrl.stepsGroup,
        )
        self.setup_panel_ctrl.advanced_requested.connect(self._show_advanced_dialog)

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
        self._install_pipeline_step_test_panel()

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
        self._sync_advanced_buttons()
        self._setup_units_menu()
        try:
            self.main_controller.load_startup_preview()
        except Exception:
            logger.debug("Startup preview load failed", exc_info=True)

    def _setup_units_menu(self):
        """Create a Config > Units submenu to toggle between SI and CGS."""
        from PySide6.QtGui import QActionGroup
        from PySide6.QtWidgets import QMenu

        menu_bar = self.menuBar()
        units_menu = getattr(self, "menuUnits", None)
        if units_menu is None:
            config_menu = getattr(self, "menuConfig", None)
            if isinstance(config_menu, QMenu):
                units_menu = QMenu("&Units", self)
                config_menu.addMenu(units_menu)
            else:
                units_menu = menu_bar.addMenu("&Units")
            units_menu.setObjectName("menuUnits")
            self.menuUnits = units_menu
        else:
            help_menu = getattr(self, "menuHelp", None)
            config_menu = getattr(self, "menuConfig", None)
            if (
                isinstance(config_menu, QMenu)
                and help_menu is not None
                and units_menu.menuAction() in menu_bar.actions()
            ):
                menu_bar.removeAction(units_menu.menuAction())
                config_menu.addMenu(units_menu)
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
        self.setStyleSheet(theme.get_stylesheet() + f"""
            QWidget#centralwidget {{
                background-color: {theme.BG_PRIMARY};
            }}
            QWidget#workflowBar {{
                background-color: {theme.BG_SECONDARY};
                border-bottom: 1px solid {theme.BORDER_DEFAULT};
            }}
            QWidget#workflowAnalysisHost {{
                border-right: 1px solid {theme.BORDER_DEFAULT};
            }}
            QWidget#workflowSourceHost {{
                border-right: 1px solid {theme.BORDER_DEFAULT};
            }}
            QWidget#workflowSourceStackHost {{
                background-color: transparent;
            }}
            QLabel#workflowAnalysisLabel,
            QLabel#workflowSourceLabel {{
                color: {theme.TEXT_SECONDARY};
                font-size: 11px;
                font-weight: 700;
                padding-right: 2px;
                text-transform: uppercase;
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
            QWidget#workflowBar QToolButton[workflowSourceMode="true"] {{
                min-width: 40px;
                max-width: 40px;
                min-height: 30px;
                max-height: 30px;
                padding: 0px;
                background-color: {theme.BG_PRIMARY};
                border: 1px solid {theme.BORDER_DEFAULT};
                border-radius: 7px;
            }}
            QWidget#workflowBar QToolButton[workflowSourceMode="true"]:hover {{
                background-color: {theme.BG_HOVER};
            }}
            QWidget#workflowBar QToolButton[workflowSourceMode="true"]:checked {{
                background-color: #DAFBE1;
                border-color: #4AC26B;
                color: #1A7F37;
            }}
            QWidget#workflowBar QRadioButton {{
                background-color: {theme.BG_PRIMARY};
                border: 1px solid {theme.BORDER_DEFAULT};
                border-radius: 5px;
                color: {theme.TEXT_SECONDARY};
                padding: 5px 8px;
            }}
            QWidget#workflowBar QRadioButton:hover {{
                background-color: {theme.BG_HOVER};
                color: {theme.TEXT_PRIMARY};
            }}
            QWidget#workflowBar QRadioButton:checked {{
                background-color: #DAFBE1;
                border-color: #4AC26B;
                color: #1A7F37;
                font-weight: 700;
            }}
            QWidget#workflowBar QRadioButton::indicator {{
                width: 0px;
                height: 0px;
            }}
            QWidget#workflowBar QLineEdit,
            QWidget#workflowBar QComboBox,
            QWidget#workflowBar QSpinBox {{
                background-color: {theme.BG_PRIMARY};
                border: 1px solid {theme.BORDER_DEFAULT};
                border-radius: 5px;
                color: {theme.TEXT_PRIMARY};
                padding: 4px 8px;
            }}
            QWidget#workflowBar QLineEdit:focus,
            QWidget#workflowBar QComboBox:focus,
            QWidget#workflowBar QSpinBox:focus {{
                border-color: {theme.ACCENT_BLUE};
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
                min-width: 32px;
                padding-left: 8px;
                padding-right: 8px;
            }}
            QTabWidget#inspectTabs {{
                background-color: {theme.BG_PRIMARY};
            }}
            """)

    def _apply_workbench_icons(self) -> None:
        """Apply resource-backed icons to the top workflow controls."""
        for button_name, icon_name in (
            ("toggleSetupBtn", "layout-sidebar"),
            ("toggleInspectBtn", "layout-table"),
            ("toggleKeyResultsBtn", "layout-results"),
        ):
            button = getattr(self, button_name, None)
            if button:
                set_button_icon(button, icon_name, size=16, clear_text=True)
                button.setToolButtonStyle(Qt.ToolButtonIconOnly)

        for button_name, icon_name in (
            ("actionOpenImageBtn", "file"),
            ("actionExportCsvBtn", "download"),
        ):
            button = getattr(self, button_name, None)
            if button:
                set_button_icon(button, icon_name, size=16)
                button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)

    def _install_workflow_setup_controls(self) -> None:
        """Move controller-owned setup controls into the top workflow bar."""
        if getattr(self, "_workflow_setup_controls_installed", False):
            return
        setup_ctrl = getattr(self, "setup_panel_ctrl", None)
        analysis_layout = getattr(self, "workflowAnalysisLayout", None)
        source_layout = getattr(self, "workflowSourceLayout", None)
        if setup_ctrl is None or analysis_layout is None or source_layout is None:
            return

        self._add_workflow_label(analysis_layout, "Analysis")
        for button in (
            setup_ctrl.sessileBtn,
            setup_ctrl.pendantBtn,
            setup_ctrl.oscillatingBtn,
            setup_ctrl.capillaryBtn,
            setup_ctrl.captiveBtn,
        ):
            if button is None:
                continue
            button.setParent(self.workflowBar)
            button.setMinimumHeight(30)
            button.setMaximumHeight(32)
            button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
            analysis_layout.addWidget(button)

        self.workflowTestPipelineBtn = QToolButton(self.workflowBar)
        self.workflowTestPipelineBtn.setObjectName("workflowTestPipelineBtn")
        self.workflowTestPipelineBtn.setText("Test")
        self.workflowTestPipelineBtn.setToolTip("Test pipeline steps")
        self.workflowTestPipelineBtn.setCheckable(True)
        self.workflowTestPipelineBtn.setAutoRaise(True)
        self.workflowTestPipelineBtn.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.workflowTestPipelineBtn.setMinimumHeight(30)
        self.workflowTestPipelineBtn.setMaximumHeight(32)
        set_button_icon(self.workflowTestPipelineBtn, "play", size=15)
        self.workflowTestPipelineBtn.hide()
        self.workflowTestPipelineBtn.toggled.connect(self._set_pipeline_test_visible)
        analysis_layout.addWidget(self.workflowTestPipelineBtn)

        self._add_workflow_label(source_layout, "Source")
        source_mode_buttons: dict[str, QToolButton] = {}
        for mode_name, object_name, tooltip, icon_name, target_radio in (
            (
                setup_ctrl.MODE_SINGLE,
                "workflowSingleSourceBtn",
                "File",
                "file",
                setup_ctrl.singleModeRadio,
            ),
            (
                setup_ctrl.MODE_BATCH,
                "workflowBatchSourceBtn",
                "Batch",
                "batch",
                setup_ctrl.batchModeRadio,
            ),
            (
                setup_ctrl.MODE_CAMERA,
                "workflowCameraSourceBtn",
                "Camera",
                "camera",
                setup_ctrl.cameraModeRadio,
            ),
        ):
            button = self._create_workflow_source_mode_button(
                object_name, tooltip, icon_name, target_radio
            )
            source_mode_buttons[mode_name] = button
            source_layout.addWidget(button)
        self.workflowSourceModeButtons = source_mode_buttons

        source_stack_host = QWidget(self.workflowBar)
        source_stack_host.setObjectName("workflowSourceStackHost")
        source_stack_host.setMinimumHeight(32)
        source_stack_host.setMaximumHeight(32)
        source_stack_host.setFixedWidth(430)
        source_stack_host.setSizePolicy(
            QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed
        )
        source_stack = QStackedLayout(source_stack_host)
        source_stack.setObjectName("workflowSourceStack")
        source_stack.setContentsMargins(0, 0, 0, 0)
        image_display = self._create_workflow_source_display("workflowImageDisplay")
        batch_display = self._create_workflow_source_display("workflowBatchDisplay")
        camera_settings_btn = self._create_workflow_camera_settings_button()
        self.workflowImageDisplay = image_display
        self.workflowBatchDisplay = batch_display
        self.workflowCameraSettingsBtn = camera_settings_btn

        source_pages: dict[str, QWidget] = {}
        source_page_layouts: dict[str, QHBoxLayout] = {}
        for mode_name in (
            setup_ctrl.MODE_SINGLE,
            setup_ctrl.MODE_BATCH,
            setup_ctrl.MODE_CAMERA,
        ):
            page = QWidget(source_stack_host)
            page.setObjectName(f"workflowSource{mode_name.title()}Page")
            page_layout = QHBoxLayout(page)
            page_layout.setContentsMargins(0, 0, 0, 0)
            page_layout.setSpacing(6)
            source_stack.addWidget(page)
            source_pages[mode_name] = page
            source_page_layouts[mode_name] = page_layout

        for widget, width in (
            (setup_ctrl.browseBtn, 34),
            (setup_ctrl.batchBrowseBtn, 34),
            (setup_ctrl.sourceIdCombo, 160),
        ):
            if widget is None:
                continue
            widget.setParent(self.workflowBar)
            widget.setMinimumHeight(30)
            widget.setMaximumHeight(32)
            if width:
                widget.setMinimumWidth(width)
                widget.setMaximumWidth(width)
            widget.show()

        if setup_ctrl.imagePathEdit is not None:
            setup_ctrl.imagePathEdit.setParent(self.workflowBar)
            setup_ctrl.imagePathEdit.hide()
            setup_ctrl.imagePathEdit.textChanged.connect(
                self._sync_workflow_source_displays
            )
        source_page_layouts[setup_ctrl.MODE_SINGLE].addWidget(image_display)
        if setup_ctrl.browseBtn is not None:
            source_page_layouts[setup_ctrl.MODE_SINGLE].addWidget(setup_ctrl.browseBtn)
        if setup_ctrl.batchPathEdit is not None:
            setup_ctrl.batchPathEdit.setParent(self.workflowBar)
            setup_ctrl.batchPathEdit.hide()
            setup_ctrl.batchPathEdit.textChanged.connect(
                self._sync_workflow_source_displays
            )
        source_page_layouts[setup_ctrl.MODE_BATCH].addWidget(batch_display)
        if setup_ctrl.batchBrowseBtn is not None:
            source_page_layouts[setup_ctrl.MODE_BATCH].addWidget(
                setup_ctrl.batchBrowseBtn
            )
        if setup_ctrl.framesSpin is not None:
            setup_ctrl.framesSpin.setParent(self.workflowBar)
            setup_ctrl.framesSpin.hide()
        source_page_layouts[setup_ctrl.MODE_CAMERA].addWidget(camera_settings_btn)

        source_layout.addWidget(source_stack_host)
        self.workflowSourceStackHost = source_stack_host
        self.workflowSourceStack = source_stack
        self._workflow_source_pages = source_pages
        self._workflow_source_page_layouts = source_page_layouts

        for group_name in ("pipelineGroup", "sourceGroup"):
            group = setup_ctrl.panel.findChild(QWidget, group_name)
            if group is not None:
                group.setVisible(False)

        for label in (
            setup_ctrl.labelImage,
            setup_ctrl.labelBatch,
            setup_ctrl.labelSourceId,
            setup_ctrl.labelFrames,
        ):
            if label is not None:
                label.setVisible(False)

        setup_ctrl.set_workflow_source_stack_mode(True)
        setup_ctrl.source_mode_changed.connect(self._sync_workflow_source_mode)
        if setup_ctrl.sourceIdCombo is not None:
            setup_ctrl.sourceIdCombo.currentTextChanged.connect(
                self._sync_workflow_source_displays
            )
        self._sync_workflow_source_mode(setup_ctrl.current_mode())
        self._sync_workflow_source_mode_buttons(setup_ctrl.current_mode())
        setup_ctrl._update_widget_states()
        self._workflow_setup_controls_installed = True

    def _install_pipeline_step_test_panel(self) -> None:
        """Install the scientific step-test rail beside the normal setup panel."""
        if getattr(self, "_pipeline_step_test_panel_installed", False):
            return
        layout = getattr(self, "setupHostLayout", None)
        if layout is None:
            return
        self.pipeline_step_test_panel = PipelineStepTestPanel(self.setupHost)
        self.pipeline_step_test_panel.hide()
        layout.addWidget(self.pipeline_step_test_panel)
        self.pipeline_step_test_ctrl = PipelineStepTestController(
            window=self,
            panel=self.pipeline_step_test_panel,
            setup_ctrl=self.setup_panel_ctrl,
            pipeline_ctrl=self.pipeline_ctrl,
            preprocessing_ctrl=self.preprocessing_ctrl,
            edge_detection_ctrl=self.edge_detection_ctrl,
            pipeline_map=PIPELINE_MAP,
            parent=self,
        )
        self.setup_panel_ctrl.pipeline_changed.connect(
            lambda _pipeline: self.pipeline_step_test_ctrl.refresh_stages()
        )
        self._pipeline_step_test_panel_installed = True

    def _set_scientific_test_available(self, available: bool) -> None:
        button = getattr(self, "workflowTestPipelineBtn", None)
        if button is None:
            return
        button.setVisible(bool(available))
        if not available:
            self._set_pipeline_test_visible(False)

    def _set_pipeline_test_visible(self, visible: bool) -> None:
        panel = getattr(self, "pipeline_step_test_panel", None)
        button = getattr(self, "workflowTestPipelineBtn", None)
        setup_panel = getattr(self, "setup_panel", None)
        if panel is None or setup_panel is None:
            return

        visible = bool(visible)
        if button is not None and button.isChecked() != visible:
            button.blockSignals(True)
            try:
                button.setChecked(visible)
            finally:
                button.blockSignals(False)

        if visible:
            ctrl = getattr(self, "pipeline_step_test_ctrl", None)
            if ctrl is not None:
                ctrl.refresh_from_live()
                ctrl.refresh_stages()
            setup_panel.hide()
            panel.show()
            if getattr(self, "setupHost", None) is not None:
                self.setupHost.setVisible(True)  # type: ignore[attr-defined]
            setup_toggle = getattr(self, "toggleSetupBtn", None)
            setup_action = getattr(self, "actionToggleSetup", None)
            for toggle in (setup_toggle, setup_action):
                if toggle is not None and hasattr(toggle, "setChecked"):
                    toggle.blockSignals(True)
                    try:
                        toggle.setChecked(True)
                    finally:
                        toggle.blockSignals(False)
        else:
            panel.hide()
            setup_panel.show()

    def _create_workflow_source_mode_button(
        self,
        object_name: str,
        tooltip: str,
        icon_name: str,
        target_radio: QAbstractButton | None,
    ) -> QToolButton:
        button = QToolButton(self.workflowBar)
        button.setObjectName(object_name)
        button.setProperty("workflowSourceMode", True)
        button.setAutoRaise(True)
        button.setCheckable(True)
        button.setToolButtonStyle(Qt.ToolButtonIconOnly)
        button.setFixedSize(40, 30)
        button.setToolTip(tooltip)
        set_button_icon(button, icon_name, size=18, clear_text=True)
        if target_radio is not None:
            button.clicked.connect(
                lambda _checked=False, radio=target_radio: radio.click()
            )
        return button

    def _create_workflow_camera_settings_button(self) -> QToolButton:
        button = QToolButton(self.workflowBar)
        button.setObjectName("workflowCameraSettingsBtn")
        button.setAutoRaise(True)
        button.setToolButtonStyle(Qt.ToolButtonIconOnly)
        button.setFixedSize(34, 30)
        button.setToolTip("Camera settings")
        set_button_icon(button, "settings", size=16, clear_text=True)
        button.clicked.connect(self._show_camera_settings_dialog)
        return button

    def _show_camera_settings_dialog(self) -> None:
        setup_ctrl = getattr(self, "setup_panel_ctrl", None)
        if setup_ctrl is None:
            return
        current = setup_ctrl.camera_settings_values()
        dialog = CameraSettingsDialog(
            cameras=setup_ctrl.camera_devices(),
            current_device=int(current["device"]),
            frames=int(current["frames"]),
            fps=int(current["fps"]),
            width=current["width"],
            height=current["height"],
            parent=self,
        )
        if dialog.exec() == dialog.Accepted:
            values = dialog.values()
            setup_ctrl.apply_camera_settings(
                device=int(values["device"]),
                frames=int(values["frames"]),
                fps=int(values["fps"]),
                width=values["width"],
                height=values["height"],
            )
            self._sync_workflow_source_displays()

    def _sync_workflow_source_mode_buttons(self, mode: str) -> None:
        buttons = getattr(self, "workflowSourceModeButtons", {})
        for mode_name, button in buttons.items():
            button.blockSignals(True)
            try:
                button.setChecked(mode_name == mode)
            finally:
                button.blockSignals(False)

    def _create_workflow_source_display(self, object_name: str) -> QLineEdit:
        display = QLineEdit(self.workflowBar)
        display.setObjectName(object_name)
        display.setReadOnly(True)
        display.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        display.setMinimumHeight(30)
        display.setMaximumHeight(32)
        display.setMinimumWidth(220)
        display.setMaximumWidth(220)
        display.setPlaceholderText("./select source")
        display.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        return display

    def _compact_workflow_path(self, path: str | None) -> str:
        if not path:
            return ""
        name = Path(path).name or str(path)
        return f"./{name}"

    def _sync_workflow_source_displays(self, *_args) -> None:
        setup_ctrl = getattr(self, "setup_panel_ctrl", None)
        if setup_ctrl is None:
            return

        image_path = setup_ctrl.image_path()
        image_display = getattr(self, "workflowImageDisplay", None)
        if image_display is not None:
            image_display.setText(self._compact_workflow_path(image_path))
            image_display.setToolTip(image_path or "No image selected")

        batch_path = setup_ctrl.batch_path()
        batch_display = getattr(self, "workflowBatchDisplay", None)
        if batch_display is not None:
            batch_display.setText(self._compact_workflow_path(batch_path))
            batch_display.setToolTip(batch_path or "No batch folder selected")

        source_combo = getattr(setup_ctrl, "sourceIdCombo", None)
        if source_combo is not None:
            text = source_combo.currentText().strip()
            source_combo.setToolTip(text or "No source selected")

    def _sync_workflow_source_mode(self, mode: str) -> None:
        """Switch the stable workflow source stack to match controller state."""
        setup_ctrl = getattr(self, "setup_panel_ctrl", None)
        stack = getattr(self, "workflowSourceStack", None)
        pages = getattr(self, "_workflow_source_pages", {})
        page_layouts = getattr(self, "_workflow_source_page_layouts", {})
        if setup_ctrl is None or stack is None:
            return
        if mode not in pages:
            mode = setup_ctrl.MODE_SINGLE
        self._sync_workflow_source_mode_buttons(mode)

        source_combo = getattr(setup_ctrl, "sourceIdCombo", None)
        if source_combo is not None:
            for layout in page_layouts.values():
                layout.removeWidget(source_combo)
            if mode == setup_ctrl.MODE_BATCH:
                page_layouts[mode].insertWidget(2, source_combo)
                source_combo.show()
            elif mode == setup_ctrl.MODE_CAMERA:
                page_layouts[mode].insertWidget(0, source_combo)
                source_combo.show()
            else:
                source_combo.hide()
                source_combo.setParent(self.workflowSourceStackHost)

        stack.setCurrentWidget(pages[mode])
        self._sync_workflow_source_displays()
        self.workflowSourceStackHost.updateGeometry()
        self.workflowSourceHost.updateGeometry()
        self.workflowBar.updateGeometry()
        self._schedule_workflow_preview_refresh()

    def _schedule_workflow_preview_refresh(self) -> None:
        """Refresh preview mapping after workflow bar geometry settles."""
        QTimer.singleShot(0, self._refresh_preview_after_workflow_layout_change)
        QTimer.singleShot(40, self._refresh_preview_after_workflow_layout_change)

    def _refresh_preview_after_workflow_layout_change(self) -> None:
        for widget_name in (
            "workflowBar",
            "workflowSourceHost",
            "workflowSourceStackHost",
        ):
            widget = getattr(self, widget_name, None)
            if widget is not None:
                widget.updateGeometry()
                layout = widget.layout()
                if layout is not None:
                    layout.activate()

        preview = getattr(self, "preview_panel", None)
        image_view = getattr(preview, "image_view", None)
        if image_view is None:
            return
        viewport = getattr(image_view, "viewport", None)
        if callable(viewport):
            viewport().update()
        if getattr(image_view, "_mode", None) != "fit":
            return
        schedule_fit = getattr(image_view, "_schedule_fit_to_window", None)
        if callable(schedule_fit):
            schedule_fit()
            return
        fit = getattr(image_view, "fit_to_window", None)
        if callable(fit):
            fit()

    def _add_workflow_label(self, layout: QLayout, text: str) -> None:
        label = QLabel(text, self.workflowBar)
        label.setObjectName(f"workflow{text}Label")
        label.setStyleSheet(
            f"color: {theme.TEXT_SECONDARY}; font-size: 11px; "
            "font-weight: 700; letter-spacing: 0px;"
        )
        label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        layout.addWidget(label)

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

                def handler():
                    return self.main_controller.select_camera(True)  # type: ignore

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

    def _set_guided_advanced_visible(self, visible: bool, save: bool = True) -> None:
        if visible:
            self._show_advanced_dialog()
        else:
            self._sync_advanced_buttons()

    def _sync_advanced_buttons(self) -> None:
        setup_ctrl = getattr(self, "setup_panel_ctrl", None)
        if setup_ctrl and hasattr(setup_ctrl, "set_advanced_visible"):
            setup_ctrl.set_advanced_visible(False, save=False)

    def _show_advanced_dialog(self) -> None:
        self._sync_advanced_buttons()
        dialog = getattr(self, "advanced_workflow_dialog", None)
        if dialog is None:
            return
        dialog.show()
        if hasattr(dialog, "show_controls"):
            dialog.show_controls()
        dialog.raise_()
        dialog.activateWindow()

    def _wire_layout_controls(self) -> None:
        self._splitter_sizes_full: list[int] | None = None
        self._workbench_splitter_sizes_full: list[int] | None = None
        self._preview_results_sizes_full: list[int] | None = None
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

        focus_actions = {
            "actionFocusMeasure": "measure",
            "actionFocusScience": "science",
            "actionFocusAnalysis": "analysis",
        }
        for action_name, mode in focus_actions.items():
            action = getattr(self, action_name, None)
            if not action:
                continue
            try:
                action.triggered.connect(
                    lambda _checked=False, m=mode: self._apply_focus_preset(m)
                )
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

    def _apply_focus_preset(self, mode: str) -> None:
        if not hasattr(self, "rootSplitter"):
            return
        results_panel = getattr(self, "results_panel_ctrl", None)

        if mode == "measure":
            self._set_scientific_test_available(False)
            if results_panel and hasattr(results_panel, "show_key_results"):
                results_panel.show_key_results()
            self._set_panel_visibility(True, False, True)
            self._apply_focus_sizes(
                preview_ratio=1.0, key_results_width=320, collapse_setup=False
            )
            self._select_inspect_tab("resultsTab")
            return

        if mode == "science":
            self._set_scientific_test_available(True)
            if results_panel:
                results_panel.set_compare_methods_visible(False, refresh=False)
                results_panel.set_diagnostics_visible(True)
            self._set_panel_visibility(False, True, True)
            has_residuals = bool(
                results_panel and results_panel.has_residuals_for_current_selection()
            )
            self._select_inspect_tab("residualsTab" if has_residuals else "resultsTab")
            self._apply_focus_sizes(preview_ratio=0.66, key_results_width=300)
            return

        if mode == "analysis":
            self._set_scientific_test_available(False)
            if results_panel:
                results_panel.set_compare_methods_visible(True, refresh=False)
                results_panel.set_diagnostics_visible(False)
            self._set_panel_visibility(False, True, False)
            self._select_inspect_tab("resultsTab")
            self._apply_focus_sizes(preview_ratio=0.32, key_results_width=0)

    def _apply_focus_sizes(
        self,
        *,
        preview_ratio: float,
        key_results_width: int,
        collapse_setup: bool = True,
    ) -> None:
        if collapse_setup:
            try:
                root_total = max(1, sum(self.rootSplitter.sizes()))  # type: ignore[attr-defined]
                self.rootSplitter.setSizes([0, root_total])  # type: ignore[attr-defined]
            except Exception:
                pass

        try:
            vertical_total = max(1, sum(self.workbenchSplitter.sizes()))  # type: ignore[attr-defined]
            preview_size = int(vertical_total * preview_ratio)
            inspector_size = vertical_total - preview_size
            self.workbenchSplitter.setSizes(  # type: ignore[attr-defined]
                [max(1, preview_size), max(0, inspector_size)]
            )
        except Exception:
            pass

        try:
            preview_total = max(1, sum(self.previewResultsSplitter.sizes()))  # type: ignore[attr-defined]
            key_width = min(max(0, key_results_width), preview_total - 1)
            self.previewResultsSplitter.setSizes(  # type: ignore[attr-defined]
                [max(1, preview_total - key_width), key_width]
            )
        except Exception:
            pass

    def _select_inspect_tab(self, tab_name: str) -> None:
        tabs = getattr(self, "inspectTabs", None)
        tab = getattr(self, tab_name, None)
        if tabs is None or tab is None:
            return
        try:
            index = tabs.indexOf(tab)
            if index >= 0:
                tabs.setCurrentIndex(index)
        except Exception:
            pass

    def _set_panel_visibility(
        self,
        show_setup: bool,
        show_inspector: bool,
        show_key_results: bool | None = None,
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
            if (
                len(vertical_sizes) >= 2
                and vertical_sizes[0] > 0
                and vertical_sizes[1] > 0
            ):
                self._workbench_splitter_sizes_full = vertical_sizes
        if (
            show_key_results
            and len(preview_results_sizes) >= 2
            and preview_results_sizes[1] > 0
        ):
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
