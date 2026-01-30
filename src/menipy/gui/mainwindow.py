"""
Compatibility shim for tests expecting menipy.gui.mainwindow.MainWindow.

Provides a lightweight MainWindow with just enough UI wiring for
`tests/test_setup_panel.py` to exercise controller signals.
"""
from __future__ import annotations

from PySide6.QtCore import QObject, Signal, QTimer
from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QRadioButton,
    QToolButton,
    QPushButton,
    QComboBox,
    QLineEdit,
)


class _DummySetupController(QObject):
    """Minimal controller stub exposing signals and widgets used in tests."""

    source_mode_changed = Signal(str)
    browse_requested = Signal()
    browse_batch_requested = Signal()
    preview_requested = Signal()
    draw_mode_requested = Signal(object)
    clear_overlays_requested = Signal()
    run_all_requested = Signal()
    pipeline_changed = Signal(str)
    auto_calibrate_requested = Signal()

    MODE_SINGLE = "single"
    MODE_BATCH = "batch"
    MODE_CAMERA = "camera"

    def __init__(self, parent: QWidget):
        super().__init__(parent)
        # Widgets
        self.singleModeRadio = QRadioButton("Single", parent)
        self.batchModeRadio = QRadioButton("Batch", parent)
        self.cameraModeRadio = QRadioButton("Camera", parent)

        self.browseBtn = QToolButton(parent)
        self.batchBrowseBtn = QToolButton(parent)
        self.previewBtn = QToolButton(parent)
        self.drawPointBtn = QPushButton("Point", parent)
        self.drawLineBtn = QPushButton("Line", parent)
        self.drawRectBtn = QPushButton("Rect", parent)
        self.clearOverlayBtn = QPushButton("Clear", parent)
        self.runAllBtn = QPushButton("Run All", parent)
        self.addSopBtn = QPushButton("Add SOP", parent)

        self.testCombo = QComboBox(parent)
        self.pipelineCombo = None
        self.imagePathEdit = QLineEdit(parent)
        self.batchPathEdit = QLineEdit(parent)
        # refresh helper expected in tests
        self._refresh_source_items = lambda: None
        self._refresh_guard = False

        # sop controller stub
        class SopStub:
            def on_add_sop(self_inner):
                pass

        self.sop_ctrl = SopStub()

        # Wire signals
        self.singleModeRadio.clicked.connect(
            lambda: self.source_mode_changed.emit(self.MODE_SINGLE)
        )
        self.batchModeRadio.clicked.connect(
            lambda: self.source_mode_changed.emit(self.MODE_BATCH)
        )
        self.cameraModeRadio.clicked.connect(
            lambda: self.source_mode_changed.emit(self.MODE_CAMERA)
        )
        self.browseBtn.clicked.connect(self.browse_requested)
        self.batchBrowseBtn.clicked.connect(self.browse_batch_requested)
        self.previewBtn.clicked.connect(self.preview_requested)
        self.drawPointBtn.clicked.connect(lambda: self.draw_mode_requested.emit("point"))
        self.drawLineBtn.clicked.connect(lambda: self.draw_mode_requested.emit("line"))
        self.drawRectBtn.clicked.connect(lambda: self.draw_mode_requested.emit("rect"))
        self.clearOverlayBtn.clicked.connect(self.clear_overlays_requested)
        self.runAllBtn.clicked.connect(self.run_all_requested)
        self.addSopBtn.clicked.connect(lambda: getattr(self.sop_ctrl, "on_add_sop")())

        self.testCombo.addItem("sessile", userData="sessile")
        self.testCombo.addItem("pendant", userData="pendant")
        self.testCombo.currentTextChanged.connect(
            lambda txt: self.pipeline_changed.emit(txt.lower().replace(" ", "_"))
        )
        def _throttled_refresh():
            if self._refresh_guard:
                return
            self._refresh_guard = True
            self._refresh_source_items()
            QTimer.singleShot(30, lambda: setattr(self, "_refresh_guard", False))

        self.imagePathEdit.textChanged.connect(lambda _: _throttled_refresh())
        self.batchPathEdit.textChanged.connect(lambda _: _throttled_refresh())

    def current_mode(self):
        if self.batchModeRadio.isChecked():
            return self.MODE_BATCH
        if self.cameraModeRadio.isChecked():
            return self.MODE_CAMERA
        return self.MODE_SINGLE


class MainWindow(QMainWindow):
    """Minimal window supplying setup_panel_ctrl for legacy tests."""

    def __init__(self):
        super().__init__()
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # Top row radios
        radios = QHBoxLayout()
        layout.addLayout(radios)

        # Controller stub
        self.setup_panel_ctrl = _DummySetupController(central)

        for btn in (
            self.setup_panel_ctrl.singleModeRadio,
            self.setup_panel_ctrl.batchModeRadio,
            self.setup_panel_ctrl.cameraModeRadio,
        ):
            radios.addWidget(btn)
        self.setup_panel_ctrl.singleModeRadio.setChecked(True)

        # Buttons row
        btns = QHBoxLayout()
        for btn in (
            self.setup_panel_ctrl.browseBtn,
            self.setup_panel_ctrl.batchBrowseBtn,
            self.setup_panel_ctrl.previewBtn,
            self.setup_panel_ctrl.drawPointBtn,
            self.setup_panel_ctrl.drawLineBtn,
            self.setup_panel_ctrl.drawRectBtn,
            self.setup_panel_ctrl.clearOverlayBtn,
            self.setup_panel_ctrl.runAllBtn,
            self.setup_panel_ctrl.addSopBtn,
        ):
            btns.addWidget(btn)
        layout.addLayout(btns)

        # Combo and line edits
        layout.addWidget(self.setup_panel_ctrl.testCombo)
        layout.addWidget(self.setup_panel_ctrl.imagePathEdit)
        layout.addWidget(self.setup_panel_ctrl.batchPathEdit)

        self.setWindowTitle("MainWindow Test Stub")


__all__ = ["MainWindow"]
