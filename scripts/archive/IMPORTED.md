# Imported modules per Python file

## __init__.py
- (no imports)

## __main__.py
- menipy.cli.main

## menipy\__init__.py
- gui

## menipy\__main__.py
- gui.app.main

## menipy\cli.py
- PIL.Image
- __future__.annotations
- argparse
- common.acquisition
- common.plugin_db.PluginDB
- common.plugins.discover_and_load_from_db
- common.plugins.discover_into_db
- common.plugins.load_active_plugins
- cv2
- json
- models.datatypes.EdgeDetectionSettings
- models.datatypes.PreprocessingSettings
- numpy
- pathlib.Path
- pipelines.base.Context
- pipelines.base.PipelineError
- pipelines.runner.PipelineRunner
- typing.Optional
- typing.Tuple

## menipy\cli\__init__.py
- PIL.Image
- __future__.annotations
- argparse
- cv2
- json
- menipy.common.acquisition
- menipy.common.plugin_db.PluginDB
- menipy.common.plugins.discover_into_db
- menipy.common.plugins.load_active_plugins
- menipy.pipelines.base.Context
- menipy.pipelines.base.PipelineBase
- menipy.pipelines.base.PipelineError
- menipy.pipelines.discover.PIPELINE_MAP
- numpy
- pathlib.Path
- typing.Optional

## menipy\common\__init__.py
- (no imports)

## menipy\common\acquisition.py
- __future__.annotations
- cv2
- numpy
- typing.Optional
- typing.Sequence

## menipy\common\edge_detection.py
- __future__.annotations
- acquisition
- cv2
- importlib.metadata.entry_points
- logging
- menipy.models.config.EdgeDetectionSettings
- menipy.models.geometry.Contour
- numpy
- registry.EDGE_DETECTORS
- typing.Callable
- typing.Optional

## menipy\common\geometry.py
- __future__.annotations
- cv2
- numpy
- numpy.linalg.lstsq
- scipy.optimize.minimize_scalar

## menipy\common\metrics.py
- numpy

## menipy\common\optimization.py
- __future__.annotations

## menipy\common\outputs.py
- __future__.annotations

## menipy\common\overlay.py
- __future__.annotations
- cv2
- numpy
- typing.Any
- typing.Dict
- typing.Iterable
- typing.Literal
- typing.Tuple
- typing.Union

## menipy\common\physics.py
- __future__.annotations

## menipy\common\plugin_db.py
- __future__.annotations
- pathlib.Path
- sqlite3
- typing.Iterable
- typing.Optional
- typing.Tuple

## menipy\common\plugin_loader.py
- __future__.annotations
- menipy.common.registry
- typing.Callable

## menipy\common\plugins.py
- __future__.annotations
- argparse
- importlib.util
- json
- logging
- pathlib.Path
- plugin_db.PluginDB
- registry.EDGE_DETECTORS
- registry.SOLVERS
- registry.register_edge
- registry.register_solver
- sys
- typing.Callable
- typing.Dict
- typing.Iterable
- typing.Optional

## menipy\common\preprocessing.py
- __future__.annotations
- cv2
- logging
- menipy.models.config.PreprocessingSettings
- menipy.models.context.Context
- menipy.models.state.MarkerSet
- menipy.models.state.PreprocessingStageRecord
- menipy.models.state.PreprocessingState
- numpy
- preprocessing_helpers.PreprocessingContext
- preprocessing_helpers.PreprocessingError
- preprocessing_helpers.apply_filter
- preprocessing_helpers.convert_to_grayscale
- preprocessing_helpers.crop_to_roi
- preprocessing_helpers.detect_contact_line
- preprocessing_helpers.fill_holes
- preprocessing_helpers.normalize_intensity
- preprocessing_helpers.rescale_roi
- preprocessing_helpers.subtract_background
- typing.Any
- typing.Optional

## menipy\common\preprocessing_helpers.py
- __future__.annotations
- cv2
- dataclasses.dataclass
- dataclasses.field
- logging
- menipy.models.config.PreprocessingSettings
- menipy.models.state.MarkerSet
- menipy.models.state.PreprocessingStageRecord
- menipy.models.state.PreprocessingState
- numpy
- scipy.ndimage
- scipy.ndimage.gaussian_filter
- scipy.ndimage.median_filter
- skimage.exposure
- skimage.morphology
- skimage.transform.resize
- typing.Any
- typing.Dict
- typing.Optional
- typing.Tuple

## menipy\common\registry.py
- __future__.annotations
- typing.Any
- typing.Callable
- typing.Dict

## menipy\common\scaling.py
- __future__.annotations

## menipy\common\solver.py
- __future__.annotations
- math
- menipy.models.fit.FitConfig
- numpy
- scipy.optimize.least_squares
- typing.Callable
- typing.Iterable
- typing.Literal
- typing.Optional
- typing.Sequence

## menipy\common\units.py
- pint.UnitRegistry
- pydantic_pint.set_registry

## menipy\common\validation.py
- __future__.annotations

## menipy\common\zold_detection.py
- (no imports)

## menipy\gui\__init__.py
- app.main

## menipy\gui\__main__.py
- app.main

## menipy\gui\app.py
- PySide6.QtCore.QCoreApplication
- PySide6.QtCore.QResource
- PySide6.QtCore.Qt
- PySide6.QtCore.qInstallMessageHandler
- PySide6.QtWidgets.QApplication
- PySide6.QtWidgets.QMessageBox
- __future__.annotations
- mainwindow.MainWindow
- pathlib.Path
- resources.app_rc
- resources.icons_rc
- resources.menipy_icons
- resources.menipy_icons_rc
- sys
- traceback

## menipy\gui\logging_bridge.py
- PySide6.QtCore.QObject
- PySide6.QtCore.Signal
- PySide6.QtWidgets.QPlainTextEdit
- __future__.annotations
- logging
- typing.Optional

## menipy\gui\main_controller.py
- PySide6.QtCore.QObject
- PySide6.QtCore.QPointF
- PySide6.QtCore.Slot
- PySide6.QtGui.QColor
- PySide6.QtGui.QImage
- PySide6.QtGui.QPainterPath
- PySide6.QtGui.QPen
- PySide6.QtWidgets.QDialog
- PySide6.QtWidgets.QFileDialog
- PySide6.QtWidgets.QGraphicsPathItem
- PySide6.QtWidgets.QMessageBox
- __future__.annotations
- cv2
- logging
- menipy.gui.controllers.pipeline_controller.PipelineController
- menipy.gui.controllers.setup_panel_controller.SetupPanelController
- menipy.gui.dialogs.acquisition_config_dialog.AcquisitionConfigDialog
- menipy.gui.dialogs.edge_detection_config_dialog.EdgeDetectionConfigDialog
- menipy.gui.dialogs.geometry_config_dialog.GeometryConfigDialog
- menipy.gui.dialogs.overlay_config_dialog.OverlayConfigDialog
- menipy.gui.dialogs.physics_config_dialog.PhysicsConfigDialog
- menipy.gui.dialogs.preprocessing_config_dialog.PreprocessingConfigDialog
- menipy.gui.mainwindow.MainWindow
- menipy.gui.panels.preview_panel.PreviewPanel
- menipy.gui.panels.results_panel.ResultsPanel
- menipy.gui.services.camera_service.CameraConfig
- menipy.gui.services.camera_service.CameraController
- menipy.models.config.PhysicsParams
- numpy
- pathlib.Path
- typing.Optional
- typing.TYPE_CHECKING

## menipy\gui\main_window.py
- PySide6.QtWidgets.QMainWindow
- PySide6.QtWidgets.QMessageBox
- importlib
- logging
- menipy.gui.base_window.BaseMainWindow

## menipy\gui\mainwindow.py
- PySide6.QtCore.QByteArray
- PySide6.QtCore.QFile
- PySide6.QtCore.Qt
- PySide6.QtGui.QCloseEvent
- PySide6.QtUiTools.QUiLoader
- PySide6.QtWidgets.QLayout
- PySide6.QtWidgets.QMainWindow
- PySide6.QtWidgets.QPlainTextEdit
- PySide6.QtWidgets.QVBoxLayout
- PySide6.QtWidgets.QWidget
- __future__.annotations
- logging
- menipy.gui.controllers.edge_detection_controller.EdgeDetectionPipelineController
- menipy.gui.controllers.pipeline_controller.PipelineController
- menipy.gui.controllers.preprocessing_controller.PreprocessingPipelineController
- menipy.gui.controllers.setup_panel_controller.SetupPanelController
- menipy.gui.helpers.image_marking.ImageMarkerHelper
- menipy.gui.logging_bridge.install_qt_logging
- menipy.gui.main_controller.MainController
- menipy.gui.panels.preview_panel.PreviewPanel
- menipy.gui.panels.results_panel.ResultsPanel
- menipy.gui.plugins_panel.PluginsController
- menipy.gui.services.camera_service.CameraConfig
- menipy.gui.services.camera_service.CameraController
- menipy.gui.services.pipeline_runner.PipelineRunner
- menipy.gui.viewmodels.run_vm.RunViewModel
- menipy.gui.views.image_view.DRAW_NONE
- menipy.gui.views.ui_main_window.Ui_MainWindow
- menipy.pipelines.discover.PIPELINE_MAP
- pathlib.Path
- services.settings_service.AppSettings
- services.sop_service.SopService
- typing.List
- typing.Optional
- views.image_view.ImageView
- views.step_item_widget.StepItemWidget

## menipy\gui\overlay.py
- PySide6.QtCore.QPointF
- PySide6.QtCore.Qt
- PySide6.QtGui.QBrush
- PySide6.QtGui.QColor
- PySide6.QtGui.QImage
- PySide6.QtGui.QPainter
- PySide6.QtGui.QPen
- PySide6.QtGui.QPixmap
- numpy

## menipy\gui\plugins_panel.py
- PySide6.QtCore.Qt
- PySide6.QtGui.QAction
- PySide6.QtWidgets.QDialog
- PySide6.QtWidgets.QDockWidget
- PySide6.QtWidgets.QFileDialog
- PySide6.QtWidgets.QLabel
- PySide6.QtWidgets.QMainWindow
- PySide6.QtWidgets.QMenu
- PySide6.QtWidgets.QMenuBar
- PySide6.QtWidgets.QMessageBox
- __future__.annotations
- menipy.common.plugins.PluginDB
- menipy.common.plugins.discover_into_db
- menipy.common.plugins.load_active_plugins
- menipy.gui.dialogs.plugin_manager_dialog.PluginManagerDialog
- menipy.gui.viewmodels.plugins_vm.PluginsViewModel
- pathlib.Path
- typing.Any
- typing.Optional
- typing.Sequence

## menipy\gui\zold_drawing_alt.py
- (no imports)

## menipy\gui\controllers\edge_detection_controller.py
- PySide6.QtCore.QObject
- PySide6.QtCore.Signal
- __future__.annotations
- cv2
- logging
- menipy.common.edge_detection
- menipy.common.geometry.find_contact_points_from_contour
- menipy.models.config.EdgeDetectionSettings
- menipy.models.context.Context
- numpy
- typing.Any
- typing.Dict
- typing.Optional

## menipy\gui\controllers\pipeline_controller.py
- PySide6.QtWidgets.QMainWindow
- PySide6.QtWidgets.QMessageBox
- PySide6.QtWidgets.QPlainTextEdit
- __future__.annotations
- importlib
- logging
- menipy.gui.controllers.edge_detection_controller.EdgeDetectionPipelineController
- menipy.gui.controllers.preprocessing_controller.PreprocessingPipelineController
- menipy.models.config.PhysicsParams
- menipy.models.context.Context
- menipy.pipelines.discover.PIPELINE_MAP
- typing.Any
- typing.Dict
- typing.Mapping
- typing.Optional

## menipy\gui\controllers\preprocessing_controller.py
- PySide6.QtCore.QObject
- PySide6.QtCore.Signal
- __future__.annotations
- logging
- menipy.common.preprocessing
- menipy.models.config.PreprocessingSettings
- menipy.models.context.Context
- menipy.models.state.MarkerSet
- menipy.models.state.PreprocessingState
- numpy
- typing.Any
- typing.Dict
- typing.Optional
- typing.Tuple

## menipy\gui\controllers\setup_panel_controller.py
- PySide6.QtCore.QObject
- PySide6.QtCore.QTimer
- PySide6.QtCore.Signal
- PySide6.QtWidgets.QButtonGroup
- PySide6.QtWidgets.QComboBox
- PySide6.QtWidgets.QLineEdit
- PySide6.QtWidgets.QListWidget
- PySide6.QtWidgets.QMainWindow
- PySide6.QtWidgets.QPushButton
- PySide6.QtWidgets.QRadioButton
- PySide6.QtWidgets.QSpinBox
- PySide6.QtWidgets.QToolButton
- PySide6.QtWidgets.QWidget
- __future__.annotations
- menipy.gui.controllers.sop_controller.SopController
- menipy.gui.views.image_view.DRAW_LINE
- menipy.gui.views.image_view.DRAW_POINT
- menipy.gui.views.image_view.DRAW_RECT
- menipy.pipelines.discover.PIPELINE_MAP
- pathlib.Path
- typing.Any
- typing.Optional
- typing.Sequence

## menipy\gui\controllers\sop_controller.py
- PySide6.QtWidgets.QComboBox
- PySide6.QtWidgets.QInputDialog
- PySide6.QtWidgets.QListWidget
- PySide6.QtWidgets.QListWidgetItem
- PySide6.QtWidgets.QMessageBox
- PySide6.QtWidgets.QWidget
- __future__.annotations
- menipy.gui.services.sop_service.Sop
- typing.Any
- typing.Callable
- typing.Optional
- typing.Sequence

## menipy\gui\dialogs\acquisition_config_dialog.py
- PySide6.QtWidgets.QCheckBox
- PySide6.QtWidgets.QDialog
- PySide6.QtWidgets.QDialogButtonBox
- PySide6.QtWidgets.QLabel
- PySide6.QtWidgets.QVBoxLayout
- __future__.annotations

## menipy\gui\dialogs\edge_detection_config_dialog.py
- PySide6.QtCore.Qt
- PySide6.QtCore.Signal
- PySide6.QtGui.QImage
- PySide6.QtGui.QPixmap
- PySide6.QtWidgets.QCheckBox
- PySide6.QtWidgets.QComboBox
- PySide6.QtWidgets.QDialog
- PySide6.QtWidgets.QDialogButtonBox
- PySide6.QtWidgets.QDoubleSpinBox
- PySide6.QtWidgets.QFormLayout
- PySide6.QtWidgets.QFrame
- PySide6.QtWidgets.QHBoxLayout
- PySide6.QtWidgets.QLabel
- PySide6.QtWidgets.QListWidget
- PySide6.QtWidgets.QListWidgetItem
- PySide6.QtWidgets.QPushButton
- PySide6.QtWidgets.QSpinBox
- PySide6.QtWidgets.QStackedWidget
- PySide6.QtWidgets.QVBoxLayout
- PySide6.QtWidgets.QWidget
- __future__.annotations
- menipy.models.config.EdgeDetectionSettings
- numpy
- typing.Optional

## menipy\gui\dialogs\geometry_config_dialog.py
- PySide6.QtCore.Qt
- PySide6.QtCore.Signal
- PySide6.QtGui.QImage
- PySide6.QtGui.QPixmap
- PySide6.QtWidgets.QApplication
- PySide6.QtWidgets.QCheckBox
- PySide6.QtWidgets.QComboBox
- PySide6.QtWidgets.QDialog
- PySide6.QtWidgets.QDialogButtonBox
- PySide6.QtWidgets.QDoubleSpinBox
- PySide6.QtWidgets.QFormLayout
- PySide6.QtWidgets.QHBoxLayout
- PySide6.QtWidgets.QLabel
- PySide6.QtWidgets.QPushButton
- PySide6.QtWidgets.QSpinBox
- PySide6.QtWidgets.QVBoxLayout
- PySide6.QtWidgets.QWidget
- __future__.annotations
- numpy
- sys
- typing.Any
- typing.Dict

## menipy\gui\dialogs\overlay_config_dialog.py
- PySide6.QtCore.Qt
- PySide6.QtCore.Signal
- PySide6.QtGui.QColor
- PySide6.QtGui.QImage
- PySide6.QtGui.QPixmap
- PySide6.QtWidgets.QApplication
- PySide6.QtWidgets.QCheckBox
- PySide6.QtWidgets.QColorDialog
- PySide6.QtWidgets.QDialog
- PySide6.QtWidgets.QDialogButtonBox
- PySide6.QtWidgets.QDoubleSpinBox
- PySide6.QtWidgets.QFormLayout
- PySide6.QtWidgets.QHBoxLayout
- PySide6.QtWidgets.QLabel
- PySide6.QtWidgets.QPushButton
- PySide6.QtWidgets.QSpinBox
- PySide6.QtWidgets.QVBoxLayout
- PySide6.QtWidgets.QWidget
- __future__.annotations
- numpy
- sys
- typing.Any
- typing.Dict
- typing.Optional

## menipy\gui\dialogs\physics_config_dialog.py
- PySide6.QtWidgets.QDialog
- PySide6.QtWidgets.QDialogButtonBox
- PySide6.QtWidgets.QFormLayout
- PySide6.QtWidgets.QLineEdit
- PySide6.QtWidgets.QMessageBox
- PySide6.QtWidgets.QVBoxLayout
- __future__.annotations
- menipy.models.config.PhysicsParams
- typing.Optional

## menipy\gui\dialogs\plugin_manager_dialog.py
- PySide6.QtCore.QFile
- PySide6.QtCore.Qt
- PySide6.QtUiTools.QUiLoader
- PySide6.QtWidgets.QAbstractItemView
- PySide6.QtWidgets.QDialog
- PySide6.QtWidgets.QMessageBox
- PySide6.QtWidgets.QTableWidgetItem
- __future__.annotations
- menipy.gui.services.settings_service.AppSettings
- menipy.gui.viewmodels.plugins_vm.PluginsViewModel
- pathlib.Path
- typing.Sequence

## menipy\gui\dialogs\preprocessing_config_dialog.py
- PySide6.QtCore.Qt
- PySide6.QtCore.Signal
- PySide6.QtCore.Slot
- PySide6.QtGui.QImage
- PySide6.QtGui.QPixmap
- PySide6.QtWidgets.QCheckBox
- PySide6.QtWidgets.QComboBox
- PySide6.QtWidgets.QDialog
- PySide6.QtWidgets.QDialogButtonBox
- PySide6.QtWidgets.QDoubleSpinBox
- PySide6.QtWidgets.QFormLayout
- PySide6.QtWidgets.QFrame
- PySide6.QtWidgets.QHBoxLayout
- PySide6.QtWidgets.QLabel
- PySide6.QtWidgets.QListWidget
- PySide6.QtWidgets.QListWidgetItem
- PySide6.QtWidgets.QPushButton
- PySide6.QtWidgets.QSpinBox
- PySide6.QtWidgets.QStackedWidget
- PySide6.QtWidgets.QVBoxLayout
- PySide6.QtWidgets.QWidget
- __future__.annotations
- menipy.models.config.PreprocessingSettings
- numpy
- typing.Optional

## menipy\gui\helpers\image_marking.py
- PySide6.QtCore.QObject
- PySide6.QtCore.QPointF
- PySide6.QtCore.Qt
- PySide6.QtGui.QColor
- __future__.annotations
- dataclasses.dataclass
- dataclasses.field
- logging
- menipy.gui.controllers.preprocessing_controller.PreprocessingPipelineController
- menipy.gui.panels.preview_panel.PreviewPanel
- numpy
- typing.List
- typing.Optional
- typing.Tuple

## menipy\gui\panels\__init__.py
- (no imports)

## menipy\gui\panels\discover.py
- (no imports)

## menipy\gui\panels\preview_panel.py
- PySide6.QtCore.QLineF
- PySide6.QtCore.QRectF
- PySide6.QtCore.Signal
- PySide6.QtGui.QColor
- PySide6.QtGui.QPixmap
- PySide6.QtWidgets.QPushButton
- PySide6.QtWidgets.QToolButton
- PySide6.QtWidgets.QWidget
- __future__.annotations
- logging
- menipy.gui.views.image_view.DRAW_LINE
- menipy.gui.views.image_view.DRAW_RECT
- pathlib.Path
- typing.Any
- typing.Callable
- typing.Optional

## menipy\gui\panels\results_panel.py
- PySide6.QtWidgets.QTableWidget
- PySide6.QtWidgets.QTableWidgetItem
- PySide6.QtWidgets.QWidget
- __future__.annotations
- typing.Any
- typing.Mapping
- typing.Optional

## menipy\gui\panels\setup_panel.py
- PySide6.QtCore.QObject
- PySide6.QtCore.Signal
- PySide6.QtWidgets.QButtonGroup
- PySide6.QtWidgets.QComboBox
- PySide6.QtWidgets.QLineEdit
- PySide6.QtWidgets.QListWidget
- PySide6.QtWidgets.QMainWindow
- PySide6.QtWidgets.QPushButton
- PySide6.QtWidgets.QRadioButton
- PySide6.QtWidgets.QSpinBox
- PySide6.QtWidgets.QToolButton
- PySide6.QtWidgets.QWidget
- __future__.annotations
- menipy.gui.sop_controller.SopController
- menipy.gui.views.image_view.DRAW_LINE
- menipy.gui.views.image_view.DRAW_POINT
- menipy.gui.views.image_view.DRAW_RECT
- menipy.pipelines.discover.PIPELINE_MAP
- pathlib.Path
- typing.Any
- typing.Optional
- typing.Sequence

## menipy\gui\resources\menipy_icons_rc.py
- PySide6.QtCore

## menipy\gui\services\camera_service.py
- PySide6.QtCore.QObject
- PySide6.QtCore.QThread
- PySide6.QtCore.QTimer
- PySide6.QtCore.Qt
- PySide6.QtCore.Signal
- PySide6.QtCore.Slot
- __future__.annotations
- cv2
- dataclasses.dataclass
- typing.Optional

## menipy\gui\services\image_convert.py
- PySide6.QtGui.QImage
- PySide6.QtGui.QPixmap
- numpy

## menipy\gui\services\pipeline_runner.py
- PySide6.QtCore.QObject
- PySide6.QtCore.QRunnable
- PySide6.QtCore.QThreadPool
- PySide6.QtCore.Signal
- __future__.annotations
- menipy.common.acquisition
- menipy.pipelines.base.PipelineBase
- menipy.pipelines.base.PipelineError
- menipy.pipelines.discover.PIPELINE_MAP
- typing.Optional

## menipy\gui\services\plugin_service.py
- __future__.annotations
- menipy.common.plugin_db.PluginDB
- menipy.common.plugins.discover_into_db
- menipy.common.plugins.load_active_plugins
- pathlib.Path
- typing.Sequence

## menipy\gui\services\settings_service.py
- __future__.annotations
- dataclasses.asdict
- dataclasses.dataclass
- dataclasses.field
- json
- pathlib.Path
- typing.List
- typing.Optional

## menipy\gui\services\sop_service.py
- __future__.annotations
- dataclasses.asdict
- dataclasses.dataclass
- json
- pathlib.Path
- typing.Dict
- typing.List
- typing.Optional

## menipy\gui\viewmodels\plugins_vm.py
- PySide6.QtCore.QObject
- PySide6.QtCore.Signal
- __future__.annotations
- menipy.gui.services.plugin_service.PluginService
- menipy.gui.viewmodels.run_vm

## menipy\gui\viewmodels\results_vm.py
- (no imports)

## menipy\gui\viewmodels\run_vm.py
- PySide6.QtCore.QObject
- PySide6.QtCore.Signal
- __future__.annotations
- menipy.gui.services.image_convert.to_pixmap
- menipy.gui.services.pipeline_runner.PipelineRunner

## menipy\gui\views\__init__.py
- (no imports)

## menipy\gui\views\image_view.py
- PySide6.QtCore.QLineF
- PySide6.QtCore.QPointF
- PySide6.QtCore.QRectF
- PySide6.QtCore.QSizeF
- PySide6.QtCore.Qt
- PySide6.QtCore.Signal
- PySide6.QtGui.QColor
- PySide6.QtGui.QImage
- PySide6.QtGui.QPainter
- PySide6.QtGui.QPainterPath
- PySide6.QtGui.QPen
- PySide6.QtGui.QPixmap
- PySide6.QtGui.QTransform
- PySide6.QtWidgets.QGraphicsDropShadowEffect
- PySide6.QtWidgets.QGraphicsEllipseItem
- PySide6.QtWidgets.QGraphicsLineItem
- PySide6.QtWidgets.QGraphicsPixmapItem
- PySide6.QtWidgets.QGraphicsRectItem
- PySide6.QtWidgets.QGraphicsScene
- PySide6.QtWidgets.QGraphicsView
- __future__.annotations
- logging
- numpy
- typing.Optional
- typing.Union

## menipy\gui\views\step_item_widget.py
- PySide6.QtCore.Qt
- PySide6.QtCore.Signal
- PySide6.QtGui.QIcon
- PySide6.QtWidgets.QHBoxLayout
- PySide6.QtWidgets.QLabel
- PySide6.QtWidgets.QSizePolicy
- PySide6.QtWidgets.QToolButton
- PySide6.QtWidgets.QWidget
- __future__.annotations
- pathlib.Path
- typing.Optional

## menipy\gui\views\ui_main_window.py
- PySide6.QtCore.QCoreApplication
- PySide6.QtCore.QDate
- PySide6.QtCore.QDateTime
- PySide6.QtCore.QLocale
- PySide6.QtCore.QMetaObject
- PySide6.QtCore.QObject
- PySide6.QtCore.QPoint
- PySide6.QtCore.QRect
- PySide6.QtCore.QSize
- PySide6.QtCore.QTime
- PySide6.QtCore.QUrl
- PySide6.QtCore.Qt
- PySide6.QtGui.QAction
- PySide6.QtGui.QBrush
- PySide6.QtGui.QColor
- PySide6.QtGui.QConicalGradient
- PySide6.QtGui.QCursor
- PySide6.QtGui.QFont
- PySide6.QtGui.QFontDatabase
- PySide6.QtGui.QGradient
- PySide6.QtGui.QIcon
- PySide6.QtGui.QImage
- PySide6.QtGui.QKeySequence
- PySide6.QtGui.QLinearGradient
- PySide6.QtGui.QPainter
- PySide6.QtGui.QPalette
- PySide6.QtGui.QPixmap
- PySide6.QtGui.QRadialGradient
- PySide6.QtGui.QTransform
- PySide6.QtWidgets.QApplication
- PySide6.QtWidgets.QMainWindow
- PySide6.QtWidgets.QMenu
- PySide6.QtWidgets.QMenuBar
- PySide6.QtWidgets.QSizePolicy
- PySide6.QtWidgets.QSplitter
- PySide6.QtWidgets.QStatusBar
- PySide6.QtWidgets.QTabWidget
- PySide6.QtWidgets.QVBoxLayout
- PySide6.QtWidgets.QWidget

## menipy\math\jurin.py
- (no imports)

## menipy\math\rayleigh_lamb.py
- (no imports)

## menipy\math\young_laplace.py
- (no imports)

## menipy\models\__init__.py
- drop_extras.apex_curvature_m_inv
- drop_extras.apparent_weight_mN
- drop_extras.projected_area_mm2
- drop_extras.surface_area_mm2
- drop_extras.vmax_uL
- drop_extras.worthington_number
- menipy.common.geometry.fit_circle
- physics.solve_young_laplace
- properties.contact_angle_from_mask
- properties.droplet_volume
- properties.estimate_surface_tension
- surface_tension.bond_number
- surface_tension.jennings_pallas_beta
- surface_tension.surface_tension
- surface_tension.volume_from_contour

## menipy\models\config.py
- __future__.annotations
- pydantic.BaseModel
- pydantic.ConfigDict
- pydantic.Field
- pydantic.field_validator
- typing.Literal
- typing.Optional
- unit_types.Density
- unit_types.Length
- unit_types.SurfaceTension

## menipy\models\context.py
- __future__.annotations
- config.EdgeDetectionSettings
- config.PreprocessingSettings
- fit.Fit
- frame.Frame
- geometry.Contour
- geometry.Geometry
- pydantic.BaseModel
- pydantic.Field
- result.Result
- state.MarkerSet
- typing.Any
- typing.Dict
- typing.List
- typing.Optional
- typing.Tuple

## menipy\models\datatypes.py
- __future__.annotations
- dataclasses.dataclass
- dataclasses.field
- datetime.datetime
- numpy
- pydantic.BaseModel
- pydantic.ConfigDict
- pydantic.Field
- pydantic.field_validator
- result.CapillaryRiseFit
- result.OscillationFit
- result.YoungLaplaceFit
- typing.Any
- typing.ContourArray
- typing.Dict
- typing.FloatVec
- typing.ImageAny
- typing.List
- typing.Literal
- typing.Optional
- typing.Tuple
- typing.Union

## menipy\models\drop_extras.py
- math
- numpy

## menipy\models\fit.py
- __future__.annotations
- dataclasses.dataclass
- dataclasses.field
- math
- pydantic.BaseModel
- pydantic.Field
- pydantic_numpy.typing.Np1DArrayFp64
- pydantic_numpy.typing.NpNDArrayFp64
- typing.FloatVec
- typing.List
- typing.Literal
- typing.Optional
- typing.Tuple

## menipy\models\frame.py
- __future__.annotations
- datetime.datetime
- numpy
- pydantic.BaseModel
- pydantic.Field
- pydantic.field_validator
- pydantic_numpy.typing.Np2DArrayUint8
- pydantic_numpy.typing.Np3DArrayUint8
- typing.Optional
- typing.Tuple
- typing.Union

## menipy\models\geometry.py
- __future__.annotations
- numpy
- pydantic.BaseModel
- pydantic.Field
- pydantic.field_validator
- typing.ContourArray
- typing.Literal
- typing.Optional
- typing.Tuple

## menipy\models\physics.py
- numpy
- scipy.integrate.solve_ivp
- typing.Callable

## menipy\models\properties.py
- __future__.annotations
- cv2
- menipy.common.geometry.fit_circle
- numpy
- typing.Optional

## menipy\models\result.py
- __future__.annotations
- config.PhysicsParams
- datetime.datetime
- fit.Confidence
- fit.Residuals
- fit.SolverInfo
- frame.Calibration
- frame.Frame
- geometry.Contour
- geometry.Geometry
- pydantic.BaseModel
- pydantic.Field
- pydantic_numpy.typing.Np1DArrayFp64
- typing.List
- typing.Literal
- typing.Optional
- typing.Union

## menipy\models\state.py
- __future__.annotations
- datetime.datetime
- frame.Calibration
- frame.CameraMeta
- frame.Frame
- geometry.Contour
- numpy
- pydantic.BaseModel
- pydantic.ConfigDict
- pydantic.Field
- result.CapillaryRiseFit
- result.OscillationFit
- result.YoungLaplaceFit
- typing.Any
- typing.ContourArray
- typing.Dict
- typing.FloatVec
- typing.ImageAny
- typing.List
- typing.Literal
- typing.Optional
- typing.Tuple

## menipy\models\surface_tension.py
- __future__.annotations
- numpy
- numpy.typing.NDArray

## menipy\models\typing.py
- __future__.annotations
- numpy.typing
- pydantic_numpy.typing.Np1DArrayFp64
- pydantic_numpy.typing.Np2DArrayFp64
- pydantic_numpy.typing.Np2DArrayUint8
- pydantic_numpy.typing.Np3DArrayUint8
- typing.Union

## menipy\models\unit_types.py
- pint.facets.plain.PlainQuantity
- pydantic_pint.PydanticPintQuantity
- typing.Annotated

## menipy\pipelines\__init__.py
- (no imports)

## menipy\pipelines\base.py
- __future__.annotations
- logging
- menipy.common.edge_detection
- menipy.common.overlay
- menipy.common.plugins._load_module_from_path
- menipy.common.preprocessing
- menipy.common.solver
- menipy.models.config.EdgeDetectionSettings
- menipy.models.config.PreprocessingSettings
- menipy.models.context.Context
- menipy.models.fit.FitConfig
- menipy.models.frame.Frame
- menipy.models.geometry.Contour
- menipy.models.geometry.Geometry
- numpy
- pathlib.Path
- time
- typing.Any
- typing.Callable
- typing.ClassVar
- typing.Dict
- typing.Optional
- typing.Union

## menipy\pipelines\discover.py
- importlib
- menipy.pipelines.base.PipelineBase
- pathlib.Path

## menipy\pipelines\runner.py
- __future__.annotations
- base.PipelineBase
- base.PipelineError
- discover.PIPELINE_MAP
- menipy.models.config.EdgeDetectionSettings
- menipy.models.config.PreprocessingSettings
- menipy.models.context.Context
- typing.Any
- typing.Optional

## menipy\pipelines\capillary_rise\__init__.py
- stages.CapillaryRisePipeline

## menipy\pipelines\capillary_rise\acquisition.py
- (no imports)

## menipy\pipelines\capillary_rise\edge_detection.py
- (no imports)

## menipy\pipelines\capillary_rise\geometry.py
- (no imports)

## menipy\pipelines\capillary_rise\optimization.py
- (no imports)

## menipy\pipelines\capillary_rise\outputs.py
- (no imports)

## menipy\pipelines\capillary_rise\overlay.py
- (no imports)

## menipy\pipelines\capillary_rise\physics.py
- (no imports)

## menipy\pipelines\capillary_rise\preprocessing.py
- (no imports)

## menipy\pipelines\capillary_rise\scaling.py
- (no imports)

## menipy\pipelines\capillary_rise\solver.py
- (no imports)

## menipy\pipelines\capillary_rise\stages.py
- __future__.annotations
- menipy.common.edge_detection
- menipy.common.overlay
- menipy.common.plugins._load_module_from_path
- menipy.common.solver
- menipy.models.config.EdgeDetectionSettings
- menipy.models.context.Context
- menipy.models.fit.FitConfig
- menipy.pipelines.base.PipelineBase
- numpy
- pathlib.Path
- typing.Optional

## menipy\pipelines\capillary_rise\validation.py
- (no imports)

## menipy\pipelines\captive_bubble\__init__.py
- stages.CaptiveBubblePipeline

## menipy\pipelines\captive_bubble\acquisition.py
- (no imports)

## menipy\pipelines\captive_bubble\edge_detection.py
- (no imports)

## menipy\pipelines\captive_bubble\geometry.py
- (no imports)

## menipy\pipelines\captive_bubble\optimization.py
- (no imports)

## menipy\pipelines\captive_bubble\outputs.py
- (no imports)

## menipy\pipelines\captive_bubble\overlay.py
- (no imports)

## menipy\pipelines\captive_bubble\physics.py
- (no imports)

## menipy\pipelines\captive_bubble\preprocessing.py
- (no imports)

## menipy\pipelines\captive_bubble\scaling.py
- (no imports)

## menipy\pipelines\captive_bubble\solver.py
- (no imports)

## menipy\pipelines\captive_bubble\stages.py
- __future__.annotations
- menipy.common.plugins._load_module_from_path
- menipy.models.geometry.CaptiveBubbleGeometry
- menipy.pipelines.base.Context
- menipy.pipelines.base.EdgeDetectionSettings
- menipy.pipelines.base.FitConfig
- menipy.pipelines.base.PipelineBase
- menipy.pipelines.base.common_solver
- menipy.pipelines.base.edged
- menipy.pipelines.base.ovl
- numpy
- pathlib.Path
- typing.Optional

## menipy\pipelines\captive_bubble\validation.py
- (no imports)

## menipy\pipelines\oscillating\__init__.py
- stages.OscillatingPipeline

## menipy\pipelines\oscillating\acquisition.py
- (no imports)

## menipy\pipelines\oscillating\edge_detection.py
- (no imports)

## menipy\pipelines\oscillating\geometry.py
- (no imports)

## menipy\pipelines\oscillating\optimization.py
- (no imports)

## menipy\pipelines\oscillating\outputs.py
- (no imports)

## menipy\pipelines\oscillating\overlay.py
- (no imports)

## menipy\pipelines\oscillating\physics.py
- (no imports)

## menipy\pipelines\oscillating\preprocessing.py
- (no imports)

## menipy\pipelines\oscillating\scaling.py
- (no imports)

## menipy\pipelines\oscillating\solver.py
- (no imports)

## menipy\pipelines\oscillating\stages.py
- __future__.annotations
- menipy.common.edge_detection
- menipy.common.overlay
- menipy.common.plugins._load_module_from_path
- menipy.common.solver
- menipy.models.context.Context
- menipy.models.fit.FitConfig
- menipy.pipelines.base.PipelineBase
- numpy
- pathlib.Path
- typing.List
- typing.Optional

## menipy\pipelines\oscillating\validation.py
- (no imports)

## menipy\pipelines\pendant\__init__.py
- stages.PendantPipeline

## menipy\pipelines\pendant\acquisition.py
- (no imports)

## menipy\pipelines\pendant\drawing.py
- __future__.annotations
- geometry.PendantMetrics
- menipy.gui.overlay.draw_analysis_overlay

## menipy\pipelines\pendant\edge_detection.py
- (no imports)

## menipy\pipelines\pendant\geometry.py
- __future__.annotations
- dataclasses.dataclass
- menipy.common.edge_detection.extract_external_contour
- menipy.common.metrics.find_apex_index
- metrics.compute_pendant_metrics
- numpy

## menipy\pipelines\pendant\metrics.py
- __future__.annotations
- menipy.common.geometry.fit_circle
- menipy.models.drop_extras.surface_area_mm2
- menipy.models.surface_tension.jennings_pallas_beta
- menipy.models.surface_tension.surface_tension
- menipy.models.surface_tension.volume_from_contour
- numpy

## menipy\pipelines\pendant\optimization.py
- (no imports)

## menipy\pipelines\pendant\outputs.py
- (no imports)

## menipy\pipelines\pendant\overlay.py
- (no imports)

## menipy\pipelines\pendant\physics.py
- (no imports)

## menipy\pipelines\pendant\preprocessing.py
- (no imports)

## menipy\pipelines\pendant\scaling.py
- (no imports)

## menipy\pipelines\pendant\solver.py
- (no imports)

## menipy\pipelines\pendant\stages.py
- __future__.annotations
- menipy.common.edge_detection
- menipy.common.overlay
- menipy.common.plugins._load_module_from_path
- menipy.common.solver
- menipy.models.config.EdgeDetectionSettings
- menipy.models.context.Context
- menipy.models.fit.FitConfig
- menipy.models.geometry.Contour
- menipy.models.geometry.Geometry
- menipy.pipelines.base.PipelineBase
- numpy
- pathlib.Path
- typing.Optional

## menipy\pipelines\pendant\validation.py
- (no imports)

## menipy\pipelines\sessile\__init__.py
- stages.SessilePipeline

## menipy\pipelines\sessile\acquisition.py
- (no imports)

## menipy\pipelines\sessile\drawing.py
- __future__.annotations
- geometry.SessileMetrics
- menipy.gui.overlay.draw_analysis_overlay

## menipy\pipelines\sessile\edge_detection.py
- (no imports)

## menipy\pipelines\sessile\geometry.py
- __future__.annotations
- dataclasses.dataclass
- menipy.common.edge_detection.extract_external_contour
- menipy.common.metrics.find_apex_index
- menipy.pipelines.base.PipelineBase
- metrics.compute_sessile_metrics
- numpy

## menipy\pipelines\sessile\metrics.py
- __future__.annotations
- menipy.common.geometry.circle_fit_angle_at_point
- menipy.common.geometry.detect_baseline_ransac
- menipy.common.geometry.estimate_contact_angle_circle_fit
- menipy.common.geometry.estimate_contact_angle_tangent
- menipy.common.geometry.find_contact_points_from_contour
- menipy.common.geometry.refine_apex_curvature
- menipy.common.geometry.tangent_angle_at_point
- menipy.models.drop_extras.surface_area_mm2
- menipy.models.surface_tension.volume_from_contour
- numpy

## menipy\pipelines\sessile\optimization.py
- (no imports)

## menipy\pipelines\sessile\outputs.py
- (no imports)

## menipy\pipelines\sessile\overlay.py
- (no imports)

## menipy\pipelines\sessile\physics.py
- (no imports)

## menipy\pipelines\sessile\preprocessing.py
- (no imports)

## menipy\pipelines\sessile\scaling.py
- (no imports)

## menipy\pipelines\sessile\solver.py
- (no imports)

## menipy\pipelines\sessile\stages.py
- __future__.annotations
- cv2
- logging
- menipy.common.edge_detection
- menipy.common.overlay
- menipy.common.plugins._load_module_from_path
- menipy.common.solver
- menipy.models.config.EdgeDetectionSettings
- menipy.models.context.Context
- menipy.models.fit.FitConfig
- menipy.models.geometry.Contour
- menipy.models.geometry.Geometry
- menipy.pipelines.base.PipelineBase
- metrics.compute_sessile_metrics
- numpy
- pathlib.Path
- typing.Optional

## menipy\pipelines\sessile\validation.py
- (no imports)

## menipy\viz\plots.py
- (no imports)
