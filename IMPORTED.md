# Imported modules per Python file

## src/__init__.py
- (no imports)

## src/__main__.py
- menipy.gui.main, src/menipy/gui/__init__.py

## src/menipy/__init__.py
- (no imports)

## src/menipy/analysis/__init__.py
- commons.compute_drop_metrics, src/menipy/analysis/commons.py
- commons.find_apex_index, src/menipy/analysis/commons.py
- commons.extract_external_contour, src/menipy/analysis/commons.py
- plotting.save_contour_sides_image, src/menipy/analysis/plotting.py
- plotting.save_contour_side_profiles, src/menipy/analysis/plotting.py
- pendant.compute_metrics, src/menipy/analysis/pendant.py
- sessile.compute_metrics, src/menipy/analysis/sessile.py
- sessile_alt.compute_metrics, src/menipy/analysis/sessile_alt.py
- detection.needle.detect_vertical_edges, src/menipy/detection/needle.py

## src/menipy/analysis/commons.py
- __future__.annotations
- cv2
- numpy
- physics.contact_geom.line_params, src/menipy/physics/contact_geom.py
- physics.contact_geom.contour_line_intersections, src/menipy/physics/contact_geom.py
- models.geometry.horizontal_intersections, src/menipy/models/geometry.py

## src/menipy/analysis/drop.py
- cv2
- numpy
- models.geometry.horizontal_intersections, src/menipy/models/geometry.py
- physics.contact_geom.line_params, src/menipy/physics/contact_geom.py
- physics.contact_geom.contour_line_intersections, src/menipy/physics/contact_geom.py

## src/menipy/analysis/pendant.py
- __future__.annotations
- numpy
- commons.compute_drop_metrics, src/menipy/analysis/commons.py

## src/menipy/analysis/plotting.py
- __future__.annotations
- numpy
- cv2
- datetime.datetime
- pathlib.Path
- importlib
- matplotlib
- matplotlib.pyplot

## src/menipy/analysis/sessile.py
- __future__.annotations
- numpy
- commons.compute_drop_metrics, src/menipy/analysis/commons.py
- commons.find_apex_index, src/menipy/analysis/commons.py
- physics.contact_geom.geom_metrics, src/menipy/physics/contact_geom.py
- physics.contact_geom.line_params, src/menipy/physics/contact_geom.py
- scipy.interpolate.UnivariateSpline
- scipy.interpolate.CubicSpline

## src/menipy/analysis/sessile_alt.py
- __future__.annotations
- numpy
- commons.compute_drop_metrics, src/menipy/analysis/commons.py
- commons.find_apex_index, src/menipy/analysis/commons.py
- detectors.geometry_alt.geom_metrics_alt, src/menipy/detectors/geometry_alt.py
- physics.contact_geom.line_params, src/menipy/physics/contact_geom.py
- scipy.interpolate.UnivariateSpline
- scipy.interpolate.CubicSpline

## src/menipy/batch.py
- __future__.annotations
- pathlib.Path
- typing.Iterable
- pandas
- processing.reader.load_image, src/menipy/processing/reader.py
- processing.segmentation.otsu_threshold, src/menipy/processing/segmentation.py
- processing.segmentation.morphological_cleanup, src/menipy/processing/segmentation.py
- processing.segmentation.external_contour_mask, src/menipy/processing/segmentation.py
- processing.segmentation.find_contours, src/menipy/processing/segmentation.py

## src/menipy/calibration/__init__.py
- calibrator.Calibration, src/menipy/calibration/calibrator.py
- calibrator.get_calibration, src/menipy/calibration/calibrator.py
- calibrator.set_calibration, src/menipy/calibration/calibrator.py
- calibrator.mm_to_pixels, src/menipy/calibration/calibrator.py
- calibrator.pixels_to_mm, src/menipy/calibration/calibrator.py
- calibrator.calibrate_from_points, src/menipy/calibration/calibrator.py
- calibrator.auto_calibrate, src/menipy/calibration/calibrator.py

## src/menipy/calibration/calibrator.py
- __future__.annotations
- dataclasses.dataclass
- math.hypot
- cv2
- numpy

## src/menipy/cli.py
- __future__.annotations

## src/menipy/common/plugins.py
- (no imports)

## src/menipy/contact_angle.py
- math
- numpy

## src/menipy/detection/__init__.py
- (no imports)

## src/menipy/detection/droplet.py
- __future__.annotations
- dataclasses.dataclass
- typing.Tuple
- cv2
- numpy
- physics.contact_geom.contour_line_intersections, src/menipy/physics/contact_geom.py
- physics.contact_geom.geom_metrics, src/menipy/physics/contact_geom.py
- physics.contact_geom.line_params, src/menipy/physics/contact_geom.py

## src/menipy/detection/needle.py
- __future__.annotations
- cv2
- numpy

## src/menipy/detection/substrate.py
- __future__.annotations
- typing.Literal
- cv2
- numpy
- skimage.measure.LineModelND
- skimage.measure.ransac

## src/menipy/detectors/__init__.py
- geometry_alt.*, src/menipy/detectors/geometry_alt.py

## src/menipy/detectors/geometry_alt.py
- __future__.annotations
- numpy
- physics.contact_geom.line_params, src/menipy/physics/contact_geom.py
- physics.contact_geom.contour_line_intersections, src/menipy/physics/contact_geom.py

## src/menipy/gui/__init__.py
- ui.MainWindow, src/menipy/ui/__init__.py
- image_view.ImageView, src/menipy/gui/image_view.py
- calibration_dialog.CalibrationDialog, src/menipy/gui/calibration_dialog.py
- controls.ZoomControl, src/menipy/gui/controls.py
- controls.ParameterPanel, src/menipy/gui/controls.py
- controls.MetricsPanel, src/menipy/gui/controls.py
- controls.CalibrationTab, src/menipy/gui/controls.py
- controls.AnalysisTab, src/menipy/gui/controls.py
- overlay.draw_drop_overlay, src/menipy/gui/overlay.py
- items.SubstrateLineItem, src/menipy/gui/items.py
- PySide6.QtWidgets.QApplication

## src/menipy/gui/base_window.py
- pathlib.Path
- numpy
- cv2
- typing.Any
- PySide6.QtCore.Qt
- PySide6.QtCore.QRectF
- PySide6.QtCore.QLineF
- PySide6.QtGui.QImage
- PySide6.QtGui.QPixmap
- PySide6.QtGui.QAction
- PySide6.QtGui.QPainter
- PySide6.QtGui.QPainterPath
- PySide6.QtGui.QPen
- PySide6.QtGui.QBrush
- PySide6.QtGui.QColor
- PySide6.QtWidgets.QApplication
- PySide6.QtWidgets.QMainWindow
- PySide6.QtWidgets.QFileDialog
- PySide6.QtWidgets.QMessageBox
- PySide6.QtWidgets.QSplitter
- PySide6.QtWidgets.QWidget
- PySide6.QtWidgets.QVBoxLayout
- PySide6.QtWidgets.QComboBox
- PySide6.QtWidgets.QPushButton
- PySide6.QtWidgets.QTabWidget
- PySide6.QtWidgets.QSlider
- pandas
- controls.ZoomControl, src/menipy/gui/controls.py
- controls.ParameterPanel, src/menipy/gui/controls.py
- controls.MetricsPanel, src/menipy/gui/controls.py
- controls.CalibrationTab, src/menipy/gui/controls.py
- controls.AnalysisTab, src/menipy/gui/controls.py
- image_view.ImageView, src/menipy/gui/image_view.py
- processing.reader.load_image, src/menipy/processing/reader.py
- processing.detect_droplet, src/menipy/processing/__init__.py
- processing.detect_sessile_droplet, src/menipy/processing/__init__.py
- processing.detect_pendant_droplet, src/menipy/processing/__init__.py
- processing.segmentation, src/menipy/processing/segmentation.py
- processing.detect_substrate_line, src/menipy/processing/__init__.py
- processing.segmentation.find_contours, src/menipy/processing/segmentation.py
- calibration.get_calibration, src/menipy/calibration/__init__.py
- calibration.pixels_to_mm, src/menipy/calibration/__init__.py
- calibration.auto_calibrate, src/menipy/calibration/__init__.py
- calibration.calibrate_from_points, src/menipy/calibration/__init__.py
- models.properties.droplet_volume, src/menipy/models/properties.py
- models.properties.estimate_surface_tension, src/menipy/models/properties.py
- models.properties.contact_angle_from_mask, src/menipy/models/properties.py
- models.geometry.fit_circle, src/menipy/models/geometry.py
- analysis.detect_vertical_edges, src/menipy/analysis/__init__.py
- analysis.extract_external_contour, src/menipy/analysis/__init__.py
- pipelines.analyze_pendant, src/menipy/pipelines/__init__.py
- pipelines.analyze_sessile_alt, src/menipy/pipelines/__init__.py
- pipelines.draw_pendant_overlays, src/menipy/pipelines/__init__.py
- pipelines.draw_sessile_overlays_alt, src/menipy/pipelines/__init__.py
- pipelines.pendant.HelperBundle, src/menipy/pipelines/pendant/__init__.py
- pipelines.sessile.HelperBundle, src/menipy/pipelines/sessile/__init__.py
- items.SubstrateLineItem, src/menipy/gui/items.py
- physics.contact_geom.geom_metrics, src/menipy/physics/contact_geom.py
- detectors.geometry_alt.side_of_polyline, src/menipy/detectors/geometry_alt.py

## src/menipy/gui/calibration_dialog.py
- __future__.annotations
- dataclasses.dataclass
- typing.Tuple
- PySide6.QtCore.Qt
- PySide6.QtCore.QLineF
- PySide6.QtGui.QImage
- PySide6.QtGui.QPixmap
- PySide6.QtGui.QPen
- PySide6.QtGui.QColor
- PySide6.QtWidgets.QDialog
- PySide6.QtWidgets.QDialogButtonBox
- PySide6.QtWidgets.QDoubleSpinBox
- PySide6.QtWidgets.QGraphicsScene
- PySide6.QtWidgets.QGraphicsView
- PySide6.QtWidgets.QLabel
- PySide6.QtWidgets.QVBoxLayout
- calibration.calibrate_from_points, src/menipy/calibration/__init__.py

## src/menipy/gui/controls.py
- PySide6.QtCore.Qt
- PySide6.QtCore.Signal
- PySide6.QtWidgets.QWidget
- PySide6.QtWidgets.QSlider
- PySide6.QtWidgets.QVBoxLayout
- PySide6.QtWidgets.QLabel
- PySide6.QtWidgets.QFormLayout
- PySide6.QtWidgets.QDoubleSpinBox
- PySide6.QtWidgets.QCheckBox
- PySide6.QtWidgets.QPushButton
- PySide6.QtWidgets.QComboBox
- PySide6.QtWidgets.QFrame

## src/menipy/gui/image_view.py
- __future__.annotations
- PySide6.QtCore.QPointF
- PySide6.QtGui.QPixmap
- PySide6.QtGui.QTransform
- PySide6.QtWidgets.QGraphicsView
- PySide6.QtWidgets.QGraphicsScene
- PySide6.QtWidgets.QGraphicsPixmapItem

## src/menipy/gui/items.py
- PySide6.QtCore
- PySide6.QtGui
- PySide6.QtWidgets

## src/menipy/gui/overlay.py
- cv2
- numpy
- PySide6.QtGui.QImage
- PySide6.QtGui.QPixmap

## src/menipy/gui.py
- __future__.annotations
- PySide6.QtWidgets.QApplication
- ui.MainWindow, src/menipy/ui/__init__.py

## src/menipy/metrics/__init__.py
- (no imports)

## src/menipy/metrics/metrics.py
- __future__.annotations

## src/menipy/models/__init__.py
- geometry.fit_circle, src/menipy/models/geometry.py
- physics.solve_young_laplace, src/menipy/models/physics.py
- properties.droplet_volume, src/menipy/models/properties.py
- properties.estimate_surface_tension, src/menipy/models/properties.py
- properties.contact_angle_from_mask, src/menipy/models/properties.py
- surface_tension.jennings_pallas_beta, src/menipy/models/surface_tension.py
- surface_tension.surface_tension, src/menipy/models/surface_tension.py
- surface_tension.bond_number, src/menipy/models/surface_tension.py
- surface_tension.volume_from_contour, src/menipy/models/surface_tension.py
- drop_extras.vmax_uL, src/menipy/models/drop_extras.py
- drop_extras.worthington_number, src/menipy/models/drop_extras.py
- drop_extras.apex_curvature_m_inv, src/menipy/models/drop_extras.py
- drop_extras.projected_area_mm2, src/menipy/models/drop_extras.py
- drop_extras.surface_area_mm2, src/menipy/models/drop_extras.py
- drop_extras.apparent_weight_mN, src/menipy/models/drop_extras.py

## src/menipy/models/drop_extras.py
- (no imports)

## src/menipy/models/datatypes.py
- numpy
- math

## src/menipy/models/geometry.py
- typing.Tuple
- numpy
- numpy.linalg.lstsq

## src/menipy/models/physics.py
- typing.Callable
- numpy
- scipy.integrate.solve_ivp

## src/menipy/models/properties.py
- __future__.annotations
- typing.Optional
- cv2
- numpy
- processing.segmentation.find_contours, src/menipy/processing/segmentation.py
- geometry.fit_circle, src/menipy/models/geometry.py

## src/menipy/models/surface_tension.py
- __future__.annotations
- numpy
- numpy.typing.NDArray

## src/menipy/physics/__init__.py
- (no imports)

## src/menipy/physics/contact_geom.py
- __future__.annotations
- math
- typing.Any
- numpy

## src/menipy/pipelines/base.py
- __future__.annotations
- logging
- time
- typing.Any
- typing.Callable
- typing.Optional
- typing.Dict
- menipy.models.datatypes.Context, src/menipy/models/datatypes.py

## src/menipy/pipelines/__init__.py
- (no imports)

## src/menipy/pipelines/discover.py
- importlib
- pathlib.Path
- menipy.pipelines.base.PipelineBase, src/menipy/pipelines/base.py

## src/menipy/pipelines/pendant/__init__.py
- geometry.analyze, src/menipy/pipelines/pendant/geometry.py
- geometry.PendantMetrics, src/menipy/pipelines/pendant/geometry.py
- geometry.HelperBundle, src/menipy/pipelines/pendant/geometry.py
- drawing.draw_overlays, src/menipy/pipelines/pendant/drawing.py

## src/menipy/pipelines/pendant/drawing.py
- __future__.annotations
- numpy
- PySide6.QtGui.QPixmap
- gui.overlay.draw_drop_overlay, src/menipy/gui/overlay.py
- geometry.PendantMetrics, src/menipy/pipelines/pendant/geometry.py

## src/menipy/pipelines/pendant/geometry.py
- __future__.annotations
- dataclasses.dataclass
- numpy
- analysis.extract_external_contour, src/menipy/analysis/__init__.py
- analysis.compute_pendant_metrics, src/menipy/analysis/__init__.py

## src/menipy/pipelines/sessile/__init__.py
- geometry_alt.analyze, src/menipy/pipelines/sessile/geometry_alt.py
- geometry_alt.SessileMetrics, src/menipy/pipelines/sessile/geometry_alt.py
- geometry_alt.HelperBundle, src/menipy/pipelines/sessile/geometry_alt.py
- drawing_alt.draw_overlays, src/menipy/pipelines/sessile/drawing_alt.py

## src/menipy/pipelines/sessile/geometry_alt.py
- __future__.annotations
- dataclasses.dataclass
- typing.Literal
- numpy
- cv2
- analysis.compute_sessile_metrics_alt, src/menipy/analysis/__init__.py
- physics.contact_geom.line_params, src/menipy/physics/contact_geom.py
- detectors.geometry_alt.split_contour_by_line, src/menipy/detectors/geometry_alt.py
- gui.overlay.draw_drop_overlay, src/menipy/gui/overlay.py

## src/menipy/plugins.py
- __future__.annotations
- importlib.metadata
- warnings
- sharpen_plugin.sharpen_filter, src/menipy/sharpen_plugin.py

## src/menipy/processing/__init__.py
- substrate.detect_substrate_line, src/menipy/processing/substrate.py
- substrate.SubstrateNotFoundError, src/menipy/processing/substrate.py
- classification.classify_drop_mode, src/menipy/processing/classification.py

## src/menipy/processing/classification.py
- numpy
- typing.Literal
- detection.Droplet, src/menipy/processing/detection.py
- calibration.get_calibration, src/menipy/calibration/__init__.py

## src/menipy/processing/detection.py
- __future__.annotations
- dataclasses.dataclass
- typing.Tuple
- cv2
- numpy
- __future__.annotations
- typing.Tuple
- numpy
- models.properties.droplet_volume, src/menipy/models/properties.py
- region.close_droplet, src/menipy/processing/region.py
- region._signed_distance, src/menipy/processing/region.py

## src/menipy/processing/reader.py
- pathlib.Path
- typing.Union
- cv2
- numpy

## src/menipy/processing/region.py
- __future__.annotations
- typing.Literal
- typing.Tuple
- cv2
- numpy

## src/menipy/processing/segmentation.py
- cv2
- numpy
- skimage.filters
- skimage.morphology

## src/menipy/processing/substrate.py
- __future__.annotations
- typing.Literal
- cv2
- numpy
- skimage.measure.LineModelND
- skimage.measure.ransac
- geometry.clip_line_to_roi, src/menipy/processing/geometry.py

## src/menipy/sharpen_plugin.py
- __future__.annotations
- numpy
- cv2

## src/menipy/ui/__init__.py
- main_window.MainWindow, src/menipy/ui/main_window.py

## src/menipy/utils.py
- __future__.annotations

## src/menipy/viewmodels/run_vm.py
- (no imports)
