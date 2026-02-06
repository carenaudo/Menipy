"""Dialog for testing and visualizing detector plugins."""
from pathlib import Path
import cv2
import numpy as np
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QComboBox, QPushButton, 
    QLabel, QFileDialog, QCheckBox, QFrame, QSizePolicy, QMessageBox, QWidget
)
from PySide6.QtGui import QImage, QPainter, QPen, QColor, QFont
from PySide6.QtCore import Qt, QRect, QLineF

from menipy.common.registry import (
    EDGE_DETECTORS, NEEDLE_DETECTORS, ROI_DETECTORS, 
    SUBSTRATE_DETECTORS, DROP_DETECTORS, APEX_DETECTORS
)
from menipy.models.config import EdgeDetectionSettings

class DetectorTestCanvas(QWidget):
    """Canvas widget to display image and detection results."""
    def __init__(self, parent=None):
        """Initialize.

        Parameters
        ----------
        parent : type
        Description.
        """
        super().__init__(parent)
        self.image = None
        self.result = None
        self.result_type = None  # 'contour', 'line', 'point', 'mask'
        
        self.show_overlay = True
        
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(400, 300)

    def set_image(self, image: np.ndarray):
        self.image = image
        self.result = None
        self.repaint()

    def set_result(self, result, result_type: str, debug_info: list = None):
        self.result = result
        self.result_type = result_type
        self.debug_info = debug_info
        self.repaint()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        if self.image is not None:
            # Draw Image
            h, w, ch = self.image.shape
            bytes_per_line = ch * w
            img_data = self.image.copy()
            # Handle grayscale vs BGR
            if len(img_data.shape) == 2:
                img_data = cv2.cvtColor(img_data, cv2.COLOR_GRAY2RGB)
            else:
                img_data = cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB)
                
            qimg = QImage(img_data.data, w, h, ch * w, QImage.Format_RGB888)
            
            scaled_img = qimg.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            
            x_off = (self.width() - scaled_img.width()) // 2
            y_off = (self.height() - scaled_img.height()) // 2
            
            painter.drawImage(x_off, y_off, scaled_img)

            # Scales
            scale_x = scaled_img.width() / w
            scale_y = scaled_img.height() / h

            def to_screen(x, y):
                return (int(x * scale_x) + x_off, int(y * scale_y) + y_off)

            # Overlay
            if self.show_overlay:
                if getattr(self, 'debug_info', None):
                    self._draw_debug_info(painter, to_screen)
                if self.result is not None:
                    self._draw_overlay(painter, to_screen, w, h)

    def _draw_debug_info(self, painter, to_screen):
        # Draw all debug candidates in gray/blue
        painter.setPen(QPen(QColor(100, 100, 255, 150), 1))
        font = painter.font()
        font.setPointSize(8)
        painter.setFont(font)
        
        for item in self.debug_info:
            # item is (xy, area, label)
            if len(item) >= 3:
                xy, area, label = item
                if isinstance(xy, np.ndarray) and xy.ndim == 2:
                    pts = [to_screen(p[0], p[1]) for p in xy]
                    if len(pts) > 1:
                        for i in range(len(pts) - 1):
                            painter.drawLine(pts[i][0], pts[i][1], pts[i+1][0], pts[i+1][1])
                        # Draw label at first point
                        painter.drawText(pts[0][0], pts[0][1], label)

    def _draw_overlay(self, painter, to_screen, img_w, img_h):
        painter.setPen(QPen(QColor(0, 255, 0), 2))
        
        if self.result_type == 'contour':
            # Expecting (N, 2) array of (x, y)
            contour = self.result
            if isinstance(contour, np.ndarray) and contour.ndim == 2 and contour.shape[1] == 2:
                pts = [to_screen(p[0], p[1]) for p in contour]
                for i in range(len(pts) - 1):
                    painter.drawLine(pts[i][0], pts[i][1], pts[i+1][0], pts[i+1][1])
                    
        elif self.result_type == 'mask':
            # Expecting binary image mask
            # For visualization, we can find contours of the mask and draw them
            mask = self.result
            if isinstance(mask, np.ndarray):
                # Ensure valid type for findContours (uint8)
                if mask.dtype != np.uint8:
                    # Normalize if needed or just cast
                    if mask.dtype == bool:
                         mask = mask.astype(np.uint8) * 255
                    else:
                         # Assume it might be float 0..1 or 0..255
                         if mask.max() <= 1.0:
                             mask = (mask * 255).astype(np.uint8)
                         else:
                             mask = mask.astype(np.uint8)
                
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours:
                    # cnt is (N, 1, 2)
                    pts = [to_screen(p[0][0], p[0][1]) for p in cnt]
                    for i in range(len(pts) - 1):
                        painter.drawLine(pts[i][0], pts[i][1], pts[i+1][0], pts[i+1][1])

        elif self.result_type == 'line':
            # Expecting y-coordinate (int or float) or (pt1, pt2)
            if isinstance(self.result, (int, float)):
                y = float(self.result)
                p1 = to_screen(0, y)
                p2 = to_screen(img_w, y)
                painter.drawLine(p1[0], p1[1], p2[0], p2[1])
                painter.drawText(p1[0] + 5, p1[1] - 5, f"Y={y:.1f}")

        elif self.result_type == 'point':
            # Expecting (x, y)
            pt = self.result
            if len(pt) == 2:
                sx, sy = to_screen(pt[0], pt[1])
                painter.setPen(QPen(QColor(255, 0, 0), 2))
                painter.drawLine(sx - 5, sy, sx + 5, sy)
                painter.drawLine(sx, sy - 5, sx, sy + 5)
                painter.drawText(sx + 5, sy - 5, f"({pt[0]:.1f}, {pt[1]:.1f})")


class DetectorTestDialog(QDialog):
    """Dialog to test detector plugins."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Detector Test Utility")
        self.resize(1000, 700)
        
        self.image = None
        self.image_path = None
        
        self.detector_categories = {
            "Drop Detector": DROP_DETECTORS,
            "Edge Detector": EDGE_DETECTORS,
            "Needle Detector": NEEDLE_DETECTORS,
            "Substrate Detector": SUBSTRATE_DETECTORS,
            "ROI Detector": ROI_DETECTORS,
            "Apex Detector": APEX_DETECTORS
        }
        
        self._setup_ui()
        self._populate_detectors()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        
        # 1. Canvas
        self.canvas = DetectorTestCanvas()
        layout.addWidget(self.canvas, stretch=1)
        
        # 2. Controls Area
        controls = QFrame()
        # controls.setStyleSheet("background-color: #f0f0f0; border-top: 1px solid #ccc;") 
        # Use theme-compatible style or just a border
        controls.setFrameShape(QFrame.StyledPanel)
        
        c_layout = QHBoxLayout(controls)
        c_layout.setContentsMargins(10, 10, 10, 10)
        
        # Image Loading
        btn_load = QPushButton("📂 Load")
        btn_load.clicked.connect(self._on_load_image)
        c_layout.addWidget(btn_load)
        
        # Category Selector
        c_layout.addWidget(QLabel("Type:"))
        self.combo_category = QComboBox()
        self.combo_category.addItems(self.detector_categories.keys())
        self.combo_category.currentTextChanged.connect(self._on_category_changed)
        c_layout.addWidget(self.combo_category)
        
        # Algorithm Selector
        c_layout.addWidget(QLabel("Algorithm:"))
        self.combo_algo = QComboBox()
        self.combo_algo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        c_layout.addWidget(self.combo_algo, stretch=1)
        
        # Run Button
        btn_run = QPushButton("▶ Run")
        btn_run.clicked.connect(self._on_run_detector)
        c_layout.addWidget(btn_run)
        
        # Options
        self.chk_overlay = QCheckBox("Overlay")
        self.chk_overlay.setChecked(True)
        self.chk_overlay.toggled.connect(self._on_toggle_overlay)
        c_layout.addWidget(self.chk_overlay)
        
        self.chk_debug = QCheckBox("Debug")
        self.chk_debug.setToolTip("Show all candidate contours (Snake only)")
        c_layout.addWidget(self.chk_debug)
        
        c_layout.addStretch()
        
        layout.addWidget(controls)
        
        # Status Label
        self.lbl_status = QLabel("Ready")
        layout.addWidget(self.lbl_status)

    def _populate_detectors(self):
        self._on_category_changed(self.combo_category.currentText())

    def _on_category_changed(self, category_name):
        """_on_category_changed."""
        self.combo_algo.clear()
        registry = self.detector_categories.get(category_name)
        if registry:
            # We can use registry.keys() or iterate
            # The registry object in menipy behaves like a dict but we should check
            # Based on registry.py it implements methods like keys()
            items = sorted(list(registry.keys()))
            self.combo_algo.addItems(items)

    def _on_load_image(self):
        """_on_load_image."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Test Image", "", 
            "Images (*.png *.jpg *.jpeg *.tiff *.bmp);;All Files (*)"
        )
        if path:
            self._load_image_file(path)

    def _load_image_file(self, path):
        """_load_image_file."""
        # Read using cv2
        img = cv2.imread(path)
        if img is None:
            QMessageBox.warning(self, "Error", f"Could not read image: {path}")
            return
            
        self.image = img
        self.image_path = path
        self.canvas.set_image(img)
        self.lbl_status.setText(f"Loaded: {path} ({img.shape[1]}x{img.shape[0]})")

    def _on_run_detector(self):
        """_on_run_detector."""
        if self.image is None:
            QMessageBox.warning(self, "Warning", "Please load an image first.")
            return

        cat_name = self.combo_category.currentText()
        algo_name = self.combo_algo.currentText()
        
        registry = self.detector_categories.get(cat_name)
        if not registry:
            return
            
        detector_fn = registry.get(algo_name)
        if not detector_fn:
            QMessageBox.warning(self, "Error", f"Could not find algorithm '{algo_name}'")
            return
            
        try:
            # Most detectors take image as first arg.
            # However some might need grayscale.
            # We'll try to be smart or generic.
            
            # Prepare input
            input_img = self.image
            # If edge detector usually expects grayscale
            if cat_name == "Edge Detector":
                if len(input_img.shape) == 3:
                     input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
            
            self.lbl_status.setText(f"Running {algo_name}...")
            
            # Execute
            result = None
            
            # Helper to invoke with flexible args
            # Try passing settings if it's an edge detector
            debug_mode = self.chk_debug.isChecked()
            debug_info = []
            
            if cat_name == "Edge Detector":
                settings = EdgeDetectionSettings()
                # Try calling with return_debug first
                if debug_mode:
                    try:
                        result = detector_fn(input_img, settings, return_debug=True)
                        # Check if tuple returned
                        if isinstance(result, tuple) and len(result) == 2:
                            result, debug_info = result
                    except TypeError:
                         # Fallback to no debug arg
                         try:
                            result = detector_fn(input_img, settings)
                         except TypeError:
                            result = detector_fn(input_img)
                else:
                    try:
                        result = detector_fn(input_img, settings)
                    except TypeError:
                         # Fallback for functions not accepting settings
                         result = detector_fn(input_img)
            else:
                 result = detector_fn(input_img)
            
            # Determine visualization type
            vis_type = 'mask' # Default fallback
            
            if cat_name == "Drop Detector":
                # Plugins like detect_drop return contour or mask?
                # Usually contour (N, 2) or mask
                if isinstance(result, np.ndarray):
                    if result.ndim == 2 and result.shape[1] == 2:
                        vis_type = 'contour'
                    else:
                        vis_type = 'mask'
                        
            elif cat_name == "Edge Detector":
                # Plugins in edge_detectors.py return (N, 2) coords
                vis_type = 'contour'
                
            elif cat_name == "Substrate Detector":
                 # Usually returns int (y-coord) or line params
                 vis_type = 'line'
                 
            elif cat_name == "Apex Detector":
                vis_type = 'point'
            
            elif cat_name == "ROI Detector":
                 # Returns (x, y, w, h) ?
                 # If so we need to handle it. 
                 # Let's assume it might return a mask or rect.
                 if isinstance(result, tuple) and len(result) == 4:
                     # Convert rect to contour for drawing
                     x, y, w, h = result
                     rect_cnt = np.array([
                         [x, y], [x+w, y], [x+w, y+h], [x, y+h], [x, y]
                     ])
                     result = rect_cnt
                     vis_type = 'contour'
                 else:
                     vis_type = 'mask'
            
            self.canvas.set_result(result, vis_type, debug_info)
            self.lbl_status.setText(f"Finished {algo_name}. Result type: {type(result)}")
            
        except Exception as e:
            QMessageBox.critical(self, "Detector Error", str(e))
            self.lbl_status.setText(f"Error: {e}")

    def _on_toggle_overlay(self, checked):
        """_on_toggle_overlay."""
        self.canvas.show_overlay = checked
        self.canvas.repaint()
