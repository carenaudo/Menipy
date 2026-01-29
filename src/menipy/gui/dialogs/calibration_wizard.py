"""
Calibration Wizard

Dialog for performing spatial calibration (pixels -> mm).
"""
from PySide6.QtCore import Qt, Signal, QPointF
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QFrame,
    QPushButton, QStackedWidget, QDoubleSpinBox, QMessageBox,
    QFileDialog, QWidget
)
import numpy as np

from menipy.gui import theme
from menipy.gui.widgets.interactive_image_viewer import InteractiveImageViewer


class CalibrationWizard(QDialog):
    """
    Wizard for calibrating spatial scale.
    
    Stages:
    1. Select Image (skip if provided)
    2. Measure Reference (draw line)
    3. Enter Dimension
    4. Result & Apply
    
    Signals:
        calibration_completed(scale_factor_px_mm): Emitted when calibration finishes.
    """
    
    calibration_completed = Signal(float)
    
    def __init__(self, parent=None, pixmap: QPixmap = None, image_data: np.ndarray = None):
        super().__init__(parent)
        self.setWindowTitle("Calibration Wizard")
        self.resize(900, 700)
        self._pixmap = pixmap
        self._image_data = image_data # Raw numpy array (preferred for detection)
        
        self._current_dist_px = 0.0
        
        # State
        self._calibration_result = 0.0  # px/mm
        
        self._setup_ui()
        if self._pixmap:
            self._viewer.set_image(self._pixmap)
            # Skip to measure page or Auto-Detect page depending on flow
            # If we have image data, Auto-Detect makes sense as first step (Page 1)
            # Page 0 is Load Image.
            self._pages.setCurrentIndex(1)
            self._update_buttons()
    
    def _setup_ui(self):
        self.setStyleSheet(theme.get_stylesheet())
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)
        
        # Header
        self._header = QLabel("Calibration Wizard")
        self._header.setProperty("title", True)
        layout.addWidget(self._header)
        
        # Progress Steps (simplified)
        self._step_label = QLabel("Step 1: Measurement")
        self._step_label.setStyleSheet(f"color: {theme.ACCENT_BLUE}; font-weight: bold;")
        layout.addWidget(self._step_label)
        
        # Content Pages
        self._pages = QStackedWidget()
        
        # Page 0: Image Selection (if needed)
        self._page_image = self._create_page_image()
        self._pages.addWidget(self._page_image)
        
        # Page 1: Auto Detect (New)
        self._page_auto_detect = self._create_page_auto_detect()
        self._pages.addWidget(self._page_auto_detect)
        
        # Page 2: Measurement (Interactive)
        self._page_measure = self._create_page_measure()
        self._pages.addWidget(self._page_measure)
        
        # Page 2: Dimension Input
        self._page_input = self._create_page_input()
        self._pages.addWidget(self._page_input)
        
        layout.addWidget(self._pages, stretch=1)
        
        # Footer Navigation
        footer = QHBoxLayout()
        footer.addStretch()
        
        self._btn_back = QPushButton("Back")
        self._btn_back.setProperty("secondary", True)
        self._btn_back.clicked.connect(self._go_back)
        footer.addWidget(self._btn_back)
        
        self._btn_next = QPushButton("Next")
        self._btn_next.clicked.connect(self._go_next)
        footer.addWidget(self._btn_next)
        
        layout.addLayout(footer)
        
        self._update_buttons()
        
    def _create_page_image(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        
        lbl = QLabel("Please load an image containing a known reference object (ruler, coin, etc).")
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(lbl)
        
        btn = QPushButton("Load Image")
        btn.clicked.connect(self._load_image)
        layout.addWidget(btn, alignment=Qt.AlignmentFlag.AlignCenter)
        
        return container
        
    def _create_page_measure(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        
        instruction = QLabel("Click and drag to draw a line along the known reference length.")
        instruction.setStyleSheet(f"color: {theme.TEXT_SECONDARY};")
        layout.addWidget(instruction)
        
        self._viewer = InteractiveImageViewer()
        self._viewer.set_tool(InteractiveImageViewer.TOOL_LINE)
        self._viewer.line_drawn.connect(self._on_line_drawn)
        
        # Frame for viewer
        frame = QFrame()
        frame.setStyleSheet(f"border: 1px solid {theme.BORDER_DEFAULT};")
        frame_layout = QVBoxLayout(frame)
        frame_layout.setContentsMargins(0,0,0,0)
        frame_layout.addWidget(self._viewer)
        layout.addWidget(frame, stretch=1)
        
        self._measure_status = QLabel("Draw a line to measure...")
        self._measure_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._measure_status)
        
        return container
        
    def _create_page_auto_detect(self) -> QWidget:
        """Page for auto-detecting needle width."""
        container = QWidget()
        layout = QVBoxLayout(container)
        
        lbl = QLabel("Automatic Needle Detection")
        lbl.setStyleSheet(f"font-size: 16px; font-weight: bold; color: {theme.TEXT_PRIMARY};")
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(lbl)
        
        desc = QLabel("The wizard can attempt to automatically detect the needle width in pixels.\n"
                      "If successful, you can skip manual measurement.")
        desc.setAlignment(Qt.AlignmentFlag.AlignCenter)
        desc.setStyleSheet(f"color: {theme.TEXT_SECONDARY};")
        layout.addWidget(desc)
        
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        
        self._btn_run_detect = QPushButton("🔍 Run Auto-Detect")
        self._btn_run_detect.setMinimumHeight(40)
        self._btn_run_detect.clicked.connect(self._run_needle_detection)
        btn_layout.addWidget(self._btn_run_detect)
        
        btn_layout.addStretch()
        layout.addLayout(btn_layout)
        
        self._auto_status = QLabel("")
        self._auto_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._auto_status)
        
        layout.addStretch()
        return container

    def _run_needle_detection(self):
        """Run the needle detection logic."""
        if not self._pixmap:
            return
            
        self._auto_status.setText("Running detection...")
        
        try:
            if self._image_data is not None:
                 # Assume input is BGR or RGB depending on source.
                 # menipy usually loads images via cv2 (BGR) or QImage (RGB/RGBA).
                 # If passed from SessileDropWindow (cv2.imread), it is BGR.
                 # Context expects BGR or gray usually.
                 rgb = self._image_data
            elif self._pixmap:
                # Convert pixmap to numpy (RGBA) -> BGR (standard for openCV)
                image = self._pixmap.toImage()
                width = image.width()
                height = image.height()
                
                ptr = image.bits()
                # ptr.setsize(image.sizeInBytes()) # Not needed for memoryview
                
                import numpy as np
                # Expecting 4 bytes per pixel (RGBA) from QImage
                flattened = np.array(ptr).reshape(height, width, 4)
                # Convert RGBA to BGR for OpenCV
                import cv2
                rgb = cv2.cvtColor(flattened, cv2.COLOR_RGBA2BGR)
            else:
                return
            
            # Create Context
            from menipy.models.context import Context
            ctx = Context(image=rgb)
            ctx.pipeline_name = "sessile" # Default to sessile or try to infer?
            # Assuming sessile for now as it's the main use case. 
            # Could pass pipeline type into wizard logic if needed.
            
            # Use the robust AutoCalibrator from the main implementation
            try:
                from menipy.common.auto_calibrator import run_auto_calibration
                
                # Run full auto-calibration (detects substrate, needle, drop, ROI)
                # This matches the "Auto-Calibrate" logic in the main application
                calib_result = run_auto_calibration(
                    rgb, 
                    pipeline_name="sessile"
                )
                
                if calib_result and calib_result.needle_rect:
                    x, y, w, h = calib_result.needle_rect
                    self._current_dist_px = float(w)
                    
                    # Also update feedback with confidence if available
                    conf = calib_result.confidence_scores.get("needle", 0.0)
                    self._auto_status.setText(f"Success! Detected width: {w:.1f} px (Conf: {conf*100:.0f}%)")
                    self._auto_status.setStyleSheet(f"color: {theme.SUCCESS_GREEN}; font-weight: bold;")
                    
                    # Draw visual feedback
                    mid_y = y + h - 5
                    p1 = QPointF(x, mid_y)
                    p2 = QPointF(x + w, mid_y)
                    self._viewer.set_static_line(p1, p2)
                    
                    # Advance to measure/dimension page
                    self._pages.setCurrentIndex(2)
                    self._update_buttons()
                    return
                    
                else:
                     self._auto_status.setText("Needle not detected.")
                     self._auto_status.setStyleSheet(f"color: {theme.WARNING_ORANGE};")
                     return

            except ImportError:
                 # Fallback if module not found (detected during dev/testing)
                 self._auto_status.setText("Error: AutoCalibrator module not found.")
                 return
            except Exception as e:
                 self._auto_status.setText(f"Detection error: {e}")
                 import traceback
                 traceback.print_exc()
                 return
                 
        except Exception as e:
            self._auto_status.setText(f"Error: {e}")
            import traceback
            traceback.print_exc()

    def _on_db_requested(self):
        """Handle DB button click."""
        from menipy.gui.dialogs.material_dialog import MaterialDialog
        dialog = MaterialDialog(self, selection_mode=True, table_type="needles")
        dialog.item_selected.connect(self._on_needle_selected)
        dialog.exec()
        
    def _on_needle_selected(self, data: dict):
        if diameter := data.get("outer_diameter"):
            self._length_spin.setValue(float(diameter))

        
    def _create_page_input(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(20)
        
        lbl = QLabel("Enter the physical length of the line you drew:")
        layout.addWidget(lbl)
        
        input_row = QHBoxLayout()
        input_row.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self._length_spin = QDoubleSpinBox()
        self._length_spin.setRange(0.001, 1000.0)
        self._length_spin.setDecimals(3)
        self._length_spin.setValue(1.0)
        self._length_spin.setFixedWidth(150)
        self._length_spin.valueChanged.connect(self._calculate_result)
        input_row.addWidget(self._length_spin)
        
        self._unit_label = QLabel("mm")
        input_row.addWidget(self._unit_label)
        
        # DB Button
        db_btn = QPushButton("📚 DB")
        db_btn.setFixedSize(50, 24)
        db_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        db_btn.setProperty("secondary", True)
        db_btn.clicked.connect(self._on_db_requested)
        input_row.addWidget(db_btn)
        
        layout.addLayout(input_row)
        
        # Result preview
        self._result_preview = QLabel("Result: -- px/mm")
        self._result_preview.setStyleSheet(f"font-size: 18px; color: {theme.ACCENT_BLUE}; font-weight: bold;")
        layout.addWidget(self._result_preview)
        
        return container
        
    def _update_buttons(self):
        idx = self._pages.currentIndex()
        count = self._pages.count()
        
        self._btn_back.setEnabled(idx > 0)
        
        if idx == count - 1:
            self._btn_next.setText("Finish")
            self._step_label.setText(f"Step {idx+1}: Confirm")
        else:
            self._btn_next.setText("Next")
            
            if idx == 0:
                self._step_label.setText("Step 1: Load Image")
                self._btn_next.setEnabled(self._pixmap is not None)
            elif idx == 1:
                self._step_label.setText("Step 2: Measure Reference")
                self._btn_next.setEnabled(self._current_dist_px > 0)
    
    def _load_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load Image", "", "Images (*.png *.jpg *.tif)")
        if path:
            self._pixmap = QPixmap(path)
            self._viewer.set_image(self._pixmap)
            self._update_buttons()
            
    def _on_line_drawn(self, p1, p2, dist):
        self._current_dist_px = dist
        self._measure_status.setText(f"Measured Distance: {dist:.2f} px")
        self._calculate_result()
        self._update_buttons()
        
    def _calculate_result(self):
        phys_len = self._length_spin.value()
        if phys_len > 0 and self._current_dist_px > 0:
            res = self._current_dist_px / phys_len
            self._calibration_result = res
            self._result_preview.setText(f"Result: {res:.2f} px/mm")
            
    def _go_next(self):
        idx = self._pages.currentIndex()
        if idx < self._pages.count() - 1:
            self._pages.setCurrentIndex(idx + 1)
            self._update_buttons()
        else:
            # Finish
            if self._calibration_result > 0:
                self.calibration_completed.emit(self._calibration_result)
                self.accept()
            else:
                QMessageBox.warning(self, "Error", "Invalid calibration result.")
                
    def _go_back(self):
        idx = self._pages.currentIndex()
        if idx > 0:
            self._pages.setCurrentIndex(idx - 1)
            self._update_buttons()
