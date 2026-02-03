"""
Script: test_sessile_qt.py
Description: 
    Standalone test script for the Sessile Drop Pipeline with Qt visualization.
    This script replicates the core logic of the main application's Sessile Drop Analysis,
    allowing developers to debug detection algorithms, verify pipeline stages, and visually
    inspect results (contour, contact angle, etc.) without running the full GUI.

features:
    - Auto-calibration using `AutoCalibrator` (detects substrate, needle, drop).
    - Manual substrate override (`--substrate-y`) for challenging images.
    - Full pipeline execution (`SessilePipeline`).
    - geometric verification (Apex, Contact Points).
    - Physics-based solving (Young-Laplace).
    - Rich visualization using PySide6 (QPainter).

Usage:
    python scripts/test_sessile_qt.py "path/to/image.png"
    python scripts/test_sessile_qt.py "path/to/image.png" --substrate-y 300
"""

import sys
import argparse
from pathlib import Path
import cv2
import numpy as np

# Ensure src is in path so we can import 'menipy' packages
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel
from PySide6.QtGui import QImage, QPainter, QPen, QColor, QFont
from PySide6.QtCore import Qt, QRect, QLineF

from menipy.pipelines.sessile.stages import SessilePipeline
from menipy.models.context import Context
from menipy.models.geometry import Contour
from menipy.common.auto_calibrator import run_auto_calibration
import logging

# Configure logging to see AutoCalibrator detection steps
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("menipy.common.auto_calibrator")
logger.setLevel(logging.DEBUG)

class ResultWidget(QWidget):
    """
    Custom Qt Widget to display the processed image and overlay detection results.
    It reads data from the pipeline `Context` object.
    """
    def __init__(self, ctx: Context, parent=None):
        super().__init__(parent)
        self.ctx = ctx
        self.image = ctx.image  # Original loaded image
        self.setMinimumSize(800, 600)
        
    def paintEvent(self, event):
        """
        Main drawing loop. Renders the image and overlays:
        - Drop Contour (Green)
        - Substrate Line (Blue Dashed)
        - Apex (Red Cross)
        - Contact Points (Magenta)
        - Metrics Text Box
        """
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # 1. Draw Image
        if self.image is not None:
            # Convert OpenCV (BGR) to QImage (RGB) for Qt
            h, w, ch = self.image.shape
            bytes_per_line = ch * w
            img_data = self.image.copy()
            img_data = cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB)
            qimg = QImage(img_data.data, w, h, bytes_per_line, QImage.Format_RGB888)
            
            # Scale image to fit the window while maintaining aspect ratio
            scaled_img = qimg.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            
            # Calculate centering offsets
            x_off = (self.width() - scaled_img.width()) // 2
            y_off = (self.height() - scaled_img.height()) // 2
            
            painter.drawImage(x_off, y_off, scaled_img)
            
            # Calculate scale factors to map original image coordinates to screen coordinates
            scale_x = scaled_img.width() / w
            scale_y = scaled_img.height() / h
            
            def to_screen(x, y):
                """Transforms image coordinates (x, y) to widget screen coordinates."""
                return (int(x * scale_x) + x_off, int(y * scale_y) + y_off)

            # 2. Draw Contour (Green)
            if hasattr(self.ctx, "contour") and self.ctx.contour is not None:
                contour = self.ctx.contour.xy
                if contour is not None and len(contour) > 0:
                    pen = QPen(QColor(0, 255, 0), 2)
                    painter.setPen(pen)
                    pts = [to_screen(p[0], p[1]) for p in contour]
                    for i in range(len(pts) - 1):
                        painter.drawLine(pts[i][0], pts[i][1], pts[i+1][0], pts[i+1][1])
            
            # 3. Draw Geometry (Baseline, Apex)
            if hasattr(self.ctx, "geometry") and self.ctx.geometry:
                base_y = self.ctx.geometry.baseline_y
                apex_xy = self.ctx.geometry.apex_xy
                
                # Draw Baseline (Blue Dashed Line)
                if base_y is not None:
                    p1 = to_screen(0, base_y)
                    p2 = to_screen(w, base_y)
                    
                    # Text Label
                    painter.setPen(QPen(QColor(0, 255, 255), 1))
                    painter.drawText(p1[0] + 5, p1[1] - 5, f"Baseline Y={base_y:.1f}")

                    # Line
                    pen = QPen(QColor(0, 100, 255), 2)
                    pen.setStyle(Qt.DashLine)
                    painter.setPen(pen)
                    painter.drawLine(QLineF(p1[0], p1[1], p2[0], p2[1]))
                
                # Draw Apex (Red Cross)
                if apex_xy:
                    ax, ay = to_screen(apex_xy[0], apex_xy[1])
                    painter.setPen(QPen(QColor(255, 0, 0), 2))
                    painter.drawLine(int(ax) - 4, int(ay), int(ax) + 4, int(ay))
                    painter.drawLine(int(ax), int(ay) - 4, int(ax), int(ay) + 4)
                    painter.drawText(int(ax) + 5, int(ay) - 5, f"Apex ({apex_xy[0]:.1f}, {apex_xy[1]:.1f})")

            # 4. Draw Contact Points (Magenta)
            if hasattr(self.ctx, "geometry") and getattr(self.ctx.geometry, "contact_line", None):
                 contact_line = self.ctx.geometry.contact_line
                 if contact_line:
                     c1, c2 = contact_line
                     for i, (cx, cy) in enumerate([c1, c2]):
                         sx, sy = to_screen(cx, cy)
                         painter.setPen(QPen(QColor(255, 0, 255), 3)) 
                         painter.drawPoint(int(sx), int(sy))
                         label = "CP_L" if i==0 else "CP_R"
                         painter.drawText(int(sx)+5, int(sy)-5, f"{label} ({cx:.1f}, {cy:.1f})")

            # 5. Draw Metrics Box (Top-Left overlay)
            painter.setPen(QPen(QColor(255, 255, 255), 1))
            painter.setFont(QFont("Consolas", 10))
            
            results = getattr(self.ctx, "results", {})
            metrics_text = [
                f"Contact Angle: {results.get('contact_angle_deg', 0):.2f}°",
                f"Volume: {results.get('volume_uL', 0):.2f} uL",
                f"Surface Tension: {results.get('surface_tension_mN_m', 0):.2f} mN/m",
                f"Diameter: {results.get('diameter_mm', 0):.2f} mm",
                f"Height: {results.get('height_mm', 0):.2f} mm",
                f"Solver: {self.ctx.fit.get('solver', {}).get('success') if hasattr(self.ctx, 'fit') else 'N/A'}"
            ]
            
            # Semi-transparent background
            bg_rect = QRect(10, 10, 250, 20 + len(metrics_text) * 15)
            painter.fillRect(bg_rect, QColor(0, 0, 0, 150))
            
            y_txt = 30
            for line in metrics_text:
                painter.drawText(20, y_txt, line)
                y_txt += 15


def run_test(image_path: str, substrate_y: int | None = None, **kwargs):
    """
    Main logic function:
    1. Loads image.
    2. Runs AutoCalibrator to detect features (unless overridden).
    3. Configures SessilePipeline context.
    4. Runs the Pipeline.
    5. Opens ResultWidget.
    """
    app = QApplication(sys.argv)
    
    display_img = cv2.imread(image_path)
    if display_img is None:
        print("Failed to load image for display")
        return

    # ---------------------------------------------------------
    # 1. Run Auto-Calibration (Wizard Logic)
    # ---------------------------------------------------------
    print("Running AutoCalibrator...")
    calib_params = {}
    if "margin_fraction" in kwargs:
        calib_params["margin_fraction"] = kwargs["margin_fraction"]
    if "adaptive_block_size" in kwargs:
         calib_params["adaptive_block_size"] = kwargs["adaptive_block_size"]
    
    # AutoCalibrator attempts to find substrate line and drop contour automatically
    calib_result = run_auto_calibration(display_img, pipeline_name="sessile", **calib_params)
    
    # ---------------------------------------------------------
    # 2. Handle Manual Substrate Override
    # ---------------------------------------------------------
    if substrate_y is not None:
        print(f"Overriding Substrate Y: {substrate_y}")
        h, w = display_img.shape[:2]
        calib_result.substrate_line = ((0, substrate_y), (w, substrate_y))
        # Note: If we override the substrate, the auto-detected contour might be invalid 
        # (e.g., clipped at the wrong height). We handle this below.
    
    # ---------------------------------------------------------
    # 3. Configure Pipeline Arguments
    # ---------------------------------------------------------
    pipeline_kwargs = {
        "image_path": image_path,
        "preprocessing_settings": None, 
        "px_per_mm": 133.0, 
        "auto_detect_features": False, # CRITICAL: Disable pipeline's internal auto-detection loop to respect our inputs
    }
    
    # Inject detected or manual substrate line into the pipeline context
    if calib_result.substrate_line:
        pipeline_kwargs["substrate_line"] = calib_result.substrate_line
        print(f"Injected substrate: {calib_result.substrate_line}")
        
    if calib_result.contact_points:
         pipeline_kwargs["contact_points"] = calib_result.contact_points
         
    # Logic for Contour Injection:
    # - If we stick with auto-detected substrate, we trust the auto-detected contour.
    # - If we manually MOVED the substrate, we must discard the old contour and let the pipeline
    #   re-run edge detection (Canny) and re-clip it against the *new* substrate line.
    valid_contour_injected = False
    if calib_result.drop_contour is not None and len(calib_result.drop_contour) > 0:
         if substrate_y is None: # No manual override
             # Inject the pre-calculated contour. The pipeline will skip edge detection.
             pipeline_kwargs["drop_contour"] = calib_result.drop_contour
             pipeline_kwargs["contour"] = Contour(xy=calib_result.drop_contour)
             valid_contour_injected = True
         else:
             print("Manual substrate set: Skipping injection of cached contour to allow re-detection.")

    if not valid_contour_injected:
         print("Warning: Drop NOT detected by AutoCalibrator (or skipped due to override).")
         # Pipeline will fallback to its default edge detection (usually Canny)
         pipeline_kwargs["edge_detection_settings"] = None 
         
         # Save debug image if we had a binary mask but decided not to use it
         if calib_result.binary_mask is not None and substrate_y is None:
             cv2.imwrite("debug_binary.png", calib_result.binary_mask)
             print("Saved debug_binary.png for inspection.")

    print("Running Sessile Pipeline...")
    if "substrate_line" in pipeline_kwargs:
        print(f"DEBUG: Pipeline kwarg substrate_line={pipeline_kwargs['substrate_line']}")

    # ---------------------------------------------------------
    # 4. Execute Pipeline
    # ---------------------------------------------------------
    pipeline = SessilePipeline()
    try:
        ctx = pipeline.run(**pipeline_kwargs)
        print("Pipeline finished.")
        if hasattr(ctx, "error") and ctx.error:
            print(f"Pipeline Error: {ctx.error}")
    except Exception as e:
        print(f"Crash: {e}")
        import traceback
        traceback.print_exc()
        return

    # ---------------------------------------------------------
    # 5. Display Results
    # ---------------------------------------------------------
    window = ResultWidget(ctx)
    window.setWindowTitle(f"Sessile Pipeline Test - {Path(image_path).name}")
    window.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Sessile Pipeline with QPainter Overlay")
    parser.add_argument("image_path", help="Path to input image")
    parser.add_argument("--substrate-y", type=int, help="Manually set substrate Y coordinate", default=None)
    parser.add_argument("--margin-fraction", type=float, help="Fraction of width for margin analysis (default 0.05)", default=0.05)
    parser.add_argument("--block-size", type=int, help="Adaptive threshold block size (odd int, default 21)", default=21)
    
    args = parser.parse_args()
    
    if not Path(args.image_path).exists():
        print(f"Error: Image not found at {args.image_path}")
        sys.exit(1)
        
    run_test(
        args.image_path, 
        substrate_y=args.substrate_y,
        margin_fraction=args.margin_fraction,
        adaptive_block_size=args.block_size
    )

