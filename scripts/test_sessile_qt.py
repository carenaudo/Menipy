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
from dataclasses import dataclass, field
from contextlib import contextmanager
from statistics import mean, median
import time
import timeit
import csv
import cv2
import numpy as np
from scipy.signal import savgol_filter
from skimage.segmentation import active_contour
from skimage.filters import gaussian as skimage_gaussian

# Ensure src is in path so we can import 'menipy' packages
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from PySide6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, 
                               QMainWindow, QCheckBox, QLabel, QFrame, QSizePolicy)
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

DEBUG_LOG = False

def dprint(msg: str):
    """Print debug messages when DEBUG_LOG flag is enabled."""
    if DEBUG_LOG:
        print(msg)


def zero_crossing_detection(image: np.ndarray) -> np.ndarray:
    """
    Detect zero crossings in a signed image (e.g., Laplacian output).
    
    Zero crossings occur where the sign of the image changes between adjacent pixels,
    indicating the presence of an edge. This is used after applying the Laplacian of
    Gaussian (LoG) filter to find precise edge locations.
    
    Args:
        image: Input image with signed values (typically float64 from Laplacian)
        
    Returns:
        Binary edge image (uint8) with 255 at zero-crossing locations
    """
    z_c_image = np.zeros(image.shape, dtype=np.uint8)
    
    # Check for sign changes in horizontal direction
    sign_h = np.sign(image)
    zero_cross_h = np.abs(np.diff(sign_h, axis=1)) > 0
    
    # Check for sign changes in vertical direction
    sign_v = np.sign(image)
    zero_cross_v = np.abs(np.diff(sign_v, axis=0)) > 0
    
    # Combine: Mark pixel as edge if there's a zero crossing to its right or below
    z_c_image[:, :-1] |= (zero_cross_h * 255).astype(np.uint8)
    z_c_image[:-1, :] |= (zero_cross_v * 255).astype(np.uint8)
    
    return z_c_image


def log_edge_detection(gray_image: np.ndarray, 
                        sigma: float = 1.0, 
                        use_zero_crossing: bool = False,
                        min_gradient: float = 5.0) -> np.ndarray:
    """
    Apply Laplacian of Gaussian (LoG) edge detection.
    
    The LoG filter combines Gaussian smoothing (to reduce noise) with the Laplacian
    operator (to detect edges as second-derivative zero-crossings). This is a classic
    edge detection technique that is less sensitive to noise than simple Laplacian.
    
    Args:
        gray_image: Grayscale input image
        sigma: Standard deviation for Gaussian blur (controls edge scale)
        use_zero_crossing: If True, find precise edges via zero-crossing detection.
                          If False, return absolute Laplacian response.
        min_gradient: Minimum gradient magnitude to consider a zero-crossing valid.
                     Higher values filter out weak/noisy edges.
        
    Returns:
        Binary edge image (uint8)
    """
    # Calculate kernel size from sigma (must be odd)
    ksize = int(6 * sigma + 1)
    if ksize % 2 == 0:
        ksize += 1
    ksize = max(3, ksize)  # Minimum kernel size of 3
    
    # Step 1: Apply Gaussian Blur to reduce noise
    blur = cv2.GaussianBlur(gray_image, (ksize, ksize), sigma)
    
    # Step 2: Apply Laplacian operator with CV_64F depth to capture negative values
    laplacian = cv2.Laplacian(blur, cv2.CV_64F)
    
    if use_zero_crossing:
        # Step 3a: Detect zero crossings for precise edge localization
        # Only keep zero crossings where the gradient magnitude is significant
        abs_lap = np.abs(laplacian)
        
        # Find zero crossings
        z_c_image = np.zeros(laplacian.shape, dtype=np.uint8)
        
        # Check horizontal zero crossings (sign change between adjacent pixels)
        sign_h = np.sign(laplacian)
        zero_cross_h = (sign_h[:, :-1] * sign_h[:, 1:]) < 0
        
        # Check vertical zero crossings
        zero_cross_v = (sign_h[:-1, :] * sign_h[1:, :]) < 0
        
        # Only mark as edge if gradient is strong enough (filter weak edges)
        # Use the max of adjacent absolute values as the "edge strength"
        grad_h = np.maximum(abs_lap[:, :-1], abs_lap[:, 1:])
        grad_v = np.maximum(abs_lap[:-1, :], abs_lap[1:, :])
        
        # Apply threshold
        z_c_image[:, :-1] |= ((zero_cross_h & (grad_h > min_gradient)) * 255).astype(np.uint8)
        z_c_image[:-1, :] |= ((zero_cross_v & (grad_v > min_gradient)) * 255).astype(np.uint8)
        
        # Morphological cleanup: close gaps to form continuous contours
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        edges = cv2.morphologyEx(z_c_image, cv2.MORPH_CLOSE, kernel)
        
        # Dilate to thicken edges for better flood-fill
        edges = cv2.dilate(edges, kernel, iterations=1)
        
        # Now flood-fill from corners to find background
        # This helps distinguish the drop (interior) from noise
        h, w = edges.shape
        filled = edges.copy()
        
        # Create a mask with 2-pixel border for floodFill
        mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
        
        # Flood fill from all corners (assuming they are background)
        for seed in [(0, 0), (w-1, 0), (0, h-1), (w-1, h-1)]:
            if filled[seed[1], seed[0]] == 0:  # Only fill if not already an edge
                cv2.floodFill(filled, mask, seed, 128)
        
        # Background is now marked as 128, foreground blob should be 0 (unfilled)
        # Create binary mask: unfilled interior = 255, rest = 0
        interior = np.where(filled == 0, 255, 0).astype(np.uint8)
        
        # Apply opening to remove small artifacts inside the blob
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        interior = cv2.morphologyEx(interior, cv2.MORPH_OPEN, kernel_small)
        
        # The interior mask is our result - contour extraction will get the boundary
        edges = interior
        
    else:
        # Step 3b: Convert to absolute and scale (simpler but less precise)
        edges = cv2.convertScaleAbs(laplacian)
        # Threshold to binary
        _, edges = cv2.threshold(edges, 20, 255, cv2.THRESH_BINARY)
        
        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    
    return edges



def init_snake_from_contour(contour: np.ndarray, expand: float = 1.2) -> np.ndarray:
    """
    Initialize snake by expanding detected contour outward from its centroid.
    
    This provides a starting point slightly outside the detected edge,
    allowing the snake to converge inward onto the true boundary.
    
    Args:
        contour: (N, 2) array of contour points
        expand: Expansion factor (>1 expands outward, <1 contracts inward)
        
    Returns:
        (N, 2) array of expanded contour points
    """
    # Calculate centroid using OpenCV moments
    contour_cv = contour.reshape(-1, 1, 2).astype(np.float32)
    M = cv2.moments(contour_cv)
    
    if M['m00'] != 0:
        cx = M['m10'] / M['m00']
        cy = M['m01'] / M['m00']
    else:
        cx, cy = contour[0]
    
    center = np.array([cx, cy])
    expanded = center + expand * (contour - center)
    return expanded


def snake_contour_detection(
    gray_image: np.ndarray,
    initial_contour: np.ndarray,
    alpha: float = 0.015,
    beta: float = 10.0,
    gamma: float = 0.001,
    max_iterations: int = 500,
    convergence: float = 0.1,
    gaussian_sigma: float = 2.0,
    expand_factor: float = 1.15,
    w_line: float = 0.0,
    w_edge: float = 1.0,
    max_px_move: float = 1.0
) -> np.ndarray:
    """
    Refine contour using scikit-image active_contour (snake).
    
    The snake algorithm iteratively evolves an initial contour to minimize
    an energy functional that balances internal forces (smoothness) with
    external forces (image gradients).
    
    Args:
        gray_image: Grayscale image (uint8)
        initial_contour: (N, 2) array - initial contour points
        alpha: Length weight (higher = shorter contour)
        beta: Smoothness weight (higher = smoother curves)
        gamma: Time step for evolution
        max_iterations: Maximum iterations
        convergence: Convergence threshold
        gaussian_sigma: Sigma for image smoothing before snake
        expand_factor: Factor to expand initial contour outward
        
    Returns:
        Refined (N, 2) contour array
    """
    # Normalize and smooth image for gradient computation
    img_normalized = gray_image.astype(np.float64) / 255.0
    img_smooth = skimage_gaussian(img_normalized, sigma=gaussian_sigma)
    
    # Initialize snake from expanded contour
    init_snake = init_snake_from_contour(initial_contour, expand=expand_factor)
    
    # Apply active contour
    snake = active_contour(
        img_smooth,
        init_snake,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        w_line=w_line,
        w_edge=w_edge,
        max_px_move=max_px_move,
        max_num_iter=max_iterations,
        convergence=convergence
    )
    
    return snake


@dataclass
class BenchResult:
    name: str
    elapsed_ms: float
    samples_ms: list[float] = field(default_factory=list)

    def stats(self) -> dict[str, float]:
        if not self.samples_ms:
            return {}
        return {
            "min_ms": min(self.samples_ms),
            "median_ms": median(self.samples_ms),
            "mean_ms": mean(self.samples_ms),
        }


class Benchmarker:
    def __init__(self, enabled: bool = False, repeat: int = 3, number: int = 1):
        self.enabled = enabled
        self.repeat = max(1, int(repeat))
        self.number = max(1, int(number))
        self.results: list[BenchResult] = []

    def timed(self, name: str, func, *, benchmark: bool = True):
        start = time.perf_counter()
        result = func()
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        samples_ms: list[float] = []
        if self.enabled and benchmark:
            timer = timeit.Timer(func, timer=time.perf_counter)
            samples = timer.repeat(repeat=self.repeat, number=self.number)
            samples_ms = [(s * 1000.0) / self.number for s in samples]

        self.results.append(BenchResult(name=name, elapsed_ms=elapsed_ms, samples_ms=samples_ms))
        return result

    @contextmanager
    def span(self, name: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            self.results.append(BenchResult(name=name, elapsed_ms=elapsed_ms))

    def print_summary(self, *, pipeline_timings: dict | None = None):
        if not self.results and not pipeline_timings:
            return

        print("\nBenchmarks (step timings):")
        header = "step | elapsed_ms | bench_min_ms | bench_median_ms | bench_mean_ms"
        print(header)
        print("-" * len(header))

        for result in self.results:
            stats = result.stats()
            if stats:
                row = (
                    f"{result.name} | "
                    f"{result.elapsed_ms:10.2f} | "
                    f"{stats['min_ms']:12.2f} | "
                    f"{stats['median_ms']:15.2f} | "
                    f"{stats['mean_ms']:13.2f}"
                )
            else:
                row = f"{result.name} | {result.elapsed_ms:10.2f} | {'':12} | {'':15} | {'':13}"
            print(row)

        if pipeline_timings:
            print("\nPipeline stage timings (ctx.timings_ms):")
            for stage, ms in pipeline_timings.items():
                print(f"{stage}: {ms:.2f} ms")

    def write_csv(self, path: Path, *, pipeline_timings: dict | None = None) -> None:
        rows = []
        for result in self.results:
            stats = result.stats()
            rows.append(
                {
                    "step": result.name,
                    "elapsed_ms": f"{result.elapsed_ms:.6f}",
                    "bench_min_ms": f"{stats['min_ms']:.6f}" if stats else "",
                    "bench_median_ms": f"{stats['median_ms']:.6f}" if stats else "",
                    "bench_mean_ms": f"{stats['mean_ms']:.6f}" if stats else "",
                }
            )

        if pipeline_timings:
            for stage, ms in pipeline_timings.items():
                rows.append(
                    {
                        "step": f"pipeline.{stage}",
                        "elapsed_ms": f"{ms:.6f}",
                        "bench_min_ms": "",
                        "bench_median_ms": "",
                        "bench_mean_ms": "",
                    }
                )

        if not rows:
            return

        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "step",
                    "elapsed_ms",
                    "bench_min_ms",
                    "bench_median_ms",
                    "bench_mean_ms",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)

class ResultWidget(QWidget):
    """
    Custom Qt Widget to display the processed image and overlay detection results.
    It reads data from the pipeline `Context` object.
    """
    def __init__(self, ctx: Context, parent=None):
        super().__init__(parent)
        self.ctx = ctx
        self.image = ctx.image  # Original loaded image
        # Let layout manage size
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(400, 300) # Smaller minimum to allow controls to show
        
        # Visibility flags
        self.show_contours = True
        # Default show_savgol to True ONLY if it was enabled in the run
        self.show_savgol = False
        if hasattr(self.ctx, "active_filters") and "Savgol" in self.ctx.active_filters:
            self.show_savgol = True
            
        self.show_contacts = True
        self.show_text = True
        self.show_tangents = True # Default On
        self.show_roi = True
        self.show_all_contours = False  # Debug: show all detected contours
        
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
            if self.show_contours and hasattr(self.ctx, "contour") and self.ctx.contour is not None:
                contour = self.ctx.contour.xy
                if contour is not None and len(contour) > 0:
                    pen = QPen(QColor(0, 255, 0), 2)
                    painter.setPen(pen)
                    pts = [to_screen(p[0], p[1]) for p in contour]
                    for i in range(len(pts) - 1):
                        painter.drawLine(pts[i][0], pts[i][1], pts[i+1][0], pts[i+1][1])
            
            # 2b. Draw ALL detected contours (Debug mode)
            if self.show_all_contours and hasattr(self.ctx, "all_contours") and self.ctx.all_contours:
                # Color palette for different contours
                colors = [
                    QColor(255, 0, 0),    # Red
                    QColor(255, 165, 0),  # Orange
                    QColor(255, 255, 0),  # Yellow
                    QColor(0, 255, 255),  # Cyan  
                    QColor(255, 0, 255),  # Magenta
                    QColor(128, 0, 255),  # Purple
                    QColor(0, 128, 255),  # Light Blue
                    QColor(255, 128, 0),  # Dark Orange
                ]
                
                for idx, (cnt, area, label) in enumerate(self.ctx.all_contours):
                    color = colors[idx % len(colors)]
                    pen = QPen(color, 2)
                    painter.setPen(pen)
                    
                    if cnt is not None and len(cnt) > 0:
                        pts = [to_screen(p[0], p[1]) for p in cnt]
                        for i in range(len(pts) - 1):
                            painter.drawLine(pts[i][0], pts[i][1], pts[i+1][0], pts[i+1][1])
                        
                        # Draw label with area
                        cx = int(np.mean([p[0] for p in pts]))
                        cy = int(np.mean([p[1] for p in pts]))
                        painter.drawText(cx, cy, f"#{idx}: {label} ({area:.0f})")
            
            # 3. Draw Geometry (Baseline, Apex)
            if hasattr(self.ctx, "geometry") and self.ctx.geometry:
                base_y = self.ctx.geometry.baseline_y
                apex_xy = self.ctx.geometry.apex_xy
                # Prefer smoothed apex if available
                if self.show_savgol and hasattr(self.ctx, "savgol_results") and self.ctx.savgol_results:
                    apex_xy = self.ctx.savgol_results.get("apex", apex_xy)
                
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
                f"Solver: {self.ctx.fit.get('solver', {}).get('success') if hasattr(self.ctx, 'fit') else 'N/A'}",

                f"Detector: {getattr(self.ctx, 'detector_name', 'Unknown')}"
            ]
            
            # Show Savgol result if available
            if hasattr(self.ctx, "savgol_results") and self.ctx.savgol_results:
                svg = self.ctx.savgol_results
                metrics_text.append(f"SA L: {svg['left_angle']:.1f}°")
                metrics_text.append(f"SA R: {svg['right_angle']:.1f}°")
                
            # Show Active Filters
            if hasattr(self.ctx, "active_filters") and self.ctx.active_filters:
                filters_str = ", ".join(self.ctx.active_filters)
                metrics_text.append(f"Filters: {filters_str}")

            if self.show_text: # Check visibility
                # Semi-transparent background
                bg_rect = QRect(10, 10, 250, 20 + len(metrics_text) * 15)
                painter.fillRect(bg_rect, QColor(0, 0, 0, 150))
                
                y_txt = 30
                for line in metrics_text:
                    painter.drawText(20, y_txt, line)
                    y_txt += 15
            
            # Draw smoothed curve if available
            if hasattr(self.ctx, "savgol_results") and self.ctx.savgol_results:
                 svg = self.ctx.savgol_results
                 x_smooth = svg['x_smooth']
                 y_smooth = svg['y_smooth']
                 
                 if self.show_savgol: # Check visibility
                     pen = QPen(QColor(0, 100, 255), 2)  # Orange/Blue-ish for smoothed
                     painter.setPen(pen)
                     
                     pts_smooth = [to_screen(x, y) for x, y in zip(x_smooth, y_smooth)]
                     for i in range(len(pts_smooth) - 1):
                         painter.drawLine(pts_smooth[i][0], pts_smooth[i][1], pts_smooth[i+1][0], pts_smooth[i+1][1])
                 
                 if self.show_contacts or self.show_tangents:
                     # Draw intersection points (Contact Points) and/or tangents
                     pen_cp = QPen(QColor(255, 0, 0), 2)
                     lc = svg.get('left_contact')
                     rc = svg.get('right_contact')
                     
                     if lc is not None:
                         l_pt = to_screen(lc[0], lc[1])
                         if self.show_contacts:
                             painter.setPen(pen_cp)
                             painter.drawLine(l_pt[0]-5, l_pt[1]-5, l_pt[0]+5, l_pt[1]+5)
                             painter.drawLine(l_pt[0]-5, l_pt[1]+5, l_pt[0]+5, l_pt[1]-5)
                         
                         if self.show_tangents and 'left_slope' in svg:
                             m = svg['left_slope']
                             if not np.isnan(m):
                                 length = 60
                                 dx_val = length / np.sqrt(1 + m**2)
                                 dy_val = m * dx_val
                                 
                                 if not (np.isnan(dx_val) or np.isnan(dy_val)):
                                     p_start = to_screen(lc[0], lc[1])
                                     p_end = to_screen(lc[0] + dx_val, lc[1] + dy_val)
                                                                  
                                     pen_tan = QPen(QColor(0, 255, 255), 2, Qt.DashLine)
                                     painter.setPen(pen_tan)
                                     painter.drawLine(p_start[0], p_start[1], p_end[0], p_end[1])
                             
                             # Draw Text
                             angle_l = svg.get('left_angle', 0)
                             painter.setPen(QPen(QColor(0, 0, 0), 1))
                             painter.drawText(p_end[0]-40, p_end[1]-10, f"{angle_l:.1f}°")

                         
                     if rc is not None:
                         r_pt = to_screen(rc[0], rc[1])
                         if self.show_contacts:
                             painter.setPen(pen_cp)
                             painter.drawLine(r_pt[0]-5, r_pt[1]-5, r_pt[0]+5, r_pt[1]+5)
                             painter.drawLine(r_pt[0]-5, r_pt[1]+5, r_pt[0]+5, r_pt[1]-5)
                         
                         if self.show_tangents and 'right_slope' in svg:
                             m = svg['right_slope']
                             if not np.isnan(m):
                                 length = 60
                                 dx_val = length / np.sqrt(1 + m**2)
                                 dy_val = m * dx_val
                                 
                                 if not (np.isnan(dx_val) or np.isnan(dy_val)):
                                     p_start = to_screen(rc[0], rc[1])
                                     p_end = to_screen(rc[0] - dx_val, rc[1] - dy_val)
                                     
                                     pen_tan = QPen(QColor(0, 255, 255), 2, Qt.DashLine)
                                     painter.setPen(pen_tan)
                                     painter.drawLine(p_start[0], p_start[1], p_end[0], p_end[1])
                             
                             # Draw Text
                             angle_r = svg.get('right_angle', 0)
                             painter.setPen(QPen(QColor(0, 0, 0), 1))
                             painter.drawText(p_end[0]+10, p_end[1]-10, f"{angle_r:.1f}°")
                             


class TestViewerWindow(QMainWindow):
    def __init__(self, ctx):
        super().__init__()
        self.setWindowTitle("Sessile Test Viewer")
        self.resize(1000, 800)
        
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        
        layout = QVBoxLayout(main_widget)
        
        # Viewer Widget
        self.viewer = ResultWidget(ctx)
        layout.addWidget(self.viewer, stretch=1)
        
        # Control Panel Wrapper
        controls_frame = QFrame()
        controls_frame.setStyleSheet("background-color: #f0f0f0; border-top: 1px solid #ccc; color: black;")
        controls_layout = QHBoxLayout(controls_frame)
        controls_layout.setContentsMargins(10, 5, 10, 5)
        
        layout.addWidget(controls_frame, stretch=0)
        
        self.cb_contours = QCheckBox("Contour (Green)")
        self.cb_contours.setChecked(True)
        self.cb_contours.toggled.connect(self.toggle_contours)
        controls_layout.addWidget(self.cb_contours)
        
        self.cb_savgol = QCheckBox("Savgol (Blue)")
        self.cb_savgol.setChecked(self.viewer.show_savgol)
        self.cb_savgol.toggled.connect(self.toggle_savgol)
        controls_layout.addWidget(self.cb_savgol)
        
        self.cb_contacts = QCheckBox("Contact Pts (Red)")
        self.cb_contacts.setChecked(True)
        self.cb_contacts.toggled.connect(self.toggle_contacts)
        controls_layout.addWidget(self.cb_contacts)
        
        self.cb_tangents = QCheckBox("Tangents (Cyan)")
        self.cb_tangents.setChecked(True)
        self.cb_tangents.toggled.connect(self.toggle_tangents)
        controls_layout.addWidget(self.cb_tangents)

        self.cb_text = QCheckBox("Metrics Text")
        self.cb_text.setChecked(True)
        self.cb_text.toggled.connect(self.toggle_text)
        controls_layout.addWidget(self.cb_text)
        
        self.cb_all_contours = QCheckBox("All Contours (Debug)")
        self.cb_all_contours.setChecked(False)
        self.cb_all_contours.toggled.connect(self.toggle_all_contours)
        controls_layout.addWidget(self.cb_all_contours)
        
        controls_layout.addStretch()

    def toggle_contours(self, checked):
        self.viewer.show_contours = checked
        self.viewer.repaint()

    def toggle_savgol(self, checked):
        self.viewer.show_savgol = checked
        self.viewer.repaint()

    def toggle_contacts(self, checked):
        self.viewer.show_contacts = checked
        self.viewer.repaint()

    def toggle_text(self, checked):
        self.viewer.show_text = checked
        self.viewer.repaint()

    def toggle_tangents(self, checked):
        self.viewer.show_tangents = checked
        self.viewer.repaint()

    def toggle_all_contours(self, checked):
        self.viewer.show_all_contours = checked
        self.viewer.repaint()


def filter_monotonic_contour(points):
    """
    Filter contour to ensure it is monotonic in X.
     """
    # Group by X
    sorted_pts = points[np.argsort(points[:, 0])]
    unique_x, indices = np.unique(sorted_pts[:, 0], return_inverse=True)
    
    # For each unique X, find min Y
    filtered_points = []
    for i in range(len(unique_x)):
        # Indices in sorted_pts corresponding to unique_x[i]
        mask = (indices == i)
        y_values = sorted_pts[mask, 1]
        min_y = np.min(y_values)
        filtered_points.append([unique_x[i], min_y])
        
    return np.array(filtered_points)
    
# ... (rest of functions remain roughly same, just updating main launch)


# ... (find_contact_points, calculate_contact_angles unchanged) ...

# ... (run_test logic up to showing widget) ...

def run_test(image_path: str, substrate_y: int | None = None, **kwargs):
    # ... (all setup code) ...
    # ---------------------------------------------------------
    # 5. Show Results using Main Window
    # ---------------------------------------------------------
    win = TestViewerWindow(ctx) # <--- Changed line
    win.show()
    
    sys.exit(app.exec())



def filter_monotonic_contour(points):
    """
    Filter contour to ensure it is monotonic in X.
    If multiple Y values exist for the same X, keep the minimum Y (upper point).
    """
    # Group by X
    sorted_pts = points[np.argsort(points[:, 0])]
    unique_x, indices = np.unique(sorted_pts[:, 0], return_inverse=True)
    
    # For each unique X, find min Y
    filtered_points = []
    for i in range(len(unique_x)):
        # Indices in sorted_pts corresponding to unique_x[i]
        mask = (indices == i)
        y_values = sorted_pts[mask, 1]
        min_y = np.min(y_values)
        filtered_points.append([unique_x[i], min_y])
        
    return np.array(filtered_points)


def smooth_and_analyze_droplet(dome_points, substrate_y, window=21, poly=3, filter_monotonic=False, filter_substrate=False, smoothing_enabled=True):
    """
    Smooth droplet contour and find contact points (User provided logic)
    """
    # 0. Optional: Filter points below substrate (Y > substrate_y) explicitly
    if filter_substrate:
        # This prevents noise/artifacts from below the baseline affecting smoothing
        mask = dome_points[:, 1] <= substrate_y
        dome_points = dome_points[mask]
        
    if len(dome_points) < 3:
        return None

    if filter_monotonic:
        dome_points = filter_monotonic_contour(dome_points)
    
    # Ensure window is odd
    if window % 2 == 0:
        window += 1
    
    # Sort by x for monotonic curve
    order = np.argsort(dome_points[:, 0])
    x = dome_points[order, 0]
    y = dome_points[order, 1]
    
    if smoothing_enabled:
        # Handle cases where window is too large for data
        if len(x) < window:
            window = len(x) if len(x) % 2 == 1 else len(x) - 1
        
        if window < poly + 2:
            return None # Not enough points

        # Smooth y-coordinates and first derivative (for slopes)
        try:
            y_smooth = savgol_filter(y, window_length=window, polyorder=poly)
            y_deriv = savgol_filter(y, window_length=window, polyorder=poly, deriv=1, delta=1.0)
        except ValueError:
            return None
    else:
        # RAW MODE: No Savgol smoothing
        y_smooth = y
        # Calculate derivative using component differences to handle vertical segments
        # dy/dx = (dy/dt) / (dx/dt)
        dx_dt = np.gradient(x)
        dy_dt = np.gradient(y)
        
        y_deriv = np.zeros_like(dy_dt)
        # Avoid divide by zero where dx ~ 0 (vertical)
        valid_dx = np.abs(dx_dt) > 1e-6
        y_deriv[valid_dx] = dy_dt[valid_dx] / dx_dt[valid_dx]
        # For vertical segments, use large value with correct sign
        # (Though typically we shouldn't have many if filtered)
        y_deriv[~valid_dx] = np.sign(dy_dt[~valid_dx]) * 1e6 
    
    # Find apex: if several points share the minimum Y, pick the middle X of that flat top
    min_y = np.min(y_smooth)
    min_idx = np.where(np.isclose(y_smooth, min_y))[0]
    if len(min_idx) > 1:
        apex_idx = int(min_idx[len(min_idx)//2])  # middle index of flat region
        apex_x = float(np.median(x[min_idx]))
        apex_y = float(min_y)
    else:
        apex_idx = int(min_idx[0])
        apex_x = float(x[apex_idx])
        apex_y = float(y_smooth[apex_idx])
    apex = (apex_x, apex_y)
    
    # Find contact points using robust whole-curve intersection
    # Pass apex_idx to search OUTWARDS for better noise immunity
    left_contact, right_contact = find_contact_points(x, y_smooth, substrate_y, apex_idx=apex_idx)
    
    # If standard intersection failed, fall back to "closest point" logic per side
    if left_contact is None:
        left_subset_mask = x <= x[apex_idx]
        if np.any(left_subset_mask):
            lx = x[left_subset_mask]
            ly = y_smooth[left_subset_mask]
            ld = y_deriv[left_subset_mask]
            closest_i = np.argmin(np.abs(ly - substrate_y))
            left_contact = np.array([lx[closest_i], ly[closest_i]])
            left_slope = ld[closest_i]
        else:
             left_contact = np.array([x[0], y_smooth[0]])
             left_slope = y_deriv[0]
    else:
        idx_l = np.argmin(np.abs(x - left_contact[0]))
        left_slope = y_deriv[idx_l]

    if right_contact is None:
        right_subset_mask = x >= x[apex_idx]
        if np.any(right_subset_mask):
            rx = x[right_subset_mask]
            ry = y_smooth[right_subset_mask]
            rd = y_deriv[right_subset_mask]
            closest_i = np.argmin(np.abs(ry - substrate_y))
            right_contact = np.array([rx[closest_i], ry[closest_i]])
            right_slope = rd[closest_i]
        else:
             right_contact = np.array([x[-1], y_smooth[-1]])
             right_slope = y_deriv[-1]
    else:
        idx_r = np.argmin(np.abs(x - right_contact[0]))
        right_slope = y_deriv[idx_r]

    # --- EXTRAPOLATION FIX For Snake/Rounded Corners ---
    # If the contact point is "floating" (y < substrate_y - epsilon)
    # OR if the slope is suspiciously flat (abs(slope) < 0.2 approx 11 deg)
    # we extrapolate the true contact from slightly higher up.
    
    # Left Extrapolation
    if left_contact is not None:
        # Check if we need to extrapolate
        dist_y = substrate_y - left_contact[1]
        is_floating = dist_y > 1.0 
        is_flat = abs(left_slope) < 0.2
        
        if is_floating or is_flat:
            # Find a point higher up (e.g., 5-15 pixels above contact)
            target_y_start = left_contact[1] - 5
            target_y_end = left_contact[1] - 20
            
            # Find points in this range on the left arc
            mask = (x <= apex[0]) & (y_smooth < target_y_start) & (y_smooth > target_y_end)
            if np.sum(mask) > 3:
                # Use median slope in this region to be robust
                region_slopes = y_deriv[mask]
                region_x = x[mask]
                region_y = y_smooth[mask]
                
                # Use the point closest to bottom of range for anchor
                anchor_idx = np.argmax(region_y) # Largest Y (lowest point in range)
                anchor_x = region_x[anchor_idx]
                anchor_y = region_y[anchor_idx]
                anchor_slope = region_slopes[anchor_idx]
                
                if abs(anchor_slope) > 0.05: # Avoid div by zero
                    # Extrapolate: x = x0 + (y_target - y0) / m
                    # y_target = substrate_y
                    new_x = anchor_x + (substrate_y - anchor_y) / anchor_slope
                    
                    dprint(f"Extrapolating Left CP: Old=({left_contact[0]:.1f}, {left_contact[1]:.1f}) m={left_slope:.3f} -> New=({new_x:.1f}, {substrate_y}) m={anchor_slope:.3f}")
                    left_contact = np.array([new_x, float(substrate_y)])
                    left_slope = anchor_slope

    # Right Extrapolation
    if right_contact is not None:
        dist_y = substrate_y - right_contact[1]
        is_floating = dist_y > 1.0 
        is_flat = abs(right_slope) < 0.2
        
        if is_floating or is_flat:
            target_y_start = right_contact[1] - 5
            target_y_end = right_contact[1] - 20
            
            mask = (x >= apex[0]) & (y_smooth < target_y_start) & (y_smooth > target_y_end)
            if np.sum(mask) > 3:
                region_slopes = y_deriv[mask]
                region_x = x[mask]
                region_y = y_smooth[mask]
                
                anchor_idx = np.argmax(region_y) 
                anchor_x = region_x[anchor_idx]
                anchor_y = region_y[anchor_idx]
                anchor_slope = region_slopes[anchor_idx]
                
                if abs(anchor_slope) > 0.05:
                    new_x = anchor_x + (substrate_y - anchor_y) / anchor_slope
                    
                    dprint(f"Extrapolating Right CP: Old=({right_contact[0]:.1f}, {right_contact[1]:.1f}) m={right_slope:.3f} -> New=({new_x:.1f}, {substrate_y}) m={anchor_slope:.3f}")
                    right_contact = np.array([new_x, float(substrate_y)])
                    right_slope = anchor_slope
    
    # Calculate angles in degrees (simple atan of slope)
    left_angle = np.degrees(np.arctan(abs(left_slope)))
    right_angle = np.degrees(np.arctan(abs(right_slope)))
    
    return {
        'x_smooth': x,
        'y_smooth': y_smooth,
        'apex': apex,
        'left_contact': left_contact,
        'right_contact': right_contact,
        'left_slope': float(left_slope),
        'right_slope': float(right_slope),
        'left_angle': float(left_angle),
        'right_angle': float(right_angle)
    }


def find_contact_points(x, y, substrate_y, apex_idx=None):
    """
    Find left & right contact points as line–curve intersections.
    Search direction is from Apex OUTWARDS to avoid picking up baseline noise far from the drop.
    """
    # Define line as y = m*x + b where m=0, b=substrate_y
    d = y - substrate_y
    
    if apex_idx is None:
         # Fallback if no apex provided: midpoint
         apex_idx = len(x) // 2

    def locate_contact(indices):
        # Scan indices in order. Look for sign change.
        # d[i] * d[i+1] < 0
        
        # We iterate through the provided indices. 
        # Note: indices must be contiguous in the array to use i, i+1 logic?
        # No, we just iterate the indices and check neighbors in the original array?
        # Simpler: Extract the subarray of 'd' corresponding to this side, 
        # then find the first crossing.
        
        # LEFT side: indices usually [apex_idx, apex_idx-1, ..., 0] -> d[apex]...d[0]
        # RIGHT side: indices usually [apex_idx, ..., len]
        
        for k in range(len(indices) - 1):
             i = indices[k]
             # For outward search, we compare i and the NEXT step in the search (indices[k+1])?
             # Or do we strictly look for crossing in the signal?
             # The signal is defined on x. We want to find x where d crosses 0.
             
             # If we are searching Left from Apex:
             # We check segment (x[i-1], x[i])?
             # Let's iterate index i moving away from apex.
             
             # Check crossing between i and i_next
             i_curr = indices[k]
             i_next = indices[k+1]
             
             # We assume x is sorted. so i_curr and i_next are adjacent in x (diff is 1).
             # Check if sign changes between them (or if we hit 0)
             # We want to catch the transition from Drop -> Substrate (Neg -> Zero/Pos)
             # If product is 0, it means one of them is 0.
             if np.sign(d[i_curr]) * np.sign(d[i_next]) <= 0:
                 # Found crossing
                 # Linear Approx:
                 # x = x_curr + t * (x_next - x_curr)
                 denom = d[i_next] - d[i_curr]
                 if denom == 0:
                     t = 0.5
                 else:
                     t = -d[i_curr] / denom
                 
                 x_c = x[i_curr] + t * (x[i_next] - x[i_curr])
                 y_c = substrate_y
                 return np.array([x_c, y_c])
        return None

    # Search Left: From Apex down to 0
    # indices: apex, apex-1, ..., 0
    left_indices = np.arange(apex_idx, -1, -1)
    left_cp = locate_contact(left_indices)
    
    # Search Right: From Apex up to end
    # indices: apex, apex+1, ..., len-1
    right_indices = np.arange(apex_idx, len(x))
    right_cp = locate_contact(right_indices)

    return left_cp, right_cp


def calculate_contact_angles(x, y_smooth, left_contact, right_contact, window=21, poly=3):
    """
    Calculate left and right contact angles using Savgol derivatives
    """
    # Get first derivative (slope)
    dy_dx = savgol_filter(y_smooth, window_length=window, polyorder=poly, deriv=1, delta=1.0)
    
    # Find indices closest to contact points
    left_idx = np.argmin(np.abs(x - left_contact[0]))
    right_idx = np.argmin(np.abs(x - right_contact[0]))
    
    # Slope at contact points
    slope_left = dy_dx[left_idx]
    slope_right = dy_dx[right_idx]
    
    # Convert to angles (in degrees)
    # Note: y-axis points down in images, so negate calculations appropriately
    # dy/dx in image coords: positive y is down.
    # Angle is usually measured inside the liquid.
    
    # For sessile drop:
    # Left side: slope is negative (going 'up' in geometric, 'down' in pixel Y as X increases? No.)
    # In pixels: Apex is low Y (min). Left side Y decreases as X approaches apex.
    # So left slope (dy/dx) is negative. 
    # Right side Y increases as X leaves apex. Slope positive.
    
    # Visual angle: 
    # tan(theta) = -slope ??
    # Let's stick to simple ATAN and standard conversion
    
    angle_left = np.degrees(np.arctan(-slope_left))
    angle_right = np.degrees(np.arctan(slope_right))
    
    # Ensure positive
    # If slope_left is negative (normal), -slope is positive.
    
    return angle_left, angle_right


def score_contour(cnt, roi_h, roi_y, needle_rect=None):
    """
    Score a contour based on area and position (Sessile bias: lower is better).
    Returns (score, centroid_y, area). score < 0 means invalid/skip.
    """
    area = cv2.contourArea(cnt)
    if area < 50: # Minimal noise filter
        return -1.0, 0.0, area

    # Get bounding box and centroid
    x, y, w, h = cv2.boundingRect(cnt)
    M = cv2.moments(cnt)
    if M["m00"] > 0:
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
    else:
        cx = x + w / 2
        cy = y + h / 2
    
    # Check needle overlap
    if needle_rect:
        nx, ny, nw, nh = needle_rect
        # Convert needle bottom to ROI-relative coordinate
        # needle_rect is likely in full image coords? 
        # Actually checking usage in run_test:
        # "needle_rect = calib_result.needle_rect ... if cy < needle_bottom_roi"
        # calib_result.needle_rect is in full image coords.
        
        needle_bottom_roi = (ny + nh) - roi_y
        
        # Skip if centroid is cleanly above the bottom of the needle
        # And within the top half of the ROI (to avoid filtering drops that just touch the line)
        if cy < needle_bottom_roi and cy < roi_h * 0.5:
             return -1.0, cy, area

    # Position score: cy / roi_h (0=top, 1=bottom)
    # Sessile drops are at the bottom.
    position_score = cy / roi_h 
    
    # Combined score
    if position_score > 0.3:
        # Boost contours in the lower part
        score = area * (1 + 2 * position_score)
    else:
        # Penalize top contours
        score = area * 0.2
        
    return score, cy, area



def run_test(
    image_path: str,
    substrate_y: int | None = None,
    benchmark: bool = False,
    bench_repeat: int = 3,
    bench_number: int = 1,
    benchmark_csv: str | None = None,
    **kwargs,
):
    """
    Main logic function:
    1. Loads image.
    2. Runs AutoCalibrator to detect features (unless overridden).
    3. Configures SessilePipeline context.
    4. Runs the Pipeline.
    5. Opens ResultWidget.
    """
    bench = Benchmarker(enabled=benchmark, repeat=bench_repeat, number=bench_number)

    app = bench.timed("qt_app_init", lambda: QApplication(sys.argv), benchmark=False)
    
    display_img = bench.timed("load_image", lambda: cv2.imread(image_path))
    if display_img is None:
        print("Failed to load image for display")
        if benchmark:
            bench.print_summary()
        if benchmark_csv:
            bench.write_csv(Path(benchmark_csv))
        return

    # ---------------------------------------------------------
    # 1. Run Auto-Calibration (Wizard Logic)
    # ---------------------------------------------------------
    dprint("Running AutoCalibrator...")
    calib_params = {}
    if "margin_fraction" in kwargs:
        calib_params["margin_fraction"] = kwargs["margin_fraction"]
    if "adaptive_block_size" in kwargs:
         calib_params["adaptive_block_size"] = kwargs["adaptive_block_size"]
    
    # AutoCalibrator attempts to find substrate line and drop contour automatically
    calib_result = bench.timed(
        "auto_calibration",
        lambda: run_auto_calibration(
            display_img, pipeline_name="sessile", **calib_params
        ),
    )
    
    # ---------------------------------------------------------
    # 2. Handle Manual Substrate Override
    # ---------------------------------------------------------
    if substrate_y is not None:
        with bench.span("manual_substrate_override"):
            print(f"Overriding Substrate Y: {substrate_y}")
            h, w = display_img.shape[:2]
            calib_result.substrate_line = ((0, substrate_y), (w, substrate_y))
            # Note: If we override the substrate, the auto-detected contour might be invalid 
            # (e.g., clipped at the wrong height). We handle this below.
    
    # ---------------------------------------------------------
    # 3. Configure Pipeline Arguments
    # ---------------------------------------------------------
    with bench.span("pipeline_setup"):
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
    # - If detector is "auto", we trust the auto-detected contour from calibration.
    # - If detector allows manual override (canny, otsu, etc.), we calculate it here and inject it.
    valid_contour_injected = False
    
    detector_name = kwargs.get("detector", "auto")
    dprint(f"DEBUG: Using detector strategy: {detector_name}")
    
    if detector_name == "auto":
        with bench.span("auto_contour_injection"):
            # Default behavior: use auto-calibrator result
            if calib_result.drop_contour is not None and len(calib_result.drop_contour) > 0:
                if substrate_y is None: # No manual override of substrate
                    pipeline_kwargs["drop_contour"] = calib_result.drop_contour
                    pipeline_kwargs["contour"] = Contour(xy=calib_result.drop_contour)
                    valid_contour_injected = True
                    print("Using AutoCalibrator contour.")
                else:
                    print("Manual substrate set: Skipping AutoCalibrator contour.")
    else:
        # Custom detection logic requested
        with bench.span(f"custom_detection_{detector_name}"):
            print(f"Running custom detection: {detector_name}")
            
            # Prepare image (grayscale)
            gray = cv2.cvtColor(display_img, cv2.COLOR_BGR2GRAY)
            
            # Determine ROI (use calibration result or full image)
            roi_x, roi_y, roi_w, roi_h = 0, 0, gray.shape[1], gray.shape[0]
            if calib_result.roi_rect:
                roi_x, roi_y, roi_w, roi_h = calib_result.roi_rect
                dprint(f"Using ROI: {calib_result.roi_rect}")
                
            roi_gray = gray[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]

            
            # Apply selected algorithm
            edges = None
            detected_xy = np.empty((0, 2), float)
            
            if detector_name == "canny":
                # Canny Edge Detection (standard)
                # Use settings similar to pipeline defaults or auto-calibrator
                enhanced = cv2.GaussianBlur(roi_gray, (5, 5), 0)
                edges = cv2.Canny(enhanced, 50, 150)
                
            elif detector_name == "otsu":
                # Otsu Thresholding
                # Use BINARY_INV because drop is dark on light background.
                # We want Drop=White (Foreground), Background=Black.
                blur = cv2.GaussianBlur(roi_gray, (5, 5), 0)
                _, edges = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                
            elif detector_name == "adaptive":
                # Adaptive Thresholding
                blur = cv2.GaussianBlur(roi_gray, (5, 5), 0)
                edges = cv2.adaptiveThreshold(
                    blur, 255,
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY_INV,
                    kwargs.get("adaptive_block_size", 21), 
                    2
                )
                
            elif detector_name == "threshold":
                # Simple Binary Thresholding
                # Inverse for Dark Drop
                _, edges = cv2.threshold(roi_gray, 128, 255, cv2.THRESH_BINARY_INV)
            
            elif detector_name == "log":
                # Laplacian of Gaussian (LoG) - absolute response
                sigma = kwargs.get("log_sigma", 1.0)
                min_grad = kwargs.get("log_min_gradient", 5.0)
                edges = log_edge_detection(roi_gray, sigma=sigma, use_zero_crossing=False, min_gradient=min_grad)
                
            elif detector_name == "log_zero_cross":
                # Laplacian of Gaussian with Zero-Crossing Detection
                # More precise edge localization using sign changes
                sigma = kwargs.get("log_sigma", 1.0)
                min_grad = kwargs.get("log_min_gradient", 5.0)
                edges = log_edge_detection(roi_gray, sigma=sigma, use_zero_crossing=True, min_gradient=min_grad)
            
            elif detector_name == "snake":
                # Snake (Active Contour) - refines an initial contour
                # Store ALL contours for debug visualization
                all_contours_debug = []
                
                # Try multiple methods to find initial contour
                # Method 1: Otsu thresholding (works well for high-contrast drops)
                _, otsu_mask = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                otsu_contours, _ = cv2.findContours(otsu_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                
                # Method 2: Canny with lower thresholds
                enhanced = cv2.GaussianBlur(roi_gray, (5, 5), 0)
                canny_edges = cv2.Canny(enhanced, 30, 100)  # Lower thresholds
                canny_contours, _ = cv2.findContours(canny_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                
                # Combine and deduplicate contours
                all_raw_contours = []
                for c in otsu_contours:
                    all_raw_contours.append((c, "otsu"))
                for c in canny_contours:
                    all_raw_contours.append((c, "canny"))
                
                # Store all for debug
                for idx, (c, src) in enumerate(all_raw_contours):
                    area = cv2.contourArea(c)
                    xy = c.reshape(-1, 2).astype(float)
                    xy[:, 0] += roi_x
                    xy[:, 1] += roi_y
                    label = f"{src}" if area > 500 else f"{src}-filtered"
                    all_contours_debug.append((xy, area, label))
                
                dprint(f"Snake: Found {len(otsu_contours)} Otsu + {len(canny_contours)} Canny contours")
                
                # Filter and Score contours
                scored_candidates = []
                needle_rect = calib_result.needle_rect if calib_result.needle_rect else None
                
                for c, src in all_raw_contours:
                    score, cy, area = score_contour(c, roi_h, roi_y, needle_rect)
                    
                    if score > 0:
                        scored_candidates.append((c, score, area, src))
                        dprint(f"Snake Candidate ({src}): score={score:.0f}, area={area:.0f}, y={cy:.1f}")
                    
                        # Debug info
                        xy = c.reshape(-1, 2).astype(float)
                        xy[:, 0] += roi_x
                        xy[:, 1] += roi_y
                        label = f"{src}-score:{int(score)}"
                        all_contours_debug.append((xy, area, label))
                
                if scored_candidates:
                    # Select the highest scored contour
                    scored_candidates.sort(key=lambda x: x[1], reverse=True)
                    best, best_score, best_area, best_src = scored_candidates[0]
                    initial_xy = best.reshape(-1, 2).astype(float)
                    
                    dprint(f"Snake: Selected '{best_src}' (score={best_score:.0f}, area={best_area:.0f}) as initial contour")
                    
                    # Apply snake refinement
                    # Mask reflection: Ensure snake doesn't go below substrate
                    snake_input_img = roi_gray.copy()
                    
                    eff_sub_y = substrate_y
                    if eff_sub_y is None and calib_result.substrate_line:
                         eff_sub_y = int((calib_result.substrate_line[0][1] + calib_result.substrate_line[1][1]) / 2)
                    
                    if eff_sub_y is not None:
                        sub_roi = eff_sub_y - roi_y
                        
                        # 1. Clamp initial contour to be above substrate (ROI coords)
                        # This prevents starting with points deep in the masked area
                        if 0 <= sub_roi < roi_h:
                            initial_xy[:, 1] = np.minimum(initial_xy[:, 1], sub_roi - 1)
                        
                        # 2. Mask image
                        if 0 <= sub_roi < roi_h:
                            dprint(f"Snake: Masking reflection below y={sub_roi} (ROI coords)")
                            # Fill below substrate with median background color or white (255)
                            snake_input_img[sub_roi:, :] = 255

                    
                    # Note: Otsu/Canny give edge-aligned contours, so expand_factor=1.0 (no expansion)
                    # CRITICAL: skimage.active_contour expects (row, col) -> (y, x)
                    # But OpenCV contours are (x, y). We must swap.
                    init_snake_rc = initial_xy[:, ::-1] # Swap columns: (x, y) -> (y, x)
                    
                    snake_rc = snake_contour_detection(
                        snake_input_img,
                        init_snake_rc,
                        alpha=kwargs.get("snake_alpha", 0.015),
                        beta=kwargs.get("snake_beta", 10.0),
                        gamma=kwargs.get("snake_gamma", 0.001),
                        max_px_move=kwargs.get("snake_max_px_move", 1.0),
                        max_iterations=kwargs.get("snake_iterations", 100),
                        w_line=kwargs.get("snake_w_line", -1.0), # Attract to dark lines
                        w_edge=kwargs.get("snake_w_edge", 1.0),
                    )
                    
                    # Convert back to (x, y) and absolute coords
                    snake_xy = snake_rc[:, ::-1]
                    snake_xy[:, 0] += roi_x
                    snake_xy[:, 1] += roi_y
                    detected_xy = snake_xy
                else:
                    dprint("Snake: No valid initial contours found (all filtered or too small)")
                
                # Store all contours for debug visualization
                pipeline_kwargs["_all_contours_debug"] = all_contours_debug
                
            # Extract contour from edges/mask
            if detector_name != "snake" and edges is not None:
                # Find all external contours
                contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                if contours:
                    dprint(f"Found {len(contours)} raw contours")
                    
                    # Filter by minimum length
                    valid_cnts = [c for c in contours if len(c) > 50]
                    dprint(f"After length filter: {len(valid_cnts)} contours")
                    
                    # Store ALL valid contours for debug
                    all_raw_contours = []
                    for i, c in enumerate(valid_cnts):
                        area = cv2.contourArea(c)
                        xy = c.reshape(-1, 2).astype(float)
                        xy[:, 0] += roi_x
                        xy[:, 1] += roi_y
                        all_raw_contours.append((xy, area, f"raw-{i}"))
                    
                    pipeline_kwargs["_all_contours_debug"] = all_raw_contours

                    if valid_cnts:
                        # Score each contour based on:
                        # 1. Area (larger is better for droplet)
                        # 2. Position (lower in ROI is better for sessile drop - closer to substrate)
                        # 3. Not overlapping with needle region
                        
                        scored_cnts = []
                        needle_rect = calib_result.needle_rect if calib_result.needle_rect else None
                        
                        for cnt in valid_cnts:
                            score, cy, area = score_contour(cnt, roi_h, roi_y, needle_rect)
                            if score > 0:
                                scored_cnts.append((cnt, score, area, cy))
                                dprint(f"Contour: area={area:.0f}, cy={cy:.1f}, score={score:.0f}")
                            else:
                                dprint(f"Skipping contour area={area:.0f} (filtered by score)")
                        
                        if scored_cnts:
                            # Sort by score descending
                            scored_cnts.sort(key=lambda x: x[1], reverse=True)
                            best = scored_cnts[0][0]
                            
                            dprint(f"Selected contour with score={scored_cnts[0][1]:.0f}, area={scored_cnts[0][2]:.0f}")
                            
                            detected_xy = best.reshape(-1, 2).astype(float)
                            # Offset back to full image
                            detected_xy[:, 0] += roi_x
                            detected_xy[:, 1] += roi_y
                            
                            # Inject into pipeline
                            pipeline_kwargs["drop_contour"] = detected_xy
                            pipeline_kwargs["contour"] = Contour(xy=detected_xy)
                            valid_contour_injected = True
                            dprint(f"Custom detection ({detector_name}) found contour with {len(detected_xy)} points")
                        else:
                            dprint(f"Custom detection ({detector_name}) found no valid contours after scoring")
                    else:
                        dprint(f"Custom detection ({detector_name}) found no valid contours (filtered by length)")
                else:
                    dprint(f"Custom detection ({detector_name}) found no contours")

    # Final Injection Check: If a contour was detected (e.g. by Snake) but not yet injected
    if not valid_contour_injected and len(detected_xy) > 0:
         dprint(f"Custom detection ({detector_name}) found contour with {len(detected_xy)} points")
         pipeline_kwargs["drop_contour"] = detected_xy
         pipeline_kwargs["contour"] = Contour(xy=detected_xy)
         valid_contour_injected = True

    if not valid_contour_injected:
         dprint("Warning: No valid contour injected. Pipeline will attempt fallback (likely Canny).")
         pipeline_kwargs["edge_detection_settings"] = None 
    
    dprint("Running Sessile Pipeline...")
    if "substrate_line" in pipeline_kwargs:
        dprint(f"DEBUG: Pipeline kwarg substrate_line={pipeline_kwargs['substrate_line']}")

    # ---------------------------------------------------------
    # 4. Execute Pipeline
    # ---------------------------------------------------------
    try:
        pipeline = SessilePipeline()
        ctx = bench.timed("pipeline_run", lambda: pipeline.run(**pipeline_kwargs))
        ctx.detector_name = detector_name  # Store detector name for display
        
        # Transfer debug contours to context for visualization
        if "_all_contours_debug" in pipeline_kwargs:
            ctx.all_contours = pipeline_kwargs["_all_contours_debug"]
        
        dprint("Pipeline finished.")
        if hasattr(ctx, "error") and ctx.error:
            dprint(f"Pipeline Error: {ctx.error}")
        
        # ---------------------------------------------------------
        # Optional: Run Savgol Smoothing on the detected contour
        # ---------------------------------------------------------
        # ---------------------------------------------------------
        # Optional: Run Savgol Smoothing / Analysis on the detected contour
        # ---------------------------------------------------------
        # We always want analysis (contact points, tangents) even if smooth_savgol is False
        if getattr(ctx, "contour", None) is not None:
            do_smooth = kwargs.get("smooth_savgol", False)
            with bench.span("savgol_smoothing"):
                dprint(f"Running Analysis (Smoothing={do_smooth})...")
                contour_pts = ctx.contour.xy
                final_sub_y = None
                
                # Try to get substrate from context or pipeline output
                if hasattr(ctx, "substrate_line") and ctx.substrate_line:
                    final_sub_y = int((ctx.substrate_line[0][1] + ctx.substrate_line[1][1]) / 2)
                elif hasattr(ctx.geometry, "baseline_y") and ctx.geometry.baseline_y:
                    final_sub_y = int(ctx.geometry.baseline_y)
                elif substrate_y is not None:
                    final_sub_y = substrate_y
                
                if final_sub_y is not None and len(contour_pts) > 0:
                    svg_res = smooth_and_analyze_droplet(
                        contour_pts, 
                        final_sub_y, 
                        window=kwargs.get("savgol_window", 21), 
                        poly=kwargs.get("savgol_poly", 3),
                        filter_monotonic=kwargs.get("filter_monotonic", False),
                        filter_substrate=kwargs.get("filter_substrate", False),
                        smoothing_enabled=do_smooth
                    )
                    
                    if svg_res:
                        # Angles are already calculated in smooth_and_analyze_droplet
                        la = svg_res.get('left_angle', 0.0)
                        ra = svg_res.get('right_angle', 0.0)
                        
                        ctx.savgol_results = svg_res  # Store for visualization
                        dprint(f"Analysis done. Angles: L={la:.1f}, R={ra:.1f}")
                        dprint(f"DEBUG: svg_res keys: {list(svg_res.keys())}")
                        
                        # Debug Print Tangents
                        if 'left_slope' in svg_res and svg_res['left_contact'] is not None:
                            m = svg_res['left_slope']
                            lc = svg_res['left_contact']
                            length = 60
                            dx_val = length / np.sqrt(1 + m**2)
                            dy_val = m * dx_val
                            dprint(f"DEBUG: Tangent Left | Start=({lc[0]:.2f}, {lc[1]:.2f}) | Vector=({dx_val:.2f}, {dy_val:.2f}) | Slope={m:.4f}")
                             
                        if 'right_slope' in svg_res and svg_res['right_contact'] is not None:
                            m = svg_res['right_slope']
                            rc = svg_res['right_contact']
                            length = 60
                            dx_val = length / np.sqrt(1 + m**2)
                            dy_val = m * dx_val
                            # Note: Drawing uses -dx, -dy for right side to go "up" (backwards in x)
                            dprint(f"DEBUG: Tangent Right | Start=({rc[0]:.2f}, {rc[1]:.2f}) | Vector=({-dx_val:.2f}, {-dy_val:.2f}) | Slope={m:.4f}")
                    else:
                        dprint("Savgol Smoothing failed (insufficient points?)")
                else:
                    dprint("Skipping Savgol: No substrate Y or contour available.")
            
        # Store active filters for display
        active = []
        if kwargs.get("filter_monotonic", False):
            active.append("Monotonic")
        if kwargs.get("filter_substrate", False):
            active.append("Substrate")
        if kwargs.get("smooth_savgol", False):
            active.append("Savgol")
        ctx.active_filters = active
             
    except Exception as e:
        print(f"Crash: {e}")
        import traceback
        traceback.print_exc()
        return

    # ---------------------------------------------------------
    # 5. Display Results
    # ---------------------------------------------------------
    win = bench.timed("ui_create", lambda: TestViewerWindow(ctx), benchmark=False)
    win.show()

    if benchmark or benchmark_csv:
        pipeline_timings = ctx.timings_ms if hasattr(ctx, "timings_ms") else None
        if benchmark:
            bench.print_summary(pipeline_timings=pipeline_timings)
        if benchmark_csv:
            bench.write_csv(Path(benchmark_csv), pipeline_timings=pipeline_timings)
    
    sys.exit(app.exec())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Sessile Pipeline with QPainter Overlay")
    parser.add_argument("image_path", help="Path to input image")
    parser.add_argument("--substrate-y", type=int, help="Manually set substrate Y coordinate", default=None)
    parser.add_argument("--margin-fraction", type=float, help="Fraction of width for margin analysis (default 0.05)", default=0.05)
    parser.add_argument("--block-size", type=int, help="Adaptive threshold block size (odd int, default 21)", default=21)
    
    parser.add_argument("--detector", choices=["auto", "canny", "otsu", "adaptive", "threshold", "log", "log_zero_cross", "snake"], 
                        default="auto", help="Choose edge detection method (default: auto)")
    parser.add_argument("--log-sigma", type=float, default=1.0, help="LoG sigma for Gaussian blur (default 1.0)")
    parser.add_argument("--log-min-gradient", type=float, default=5.0, help="LoG min gradient for zero-crossing (default 5.0, higher=less noise)")
    
    # Snake (Active Contour) parameters
    parser.add_argument("--snake-alpha", type=float, default=0.015, help="Snake length weight (default 0.015)")
    parser.add_argument("--snake-beta", type=float, default=10.0, help="Snake smoothness weight (default 10.0)")
    parser.add_argument("--snake-gamma", type=float, default=0.001, help="Snake time step (default 0.001)")
    parser.add_argument("--snake-iterations", type=int, default=500, help="Snake max iterations (default 500)")
    parser.add_argument("--snake-sigma", type=float, default=2.0, help="Snake Gaussian sigma for image smoothing (default 2.0)")
    parser.add_argument("--snake-expand", type=float, default=1.0, help="Snake initial contour expansion factor (default 1.0, no expansion)")
    
    # Optional smoothing
    parser.add_argument("--smooth-savgol", action="store_true", help="Enable Savgol smoothing of the dome")
    parser.add_argument("--debug-log", action="store_true", help="Print debug messages to terminal")
    parser.add_argument("--filter-monotonic", action="store_true", help="Filter contour to keep only upper point (min Y) per X")
    parser.add_argument("--filter-substrate", action="store_true", help="Filter points below substrate (Y > substrate_y)")
    parser.add_argument("--savgol-window", type=int, default=21, help="Savgol window length (default 21)")
    parser.add_argument("--savgol-poly", type=int, default=3, help="Savgol poly order (default 3)")
    parser.add_argument("--benchmark", action="store_true", help="Enable step benchmarks with timeit")
    parser.add_argument("--bench-repeat", type=int, default=3, help="Benchmark repeats (default 3)")
    parser.add_argument("--bench-number", type=int, default=1, help="Benchmark iterations per repeat (default 1)")
    parser.add_argument("--benchmark-csv", type=str, default=None, help="Write benchmark timings to CSV")

    args = parser.parse_args()
    
    if not Path(args.image_path).exists():
        print(f"Error: Image not found at {args.image_path}")
        sys.exit(1)
    
    DEBUG_LOG = bool(args.debug_log)

    run_test(
        args.image_path, 
        substrate_y=args.substrate_y,
        margin_fraction=args.margin_fraction,
        adaptive_block_size=args.block_size,
        detector=args.detector,
        smooth_savgol=args.smooth_savgol,
        filter_monotonic=args.filter_monotonic,
        filter_substrate=args.filter_substrate,
        savgol_window=args.savgol_window,
        savgol_poly=args.savgol_poly,
        log_sigma=args.log_sigma,
        log_min_gradient=args.log_min_gradient,
        snake_alpha=args.snake_alpha,
        snake_beta=args.snake_beta,
        snake_gamma=args.snake_gamma,
        snake_iterations=args.snake_iterations,
        snake_sigma=args.snake_sigma,
        snake_expand=args.snake_expand,
        benchmark=args.benchmark,
        bench_repeat=args.bench_repeat,
        bench_number=args.bench_number,
        benchmark_csv=args.benchmark_csv,
    )

