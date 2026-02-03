"""
Automatic region detection for calibration wizard.

This module provides automatic detection of substrate, needle, drop contour,
and ROI regions for the calibration wizard. Detection strategies are
pipeline-specific (sessile vs pendant).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, Tuple, List

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CalibrationResult:
    """Results from automatic calibration detection."""
    
    # Substrate line as ((x1, y1), (x2, y2)) - horizontal baseline
    substrate_line: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None
    
    # Needle region as (x, y, width, height) if detected
    needle_rect: Optional[Tuple[int, int, int, int]] = None
    
    # Drop contour as Nx2 array of (x, y) points
    drop_contour: Optional[np.ndarray] = None
    
    # ROI as (x, y, width, height) encompassing the region of interest
    roi_rect: Optional[Tuple[int, int, int, int]] = None
    
    # Contact points (left and right) where drop meets substrate/needle
    contact_points: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None
    
    # Apex point (bottom for pendant, top for sessile)
    apex_point: Optional[Tuple[int, int]] = None
    
    # Confidence scores for each detection (0.0 - 1.0)
    confidence_scores: dict = field(default_factory=dict)
    
    # Enhanced image used for detection (for preview)
    enhanced_image: Optional[np.ndarray] = None
    
    # Binary mask from segmentation (for preview)
    binary_mask: Optional[np.ndarray] = None


class AutoCalibrator:
    """
    Automatic region detection for calibration wizard.
    
    Uses CLAHE for contrast enhancement and adaptive/Otsu thresholding
    for robust segmentation across different image conditions.
    """
    
    def __init__(
        self,
        image: np.ndarray,
        pipeline_name: str = "sessile",
        *,
        clahe_clip_limit: float = 2.0,
        clahe_tile_size: Tuple[int, int] = (8, 8),
        adaptive_block_size: int = 21,
        adaptive_c: int = 2,
        margin_fraction: float = 0.05,
        min_area_fraction: float = 0.005,
        roi_padding: int = 20,
    ) -> None:
        """
        Initialize the auto-calibrator.
        
        Args:
            image: Input image (BGR or grayscale)
            pipeline_name: Pipeline type ("sessile", "pendant", etc.)
            clahe_clip_limit: CLAHE clip limit for contrast enhancement
            clahe_tile_size: CLAHE tile grid size
            adaptive_block_size: Block size for adaptive thresholding
            adaptive_c: Constant subtracted from mean in adaptive thresholding
            margin_fraction: Fraction of image width for substrate detection margin
            min_area_fraction: Minimum contour area as fraction of image area
            roi_padding: Padding pixels around detected regions for ROI
        """
        self.original_image = image.copy()
        self.pipeline_name = pipeline_name.lower()
        
        # Parameters
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_tile_size = clahe_tile_size
        self.adaptive_block_size = adaptive_block_size
        self.adaptive_c = adaptive_c
        self.margin_fraction = margin_fraction
        self.min_area_fraction = min_area_fraction
        self.roi_padding = roi_padding
        
        # Derived properties
        if len(image.shape) == 3:
            self.gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            self.gray = image.copy()
        
        self.height, self.width = self.gray.shape[:2]
        self.image_area = self.height * self.width
        
        # Apply CLAHE for contrast enhancement
        clahe = cv2.createCLAHE(
            clipLimit=self.clahe_clip_limit,
            tileGridSize=self.clahe_tile_size
        )
        self.enhanced_gray = clahe.apply(self.gray)
        
        # Internal state
        self._substrate_y: Optional[int] = None
        self._needle_contour: Optional[np.ndarray] = None
        self._needle_rect: Optional[Tuple[int, int, int, int]] = None
        self._drop_contour: Optional[np.ndarray] = None
        self._binary_clean: Optional[np.ndarray] = None
    
    def detect_all(self) -> CalibrationResult:
        """
        Run full detection pipeline and return results.
        
        Routes to pipeline-specific detection strategy.
        
        Returns:
            CalibrationResult with all detected regions and confidence scores.
        """
        if self.pipeline_name == "pendant":
            return self._detect_pendant()
        else:
            return self._detect_sessile()
    
    def _detect_sessile(self) -> CalibrationResult:
        """Run sessile drop detection pipeline."""
        result = CalibrationResult()
        result.enhanced_image = self.enhanced_gray.copy()
        
        # Step 1: Detect substrate baseline
        substrate_line, substrate_conf = self._detect_substrate()
        if substrate_line:
            result.substrate_line = substrate_line
            result.confidence_scores["substrate"] = substrate_conf
            logger.info(f"Substrate detected at y={self._substrate_y} (conf={substrate_conf:.2f})")
        
        # Step 2: Segment image using adaptive thresholding
        self._segment_image_adaptive()
        result.binary_mask = self._binary_clean.copy() if self._binary_clean is not None else None
        
        # Step 3: Detect needle
        needle_rect, needle_conf = self._detect_needle_sessile()
        if needle_rect:
            result.needle_rect = needle_rect
            result.confidence_scores["needle"] = needle_conf
            logger.info(f"Needle detected: {needle_rect} (conf={needle_conf:.2f})")
        
        # Step 4: Detect drop contour
        drop_contour, contact_pts, drop_conf = self._detect_drop_sessile()
        if drop_contour is not None and len(drop_contour) > 0:
            result.drop_contour = drop_contour
            result.contact_points = contact_pts
            result.confidence_scores["drop"] = drop_conf
            logger.info(f"Drop detected with {len(drop_contour)} points (conf={drop_conf:.2f})")
        
        # Step 5: Compute ROI from detected regions
        roi_rect, roi_conf = self._compute_roi(result)
        if roi_rect:
            result.roi_rect = roi_rect
            result.confidence_scores["roi"] = roi_conf
            logger.info(f"ROI computed: {roi_rect} (conf={roi_conf:.2f})")
        
        # Compute overall confidence
        if result.confidence_scores:
            result.confidence_scores["overall"] = sum(result.confidence_scores.values()) / len(result.confidence_scores)
        
        return result
    
    def _detect_pendant(self) -> CalibrationResult:
        """
        Run pendant drop detection pipeline.
        
        Uses Otsu thresholding (better for high-contrast silhouettes).
        Detects needle by shaft lines and contact points where contour deviates.
        """
        result = CalibrationResult()
        result.enhanced_image = self.enhanced_gray.copy()
        
        # Step 1: Segment using Otsu thresholding
        self._segment_image_otsu()
        result.binary_mask = self._binary_clean.copy() if self._binary_clean is not None else None
        
        # Step 2: Find the main drop contour
        drop_cnt, drop_conf = self._find_pendant_drop_contour()
        if drop_cnt is None:
            logger.warning("Pendant: Could not find drop contour")
            return result
        
        # Step 3: Detect needle and contact points
        needle_rect, contact_pts, needle_conf = self._detect_needle_pendant(drop_cnt)
        if needle_rect:
            result.needle_rect = needle_rect
            result.contact_points = contact_pts
            result.confidence_scores["needle"] = needle_conf
            logger.info(f"Pendant needle detected: {needle_rect} (conf={needle_conf:.2f})")
        
        # Step 4: Find apex (bottom of drop)
        apex_pt, apex_conf = self._detect_apex_pendant(drop_cnt)
        if apex_pt:
            result.apex_point = apex_pt
            result.confidence_scores["apex"] = apex_conf
            logger.info(f"Pendant apex detected at {apex_pt} (conf={apex_conf:.2f})")
        
        # Step 5: Set drop contour
        drop_contour = drop_cnt.reshape(-1, 2).astype(np.float64)
        result.drop_contour = drop_contour
        result.confidence_scores["drop"] = drop_conf
        
        # Step 6: Compute ROI
        roi_rect, roi_conf = self._compute_roi_pendant(result)
        if roi_rect:
            result.roi_rect = roi_rect
            result.confidence_scores["roi"] = roi_conf
            logger.info(f"Pendant ROI computed: {roi_rect} (conf={roi_conf:.2f})")
        
        # Compute overall confidence
        if result.confidence_scores:
            result.confidence_scores["overall"] = sum(result.confidence_scores.values()) / len(result.confidence_scores)
        
        return result
    
    # ======================== Segmentation Methods ========================
    
    def _segment_image_adaptive(self) -> None:
        """Segment using adaptive thresholding (for sessile)."""
        blur = cv2.GaussianBlur(self.enhanced_gray, (5, 5), 0)
        
        binary = cv2.adaptiveThreshold(
            blur, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            self.adaptive_block_size,
            self.adaptive_c
        )
        
        # Mask below substrate line
        if self._substrate_y is not None:
            binary[self._substrate_y - 2:, :] = 0
        
        # Morphological cleanup
        kernel = np.ones((3, 3), np.uint8)
        binary_clean = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
        binary_clean = cv2.morphologyEx(binary_clean, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        self._binary_clean = binary_clean
    
    def _segment_image_otsu(self) -> None:
        """Segment using Otsu thresholding (for pendant high-contrast)."""
        blur = cv2.GaussianBlur(self.gray, (5, 5), 0)
        
        _, binary = cv2.threshold(
            blur, 0, 255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        
        # Morphological cleanup
        kernel = np.ones((5, 5), np.uint8)
        binary_clean = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
        
        self._binary_clean = binary_clean
    
    # ======================== Sessile Detection Methods ========================
    
    def _detect_substrate(self) -> Tuple[Optional[Tuple[Tuple[int, int], Tuple[int, int]]], float]:
        """Detect substrate baseline using gradient analysis on image margins."""
        margin_px = max(10, min(50, int(self.width * self.margin_fraction)))
        
        left_strip = self.enhanced_gray[:, 0:margin_px]
        right_strip = self.enhanced_gray[:, self.width - margin_px:self.width]
        
        y_left = self._find_horizon_median(left_strip)
        y_right = self._find_horizon_median(right_strip)
        
        if y_left is None and y_right is None:
            self._substrate_y = int(self.height * 0.8)
            return ((0, self._substrate_y), (self.width, self._substrate_y)), 0.3
        
        if y_left is None:
            y_left = y_right
        if y_right is None:
            y_right = y_left
        
        self._substrate_y = int((y_left + y_right) / 2)
        
        diff = abs(y_left - y_right)
        max_diff = self.height * 0.1
        consistency = max(0.0, 1.0 - diff / max_diff)
        confidence = 0.5 + 0.5 * consistency
        
        substrate_line = ((0, self._substrate_y), (self.width, self._substrate_y))
        return substrate_line, confidence
    
    def _find_horizon_median(self, strip_gray: np.ndarray) -> Optional[int]:
        """Find horizon line in a vertical strip using gradient analysis."""
        detected_ys: List[int] = []
        h, w = strip_gray.shape
        min_limit, max_limit = int(h * 0.05), int(h * 0.95)
        
        for col in range(w):
            col_data = strip_gray[:, col].astype(float)
            grad = np.diff(col_data)
            valid_grad = grad[min_limit:max_limit]
            
            if len(valid_grad) == 0:
                continue
            
            best_idx = np.argmin(valid_grad)
            best_y = best_idx + min_limit
            detected_ys.append(best_y)
        
        if not detected_ys:
            return None
        
        return int(np.median(detected_ys))
    
    def _detect_needle_sessile(self) -> Tuple[Optional[Tuple[int, int, int, int]], float]:
        """Detect needle region (contour touching top border) for sessile."""
        if self._binary_clean is None:
            return None, 0.0
        
        contours, _ = cv2.findContours(
            self._binary_clean,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            return None, 0.0
        
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            
            if y < 5:
                self._needle_contour = cnt
                self._needle_rect = (x, y, w, h)  # Store for drop detection filter
                aspect_ratio = h / max(w, 1)
                if aspect_ratio > 2:
                    confidence = min(1.0, 0.7 + 0.1 * min(aspect_ratio / 5, 3))
                else:
                    confidence = 0.5
                
                return (x, y, w, h), confidence
        
        return None, 0.0
    
    def _detect_drop_sessile(self) -> Tuple[Optional[np.ndarray], Optional[Tuple[Tuple[int, int], Tuple[int, int]]], float]:
        """Detect drop contour and contact points for sessile.
        
        For sessile drops, the contour must:
        - Be well below the needle (50px gap minimum)
        - Touch the substrate line
        - Not be rectangular (ROI boundaries)
        """
        if self._binary_clean is None:
            return None, None, 0.0
        
        contours, _ = cv2.findContours(
            self._binary_clean,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            return None, None, 0.0
        
        center_x = self.width // 2
        min_area = self.image_area * self.min_area_fraction
        substrate_touch_tolerance = 15  # pixels
        
        # Separate contours: those touching substrate vs floating
        substrate_contours = []
        floating_contours = []
        
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = cv2.contourArea(cnt)
            top_y = y
            bottom_y = y + h
            
            # Skip very small contours
            if area < min_area:
                continue
            
            # Skip contours touching image edges (except bottom/substrate)
            if x < 5 or (x + w) > (self.width - 5):
                continue
            
            # ===== NEEDLE FILTER =====
            # Skip contours too close to needle (pendant drops)
            if self._needle_rect is not None:
                n_x, n_y, n_w, n_h = self._needle_rect
                needle_bottom = n_y + n_h
                needle_center_x = n_x + n_w // 2
                
                # Sessile drop must start well BELOW needle bottom (50px gap)
                min_gap_from_needle = 50
                if top_y < needle_bottom + min_gap_from_needle:
                    logger.debug(f"Skipping contour near needle (top_y={top_y}, needle_bottom={needle_bottom})")
                    continue
                
                # Skip contours horizontally aligned with needle if close to it
                cnt_center_x = x + w // 2
                if abs(cnt_center_x - needle_center_x) < n_w and top_y < needle_bottom + 100:
                    logger.debug("Skipping contour aligned with needle")
                    continue
            
            # ===== RECTANGULARITY FILTER =====
            # Skip rectangular contours (likely ROI boundaries, not droplets)
            rect_area = w * h
            if rect_area > 0:
                rectangularity = area / rect_area
                # Perfect rectangle = 1.0, circle/ellipse ≈ 0.78
                if rectangularity > 0.85:
                    logger.debug(f"Skipping rectangular contour (rect={rectangularity:.2f})")
                    continue
            
            # ===== REFLECTION FILTER =====
            # Center must be ABOVE substrate (not a reflection below substrate)
            if self._substrate_y is not None:
                cnt_cy = y + h // 2
                if cnt_cy > self._substrate_y:
                    continue
            
            # Calculate metrics for sorting
            cnt_center_x = x + w // 2
            distance_from_center = abs(cnt_center_x - center_x)
            
            # ===== SUBSTRATE CONTACT CHECK =====
            if self._substrate_y is not None:
                distance_to_substrate = abs(bottom_y - self._substrate_y)
                touches_substrate = distance_to_substrate <= substrate_touch_tolerance
                
                if touches_substrate:
                    substrate_contours.append((cnt, area, distance_from_center, distance_to_substrate))
                else:
                    floating_contours.append((cnt, area, distance_from_center, distance_to_substrate))
            else:
                floating_contours.append((cnt, area, distance_from_center, 0))
        
        # Prefer substrate-touching contours
        if substrate_contours:
            # Sort by: closest to substrate, then largest area, then closest to center
            substrate_contours.sort(key=lambda x: (x[3], -x[1], x[2]))
            valid_contours = substrate_contours
            logger.info(f"Found {len(substrate_contours)} substrate-touching contour(s)")
        elif floating_contours:
            floating_contours.sort(key=lambda x: (-x[1], x[2]))
            valid_contours = floating_contours
            logger.warning("No substrate-touching contours found, using floating contours")
        else:
            # No valid contours found after filtering - try fallback
            if contours:
                largest = max(contours, key=cv2.contourArea)
                if cv2.contourArea(largest) > min_area:
                    # Set as best_cnt and let the contact point detection below handle it
                    self._drop_contour = largest
                    best_cnt = largest
                    valid_contours = [(largest, cv2.contourArea(largest), 0, 0)]
                    logger.info("Using fallback: largest contour from unfiltered list")
                else:
                    return None, None, 0.0
            else:
                return None, None, 0.0
        
        # Select best contour from valid_contours
        best_cnt = valid_contours[0][0]
        self._drop_contour = best_cnt
        
        hull = cv2.convexHull(best_cnt)
        points = hull[:, 0, :]
        
        if self._substrate_y is not None:
            # Find contact points: leftmost and rightmost points near substrate
            substrate_tolerance = 20  # pixels tolerance for "near substrate"
            
            # Get points that are near the substrate line
            near_substrate = [pt for pt in points 
                              if abs(pt[1] - self._substrate_y) <= substrate_tolerance]
            
            # Get dome points (above substrate, including boundary points)
            # We accept points up to substrate_tolerance below substrate (handling reflection)
            # but we CLAMP them to the substrate level to ensure a flat baseline.
            dome_raw = [pt for pt in points if pt[1] <= (self._substrate_y + substrate_tolerance)]
            dome_points = []
            for pt in dome_raw:
                if pt[1] > self._substrate_y:
                    # Clamp to substrate line
                    dome_points.append(np.array([pt[0], self._substrate_y], dtype=pt.dtype))
                else:
                    dome_points.append(pt)
            
            # Find contact points from near_substrate or from the whole contour
            if near_substrate:
                # Contact points are leftmost and rightmost near substrate
                sorted_near = sorted(near_substrate, key=lambda p: p[0])
                x_left = sorted_near[0][0]
                x_right = sorted_near[-1][0]
            elif len(points) > 0:
                # Fallback: use leftmost and rightmost of the whole contour
                sorted_pts = sorted(points, key=lambda p: p[0])
                x_left = sorted_pts[0][0]
                x_right = sorted_pts[-1][0]
            else:
                # No valid points
                drop_contour = points.astype(np.float64)
                return drop_contour, None, 0.4
            
            # Contact points are at the substrate Y level
            cp_left = (int(x_left), self._substrate_y)
            cp_right = (int(x_right), self._substrate_y)
            contact_points = (cp_left, cp_right)
            
            if dome_points:
                dome_points = sorted(dome_points, key=lambda p: p[0])
                
                # Build a closed polygon:
                # 1. Start at left contact point (on substrate)
                # 2. Trace dome contour from left to right
                # 3. End at right contact point (on substrate)
                # 4. Close back to left contact point (the baseline)
                final_polygon = np.array(
                    [[x_left, self._substrate_y]] +  # Left contact point
                    [[p[0], p[1]] for p in dome_points] +  # Dome contour
                    [[x_right, self._substrate_y]] +  # Right contact point
                    [[x_left, self._substrate_y]],  # Close polygon back to start
                    dtype=np.float64
                )
            else:
                # No dome points - use convex hull but add contact points
                all_pts = sorted(points, key=lambda p: p[0])
                final_polygon = np.array(
                    [[x_left, self._substrate_y]] +  # Left contact point
                    [[p[0], p[1]] for p in all_pts] +  # All contour points
                    [[x_right, self._substrate_y]] +  # Right contact point
                    [[x_left, self._substrate_y]],  # Close polygon
                    dtype=np.float64
                )
            
            hull_area = cv2.contourArea(hull)
            cnt_area = cv2.contourArea(best_cnt)
            solidity = cnt_area / max(hull_area, 1)
            confidence = min(1.0, solidity + 0.2)
            
            logger.info(f"Contact points detected: left={cp_left}, right={cp_right}")
            return final_polygon, contact_points, confidence
        
        drop_contour = points.astype(np.float64)
        return drop_contour, None, 0.6
    
    # ======================== Pendant Detection Methods ========================
    
    def _find_pendant_drop_contour(self) -> Tuple[Optional[np.ndarray], float]:
        """Find the main drop contour for pendant (largest, centered)."""
        if self._binary_clean is None:
            return None, 0.0
        
        contours, _ = cv2.findContours(
            self._binary_clean,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            return None, 0.0
        
        img_center_x = self.width // 2
        min_area = self.image_area * 0.05  # At least 5% of image
        
        valid_contours = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
            
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                # Must be roughly centered (within 30% of center)
                if abs(cx - img_center_x) < (self.width * 0.3):
                    valid_contours.append((cnt, area))
        
        if not valid_contours:
            return None, 0.0
        
        # Select largest valid contour
        drop_cnt = max(valid_contours, key=lambda x: x[1])[0]
        self._drop_contour = drop_cnt
        
        # Confidence based on size and centering
        area = cv2.contourArea(drop_cnt)
        area_ratio = area / self.image_area
        confidence = min(1.0, 0.6 + area_ratio * 2)
        
        return drop_cnt, confidence
    
    def _detect_needle_pendant(
        self, drop_cnt: np.ndarray
    ) -> Tuple[Optional[Tuple[int, int, int, int]], Optional[Tuple[Tuple[int, int], Tuple[int, int]]], float]:
        """
        Detect needle region for pendant drop using shaft line analysis.
        
        Finds the vertical shaft lines at the top of the contour, then
        walks down to find where the contour deviates (contact points).
        """
        x, y, w, h = cv2.boundingRect(drop_cnt)
        pts = drop_cnt.reshape(-1, 2)
        
        # Define needle shaft reference (top 20 pixels)
        top_limit = y + 20
        
        # Left shaft line: median X of points in top-left quadrant
        left_shaft_pts = pts[(pts[:, 1] < top_limit) & (pts[:, 0] < (x + w/2))]
        if len(left_shaft_pts) == 0:
            return None, None, 0.0
        ref_x_left = np.median(left_shaft_pts[:, 0])
        
        # Right shaft line: median X of points in top-right quadrant
        right_shaft_pts = pts[(pts[:, 1] < top_limit) & (pts[:, 0] > (x + w/2))]
        if len(right_shaft_pts) == 0:
            return None, None, 0.0
        ref_x_right = np.median(right_shaft_pts[:, 0])
        
        # Tolerance: how many pixels "out" counts as drop starting?
        tolerance = 3
        
        # Create mask for precise scanning
        mask = np.zeros((self.height, self.width), dtype=np.uint8)
        cv2.drawContours(mask, [drop_cnt], -1, 255, 1)
        
        # Find left contact point (where contour moves left of shaft)
        contact_y_left = y
        contact_x_left = int(ref_x_left)
        
        for cy in range(y, y + h):
            row = mask[cy, 0:int(x + w/2)]  # Left half
            indices = np.where(row > 0)[0]
            if len(indices) > 0:
                current_x = indices[0]  # Leftmost pixel
                if current_x < (ref_x_left - tolerance):
                    contact_y_left = cy
                    contact_x_left = current_x
                    break
        
        # Find right contact point (where contour moves right of shaft)
        contact_y_right = y
        contact_x_right = int(ref_x_right)
        
        for cy in range(y, y + h):
            row = mask[cy, int(x + w/2):self.width]  # Right half
            indices = np.where(row > 0)[0]
            if len(indices) > 0:
                current_x = indices[-1] + int(x + w/2)  # Rightmost pixel
                if current_x > (ref_x_right + tolerance):
                    contact_y_right = cy
                    contact_x_right = current_x
                    break
        
        # Needle bottom is the higher of the two contact points
        needle_bottom = min(contact_y_left, contact_y_right)
        
        # Build needle rectangle
        needle_x = int(ref_x_left)
        needle_y = y
        needle_w = int(ref_x_right - ref_x_left)
        needle_h = needle_bottom - y
        
        if needle_w <= 0 or needle_h <= 0:
            return None, None, 0.0
        
        needle_rect = (needle_x, needle_y, needle_w, needle_h)
        contact_points = (
            (contact_x_left, contact_y_left),
            (contact_x_right, contact_y_right)
        )
        
        # Confidence based on needle aspect ratio
        aspect = needle_h / max(needle_w, 1)
        confidence = min(1.0, 0.6 + 0.1 * min(aspect, 4))
        
        return needle_rect, contact_points, confidence
    
    def _detect_apex_pendant(self, drop_cnt: np.ndarray) -> Tuple[Optional[Tuple[int, int]], float]:
        """Detect apex point (bottom of pendant drop)."""
        pts = drop_cnt.reshape(-1, 2)
        
        # Apex is the point with maximum Y (bottom of drop)
        apex_idx = np.argmax(pts[:, 1])
        apex_pt = pts[apex_idx]
        
        apex = (int(apex_pt[0]), int(apex_pt[1]))
        
        # High confidence - apex is straightforward to find
        return apex, 0.95
    
    def _compute_roi_pendant(
        self, result: CalibrationResult
    ) -> Tuple[Optional[Tuple[int, int, int, int]], float]:
        """Compute ROI for pendant drop (from needle to apex)."""
        if result.drop_contour is None:
            return None, 0.0
        
        contour = np.asarray(result.drop_contour)
        x_min = int(np.min(contour[:, 0]))
        x_max = int(np.max(contour[:, 0]))
        y_min = int(np.min(contour[:, 1]))
        y_max = int(np.max(contour[:, 1]))
        
        # Include apex with padding
        if result.apex_point:
            y_max = max(y_max, result.apex_point[1])
        
        # Apply padding
        pad = self.roi_padding
        x_min = max(0, x_min - pad)
        y_min = max(0, y_min)  # Start from top of drop (needle)
        x_max = min(self.width, x_max + pad)
        y_max = min(self.height, y_max + pad)
        
        roi_width = x_max - x_min
        roi_height = y_max - y_min
        
        if roi_width <= 0 or roi_height <= 0:
            return None, 0.0
        
        return (x_min, y_min, roi_width, roi_height), 0.9
    
    # ======================== Common Methods ========================
    
    def _compute_roi(
        self, result: CalibrationResult
    ) -> Tuple[Optional[Tuple[int, int, int, int]], float]:
        """Compute ROI rectangle encompassing detected regions (sessile)."""
        x_min, y_min = self.width, self.height
        x_max, y_max = 0, 0
        
        has_data = False
        
        if result.drop_contour is not None and len(result.drop_contour) > 0:
            contour = np.asarray(result.drop_contour)
            x_min = min(x_min, int(np.min(contour[:, 0])))
            x_max = max(x_max, int(np.max(contour[:, 0])))
            y_min = min(y_min, int(np.min(contour[:, 1])))
            y_max = max(y_max, int(np.max(contour[:, 1])))
            has_data = True
        
        if result.substrate_line and self._substrate_y is not None:
            y_max = max(y_max, self._substrate_y)
            has_data = True
        
        if result.needle_rect:
            nx, ny, nw, nh = result.needle_rect
            needle_include_y = ny + int(nh * 0.7)
            y_min = min(y_min, needle_include_y)
            x_min = min(x_min, nx)
            x_max = max(x_max, nx + nw)
            has_data = True
        
        if not has_data:
            return None, 0.0
        
        pad = self.roi_padding
        x_min = max(0, x_min - pad)
        y_min = max(0, y_min - pad)
        x_max = min(self.width, x_max + pad)
        y_max = min(self.height, y_max + pad)
        
        roi_width = x_max - x_min
        roi_height = y_max - y_min
        
        if roi_width <= 0 or roi_height <= 0:
            return None, 0.0
        
        area_ratio = (roi_width * roi_height) / self.image_area
        if area_ratio < 0.1:
            confidence = 0.5
        elif area_ratio > 0.8:
            confidence = 0.6
        else:
            confidence = 0.9
        
        return (x_min, y_min, roi_width, roi_height), confidence


def run_auto_calibration(
    image: np.ndarray,
    pipeline_name: str = "sessile",
    **kwargs
) -> CalibrationResult:
    """
    Convenience function to run auto-calibration.
    
    Args:
        image: Input image (BGR or grayscale)
        pipeline_name: Pipeline type ("sessile", "pendant", etc.)
        **kwargs: Additional parameters for AutoCalibrator
        
    Returns:
        CalibrationResult with detected regions.
    """
    calibrator = AutoCalibrator(image, pipeline_name, **kwargs)
    return calibrator.detect_all()
