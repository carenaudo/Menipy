"""
Automatic region detection for calibration wizard.

This module provides automatic detection of substrate, needle, drop contour,
and ROI regions for the calibration wizard. Detection strategies are
pipeline-specific (sessile vs pendant).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from types import SimpleNamespace

import cv2
import numpy as np

from menipy.common.detection_result import normalize_detection_result
from menipy.common.geometry_prototypes import detect_bilateral_needle
from menipy.common.sessile_detection import (
    detect_sessile_drop_contour,
    detect_sessile_needle_shaft,
    detect_sessile_substrate_line,
    segment_sessile_binary,
)

logger = logging.getLogger(__name__)


@dataclass
class CalibrationResult:
    """Results from automatic calibration detection."""

    # Substrate line as ((x1, y1), (x2, y2)) - horizontal baseline
    substrate_line: tuple[tuple[int, int], tuple[int, int]] | None = None

    # Needle region as (x, y, width, height) if detected
    needle_rect: tuple[int, int, int, int] | None = None

    # Drop contour as Nx2 array of (x, y) points
    drop_contour: np.ndarray | None = None

    # ROI as (x, y, width, height) encompassing the region of interest
    roi_rect: tuple[int, int, int, int] | None = None

    # Contact points (left and right) where drop meets substrate/needle
    contact_points: tuple[tuple[int, int], tuple[int, int]] | None = None

    # Apex point (bottom for pendant, top for sessile)
    apex_point: tuple[int, int] | None = None

    # Confidence scores for each detection (0.0 - 1.0)
    confidence_scores: dict = field(default_factory=dict)
    detector_diagnostics: dict = field(default_factory=dict)

    # Enhanced image used for detection (for preview)
    enhanced_image: np.ndarray | None = None

    # Binary mask from segmentation (for preview)
    binary_mask: np.ndarray | None = None


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
        clahe_tile_size: tuple[int, int] = (8, 8),
        adaptive_block_size: int = 21,
        adaptive_c: int = 2,
        margin_fraction: float = 0.05,
        min_area_fraction: float = 0.005,
        roi_padding: int = 20,
        experimental_geometry_mode: str = "off",
        needle_geometry_method: str = "legacy",
        onnx_proposal_mode: str = "off",
        segmentation_provider: str = "mobilesam",
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
        self.experimental_geometry_mode = str(experimental_geometry_mode)
        self.needle_geometry_method = str(needle_geometry_method)
        self.onnx_proposal_mode = str(onnx_proposal_mode)
        self.segmentation_provider = str(segmentation_provider)

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
            clipLimit=self.clahe_clip_limit, tileGridSize=self.clahe_tile_size
        )
        self.enhanced_gray = clahe.apply(self.gray)

        # Internal state
        self._substrate_y: int | None = None
        self._needle_contour: np.ndarray | None = None
        self._needle_rect: tuple[int, int, int, int] | None = None
        self._drop_contour: np.ndarray | None = None
        self._binary_clean: np.ndarray | None = None

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
        substrate_outcome = normalize_detection_result(
            substrate_line, feature="substrate", confidence=substrate_conf
        )
        result.detector_diagnostics["substrate"] = substrate_outcome.to_diagnostics()
        if substrate_line:
            result.substrate_line = substrate_line
            result.confidence_scores["substrate"] = substrate_conf
            logger.info(
                f"Substrate detected at y={self._substrate_y} (conf={substrate_conf:.2f})"
            )

        # Step 2: Segment image using adaptive thresholding
        self._segment_image_adaptive()
        result.binary_mask = (
            self._binary_clean.copy() if self._binary_clean is not None else None
        )

        # Step 3: Detect needle
        needle_rect, needle_conf = self._detect_needle_sessile()
        if self.needle_geometry_method == "bilateral_robust" or self.experimental_geometry_mode == "shadow":
            experimental = detect_bilateral_needle(self.original_image)
            result.detector_diagnostics["needle_bilateral"] = experimental.to_diagnostics()
            if self.needle_geometry_method == "bilateral_robust" and experimental.accepted:
                needle_rect = experimental.value["needle_rect"]
                needle_conf = experimental.confidence
            elif self.needle_geometry_method == "bilateral_robust":
                needle_rect = None
                needle_conf = 0.0
        needle_outcome = normalize_detection_result(
            needle_rect, feature="needle", confidence=needle_conf
        )
        result.detector_diagnostics["needle"] = needle_outcome.to_diagnostics()
        if needle_rect:
            result.needle_rect = needle_rect
            result.confidence_scores["needle"] = needle_conf
            logger.info(f"Needle detected: {needle_rect} (conf={needle_conf:.2f})")

        # Step 4: Detect drop contour
        drop_contour, contact_pts, drop_conf = self._detect_drop_sessile()
        drop_outcome = normalize_detection_result(
            drop_contour,
            feature="drop",
            confidence=drop_conf,
            mask=self._binary_clean,
            metrics={"contact_points_detected": contact_pts is not None},
        )
        result.detector_diagnostics["drop"] = drop_outcome.to_diagnostics()
        if drop_contour is not None and len(drop_contour) > 0:
            result.drop_contour = drop_contour
            result.contact_points = contact_pts
            result.confidence_scores["drop"] = drop_conf
            logger.info(
                f"Drop detected with {len(drop_contour)} points (conf={drop_conf:.2f})"
            )
            apex_y = float(np.min(drop_contour[:, 1]))
            apex_band = drop_contour[np.abs(drop_contour[:, 1] - apex_y) <= 1.0]
            apex_x = float(np.median(apex_band[:, 0]))
            result.apex_point = (int(round(apex_x)), int(round(apex_y)))
            result.confidence_scores["apex"] = 0.95
            result.detector_diagnostics["apex"] = normalize_detection_result(
                result.apex_point, feature="apex", confidence=0.95
            ).to_diagnostics()

        # Step 5: Compute ROI from detected regions
        roi_rect, roi_conf = self._compute_roi(result)
        if roi_rect:
            result.roi_rect = roi_rect
            result.confidence_scores["roi"] = roi_conf
            logger.info(f"ROI computed: {roi_rect} (conf={roi_conf:.2f})")

        # Compute overall confidence
        if result.confidence_scores:
            result.confidence_scores["overall"] = sum(
                result.confidence_scores.values()
            ) / len(result.confidence_scores)

        self._attach_onnx_shadow(result)
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
        result.binary_mask = (
            self._binary_clean.copy() if self._binary_clean is not None else None
        )

        # Step 2: Find the main drop contour
        drop_cnt, drop_conf = self._find_pendant_drop_contour()
        result.detector_diagnostics["drop"] = normalize_detection_result(
            drop_cnt, feature="drop", confidence=drop_conf, mask=self._binary_clean
        ).to_diagnostics()
        if drop_cnt is None:
            logger.warning("Pendant: Could not find drop contour")
            return result

        # Step 3: Detect needle and contact points
        needle_rect, contact_pts, needle_conf = self._detect_needle_pendant(drop_cnt)
        if self.needle_geometry_method == "bilateral_robust" or self.experimental_geometry_mode == "shadow":
            experimental = detect_bilateral_needle(self.original_image, drop_cnt)
            result.detector_diagnostics["needle_bilateral"] = experimental.to_diagnostics()
            if self.needle_geometry_method == "bilateral_robust" and experimental.accepted:
                needle_rect = experimental.value["needle_rect"]
                contact_pts = experimental.value["contact_points"]
                needle_conf = experimental.confidence
            elif self.needle_geometry_method == "bilateral_robust":
                needle_rect = None
                contact_pts = None
                needle_conf = 0.0
        result.detector_diagnostics["needle"] = normalize_detection_result(
            needle_rect,
            feature="needle",
            confidence=needle_conf,
            metrics={"contact_points_detected": contact_pts is not None},
        ).to_diagnostics()
        if needle_rect:
            result.needle_rect = needle_rect
            result.contact_points = contact_pts
            result.confidence_scores["needle"] = needle_conf
            logger.info(
                f"Pendant needle detected: {needle_rect} (conf={needle_conf:.2f})"
            )

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
            result.confidence_scores["overall"] = sum(
                result.confidence_scores.values()
            ) / len(result.confidence_scores)

        self._attach_onnx_shadow(result)
        return result

    def _attach_onnx_shadow(self, result: CalibrationResult) -> None:
        """Record proposal diagnostics without changing calibration fields."""
        if self.onnx_proposal_mode != "shadow":
            return
        from menipy.common.onnx_shadow import run_shadow_segmentation

        proxy = SimpleNamespace(
            image=self.original_image,
            frames=None,
            drop_contour=result.drop_contour,
            detected_contour=result.drop_contour,
            contour=None,
            needle_rect=result.needle_rect,
            onnx_proposal_mode="shadow",
            segmentation_provider=self.segmentation_provider,
            onnx_proposal_classes=["droplet", "needle"],
            onnx_proposals={},
        )
        run_shadow_segmentation(proxy, self.pipeline_name)
        result.detector_diagnostics["onnx_proposals"] = proxy.onnx_proposals

    # ======================== Segmentation Methods ========================

    def _segment_image_adaptive(self) -> None:
        """Segment using adaptive thresholding (for sessile)."""
        self._binary_clean = segment_sessile_binary(
            self.original_image,
            substrate_y=self._substrate_y,
            clahe_clip_limit=self.clahe_clip_limit,
            clahe_tile_size=self.clahe_tile_size,
            adaptive_block_size=self.adaptive_block_size,
            adaptive_c=self.adaptive_c,
        )

    def _segment_image_otsu(self) -> None:
        """Segment using Otsu thresholding (for pendant high-contrast)."""
        blur = cv2.GaussianBlur(self.gray, (5, 5), 0)

        _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Morphological cleanup
        kernel = np.ones((5, 5), np.uint8)
        binary_clean = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

        self._binary_clean = binary_clean

    # ======================== Sessile Detection Methods ========================

    def _detect_substrate(
        self,
    ) -> tuple[tuple[tuple[int, int], tuple[int, int]] | None, float]:
        """Detect substrate baseline using gradient analysis on image margins."""
        substrate_line, confidence = detect_sessile_substrate_line(
            self.original_image,
            clahe_clip_limit=self.clahe_clip_limit,
            clahe_tile_size=self.clahe_tile_size,
            side_margin_fraction=self.margin_fraction,
        )
        if substrate_line is None:
            self._substrate_y = int(self.height * 0.8)
            return ((0, self._substrate_y), (self.width, self._substrate_y)), 0.3

        self._substrate_y = int(substrate_line[0][1])
        return substrate_line, confidence

    def _find_horizon_median(self, strip_gray: np.ndarray) -> int | None:
        """Find horizon line in a vertical strip using gradient analysis."""
        detected_ys: list[int] = []
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

    def _detect_needle_sessile(
        self,
    ) -> tuple[tuple[int, int, int, int] | None, float]:
        """Detect needle region (contour touching top border) for sessile."""
        shaft_rect, shaft_confidence, _ = detect_sessile_needle_shaft(
            self.original_image, substrate_y=self._substrate_y
        )
        if shaft_rect is not None:
            self._needle_rect = shaft_rect
            return shaft_rect, shaft_confidence

        if self._binary_clean is None:
            return None, 0.0

        contours, _ = cv2.findContours(
            self._binary_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return None, 0.0

        candidates = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if y >= 5 or x <= 2 or x + w >= self.width - 2 or w > self.width * 0.2:
                continue
            aspect_ratio = h / max(w, 1)
            center_error = abs((x + w / 2) - self.width / 2) / self.width
            score = min(1.0, aspect_ratio / 5.0) * 0.6 + (1.0 - center_error) * 0.4
            candidates.append((score, cnt, (x, y, w, h)))

        if candidates:
            confidence, cnt, rect = max(candidates, key=lambda item: item[0])
            self._needle_contour = cnt
            self._needle_rect = rect
            return rect, float(np.clip(confidence, 0.0, 1.0))

        return None, 0.0

    def _detect_drop_sessile(
        self,
    ) -> tuple[
        np.ndarray | None, tuple[tuple[int, int], tuple[int, int]] | None, float
    ]:
        """Detect drop contour and contact points for sessile.

        For sessile drops, the contour must:
            - Be well below the needle (50px gap minimum)
        - Touch the substrate line
        - Not be rectangular (ROI boundaries)
        """
        if self._binary_clean is None:
            return None, None, 0.0
        detection = detect_sessile_drop_contour(
            self.original_image,
            substrate_y=self._substrate_y,
            needle_rect=self._needle_rect,
            min_area_fraction=self.min_area_fraction,
        )
        self._drop_contour = detection.contour
        if detection.binary_mask is not None:
            self._binary_clean = detection.binary_mask
        return detection.contour, detection.contact_points, detection.confidence

    # ======================== Pendant Detection Methods ========================

    def _find_pendant_drop_contour(self) -> tuple[np.ndarray | None, float]:
        """Find the main drop contour for pendant (largest, centered)."""
        if self._binary_clean is None:
            return None, 0.0

        contours, _ = cv2.findContours(
            self._binary_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
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

    def _detect_needle_pendant(self, drop_cnt: np.ndarray) -> tuple[
        tuple[int, int, int, int] | None,
        tuple[tuple[int, int], tuple[int, int]] | None,
        float,
    ]:
        """
        Detect needle region for pendant drop using shaft line analysis.

        Finds the vertical shaft lines at the top of the contour, then
        walks down to find where the contour deviates (contact points).
        """
        x, y, w, h = cv2.boundingRect(drop_cnt)
        if y >= 5:
            return None, None, 0.0
        pts = drop_cnt.reshape(-1, 2)

        # Define needle shaft reference (top 20 pixels)
        top_limit = y + 20

        # Left shaft line: median X of points in top-left quadrant
        left_shaft_pts = pts[(pts[:, 1] < top_limit) & (pts[:, 0] < (x + w / 2))]
        if len(left_shaft_pts) == 0:
            return None, None, 0.0
        ref_x_left = np.median(left_shaft_pts[:, 0])

        # Right shaft line: median X of points in top-right quadrant
        right_shaft_pts = pts[(pts[:, 1] < top_limit) & (pts[:, 0] > (x + w / 2))]
        if len(right_shaft_pts) == 0:
            return None, None, 0.0
        ref_x_right = np.median(right_shaft_pts[:, 0])

        # Tolerance: how many pixels "out" counts as drop starting?
        tolerance = 0

        # Create mask for precise scanning
        mask = np.zeros((self.height, self.width), dtype=np.uint8)
        cv2.drawContours(mask, [drop_cnt], -1, 255, 1)

        # Find left contact point (where contour moves left of shaft)
        contact_y_left = y
        contact_x_left = int(ref_x_left)

        for cy in range(y, y + h):
            row = mask[cy, 0 : int(x + w / 2)]  # Left half
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
            row = mask[cy, int(x + w / 2) : self.width]  # Right half
            indices = np.where(row > 0)[0]
            if len(indices) > 0:
                current_x = indices[-1] + int(x + w / 2)  # Rightmost pixel
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
            (contact_x_right, contact_y_right),
        )

        # Confidence based on needle aspect ratio
        aspect = needle_h / max(needle_w, 1)
        confidence = min(1.0, 0.6 + 0.1 * min(aspect, 4))

        return needle_rect, contact_points, confidence

    def _detect_apex_pendant(
        self, drop_cnt: np.ndarray
    ) -> tuple[tuple[int, int] | None, float]:
        """Detect apex point (bottom of pendant drop)."""
        pts = drop_cnt.reshape(-1, 2)

        # Apex is the point with maximum Y (bottom of drop)
        apex_y = int(np.max(pts[:, 1]))
        apex_band = pts[np.abs(pts[:, 1] - apex_y) <= 1]
        apex = (int(round(float(np.median(apex_band[:, 0])))), apex_y)

        # High confidence - apex is straightforward to find
        return apex, 0.95

    def _compute_roi_pendant(
        self, result: CalibrationResult
    ) -> tuple[tuple[int, int, int, int] | None, float]:
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
    ) -> tuple[tuple[int, int, int, int] | None, float]:
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
    image: np.ndarray, pipeline_name: str = "sessile", **kwargs
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
