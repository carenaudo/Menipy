"""
Tests for auto_calibrator module.
"""
import pytest
import numpy as np

from menipy.common.auto_calibrator import (
    AutoCalibrator,
    CalibrationResult,
    run_auto_calibration,
)


def create_test_image_sessile(
    width: int = 640,
    height: int = 480,
    *,
    substrate_y: int = 400,
    drop_center_x: int = 320,
    drop_radius_x: int = 80,
    drop_radius_y: int = 60,
    needle_width: int = 40,
    background_color: int = 200,
    foreground_color: int = 50,
) -> np.ndarray:
    """Create a synthetic sessile drop image for testing."""
    import cv2
    
    # Create light gray background
    image = np.full((height, width, 3), background_color, dtype=np.uint8)
    
    # Draw substrate (dark horizontal band at substrate_y)
    cv2.rectangle(
        image,
        (0, substrate_y - 2),
        (width, height),
        (foreground_color, foreground_color, foreground_color),
        -1
    )
    
    # Draw drop (ellipse touching substrate)
    drop_center_y = substrate_y - drop_radius_y
    cv2.ellipse(
        image,
        (drop_center_x, drop_center_y),
        (drop_radius_x, drop_radius_y),
        0, 0, 360,
        (foreground_color, foreground_color, foreground_color),
        -1
    )
    
    # Draw needle (rectangle from top touching drop)
    needle_x = drop_center_x - needle_width // 2
    needle_height = drop_center_y - drop_radius_y + 10  # Overlaps with drop
    cv2.rectangle(
        image,
        (needle_x, 0),
        (needle_x + needle_width, needle_height),
        (foreground_color + 10, foreground_color + 10, foreground_color + 10),
        -1
    )
    
    return image


class TestAutoCalibrator:
    """Tests for AutoCalibrator class."""
    
    def test_init(self):
        """Test AutoCalibrator initialization."""
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        calibrator = AutoCalibrator(image, "sessile")
        
        assert calibrator.pipeline_name == "sessile"
        assert calibrator.width == 640
        assert calibrator.height == 480
        assert calibrator.enhanced_gray is not None
    
    def test_init_grayscale_image(self):
        """Test initialization with grayscale image."""
        image = np.zeros((480, 640), dtype=np.uint8)
        calibrator = AutoCalibrator(image, "pendant")
        
        assert calibrator.gray.shape == (480, 640)
        assert calibrator.enhanced_gray.shape == (480, 640)
    
    def test_detect_all_returns_calibration_result(self):
        """Test that detect_all returns a CalibrationResult."""
        image = create_test_image_sessile()
        calibrator = AutoCalibrator(image, "sessile")
        
        result = calibrator.detect_all()
        
        assert isinstance(result, CalibrationResult)
        assert result.enhanced_image is not None
    
    def test_detect_substrate_horizontal(self):
        """Test substrate detection finds horizontal baseline."""
        image = create_test_image_sessile(substrate_y=350)
        calibrator = AutoCalibrator(image, "sessile")
        
        result = calibrator.detect_all()
        
        assert result.substrate_line is not None
        p1, p2 = result.substrate_line
        # Substrate should be roughly horizontal
        assert abs(p1[1] - p2[1]) < 20
        # Y coordinate should be near 350
        avg_y = (p1[1] + p2[1]) / 2
        assert 330 < avg_y < 370
    
    def test_detect_needle_from_top(self):
        """Test needle detection finds region touching top border."""
        image = create_test_image_sessile()
        calibrator = AutoCalibrator(image, "sessile")
        
        result = calibrator.detect_all()
        
        # Needle should be detected (touches top)
        assert result.needle_rect is not None
        x, y, w, h = result.needle_rect
        assert y < 5  # Starts near top
        assert w > 0 and h > 0
    
    def test_detect_drop_contour(self):
        """Test drop contour detection."""
        image = create_test_image_sessile()
        calibrator = AutoCalibrator(image, "sessile")
        
        result = calibrator.detect_all()
        
        assert result.drop_contour is not None
        assert len(result.drop_contour) > 3  # At least 4 points for a shape
    
    def test_compute_roi(self):
        """Test ROI computation from detected regions."""
        image = create_test_image_sessile()
        calibrator = AutoCalibrator(image, "sessile")
        
        result = calibrator.detect_all()
        
        assert result.roi_rect is not None
        x, y, w, h = result.roi_rect
        assert x >= 0 and y >= 0
        assert w > 0 and h > 0
        assert x + w <= 640
        assert y + h <= 480
    
    def test_confidence_scores(self):
        """Test that confidence scores are computed."""
        image = create_test_image_sessile()
        calibrator = AutoCalibrator(image, "sessile")
        
        result = calibrator.detect_all()
        
        assert "overall" in result.confidence_scores
        overall = result.confidence_scores["overall"]
        assert 0.0 <= overall <= 1.0
    
    def test_contact_points_at_substrate(self):
        """Test that contact points are at substrate level."""
        image = create_test_image_sessile(substrate_y=380)
        calibrator = AutoCalibrator(image, "sessile")
        
        result = calibrator.detect_all()
        
        if result.contact_points is not None and result.substrate_line is not None:
            left, right = result.contact_points
            substrate_y = (result.substrate_line[0][1] + result.substrate_line[1][1]) / 2
            
            # Contact points should be at substrate level
            assert abs(left[1] - substrate_y) < 10
            assert abs(right[1] - substrate_y) < 10
    
    def test_low_contrast_image(self):
        """Test detection on low-contrast image (CLAHE should help)."""
        # Create a very low contrast image
        image = np.full((480, 640, 3), 128, dtype=np.uint8)
        # Add subtle drop
        import cv2
        cv2.ellipse(image, (320, 350), (50, 40), 0, 0, 360, (110, 110, 110), -1)
        cv2.line(image, (0, 380), (640, 380), (115, 115, 115), 3)
        
        calibrator = AutoCalibrator(image, "sessile")
        result = calibrator.detect_all()
        
        # Should still detect something (CLAHE enhances contrast)
        assert isinstance(result, CalibrationResult)


class TestCalibrationResult:
    """Tests for CalibrationResult dataclass."""
    
    def test_default_values(self):
        """Test default values are None or empty."""
        result = CalibrationResult()
        
        assert result.substrate_line is None
        assert result.needle_rect is None
        assert result.drop_contour is None
        assert result.roi_rect is None
        assert result.contact_points is None
        assert result.confidence_scores == {}
    
    def test_with_values(self):
        """Test setting values."""
        result = CalibrationResult(
            substrate_line=((0, 400), (640, 400)),
            roi_rect=(100, 200, 400, 200),
            confidence_scores={"overall": 0.85}
        )
        
        assert result.substrate_line == ((0, 400), (640, 400))
        assert result.roi_rect == (100, 200, 400, 200)
        assert result.confidence_scores["overall"] == 0.85


class TestRunAutoCalibration:
    """Tests for run_auto_calibration convenience function."""
    
    def test_run_auto_calibration(self):
        """Test convenience function."""
        image = create_test_image_sessile()
        
        result = run_auto_calibration(image, "sessile")
        
        assert isinstance(result, CalibrationResult)
        assert result.substrate_line is not None or result.drop_contour is not None
    
    def test_custom_parameters(self):
        """Test with custom parameters."""
        image = create_test_image_sessile()
        
        result = run_auto_calibration(
            image,
            "sessile",
            clahe_clip_limit=3.0,
            roi_padding=30
        )
        
        assert isinstance(result, CalibrationResult)


class TestPendantDetection:
    """Tests for pendant drop detection."""
    
    @staticmethod
    def create_pendant_image(
        width: int = 640,
        height: int = 480,
        *,
        needle_x: int = 300,
        needle_width: int = 40,
        needle_height: int = 100,
        drop_radius_x: int = 80,
        drop_radius_y: int = 120,
        background_color: int = 200,
        foreground_color: int = 30,
    ) -> np.ndarray:
        """Create a synthetic pendant drop image for testing."""
        import cv2
        
        # Create light background
        image = np.full((height, width, 3), background_color, dtype=np.uint8)
        
        # Draw needle (rectangle at top)
        cv2.rectangle(
            image,
            (needle_x, 0),
            (needle_x + needle_width, needle_height),
            (foreground_color, foreground_color, foreground_color),
            -1
        )
        
        # Draw drop (ellipse hanging from needle)
        drop_center_x = needle_x + needle_width // 2
        drop_center_y = needle_height + drop_radius_y - 20
        cv2.ellipse(
            image,
            (drop_center_x, drop_center_y),
            (drop_radius_x, drop_radius_y),
            0, 0, 360,
            (foreground_color, foreground_color, foreground_color),
            -1
        )
        
        return image
    
    def test_pendant_detection_runs(self):
        """Test pendant detection pipeline runs without errors."""
        image = self.create_pendant_image()
        
        result = run_auto_calibration(image, "pendant")
        
        assert isinstance(result, CalibrationResult)
        assert result.binary_mask is not None
    
    def test_pendant_drop_contour_detected(self):
        """Test pendant drop contour is detected."""
        image = self.create_pendant_image()
        
        result = run_auto_calibration(image, "pendant")
        
        assert result.drop_contour is not None
        assert len(result.drop_contour) > 0
    
    def test_pendant_needle_detected(self):
        """Test pendant needle is detected with shaft line analysis."""
        image = self.create_pendant_image()
        
        result = run_auto_calibration(image, "pendant")
        
        # Needle should be detected
        assert result.needle_rect is not None
        x, y, w, h = result.needle_rect
        assert y < 10  # Starts near top
        assert w > 0 and h > 0
    
    def test_pendant_apex_detected(self):
        """Test pendant apex (bottom of drop) is detected."""
        image = self.create_pendant_image()
        
        result = run_auto_calibration(image, "pendant")
        
        assert result.apex_point is not None
        apex_x, apex_y = result.apex_point
        # Apex should be in lower portion of image
        assert apex_y > 150
    
    def test_pendant_contact_points(self):
        """Test pendant contact points are detected."""
        image = self.create_pendant_image()
        
        result = run_auto_calibration(image, "pendant")
        
        if result.contact_points:
            left, right = result.contact_points
            # Contact points should be near needle edges
            assert left[0] < right[0]
    
    def test_pendant_roi_computed(self):
        """Test ROI is computed for pendant."""
        image = self.create_pendant_image()
        
        result = run_auto_calibration(image, "pendant")
        
        assert result.roi_rect is not None
        x, y, w, h = result.roi_rect
        assert w > 0 and h > 0
    
    def test_pendant_confidence_scores(self):
        """Test confidence scores for pendant detection."""
        image = self.create_pendant_image()
        
        result = run_auto_calibration(image, "pendant")
        
        assert "drop" in result.confidence_scores
        assert "overall" in result.confidence_scores


class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_blank_image(self):
        """Test with completely blank image."""
        image = np.full((480, 640, 3), 128, dtype=np.uint8)
        
        result = run_auto_calibration(image, "sessile")
        
        assert isinstance(result, CalibrationResult)
        # Detection should handle gracefully (fallback to defaults)
    
    def test_very_small_image(self):
        """Test with small image."""
        image = np.zeros((50, 50, 3), dtype=np.uint8)
        
        result = run_auto_calibration(image, "sessile")
        
        assert isinstance(result, CalibrationResult)
    
    def test_single_channel_image(self):
        """Test with single channel grayscale."""
        image = np.zeros((480, 640), dtype=np.uint8)
        
        result = run_auto_calibration(image, "sessile")
        
        assert isinstance(result, CalibrationResult)
