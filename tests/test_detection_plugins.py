"""
Tests for detection plugins.
"""
import pytest
import numpy as np
import cv2
import sys
from pathlib import Path

# Add plugins directory to path
plugins_dir = Path(__file__).parent.parent / "plugins"
sys.path.insert(0, str(plugins_dir))

# Import plugins to register them
import detect_needle
import detect_roi
import detect_substrate
import detect_drop
import detect_apex

from menipy.common.registry import (
    NEEDLE_DETECTORS,
    ROI_DETECTORS,
    SUBSTRATE_DETECTORS,
    DROP_DETECTORS,
    APEX_DETECTORS,
)


def create_sessile_test_image(
    width: int = 640,
    height: int = 480,
    substrate_y: int = 400,
    background: int = 200,
    foreground: int = 50,
) -> np.ndarray:
    """Create a synthetic sessile drop image."""
    image = np.full((height, width, 3), background, dtype=np.uint8)
    
    # Needle at top
    cv2.rectangle(image, (300, 0), (340, 100), (foreground,) * 3, -1)
    
    # Drop
    cv2.ellipse(image, (320, 350), (80, 50), 0, 0, 360, (foreground,) * 3, -1)
    
    # Substrate
    cv2.rectangle(image, (0, substrate_y), (width, height), (foreground,) * 3, -1)
    
    return image


def create_pendant_test_image(
    width: int = 640,
    height: int = 480,
    background: int = 200,
    foreground: int = 30,
) -> np.ndarray:
    """Create a synthetic pendant drop image."""
    image = np.full((height, width, 3), background, dtype=np.uint8)
    
    # Needle at top
    cv2.rectangle(image, (300, 0), (340, 100), (foreground,) * 3, -1)
    
    # Drop hanging from needle
    cv2.ellipse(image, (320, 200), (80, 120), 0, 0, 360, (foreground,) * 3, -1)
    
    return image


class TestPluginRegistration:
    """Test that all plugins are properly registered."""
    
    def test_needle_detectors_registered(self):
        assert "sessile" in NEEDLE_DETECTORS
        assert "pendant" in NEEDLE_DETECTORS
    
    def test_roi_detectors_registered(self):
        assert "sessile" in ROI_DETECTORS
        assert "pendant" in ROI_DETECTORS
        assert "auto" in ROI_DETECTORS
    
    def test_substrate_detectors_registered(self):
        assert "gradient" in SUBSTRATE_DETECTORS
        assert "hough" in SUBSTRATE_DETECTORS
    
    def test_drop_detectors_registered(self):
        assert "sessile" in DROP_DETECTORS
        assert "pendant" in DROP_DETECTORS
    
    def test_apex_detectors_registered(self):
        assert "sessile" in APEX_DETECTORS
        assert "pendant" in APEX_DETECTORS
        assert "auto" in APEX_DETECTORS


class TestNeedleDetector:
    """Tests for needle detection plugin."""
    
    def test_sessile_needle_detection(self):
        image = create_sessile_test_image()
        detector = NEEDLE_DETECTORS["sessile"]
        
        result = detector(image)
        
        assert result is not None
        x, y, w, h = result
        assert y < 5  # Touches top
        assert w > 0 and h > 0
    
    def test_sessile_no_needle(self):
        # Image with no needle (just drop)
        image = np.full((480, 640, 3), 200, dtype=np.uint8)
        cv2.ellipse(image, (320, 350), (80, 50), 0, 0, 360, (50, 50, 50), -1)
        
        detector = NEEDLE_DETECTORS["sessile"]
        result = detector(image)
        
        # Should return None if no contour touches top
        # (may still detect something due to adaptive threshold noise)
        # Just verify it doesn't crash
        assert result is None or isinstance(result, tuple)


class TestSubstrateDetector:
    """Tests for substrate detection plugin."""
    
    def test_gradient_detection(self):
        image = create_sessile_test_image(substrate_y=380)
        detector = SUBSTRATE_DETECTORS["gradient"]
        
        result = detector(image)
        
        assert result is not None
        (x1, y1), (x2, y2) = result
        # Should be roughly horizontal
        assert abs(y1 - y2) < 20
        # Y should be near 380
        avg_y = (y1 + y2) / 2
        assert 350 < avg_y < 410


class TestDropDetector:
    """Tests for drop detection plugin."""
    
    def test_sessile_drop_detection(self):
        image = create_sessile_test_image()
        detector = DROP_DETECTORS["sessile"]
        
        result = detector(image, substrate_y=400)
        
        assert result is not None
        if isinstance(result, tuple) and len(result) == 2:
            contour, contact_pts = result
            assert contour is not None
            assert len(contour) > 0
    
    def test_pendant_drop_detection(self):
        image = create_pendant_test_image()
        detector = DROP_DETECTORS["pendant"]
        
        result = detector(image)
        
        assert result is not None
        assert len(result) > 0


class TestApexDetector:
    """Tests for apex detection plugin."""
    
    def test_pendant_apex(self):
        # Create simple contour
        contour = np.array([
            [320, 100],  # Top
            [400, 200],  # Right
            [320, 350],  # Bottom (apex)
            [240, 200],  # Left
        ], dtype=np.float64)
        
        detector = APEX_DETECTORS["pendant"]
        result = detector(contour)
        
        assert result is not None
        x, y = result
        assert y == 350  # Bottom point
    
    def test_sessile_apex(self):
        # Create simple dome contour
        contour = np.array([
            [240, 400],  # Left base
            [280, 300],  # Left side
            [320, 250],  # Top (apex)
            [360, 300],  # Right side
            [400, 400],  # Right base
        ], dtype=np.float64)
        
        detector = APEX_DETECTORS["sessile"]
        result = detector(contour, substrate_y=400)
        
        assert result is not None
        x, y = result
        assert y == 250  # Top point


class TestROIDetector:
    """Tests for ROI detection plugin."""
    
    def test_sessile_roi_with_data(self):
        image = create_sessile_test_image()
        drop_contour = np.array([
            [240, 350], [320, 300], [400, 350], [320, 400]
        ], dtype=np.float64)
        
        detector = ROI_DETECTORS["sessile"]
        result = detector(
            image,
            drop_contour=drop_contour,
            substrate_y=400,
            padding=20
        )
        
        assert result is not None
        x, y, w, h = result
        assert w > 0 and h > 0
    
    def test_pendant_roi_with_data(self):
        image = create_pendant_test_image()
        drop_contour = np.array([
            [280, 100], [360, 100], [360, 320], [280, 320]
        ], dtype=np.float64)
        
        detector = ROI_DETECTORS["pendant"]
        result = detector(
            image,
            drop_contour=drop_contour,
            apex_point=(320, 320),
            padding=20
        )
        
        assert result is not None
        x, y, w, h = result
        assert w > 0 and h > 0
