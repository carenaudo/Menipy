"""
Tests for stage-based detection preprocessor plugins.
"""

import numpy as np
import cv2
import sys
from pathlib import Path

# Add plugins directory to path
plugins_dir = Path(__file__).parent.parent / "plugins"
sys.path.insert(0, str(plugins_dir))

# Import plugins to register them
import preproc_detect_substrate
import preproc_detect_needle
import preproc_detect_drop
import preproc_detect_roi
import preproc_auto_detect

from menipy.common.registry import PREPROCESSORS


class MockContext:
    """Mock context for testing preprocessor plugins."""
    def __init__(self, pipeline_name: str = "sessile"):
        self.pipeline_name = pipeline_name
        self.auto_detect_features = True


def create_sessile_test_image(
    width: int = 640,
    height: int = 480,
    substrate_y: int = 400,
) -> np.ndarray:
    """Create a synthetic sessile drop image."""
    image = np.full((height, width, 3), 200, dtype=np.uint8)
    cv2.rectangle(image, (300, 0), (340, 100), (50, 50, 50), -1)
    cv2.ellipse(image, (320, 350), (80, 50), 0, 0, 360, (50, 50, 50), -1)
    cv2.rectangle(image, (0, substrate_y), (width, height), (50, 50, 50), -1)
    return image


def create_pendant_test_image(
    width: int = 640,
    height: int = 480,
) -> np.ndarray:
    """Create a synthetic pendant drop image."""
    image = np.full((height, width, 3), 200, dtype=np.uint8)
    cv2.rectangle(image, (300, 0), (340, 100), (30, 30, 30), -1)
    cv2.ellipse(image, (320, 200), (80, 120), 0, 0, 360, (30, 30, 30), -1)
    return image


class TestPluginRegistration:
    """Test that all preprocessor plugins are registered."""
    
    def test_detect_substrate_registered(self):
        assert "detect_substrate" in PREPROCESSORS
    
    def test_detect_needle_registered(self):
        assert "detect_needle" in PREPROCESSORS
    
    def test_detect_drop_registered(self):
        assert "detect_drop" in PREPROCESSORS
    
    def test_detect_roi_registered(self):
        """test_detect_roi_registered."""
        assert "detect_roi" in PREPROCESSORS
    
    def test_auto_detect_registered(self):
        """test_auto_detect_registered."""
        assert "auto_detect" in PREPROCESSORS


class TestSubstratePreprocessor:
    """Tests for substrate detection preprocessor."""
    
    def test_detects_substrate_line(self):
        """test_detects_substrate_line."""
        ctx = MockContext()
        ctx.image = create_sessile_test_image(substrate_y=380)
        
        ctx = PREPROCESSORS["detect_substrate"](ctx)
        
        assert hasattr(ctx, "substrate_line")
        assert ctx.substrate_line is not None
        p1, p2 = ctx.substrate_line
        avg_y = (p1[1] + p2[1]) / 2
        assert 350 < avg_y < 410


class TestNeedlePreprocessor:
    """Tests for needle detection preprocessor."""
    
    def test_detects_sessile_needle(self):
        """test_detects_sessile_needle."""
        ctx = MockContext("sessile")
        ctx.image = create_sessile_test_image()
        
        ctx = PREPROCESSORS["detect_needle"](ctx)
        
        assert hasattr(ctx, "needle_rect")
        if ctx.needle_rect:
            x, y, w, h = ctx.needle_rect
            assert y < 5  # Touches top


class TestDropPreprocessor:
    """Tests for drop detection preprocessor."""
    
    def test_detects_sessile_drop(self):
        """test_detects_sessile_drop."""
        ctx = MockContext("sessile")
        ctx.image = create_sessile_test_image()
        ctx.substrate_line = ((0, 400), (640, 400))
        
        ctx = PREPROCESSORS["detect_drop"](ctx)
        
        assert hasattr(ctx, "detected_contour")
        assert ctx.detected_contour is not None
    
    def test_detects_pendant_drop(self):
        """Test_detects_pendant_drop."""
        ctx = MockContext("pendant")
        ctx.image = create_pendant_test_image()
        
        ctx = PREPROCESSORS["detect_drop"](ctx)
        
        assert hasattr(ctx, "detected_contour")
        assert ctx.detected_contour is not None


class TestROIPreprocessor:
    """Tests for ROI detection preprocessor."""
    
    def test_computes_sessile_roi(self):
        """Test_computes_sessile_roi."""
        ctx = MockContext("sessile")
        ctx.image = create_sessile_test_image()
        ctx.substrate_line = ((0, 400), (640, 400))
        ctx.detected_contour = np.array([
            [240, 350], [320, 300], [400, 350], [320, 400]
        ], dtype=np.float64)
        
        ctx = PREPROCESSORS["detect_roi"](ctx)
        
        assert hasattr(ctx, "detected_roi")
        assert ctx.detected_roi is not None
        x, y, w, h = ctx.detected_roi
        assert w > 0 and h > 0


class TestAutoDetectPreprocessor:
    """Tests for combined auto_detect preprocessor."""
    
    def test_sessile_full_detection(self):
        """Test_sessile_full_detection."""
        ctx = MockContext("sessile")
        ctx.image = create_sessile_test_image()
        
        ctx = PREPROCESSORS["auto_detect"](ctx)
        
        assert hasattr(ctx, "substrate_line")
        assert hasattr(ctx, "detected_contour")
        assert hasattr(ctx, "detected_roi")
    
    def test_pendant_full_detection(self):
        """Test_pendant_full_detection."""
        ctx = MockContext("pendant")
        ctx.image = create_pendant_test_image()
        
        ctx = PREPROCESSORS["auto_detect"](ctx)
        
        assert hasattr(ctx, "detected_contour")
        assert hasattr(ctx, "detected_roi")
    
    def test_respects_auto_detect_flag(self):
        """Test_respects_auto_detect_flag."""
        ctx = MockContext("sessile")
        ctx.image = create_sessile_test_image()
        ctx.auto_detect_features = False
        
        ctx = PREPROCESSORS["auto_detect"](ctx)
        
        # Should not detect anything when disabled
        assert not hasattr(ctx, "substrate_line") or ctx.substrate_line is None
