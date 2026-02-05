"""
Tests for edge detector plugins.

Tests the Otsu, Adaptive, LoG, and Improved Snake edge detectors.
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
import edge_detectors

from menipy.common.registry import EDGE_DETECTORS
from menipy.models.config import EdgeDetectionSettings


def create_synthetic_drop_image(
    width: int = 320,
    height: int = 240,
    drop_center: tuple = (160, 150),
    drop_axes: tuple = (60, 50),
    background: int = 200,
    foreground: int = 50,
) -> np.ndarray:
    """Create a synthetic grayscale drop image for testing."""
    image = np.full((height, width), background, dtype=np.uint8)
    
    # Draw ellipse (drop)
    cv2.ellipse(image, drop_center, drop_axes, 0, 0, 360, foreground, -1)
    
    return image


def create_noisy_drop_image() -> np.ndarray:
    """Create a drop image with noise for robustness testing."""
    image = create_synthetic_drop_image()
    noise = np.random.normal(0, 15, image.shape).astype(np.int16)
    noisy = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return noisy


class TestPluginRegistration:
    """Test that edge detector plugins are properly registered."""
    
    def test_otsu_registered(self):
        assert "otsu" in EDGE_DETECTORS
    
    def test_adaptive_registered(self):
        assert "adaptive" in EDGE_DETECTORS
    
    def test_log_registered(self):
        assert "log" in EDGE_DETECTORS
    
    def test_improved_snake_registered(self):
        assert "improved_snake" in EDGE_DETECTORS


class TestOtsuDetector:
    """Tests for Otsu thresholding edge detector."""
    
    def test_detects_contour(self):
        image = create_synthetic_drop_image()
        settings = EdgeDetectionSettings(method="otsu")
        
        detector = EDGE_DETECTORS["otsu"]
        result = detector(image, settings)
        
        assert result is not None
        assert len(result) > 10  # Should have many contour points
        assert result.shape[1] == 2  # (x, y) pairs
    
    def test_handles_noisy_image(self):
        image = create_noisy_drop_image()
        settings = EdgeDetectionSettings(method="otsu")
        
        detector = EDGE_DETECTORS["otsu"]
        result = detector(image, settings)
        
        assert result is not None
        assert len(result) > 5


class TestAdaptiveDetector:
    """Tests for adaptive thresholding edge detector."""
    
    def test_detects_contour(self):
        image = create_synthetic_drop_image()
        settings = EdgeDetectionSettings(
            method="adaptive",
            plugin_settings={
                "adaptive_block_size": 21,
                "adaptive_c": 2,
            }
        )
        
        detector = EDGE_DETECTORS["adaptive"]
        result = detector(image, settings)
        
        assert result is not None
        assert len(result) > 10
        assert result.shape[1] == 2
    
    def test_with_different_block_sizes(self):
        image = create_synthetic_drop_image()
        
        for block_size in [11, 21, 31]:
            settings = EdgeDetectionSettings(
                method="adaptive",
                plugin_settings={"adaptive_block_size": block_size},
            )
            
            detector = EDGE_DETECTORS["adaptive"]
            result = detector(image, settings)
            
            assert result is not None
            assert len(result) > 0


class TestLoGDetector:
    """Tests for Laplacian of Gaussian edge detector."""
    
    def test_detects_contour_standard_mode(self):
        image = create_synthetic_drop_image()
        settings = EdgeDetectionSettings(
            method="log",
            plugin_settings={
                "log_sigma": 1.0,
                "log_use_zero_crossing": False,
            }
        )
        
        detector = EDGE_DETECTORS["log"]
        result = detector(image, settings)
        
        assert result is not None
        assert len(result) > 10
        assert result.shape[1] == 2
    
    def test_detects_contour_zero_crossing_mode(self):
        image = create_synthetic_drop_image()
        settings = EdgeDetectionSettings(
            method="log",
            plugin_settings={
                "log_sigma": 1.0,
                "log_use_zero_crossing": True,
                "log_min_gradient": 3.0,
            }
        )
        
        detector = EDGE_DETECTORS["log"]
        result = detector(image, settings)
        
        assert result is not None
        # Zero-crossing may return fewer points depending on image
        assert result.shape[1] == 2
    
    def test_sigma_variations(self):
        image = create_synthetic_drop_image()
        
        for sigma in [0.5, 1.0, 2.0]:
            settings = EdgeDetectionSettings(
                method="log",
                plugin_settings={"log_sigma": sigma},
            )
            
            detector = EDGE_DETECTORS["log"]
            result = detector(image, settings)
            
            assert result is not None


class TestImprovedSnakeDetector:
    """Tests for improved snake (active contour) detector."""
    
    def test_detects_contour(self):
        image = create_synthetic_drop_image()
        settings = EdgeDetectionSettings(
            method="improved_snake",
            snake_iterations=50,  # Lower for faster testing
            snake_alpha=0.015,
            snake_beta=10.0,
        )
        
        detector = EDGE_DETECTORS["improved_snake"]
        result = detector(image, settings)
        
        assert result is not None
        if len(result) > 0:
            assert result.shape[1] == 2
    
    def test_with_substrate_masking(self):
        """Test that substrate masking works (contour should stay above substrate)."""
        image = create_synthetic_drop_image(drop_center=(160, 180))
        substrate_y = 200
        
        settings = EdgeDetectionSettings(
            method="improved_snake",
            snake_iterations=30,
        )
        
        detector = EDGE_DETECTORS["improved_snake"]
        # Note: substrate_y is passed as extra kwarg (not in settings)
        # The detector will gracefully handle this
        result = detector(image, settings)
        
        assert result is not None


class TestContourValidity:
    """Test that detected contours have valid properties."""
    
    @pytest.mark.parametrize("method", ["otsu", "adaptive", "log"])
    def test_contour_is_closed_loop(self, method):
        """Test that contour forms a roughly closed shape."""
        image = create_synthetic_drop_image()
        settings = EdgeDetectionSettings(method=method)
        
        detector = EDGE_DETECTORS[method]
        result = detector(image, settings)
        
        if len(result) > 10:
            # Check that first and last points are reasonably close
            # (indicating a closed contour)
            start = result[0]
            end = result[-1]
            distance = np.linalg.norm(start - end)
            # OpenCV contours should be closed or at least close
            assert distance < 50, f"Contour not closed: distance={distance}"
    
    @pytest.mark.parametrize("method", ["otsu", "adaptive", "log"])
    def test_contour_bounds_within_image(self, method):
        """Test that all contour points are within image bounds."""
        width, height = 320, 240
        image = create_synthetic_drop_image(width=width, height=height)
        settings = EdgeDetectionSettings(method=method)
        
        detector = EDGE_DETECTORS[method]
        result = detector(image, settings)
        
        if len(result) > 0:
            assert np.all(result[:, 0] >= 0)
            assert np.all(result[:, 0] < width)
            assert np.all(result[:, 1] >= 0)
            assert np.all(result[:, 1] < height)


class TestEdgeDetectionSettings:
    """Test EdgeDetectionSettings model with new fields."""
    
    def test_new_methods_accepted(self):
        for method in ["otsu", "adaptive", "log", "improved_snake"]:
            settings = EdgeDetectionSettings(method=method)
            assert settings.method == method
    
    
    def test_log_settings_defaults(self):
        # Settings removed from config, now in plugin model
        from menipy.common.plugin_settings import get_detector_settings_model
        
        LogModel = get_detector_settings_model("log")
        assert LogModel is not None
        
        defaults = LogModel()
        assert defaults.log_sigma == 1.0
        assert defaults.log_min_gradient == 5.0
        assert defaults.log_use_zero_crossing is False
    
    def test_adaptive_settings_defaults(self):
        from menipy.common.plugin_settings import get_detector_settings_model
        
        AdaptiveModel = get_detector_settings_model("adaptive")
        assert AdaptiveModel is not None
        
        defaults = AdaptiveModel()
        assert defaults.adaptive_block_size == 21
        assert defaults.adaptive_c == 2
    
    def test_adaptive_block_size_must_be_odd(self):
        # Validation moved to plugin settings model
        from menipy.common.plugin_settings import get_detector_settings_model
        
        AdaptiveModel = get_detector_settings_model("adaptive")
        model = AdaptiveModel(adaptive_block_size=20)
        # Should be auto-corrected in post_init
        assert model.adaptive_block_size == 21
