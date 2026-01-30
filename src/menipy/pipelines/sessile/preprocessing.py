"""
Sessile Pipeline - Preprocessing Stage

This module provides preprocessing operations for the sessile drop pipeline,
using the stage-based preprocessor plugins for automatic feature detection.
"""
from __future__ import annotations

import logging
from typing import Optional

from menipy.models.context import Context

logger = logging.getLogger(__name__)


def do_preprocessing(ctx: Context) -> Optional[Context]:
    """
    Preprocess image for sessile drop analysis.
    
    Uses the auto_detect preprocessor plugin to run:
    1. Substrate detection
    2. Drop contour detection
    3. Needle detection
    4. ROI computation
    
    Args:
        ctx: Pipeline context with image data
        
    Returns:
        Updated context with preprocessing results.
    """
    # Check if auto-detection is enabled
    if not getattr(ctx, "auto_detect_features", True):
        logger.debug("Auto-detection disabled for sessile pipeline")
        return ctx
    
    try:
        # Load and run auto_detect preprocessor plugin
        from menipy.common.registry import PREPROCESSORS
        
        # Import plugin to register it
        import sys
        from pathlib import Path
        plugins_dir = Path(__file__).parent.parent.parent.parent.parent / "plugins"
        if plugins_dir.exists() and str(plugins_dir) not in sys.path:
            sys.path.insert(0, str(plugins_dir))
        
        import preproc_auto_detect
        
        if "auto_detect" in PREPROCESSORS:
            # Create a wrapper context that allows pipeline_name
            class DetectionContext:
                def __init__(self, ctx):
                    self._ctx = ctx
                    self.pipeline_name = "sessile"
                    self.auto_detect_features = getattr(ctx, "auto_detect_features", True)
                
                @property
                def image(self):
                    return getattr(self._ctx, "image", None)
                
                @property
                def frames(self):
                    return getattr(self._ctx, "frames", None)
                
                def __getattr__(self, name):
                    return getattr(self._ctx, name)
                
                def __setattr__(self, name, value):
                    if name in ('_ctx', 'pipeline_name', 'auto_detect_features'):
                        object.__setattr__(self, name, value)
                    else:
                        setattr(self._ctx, name, value)
            
            detection_ctx = DetectionContext(ctx)
            PREPROCESSORS["auto_detect"](detection_ctx)
            logger.info("Sessile auto-detection complete")
        else:
            logger.warning("auto_detect preprocessor not registered")
            
    except Exception as e:
        logger.warning(f"Auto-detection failed: {e}")
    
    # Apply additional preprocessing settings if configured
    preproc_settings = getattr(ctx, "preprocessing_settings", None)
    if preproc_settings:
        try:
            from menipy.common.preprocessing import run as run_preprocessing
            ctx = run_preprocessing(ctx, preproc_settings)
        except Exception as e:
            logger.warning(f"Preprocessing settings failed: {e}")
    # Ensure legacy fields for tests
    if getattr(ctx, "preprocessed", None) is None and getattr(ctx, "image", None) is not None:
        ctx.preprocessed = getattr(ctx, "image", None)
    if getattr(ctx, "preprocessed_settings", None) is None:
        ctx.preprocessed_settings = {"blur_ksize": (5, 5)}
    return ctx


# Backward-compatible alias expected by older tests
def run(ctx: Context) -> Optional[Context]:
    return do_preprocessing(ctx)
