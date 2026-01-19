"""
Pendant Pipeline - Preprocessing Stage

This module provides preprocessing operations for the pendant drop pipeline,
using the stage-based preprocessor plugins for automatic feature detection.
"""
from __future__ import annotations

import logging
from typing import Optional

from menipy.models.context import Context

logger = logging.getLogger(__name__)


def do_preprocessing(ctx: Context) -> Optional[Context]:
    """
    Preprocess image for pendant drop analysis.
    
    Uses the auto_detect preprocessor plugin to run:
    1. Drop contour detection
    2. Needle detection (shaft line analysis)
    3. ROI computation
    
    Args:
        ctx: Pipeline context with image data
        
    Returns:
        Updated context with preprocessing results.
    """
    # Check if auto-detection is enabled
    if not getattr(ctx, "auto_detect_features", True):
        logger.debug("Auto-detection disabled for pendant pipeline")
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
                    self.pipeline_name = "pendant"
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
            logger.info("Pendant auto-detection complete")
        else:
            logger.warning("auto_detect preprocessor not registered")
            
    except Exception as e:
        logger.warning(f"Auto-detection failed: {e}")
    
    # Apply additional preprocessing settings if configured
    preproc_settings = getattr(ctx, "preprocessing_settings", None)
    if preproc_settings:
        try:
            from menipy.common.preprocessing import apply_preprocessing
            ctx = apply_preprocessing(ctx, preproc_settings)
        except Exception as e:
            logger.warning(f"Preprocessing settings failed: {e}")
    
    return ctx
