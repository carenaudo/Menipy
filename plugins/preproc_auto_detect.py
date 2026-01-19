"""
Auto-detection preprocessor plugin.

This plugin combines all detection steps in the correct order:
1. Substrate detection (sessile only)
2. Drop contour detection  
3. Needle detection
4. ROI computation

Follows the stage-based pattern: operates on ctx and returns ctx.
"""
from __future__ import annotations

import logging

from menipy.common.registry import register_preprocessor

logger = logging.getLogger(__name__)


def auto_detect_preprocessor(ctx):
    """
    Preprocessor plugin that runs full auto-detection pipeline.
    
    Chains detection steps in correct order based on pipeline type.
    
    Args:
        ctx: Pipeline context with image data
        
    Returns:
        Updated context with all detected features.
    """
    # Check if auto-detection is enabled
    if not getattr(ctx, "auto_detect_features", True):
        logger.debug("Auto-detection disabled")
        return ctx
    
    # Get pipeline type
    pipeline = getattr(ctx, "pipeline_name", "sessile").lower()
    logger.info(f"Running auto-detection for {pipeline} pipeline")
    
    # Import detection preprocessors (they register themselves)
    try:
        from menipy.common.registry import PREPROCESSORS
        
        # Load detection plugins if not already loaded
        import sys
        from pathlib import Path
        plugins_dir = Path(__file__).parent
        if str(plugins_dir) not in sys.path:
            sys.path.insert(0, str(plugins_dir))
        
        # Import plugins (registers them)
        import preproc_detect_substrate
        import preproc_detect_drop
        import preproc_detect_needle
        import preproc_detect_roi
        
        # Run in correct order
        if pipeline == "sessile":
            # Sessile: substrate -> drop -> needle -> roi
            if "detect_substrate" in PREPROCESSORS:
                ctx = PREPROCESSORS["detect_substrate"](ctx)
            if "detect_drop" in PREPROCESSORS:
                ctx = PREPROCESSORS["detect_drop"](ctx)
            if "detect_needle" in PREPROCESSORS:
                ctx = PREPROCESSORS["detect_needle"](ctx)
            if "detect_roi" in PREPROCESSORS:
                ctx = PREPROCESSORS["detect_roi"](ctx)
        else:
            # Pendant: drop -> needle -> roi
            if "detect_drop" in PREPROCESSORS:
                ctx = PREPROCESSORS["detect_drop"](ctx)
            if "detect_needle" in PREPROCESSORS:
                ctx = PREPROCESSORS["detect_needle"](ctx)
            if "detect_roi" in PREPROCESSORS:
                ctx = PREPROCESSORS["detect_roi"](ctx)
        
        logger.info("Auto-detection complete")
        
    except Exception as e:
        logger.warning(f"Auto-detection failed: {e}")
    
    return ctx


# Register as preprocessor plugin
register_preprocessor("auto_detect", auto_detect_preprocessor)
