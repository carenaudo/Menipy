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
    # Check settings
    settings = getattr(ctx, "preprocessing_settings", None)
    should_auto_detect = getattr(ctx, "auto_detect_features", True)
    
    detect_roi = True
    detect_needle = True
    detect_substrate = True
    detect_drop = True

    if settings and hasattr(settings, "auto_detect"):
        ad = settings.auto_detect
        if not ad.enabled:
            logger.debug("Auto-detection disabled via settings")
            return ctx
        # Map settings to flags
        detect_roi = ad.detect_roi
        detect_needle = ad.detect_needle
        detect_substrate = ad.detect_substrate
        # Drop detection usually needed if others are needed, or implied
    elif not should_auto_detect:
        logger.debug("Auto-detection disabled via context flag")
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
            if detect_substrate and "detect_substrate" in PREPROCESSORS:
                ctx = PREPROCESSORS["detect_substrate"](ctx)
            if detect_drop and "detect_drop" in PREPROCESSORS:
                ctx = PREPROCESSORS["detect_drop"](ctx)
            if detect_needle and "detect_needle" in PREPROCESSORS:
                ctx = PREPROCESSORS["detect_needle"](ctx)
            if detect_roi and "detect_roi" in PREPROCESSORS:
                ctx = PREPROCESSORS["detect_roi"](ctx)
        else:
            # Pendant: drop -> needle -> roi
            if detect_drop and "detect_drop" in PREPROCESSORS:
                ctx = PREPROCESSORS["detect_drop"](ctx)
            if detect_needle and "detect_needle" in PREPROCESSORS:
                ctx = PREPROCESSORS["detect_needle"](ctx)
            if detect_roi and "detect_roi" in PREPROCESSORS:
                ctx = PREPROCESSORS["detect_roi"](ctx)
        
        logger.info("Auto-detection complete")
        
    except Exception as e:
        logger.warning(f"Auto-detection failed: {e}")
    
    return ctx


# Register as preprocessor plugin
register_preprocessor("auto_detect", auto_detect_preprocessor)
