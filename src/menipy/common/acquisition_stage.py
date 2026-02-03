# menipy/common/acquisition_stage.py
"""
Unified acquisition stage implementation for all pipelines.

This module provides a consistent image loading strategy that is shared
across all pipeline implementations to ensure consistent behavior.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from menipy.models.context import Context


def do_acquisition(ctx: Context, logger: logging.Logger) -> Context:
    """Unified acquisition stage for all pipelines.
    
    Handles image loading from multiple sources with consistent behavior:
    
    1. Return early if frames already exist
    2. Wrap numpy array in Frame if ctx.image is ndarray
    3. Load from ctx.image if it's a string path
    4. Load from ctx.image_path if provided
    
    Args:
        ctx: Pipeline context object
        logger: Logger instance for the calling pipeline
        
    Returns:
        Updated context with ctx.frames, ctx.frame, and ctx.image populated
    """
    # If frames already exist, nothing to do
    if getattr(ctx, "frames", None) and len(ctx.frames) > 0:
        logger.info(f"[do_acquisition] Frames already exist: {len(ctx.frames)}")
        return ctx
    
    # Check if we have a direct image (numpy array)
    image = getattr(ctx, "image", None)
    
    # If image is a string path, treat it as image_path
    if isinstance(image, (str, Path)):
        if not getattr(ctx, "image_path", None):
            ctx.image_path = str(image)
        image = None  # Clear so we load from path below
    
    # Wrap numpy array in Frame
    if image is not None and isinstance(image, np.ndarray):
        from menipy.models.frame import Frame
        
        frame = Frame(image=image)
        ctx.frames = [frame]
        ctx.frame = frame
        logger.info(f"[do_acquisition] Wrapped ctx.image in frame, shape: {image.shape}")
        return ctx
    
    # Try to load from file path
    image_path = getattr(ctx, "image_path", None)
    if not image_path:
        logger.warning("[do_acquisition] No image or image_path available!")
        return ctx
    
    # Try OpenCV first (most common), then fall back to acquisition module
    try:
        import cv2
        img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if img is not None:
            from menipy.models.frame import Frame
            frame = Frame(image=img)
            ctx.frames = [frame]
            ctx.frame = frame
            ctx.image = img
            logger.info(f"[do_acquisition] Loaded image from {image_path} via cv2")
            return ctx
    except Exception as e:
        logger.debug(f"[do_acquisition] cv2 load failed: {e}, trying acquisition module")
    
    # Fallback to acquisition module
    try:
        from menipy.common import acquisition as acq
        logger.info(f"[do_acquisition] Loading from file via acquisition module: {image_path}")
        frames = acq.from_file([image_path])
        if frames:
            ctx.frames = frames
            ctx.frame = frames[0]
            ctx.image = frames[0].image if hasattr(frames[0], 'image') else frames[0]
            logger.info(f"[do_acquisition] Loaded {len(frames)} frames from disk")
    except Exception as e:
        logger.error(f"[do_acquisition] Failed to load from file: {e}")
    
    return ctx
