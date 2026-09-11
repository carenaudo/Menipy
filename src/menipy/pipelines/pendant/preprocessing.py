"""
Pendant Pipeline - Preprocessing Stage

This module provides preprocessing operations for the pendant drop pipeline,
using the stage-based preprocessor plugins for automatic feature detection.
"""

from __future__ import annotations

import logging

from menipy.models.context import Context

logger = logging.getLogger(__name__)


# Geometry a caller can supply -- from calibration, a wizard, or a script --
# which automatic detection may fill in but never overwrite.
_CALLER_GEOMETRY = (
    "needle_rect",
    "contact_points",
    "apex_point",
    "roi",
    "detected_roi",
    "drop_contour",
    "detected_contour",
)


def do_preprocessing(ctx: Context) -> Context | None:
    """Preprocess an image for pendant drop analysis.

    When the ``auto_detect`` preprocessor plugin is registered it runs drop
    contour detection, needle detection (shaft line analysis) and ROI
    computation. Detection only fills in geometry the caller left unset;
    supplied geometry (see ``_CALLER_GEOMETRY``) is kept as given.

    Parameters
    ----------
    ctx : Context
        Pipeline context with image data.

    Returns
    -------
    Context or None
        Updated context with preprocessing results.
    """
    # Check if auto-detection is enabled
    if not getattr(ctx, "auto_detect_features", True):
        logger.debug("Auto-detection disabled for pendant pipeline")
        return ctx

    try:
        # Load and run auto_detect preprocessor plugin
        # Import plugin to register it
        import sys
        from pathlib import Path

        from menipy.common.registry import PREPROCESSORS

        plugins_dir = Path(__file__).parent.parent.parent.parent.parent / "plugins"
        if plugins_dir.exists() and str(plugins_dir) not in sys.path:
            sys.path.insert(0, str(plugins_dir))

        if "auto_detect" in PREPROCESSORS:
            # Create a wrapper context that allows pipeline_name
            class DetectionContext:
                def __init__(self, ctx):
                    self._ctx = ctx
                    self.pipeline_name = "pendant"
                    self.auto_detect_features = getattr(
                        ctx, "auto_detect_features", True
                    )

                @property
                def image(self):
                    return getattr(self._ctx, "image", None)

                @property
                def frames(self):
                    """frames.

                    Returns
                    -------
                    type
                    Description.
                    """
                    return getattr(self._ctx, "frames", None)

                def __getattr__(self, name):
                    return getattr(self._ctx, name)

                def __setattr__(self, name, value):
                    if name in ("_ctx", "pipeline_name", "auto_detect_features"):
                        object.__setattr__(self, name, value)
                    else:
                        setattr(self._ctx, name, value)

            # Auto-detection fills in geometry the caller did not supply; it
            # must not overwrite geometry that was supplied. Its needle
            # detector estimates the contacts with a different tolerance than
            # AutoCalibrator, so calibrated or hand-edited contacts moved and
            # changed where the contour is clipped at the needle.
            supplied = {
                name: getattr(ctx, name, None)
                for name in _CALLER_GEOMETRY
                if getattr(ctx, name, None) is not None
            }
            detection_ctx = DetectionContext(ctx)
            PREPROCESSORS["auto_detect"](detection_ctx)
            for name, value in supplied.items():
                setattr(ctx, name, value)
            logger.info(
                "Pendant auto-detection complete (kept supplied: %s)",
                ", ".join(sorted(supplied)) or "none",
            )
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
    # Ensure legacy fields for tests
    if (
        getattr(ctx, "preprocessed", None) is None
        and getattr(ctx, "image", None) is not None
    ):
        ctx.preprocessed = getattr(ctx, "image", None)
    if getattr(ctx, "preprocessed_settings", None) is None:
        # minimal defaults expected by tests
        ctx.preprocessed_settings = {"blur_ksize": (5, 5)}
    return ctx


# Backward-compatible alias expected by older tests
def run(ctx: Context) -> Context | None:
    """Run.

    Parameters
    ----------
    ctx : type
        Description.

    Returns
    -------
    type
        Description.
    """
    return do_preprocessing(ctx)
