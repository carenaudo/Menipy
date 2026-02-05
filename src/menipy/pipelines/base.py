# src/menipy/pipelines/base.py
"""
Base pipeline class with template method pattern for stage-based execution.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Callable, Optional, ClassVar

import numpy as np

# Core models
from menipy.models.context import Context
from menipy.models.config import PreprocessingSettings, EdgeDetectionSettings
from menipy.models.fit import FitConfig
from menipy.models.geometry import Contour, Geometry

# Common utilities
from menipy.common import edge_detection as edged
from menipy.common import overlay as ovl
from menipy.common import solver as common_solver
from menipy.common.plugins import _load_module_from_path

# Make common utilities available to subclasses
__all__ = [
    "Context",
    "PreprocessingSettings",
    "EdgeDetectionSettings",
    "FitConfig",
    "Contour",
    "Geometry",
    "edged",
    "ovl",
    "common_solver",
]

# ------------------------------- Base Class ----------------------------------


class PipelineError(RuntimeError):
    """Raised when a pipeline stage fails fatally."""


class PipelineBase:
    """
    Template-Method pipeline skeleton.

    Subclasses override stage hooks (do_acquisition, do_preprocessing, …).
    Each hook may mutate Context in place and/or return it.
    The base class measures per-stage runtime, captures errors, and returns Context.
    """

    name: str = "base"

    # Common plugin setup
    _repo_root: ClassVar[Path] = Path(__file__).resolve().parents[3]
    _toy_path: ClassVar[Path] = _repo_root / "plugins" / "toy_young_laplace.py"
    _toy_mod = _load_module_from_path(_toy_path, "adsa_plugins.toy_young_laplace")
    young_laplace_sphere = getattr(_toy_mod, "toy_young_laplace")
    DEFAULT_SEQ = [
        ("acquisition", None),
        ("preprocessing", None),
        ("feature_detection", None),      # NEW: detect ROI, needle, substrate
        ("contour_extraction", None),     # was: edge_detection
        ("contour_refinement", None),     # NEW: clip/smooth contour
        ("calibration", None),            # was: scaling
        ("geometric_features", None),     # was: geometry
        ("physics", None),
        ("profile_fitting", None),        # was: solver
        # optimization stage REMOVED
        ("compute_metrics", None),        # was: outputs
        ("overlay", None),
        ("validation", None),
    ]

    # ---- Stage hooks (override in subclasses as needed) ----
    def do_acquisition(self, ctx: Context) -> Optional[Context]:
        """Load raw image data from file paths or wrap existing arrays."""
        return ctx

    def do_preprocessing(self, ctx: Context) -> Optional[Context]:
        """Apply image filters and enhancements (blur, contrast, threshold)."""
        from menipy.common import preprocessing

        if self.preprocessing_settings:
            return preprocessing.run(ctx, self.preprocessing_settings)
        return ctx

    def do_feature_detection(self, ctx: Context) -> Optional[Context]:
        """Detect features like ROI, needle, substrate, and contact points.
        
        This stage runs automatic detection algorithms to identify key image
        features before contour extraction. Results are stored in ctx for use
        by subsequent stages.
        """
        return ctx

    def do_contour_extraction(self, ctx: Context) -> Optional[Context]:
        """Extract the droplet contour from the preprocessed image.
        
        Previously named 'edge_detection'. Uses Canny or registered plugins
        to extract contour points as (x, y) coordinate arrays.
        """
        from menipy.common import edge_detection

        if self.edge_detection_settings:
            return edge_detection.run(ctx, self.edge_detection_settings)
        return ctx

    def do_contour_refinement(self, ctx: Context) -> Optional[Context]:
        """Refine the extracted contour by clipping, smoothing, or filtering.
        
        Operations may include:
        - Clipping contour at substrate line
        - Removing noise points
        - Smoothing contour path
        - Interpolating missing segments
        """
        return ctx

    def do_calibration(self, ctx: Context) -> Optional[Context]:
        """Convert pixel measurements to physical units (mm).
        
        Previously named 'scaling'. Calculates px_per_mm from calibration
        objects (needle diameter) and scales contour coordinates.
        """
        return ctx

    def do_geometric_features(self, ctx: Context) -> Optional[Context]:
        """Extract geometric features from the contour.
        
        Previously named 'geometry'. Calculates:
        - Axis of symmetry
        - Apex point location
        - Baseline/substrate line
        - Tilt angle
        
        Note: Metric computation has been moved to do_compute_metrics.
        """
        return ctx

    def do_physics(self, ctx: Context) -> Optional[Context]:
        """Define physical parameters for the model (densities, gravity)."""
        return ctx

    def do_profile_fitting(self, ctx: Context) -> Optional[Context]:
        """Fit the physical model (Young-Laplace) to the contour data.
        
        Previously named 'solver'. Uses least-squares optimization to fit
        theoretical curves to measured contour points.
        """
        return ctx

    def do_compute_metrics(self, ctx: Context) -> Optional[Context]:
        """Aggregate and compute final measurement results.
        
        Previously named 'outputs'. Computes derived metrics like:
        - Surface tension (from fit parameters)
        - Volume (disk integration)
        - Diameter, height
        - Contact angles
        """
        return ctx

    def do_overlay(self, ctx: Context) -> Optional[Context]:
        """Generate visual annotations to overlay on the original image."""
        return ctx

    def do_validation(self, ctx: Context) -> Optional[Context]:
        """Quality assurance checks on analysis results.
        
        This stage should verify:
        
        1. RESIDUAL MAGNITUDES
           - Check if fit residuals are within acceptable bounds
           - Flag results with high residual RMS (> threshold)
           - Compute goodness-of-fit metrics (R², chi-squared)
        
        2. PHYSICAL PLAUSIBILITY
           - Verify surface tension is in expected range (e.g., 15-80 mN/m)
           - Check contact angles are valid (0-180°)
           - Validate volume is positive and reasonable
           - Ensure Bond number is physically meaningful
        
        3. CONTOUR QUALITY
           - Check contour point density (too few points = unreliable fit)
           - Detect contour artifacts (gaps, noise spikes)
           - Verify contour is closed/continuous
           - Check for asymmetry beyond threshold
        
        4. GEOMETRIC CONSISTENCY
           - Validate apex is at expected location
           - Check baseline detection quality
           - Verify needle/calibration object detection
        
        5. REPEATABILITY (for batch processing)
           - Compare with previous frames if available
           - Flag sudden jumps in measured values
        
        Sets ctx.qa dict with:
           - 'ok': bool - overall pass/fail
           - 'warnings': list - non-fatal issues
           - 'errors': list - fatal issues
           - 'scores': dict - quality scores per check
        """
        return ctx

    # ---- Backward compatibility aliases (deprecated) ----
    def do_edge_detection(self, ctx: Context) -> Optional[Context]:
        """DEPRECATED: Use do_contour_extraction instead."""
        return self.do_contour_extraction(ctx)

    def do_geometry(self, ctx: Context) -> Optional[Context]:
        """DEPRECATED: Use do_geometric_features instead."""
        return self.do_geometric_features(ctx)

    def do_scaling(self, ctx: Context) -> Optional[Context]:
        """DEPRECATED: Use do_calibration instead."""
        return self.do_calibration(ctx)

    def do_solver(self, ctx: Context) -> Optional[Context]:
        """DEPRECATED: Use do_profile_fitting instead."""
        return self.do_profile_fitting(ctx)

    def do_outputs(self, ctx: Context) -> Optional[Context]:
        """DEPRECATED: Use do_compute_metrics instead."""
        return self.do_compute_metrics(ctx)

    def do_optimization(self, ctx: Context) -> Optional[Context]:
        """DEPRECATED: This stage has been removed."""
        return ctx

    # ---- Orchestration -------------------------------------------------------

    def __init__(
        self,
        *,
        logger: Optional[logging.Logger] = None,
        preprocessing_settings: Optional[PreprocessingSettings] = None,
        edge_detection_settings: Optional[EdgeDetectionSettings] = None,
    ) -> None:
        self.logger = logger or logging.getLogger(f"menipy.pipelines.{self.name}")
        self.preprocessing_settings = preprocessing_settings
        self.edge_detection_settings = edge_detection_settings
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
                )
            )
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
            self.logger.propagate = False

    def _prime_ctx(self, ctx: Context, **kwargs: Any) -> Context:
        """
        Seed Context with common runtime parameters so 'acquisition' has
        what it needs even for subset runs.
        Accepted keys: image / image_path, camera / cam_id, frames, roi
        """
        # timings dict (used by _call_stage)
        if not hasattr(ctx, "timings_ms"):
            ctx.timings_ms = {}

        image = kwargs.get("image")
        image_path = kwargs.get("image_path")

        # If callers pass a file path as `image`, normalize it to image_path so downstream
        # stages don't treat a string as pixel data.
        if isinstance(image, (str, Path)):
            image_path = str(image)
            image = None

        if image is not None:
            ctx.image = image
        if image_path is not None:
            ctx.image_path = image_path

        # For single-image processing, also populate current_frame
        if (
            image is not None
            and not kwargs.get("camera")
            and isinstance(image, np.ndarray)
        ):
            from menipy.models.frame import Frame

            ctx.current_frame = Frame(image=image)
            ctx.frames = [ctx.current_frame]  # Also set frames for compatibility
        cam = kwargs.get("camera")
        if cam is None:
            cam = kwargs.get("cam_id")
        if cam is not None:
            ctx.camera_id = cam

        if "frames" in kwargs and kwargs["frames"] is not None:
            try:
                ctx.frames_requested = int(kwargs["frames"])
            except Exception:
                ctx.frames_requested = kwargs["frames"]

        if "roi" in kwargs and kwargs["roi"] is not None:
            ctx.roi = kwargs["roi"]
        # Also accept roi_rect as alias for roi
        if "roi_rect" in kwargs and kwargs["roi_rect"] is not None:
            ctx.roi = kwargs["roi_rect"]
        if "needle_rect" in kwargs and kwargs["needle_rect"] is not None:
            ctx.needle_rect = kwargs["needle_rect"]
        if "contact_line" in kwargs and kwargs["contact_line"] is not None:
            ctx.contact_line = kwargs["contact_line"]
        # Additional calibration results from CalibrationWizardDialog
        if "substrate_line" in kwargs and kwargs["substrate_line"] is not None:
            ctx.substrate_line = kwargs["substrate_line"]
        if "drop_contour" in kwargs and kwargs["drop_contour"] is not None:
            ctx.drop_contour = kwargs["drop_contour"]
        if "contact_points" in kwargs and kwargs["contact_points"] is not None:
            ctx.contact_points = kwargs["contact_points"]
        # Scale factor (pixels per mm)
        if "px_per_mm" in kwargs and kwargs["px_per_mm"] is not None:
            ctx.px_per_mm = kwargs["px_per_mm"]

        # Calibration parameters
        if "needle_diameter_mm" in kwargs and kwargs["needle_diameter_mm"] is not None:
            ctx.needle_diameter_mm = kwargs["needle_diameter_mm"]
        if "calibration_params" in kwargs and kwargs["calibration_params"] is not None:
            for key, value in kwargs["calibration_params"].items():
                setattr(ctx, key, value)

        # Prioritize kwargs over instance settings to avoid multiple value errors.
        if "preprocessing_settings" not in kwargs:
            kwargs["preprocessing_settings"] = self.preprocessing_settings
        ctx.preprocessing_settings = kwargs["preprocessing_settings"]
        if "edge_detection_settings" not in kwargs:
            kwargs["edge_detection_settings"] = self.edge_detection_settings
        ctx.edge_detection_settings = kwargs["edge_detection_settings"]

        if "measurement_id" in kwargs:
            ctx.measurement_id = kwargs["measurement_id"]
        if "measurement_sequence" in kwargs:
            ctx.measurement_sequence = kwargs["measurement_sequence"]

        # Explicitly set scale and physics if provided
        if "scale" in kwargs:
            ctx.scale = kwargs["scale"]
        if "physics" in kwargs:
            ctx.physics = kwargs["physics"]

        # Store any other kwargs into context (e.g. auto_detect_features)
        known_keys = {
            "image", "image_path", "camera", "cam_id", "frames", "roi", "roi_rect",
            "needle_rect", "contact_line", "substrate_line", "drop_contour", "contact_points",
            "px_per_mm", "needle_diameter_mm", "calibration_params",
            "preprocessing_settings", "edge_detection_settings",
            "measurement_id", "measurement_sequence", "scale", "physics"
        }
        for k, v in kwargs.items():
            if k not in known_keys:
                setattr(ctx, k, v)

        return ctx

    def _call_stage(
        self, ctx: Context, stage_name: str, fn: Callable[[Context], Optional[Context]]
    ) -> Context:
        start = time.perf_counter()

        def _ctx_summary(c: Context) -> str:
            try:
                fr = getattr(c, "frames", None)
                if fr is None:
                    nfr = 0
                elif isinstance(fr, (list, tuple)):
                    nfr = len(fr)
                else:
                    try:
                        nfr = int(getattr(fr, "shape", (None,))[0])
                    except Exception:
                        nfr = 1
                preview = bool(getattr(c, "preview", None))
                contour_obj = getattr(c, "contour", None)
                contour = False
                try:
                    if contour_obj is not None:
                        contour = getattr(contour_obj, "xy", None) is not None
                except Exception:
                    contour = True
                results = getattr(c, "results", None)
                results_k = (
                    len(results) if isinstance(results, dict) else (1 if results else 0)
                )
                return f"frames={nfr} preview={int(preview)} contour={int(bool(contour))} results_keys={results_k}"
            except Exception:
                return "summary_error"

        try:
            maybe_ctx = fn(ctx)
            if maybe_ctx is not None:
                ctx = maybe_ctx
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            # timings/log live in Context (provided by models.datatypes)
            if hasattr(ctx, "timings_ms") and isinstance(ctx.timings_ms, dict):
                ctx.timings_ms[stage_name] = elapsed_ms
            if hasattr(ctx, "error"):
                ctx.error = f"{stage_name} failed: {exc!r}"
            # log exception (pipeline logger)
            self.logger.exception("%s failed after %.2f ms", stage_name, elapsed_ms)

            try:
                ctx_id = hex(id(ctx))
            except Exception:
                ctx_id = str(id(ctx))
            summary = _ctx_summary(ctx)
            msg = f"[pipeline:{self.name} ctx={ctx_id}] {stage_name} failed after {elapsed_ms:.2f} ms: {exc!r} | {summary}"
            try:
                if hasattr(ctx, "note"):
                    ctx.note(msg)
                elif hasattr(ctx, "log") and isinstance(ctx.log, list):
                    ctx.log.append(msg)
            except Exception:
                pass
            try:
                logging.getLogger().error(msg)
            except Exception:
                pass
            raise PipelineError(getattr(ctx, "error", str(exc))) from exc
        else:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            if hasattr(ctx, "timings_ms") and isinstance(ctx.timings_ms, dict):
                ctx.timings_ms[stage_name] = elapsed_ms
            self.logger.debug("✓ %s (%.2f ms)", stage_name, elapsed_ms)

            try:
                ctx_id = hex(id(ctx))
            except Exception:
                ctx_id = str(id(ctx))
            summary = _ctx_summary(ctx)
            msg = f"[pipeline:{self.name} ctx={ctx_id}] Completed stage: {stage_name} ({elapsed_ms:.2f} ms) | {summary}"
            try:
                if hasattr(ctx, "note"):
                    ctx.note(msg)
                elif hasattr(ctx, "log") and isinstance(ctx.log, list):
                    ctx.log.append(msg)
            except Exception:
                pass

            try:
                self.logger.info(msg)
            except Exception:
                pass
            try:
                root = logging.getLogger()
                if any(h.__class__.__name__ == "QtLogHandler" for h in root.handlers):
                    root.setLevel(logging.INFO)
                    root.info(msg)
            except Exception:
                pass

        return ctx

    def build_plan(self, only: list[str] | None = None, include_prereqs: bool = True):
        seq = [(n, getattr(self, f"do_{n}", None)) for (n, _fn) in self.DEFAULT_SEQ]
        seq = [(n, fn) for (n, fn) in seq if callable(fn)]
        if not only:
            return seq
        names = [n for (n, _fn) in seq]
        if include_prereqs:
            # keep everything up to the furthest requested stage
            idx = {n: i for i, n in enumerate(names)}
            last = max(idx[n] for n in only if n in idx)
            wanted = set(names[: last + 1])
        else:
            wanted = set(only)
        return [(n, fn) for (n, fn) in seq if n in wanted]

    def run_with_plan(
        self,
        *,
        only: list[str] | None = None,
        include_prereqs: bool = True,
        **kwargs: Any,
    ) -> Context:
        ctx = Context()
        ctx = self._prime_ctx(ctx, **kwargs)
        for name, fn in self.build_plan(only=only, include_prereqs=include_prereqs):
            ctx = self._call_stage(ctx, name, fn)
        self._ctx = ctx
        return ctx

    def run(self, **kwargs: Any) -> Context:
        """
        Execute the full stage sequence and return the populated Context.
        Any **kwargs are seeded into Context for 'acquisition' to use.
        """
        ctx = Context()
        ctx = self._prime_ctx(ctx, **kwargs)

        self.logger.info("Starting pipeline: %s", self.name)

        stages: list[tuple[str, Callable[[Context], Optional[Context]]]] = [
            ("acquisition", self.do_acquisition),
            ("preprocessing", self.do_preprocessing),
            ("feature_detection", self.do_feature_detection),
            ("contour_extraction", self.do_contour_extraction),
            ("contour_refinement", self.do_contour_refinement),
            ("calibration", self.do_calibration),
            ("geometric_features", self.do_geometric_features),
            ("physics", self.do_physics),
            ("profile_fitting", self.do_profile_fitting),
            ("compute_metrics", self.do_compute_metrics),
            ("overlay", self.do_overlay),
            ("validation", self.do_validation),
        ]

        for name, fn in stages:
            ctx = self._call_stage(ctx, name, fn)

        self.logger.info("Finished pipeline: %s", self.name)
        return ctx
