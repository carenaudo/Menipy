# src/menipy/pipelines/base.py
"""Base pipeline class with template method pattern for stage-based execution."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

# Common utilities
from menipy.common import edge_detection as edged
from menipy.common import overlay as ovl
from menipy.common import solver as common_solver
from menipy.common._module_loader import load_module_from_path
from menipy.common.cancellation import cancellation_scope
from menipy.models.config import EdgeDetectionSettings, PreprocessingSettings

# Core models
from menipy.models.context import Context
from menipy.models.fit import FitConfig
from menipy.models.geometry import Contour, Geometry

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


# Stage names used before the canonical sequence below; ``None`` marks a stage
# that no longer exists. Stored SOPs and presets may still contain them.
LEGACY_STAGE_NAMES: dict[str, str | None] = {
    "edge_detection": "contour_extraction",
    "geometry": "geometric_features",
    "scaling": "calibration",
    "solver": "profile_fitting",
    "outputs": "compute_metrics",
    "optimization": None,
}


def canonical_stage_name(name: str) -> str | None:
    """Current name of a stage.

    Parameters
    ----------
    name : str
        Canonical or legacy stage name.

    Returns
    -------
    str or None
        The canonical name, or ``None`` for a removed legacy stage.
    """
    name = str(name).strip().lower()
    return LEGACY_STAGE_NAMES.get(name, name)


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
    _toy_mod = load_module_from_path(_toy_path, "menipy_plugins.toy_young_laplace")
    young_laplace_sphere = _toy_mod.toy_young_laplace
    DEFAULT_SEQ = [
        ("acquisition", None),
        ("preprocessing", None),
        ("feature_detection", None),  # NEW: detect ROI, needle, substrate
        ("contour_extraction", None),  # was: edge_detection
        ("contour_refinement", None),  # NEW: clip/smooth contour
        ("calibration", None),  # was: scaling
        ("geometric_features", None),  # was: geometry
        ("physics", None),
        ("profile_fitting", None),  # was: solver
        # optimization stage REMOVED
        ("compute_metrics", None),  # was: outputs
        ("overlay", None),
        ("validation", None),
    ]
    # Stages a run may leave out because no later stage reads their output.
    # ``overlay`` only draws (``preview``/``overlay``/``overlay_commands``);
    # the preview then shows the plain image. Every other stage feeds the next
    # one, and ``validation`` produces the accept/reject flag of the result.
    OPTIONAL_STAGES: ClassVar[frozenset[str]] = frozenset({"overlay"})

    @classmethod
    def stage_names(cls) -> list[str]:
        """Stages this pipeline runs, in order.

        Returns
        -------
        list of str
            Canonical stage names with an implementation.
        """
        return [n for n, _ in cls.DEFAULT_SEQ if callable(getattr(cls, f"do_{n}", None))]

    @classmethod
    def skipped_stages(cls, included: list[str] | tuple[str, ...]) -> list[str]:
        """Optional stages that a stage selection leaves out.

        Required stages always run, so only optional stages missing from
        ``included`` are returned.

        Parameters
        ----------
        included : sequence of str
            Selected stages; legacy names are accepted.

        Returns
        -------
        list of str
            Optional stages to skip, in pipeline order.
        """
        chosen = {canonical_stage_name(name) for name in included}
        return [
            name
            for name in cls.stage_names()
            if name in cls.OPTIONAL_STAGES and name not in chosen
        ]

    # ---- Stage hooks (override in subclasses as needed) ----
    def do_acquisition(self, ctx: Context) -> Context | None:
        """Load raw image data from file paths or wrap existing arrays."""
        return ctx

    def do_preprocessing(self, ctx: Context) -> Context | None:
        """Apply image filters and enhancements (blur, contrast, threshold)."""
        from menipy.common import preprocessing

        if self.preprocessing_settings:
            return preprocessing.run(ctx, self.preprocessing_settings)
        return ctx

    def do_feature_detection(self, ctx: Context) -> Context | None:
        """Detect features like ROI, needle, substrate, and contact points.

        This stage runs automatic detection algorithms to identify key image
        features before contour extraction. Results are stored in ctx for use
        by subsequent stages.
        """
        return ctx

    def do_contour_extraction(self, ctx: Context) -> Context | None:
        """Extract the droplet contour from the preprocessed image.

        Previously named 'edge_detection'. Uses Canny or registered plugins
        to extract contour points as (x, y) coordinate arrays.
        """
        from menipy.common import edge_detection

        if self.edge_detection_settings:
            return edge_detection.run(ctx, self.edge_detection_settings)
        return ctx

    def do_contour_refinement(self, ctx: Context) -> Context | None:
        """Refine the extracted contour by clipping, smoothing, or filtering.

        Operations may include:
            - Clipping contour at substrate line
        - Removing noise points
        - Smoothing contour path
        - Interpolating missing segments
        """
        return ctx

    def do_calibration(self, ctx: Context) -> Context | None:
        """Convert pixel measurements to physical units (mm).

        Previously named 'scaling'. Calculates px_per_mm from calibration
        objects (needle diameter) and scales contour coordinates.
        """
        return ctx

    def do_geometric_features(self, ctx: Context) -> Context | None:
        """Extract geometric features from the contour.

        Previously named 'geometry'. Calculates:
            - Axis of symmetry
        - Apex point location
        - Baseline/substrate line
        - Tilt angle

        Note: Metric computation has been moved to do_compute_metrics.
        """
        return ctx

    def do_physics(self, ctx: Context) -> Context | None:
        """Define physical parameters for the model (densities, gravity)."""
        return ctx

    def do_profile_fitting(self, ctx: Context) -> Context | None:
        """Fit the physical model (Young-Laplace) to the contour data.

        Previously named 'solver'. Uses least-squares optimization to fit
        theoretical curves to measured contour points.
        """
        return ctx

    def do_compute_metrics(self, ctx: Context) -> Context | None:
        """Aggregate and compute final measurement results.

        Previously named 'outputs'. Computes derived metrics like:
            - Surface tension (from fit parameters)
        - Volume (disk integration)
        - Diameter, height
        - Contact angles
        """
        return ctx

    def do_overlay(self, ctx: Context) -> Context | None:
        """Generate visual annotations to overlay on the original image."""
        return ctx

    def do_validation(self, ctx: Context) -> Context | None:
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
        from menipy.common.validation import build_diagnostics, validate

        ctx.qa = validate(ctx)
        ctx.results["diagnostics"] = build_diagnostics(ctx, ctx.qa)
        return ctx

    # ---- Backward compatibility aliases (deprecated) ----
    def do_edge_detection(self, ctx: Context) -> Context | None:
        """DEPRECATED: Use do_contour_extraction instead."""
        return self.do_contour_extraction(ctx)

    def do_geometry(self, ctx: Context) -> Context | None:
        """DEPRECATED: Use do_geometric_features instead."""
        return self.do_geometric_features(ctx)

    def do_scaling(self, ctx: Context) -> Context | None:
        """DEPRECATED: Use do_calibration instead."""
        return self.do_calibration(ctx)

    def do_solver(self, ctx: Context) -> Context | None:
        """DEPRECATED: Use do_profile_fitting instead."""
        return self.do_profile_fitting(ctx)

    def do_outputs(self, ctx: Context) -> Context | None:
        """DEPRECATED: Use do_compute_metrics instead."""
        return self.do_compute_metrics(ctx)

    def do_optimization(self, ctx: Context) -> Context | None:
        """DEPRECATED: This stage has been removed."""
        return ctx

    # ---- Orchestration -------------------------------------------------------

    def __init__(
        self,
        *,
        logger: logging.Logger | None = None,
        preprocessing_settings: PreprocessingSettings | None = None,
        edge_detection_settings: EdgeDetectionSettings | None = None,
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
            frame_arg = kwargs["frames"]
            if isinstance(frame_arg, (list, tuple, np.ndarray)):
                ctx.frames = (
                    list(frame_arg) if isinstance(frame_arg, tuple) else frame_arg
                )
                try:
                    ctx.frames_requested = len(frame_arg)
                except TypeError:
                    ctx.frames_requested = 1
            else:
                ctx.frames_requested = int(frame_arg)

        if "roi" in kwargs and kwargs["roi"] is not None:
            ctx.roi = kwargs["roi"]
        # Also accept roi_rect as alias for roi
        if "roi_rect" in kwargs and kwargs["roi_rect"] is not None:
            ctx.roi = kwargs["roi_rect"]
        if "detected_roi" in kwargs and kwargs["detected_roi"] is not None:
            ctx.detected_roi = kwargs["detected_roi"]
        if "needle_rect" in kwargs and kwargs["needle_rect"] is not None:
            ctx.needle_rect = kwargs["needle_rect"]
        if "contact_line" in kwargs and kwargs["contact_line"] is not None:
            ctx.contact_line = kwargs["contact_line"]
        # Additional calibration results from CalibrationWizardDialog
        if "substrate_line" in kwargs and kwargs["substrate_line"] is not None:
            ctx.substrate_line = kwargs["substrate_line"]
        if "drop_contour" in kwargs and kwargs["drop_contour"] is not None:
            ctx.drop_contour = kwargs["drop_contour"]
        if "detected_contour" in kwargs and kwargs["detected_contour"] is not None:
            ctx.detected_contour = kwargs["detected_contour"]
        if "contact_points" in kwargs and kwargs["contact_points"] is not None:
            ctx.contact_points = kwargs["contact_points"]
        if "apex_point" in kwargs and kwargs["apex_point"] is not None:
            ctx.apex_point = kwargs["apex_point"]
        # Scale factor (pixels per mm)
        if "px_per_mm" in kwargs and kwargs["px_per_mm"] is not None:
            ctx.px_per_mm = kwargs["px_per_mm"]

        # Calibration parameters
        if "needle_diameter_mm" in kwargs and kwargs["needle_diameter_mm"] is not None:
            ctx.needle_diameter_mm = kwargs["needle_diameter_mm"]
        if "calibration_params" in kwargs and kwargs["calibration_params"] is not None:
            for key, value in kwargs["calibration_params"].items():
                # Normalize gravity aliases into physics dict to keep Context schema strict.
                if key in {"gravity_m_s2", "g"}:
                    if value is not None:
                        if not isinstance(ctx.physics, dict):
                            ctx.physics = {}
                        ctx.physics["g"] = value
                    continue
                if key in type(ctx).model_fields:
                    setattr(ctx, key, value)

        # Pipeline-specific analysis settings (GUI/CLI) are additive fields on
        # Context; legacy callers may continue passing this nested mapping.
        analysis_params = kwargs.get("analysis_params") or {}
        if isinstance(analysis_params, dict):
            for key in (
                "experimental_geometry_mode",
                "needle_geometry_method",
                "pendant_initializer",
                "pendant_contour_model",
                "contact_angle_method",
                "onnx_proposal_mode",
                "segmentation_provider",
                "onnx_proposal_classes",
                "pendant_approximation_methods",
                "pendant_approximator_settings",
            ):
                if key in analysis_params and key in type(ctx).model_fields:
                    setattr(ctx, key, analysis_params[key])

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
            "image",
            "image_path",
            "camera",
            "cam_id",
            "frames",
            "roi",
            "roi_rect",
            "detected_roi",
            "needle_rect",
            "contact_line",
            "substrate_line",
            "drop_contour",
            "detected_contour",
            "contact_points",
            "apex_point",
            "px_per_mm",
            "needle_diameter_mm",
            "calibration_params",
            "analysis_params",
            "preprocessing_settings",
            "edge_detection_settings",
            "measurement_id",
            "measurement_sequence",
            "scale",
            "physics",
        }
        for k, v in kwargs.items():
            if k not in known_keys:
                if k in type(ctx).model_fields:
                    setattr(ctx, k, v)

        return ctx

    def _call_stage(
        self, ctx: Context, stage_name: str, fn: Callable[[Context], Context | None]
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
            token = ctx.cancellation_token
            with cancellation_scope(token):
                maybe_ctx = fn(ctx)
                if maybe_ctx is not None:
                    ctx = maybe_ctx
                    ctx.cancellation_token = token
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
        """Ordered ``(stage, callable)`` pairs to execute.

        Parameters
        ----------
        only : list of str, optional
            Target stages (legacy names accepted). ``None`` plans every stage.
        include_prereqs : bool, optional
            With ``True`` (default) every stage up to the last target runs,
            because each stage needs the output of the earlier ones.

        Returns
        -------
        list of tuple
            The stage plan.
        """
        seq = [(n, getattr(self, f"do_{n}", None)) for (n, _fn) in self.DEFAULT_SEQ]
        seq = [(n, fn) for (n, fn) in seq if callable(fn)]
        if not only:
            return seq
        names = [n for (n, _fn) in seq]
        targets = [canonical_stage_name(n) for n in only]
        targets = [n for n in targets if n in names]
        if not targets:
            raise PipelineError(f"None of the requested stages exist: {list(only)}")
        if include_prereqs:
            # keep everything up to the furthest requested stage
            idx = {n: i for i, n in enumerate(names)}
            last = max(idx[n] for n in targets)
            wanted = set(names[: last + 1])
        else:
            wanted = set(targets)
        return [(n, fn) for (n, fn) in seq if n in wanted]

    def _without_skipped(self, plan: list, skip_stages) -> list:
        """Drop ``skip_stages`` from ``plan``; only optional stages may go."""
        skip = {canonical_stage_name(n) for n in (skip_stages or ())} - {None}
        required = skip - set(self.OPTIONAL_STAGES)
        if required:
            raise PipelineError(
                f"Required stages cannot be skipped: {sorted(required)}"
            )
        return [(n, fn) for (n, fn) in plan if n not in skip]

    def run_with_plan(
        self,
        *,
        only: list[str] | None = None,
        include_prereqs: bool = True,
        skip_stages: list[str] | tuple[str, ...] | None = None,
        **kwargs: Any,
    ) -> Context:
        """Run a subset of the stage sequence.

        Parameters
        ----------
        only : list[str], optional
            Target stages; ``None`` runs every stage.
        include_prereqs : bool, optional
            Whether to include prerequisite stages. Default is True.
        skip_stages : sequence of str, optional
            Optional stages (see ``OPTIONAL_STAGES``) to leave out.
        **kwargs : Any
            Additional keyword arguments.

        Returns
        -------
        Context
            The result context from the pipeline execution.

        Raises
        ------
        PipelineError
            If ``skip_stages`` names a required stage.
        """
        plan = self._without_skipped(
            self.build_plan(only=only, include_prereqs=include_prereqs), skip_stages
        )
        ctx = Context()
        ctx = self._prime_ctx(ctx, **kwargs)
        try:
            for name, fn in plan:
                ctx = self._call_stage(ctx, name, fn)
        finally:
            if ctx.sequence_store is not None:
                ctx.sequence_store.close()
                ctx.sequence_store = None
        self._ctx = ctx
        return ctx

    def run(self, **kwargs: Any) -> Context:
        """
        Execute the full stage sequence and return the populated Context.
        Any **kwargs are seeded into Context for 'acquisition' to use; an
        optional ``skip_stages`` sequence leaves out optional stages (see
        ``OPTIONAL_STAGES``).
        """
        plan = self._without_skipped(self.build_plan(), kwargs.pop("skip_stages", None))
        ctx = Context()
        ctx = self._prime_ctx(ctx, **kwargs)

        self.logger.info("Starting pipeline: %s", self.name)

        try:
            for name, fn in plan:
                ctx = self._call_stage(ctx, name, fn)
        finally:
            if ctx.sequence_store is not None:
                ctx.sequence_store.close()
                ctx.sequence_store = None

        self.logger.info("Finished pipeline: %s", self.name)
        return ctx
