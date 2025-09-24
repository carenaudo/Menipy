# src/menipy/pipeline/base.py
from __future__ import annotations

import logging
import time
from typing import Any, Callable, Optional, Dict

# Context lives in models.datatypes (per your requirement)
from menipy.models.datatypes import Context, PreprocessingSettings, EdgeDetectionSettings



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
    DEFAULT_SEQ = [
            ("acquisition",  None),
            ("preprocessing", None),
            ("edge_detection", None),
            ("geometry", None),
            ("scaling", None),
            ("physics", None),
            ("solver", None),
            ("optimization", None),
            ("outputs", None),
            ("overlay", None),
            ("validation", None),
        ]
    # ---- Stage hooks (override in subclasses as needed) ----
    def do_acquisition(self, ctx: Context) -> Optional[Context]: return ctx
    def do_preprocessing(self, ctx: Context) -> Optional[Context]:
        from menipy.common import preprocessing
        if self.preprocessing_settings:
            return preprocessing.run(ctx, self.preprocessing_settings)
        return ctx
    def do_edge_detection(self, ctx: Context) -> Optional[Context]:
        from menipy.common import edge_detection
        if self.edge_detection_settings:
            return edge_detection.run(ctx, self.edge_detection_settings)
        return ctx
    def do_geometry(self, ctx: Context) -> Optional[Context]: return ctx
    def do_scaling(self, ctx: Context) -> Optional[Context]: return ctx
    def do_physics(self, ctx: Context) -> Optional[Context]: return ctx
    def do_solver(self, ctx: Context) -> Optional[Context]: return ctx
    def do_optimization(self, ctx: Context) -> Optional[Context]: return ctx
    def do_outputs(self, ctx: Context) -> Optional[Context]: return ctx
    def do_overlay(self, ctx: Context) -> Optional[Context]: return ctx
    def do_validation(self, ctx: Context) -> Optional[Context]: return ctx

    # ---- Orchestration -------------------------------------------------------

    def __init__(self, *,
                 logger: Optional[logging.Logger] = None,
                 preprocessing_settings: Optional[PreprocessingSettings] = None,
                 edge_detection_settings: Optional[EdgeDetectionSettings] = None) -> None:
        self.logger = logger or logging.getLogger(f"menipy.pipelines.{self.name}")
        self.preprocessing_settings = preprocessing_settings
        self.edge_detection_settings = edge_detection_settings
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
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

        image = kwargs.get("image") or kwargs.get("image_path")
        if image is not None:
            ctx.image_path = image

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
        if "needle_rect" in kwargs and kwargs["needle_rect"] is not None:
            ctx.needle_rect = kwargs["needle_rect"]
        if "contact_line" in kwargs and kwargs["contact_line"] is not None:
            ctx.contact_line = kwargs["contact_line"]

        if "preprocessing_settings" in kwargs and kwargs["preprocessing_settings"] is not None:
            ctx.preprocessing_settings = kwargs["preprocessing_settings"]
        if "edge_detection_settings" in kwargs and kwargs["edge_detection_settings"] is not None:
            ctx.edge_detection_settings = kwargs["edge_detection_settings"]

        return ctx

    def _call_stage(self, ctx: Context, stage_name: str, fn: Callable[[Context], Optional[Context]]) -> Context:
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
                results_k = len(results) if isinstance(results, dict) else (1 if results else 0)
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
                if any(h.__class__.__name__ == 'QtLogHandler' for h in root.handlers):
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

    def run_with_plan(self, *, only: list[str] | None = None, include_prereqs: bool = True, **kwargs: Any) -> Context:
        ctx = Context()
        ctx = self._prime_ctx(ctx, preprocessing_settings=self.preprocessing_settings, edge_detection_settings=self.edge_detection_settings, **kwargs)
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
        ctx = self._prime_ctx(ctx, preprocessing_settings=self.preprocessing_settings, edge_detection_settings=self.edge_detection_settings, **kwargs)

        self.logger.info("Starting pipeline: %s", self.name)

        stages: list[tuple[str, Callable[[Context], Optional[Context]]]] = [
            ("acquisition", self.do_acquisition),
            ("preprocessing", self.do_preprocessing),
            ("edge_detection", self.do_edge_detection),
            ("geometry", self.do_geometry),
            ("scaling", self.do_scaling),
            ("physics", self.do_physics),
            ("solver", self.do_solver),
            ("optimization", self.do_optimization),
            ("outputs", self.do_outputs),
            ("overlay", self.do_overlay),
            ("validation", self.do_validation),
        ]

        for name, fn in stages:
            ctx = self._call_stage(ctx, name, fn)

        self.logger.info("Finished pipeline: %s", self.name)
        return ctx