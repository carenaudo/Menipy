# src/menipy/pipeline/base.py
from __future__ import annotations

import logging
import time
from typing import Any, Callable, Optional, Dict

# Context lives in models.datatypes (per your requirement)
from ..models.datatypes import Context



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

    # ---- Stage hooks (override in subclasses as needed) ----
    def do_acquisition(self, ctx: Context) -> Optional[Context]: return ctx
    def do_preprocessing(self, ctx: Context) -> Optional[Context]: return ctx
    def do_edge_detection(self, ctx: Context) -> Optional[Context]: return ctx
    def do_geometry(self, ctx: Context) -> Optional[Context]: return ctx
    def do_scaling(self, ctx: Context) -> Optional[Context]: return ctx
    def do_physics(self, ctx: Context) -> Optional[Context]: return ctx
    def do_solver(self, ctx: Context) -> Optional[Context]: return ctx
    def do_optimization(self, ctx: Context) -> Optional[Context]: return ctx
    def do_outputs(self, ctx: Context) -> Optional[Context]: return ctx
    def do_overlay(self, ctx: Context) -> Optional[Context]: return ctx
    def do_validation(self, ctx: Context) -> Optional[Context]: return ctx

    # ---- Orchestration -------------------------------------------------------

    def __init__(self, *, logger: Optional[logging.Logger] = None) -> None:
        self.logger = logger or logging.getLogger(f"menipy.pipelines.{self.name}")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
            self.logger.propagate = False

    def _call_stage(self, ctx: Context, stage_name: str, fn: Callable[[Context], Optional[Context]]) -> Context:
        start = time.perf_counter()
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
            self.logger.exception("%s failed after %.2f ms", stage_name, elapsed_ms)
            raise PipelineError(getattr(ctx, "error", str(exc))) from exc
        else:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            if hasattr(ctx, "timings_ms") and isinstance(ctx.timings_ms, dict):
                ctx.timings_ms[stage_name] = elapsed_ms
            self.logger.debug("✓ %s (%.2f ms)", stage_name, elapsed_ms)
        return ctx

    def run(self, **kwargs: Any) -> Context:
        """
        Execute the full stage sequence and return the populated Context.
        Parameters in **kwargs are ignored by the base class but allow subclasses
        to accept runtime options in their overridden hooks if desired.
        """
        # Context is created by caller or here? We create a fresh one here.
        ctx = Context()  # comes from models.datatypes

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