"""Needle-in-sessile-drop hysteresis pipeline stages."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from menipy.common.needle_hysteresis import analyze_needle_hysteresis_sequence
from menipy.common.sequence_acquisition import (
    VIDEO_SUFFIXES,
    frames_from_memory,
    load_image_sequence,
    load_video,
)
from menipy.models.context import Context
from menipy.models.frame import Frame
from menipy.pipelines.base import PipelineBase

logger = logging.getLogger(__name__)


class NeedleHysteresisPipeline(PipelineBase):
    """Analyze dynamic advancing/receding angles and hysteresis with an immersed needle."""

    name = "needle_hysteresis"
    ui_metadata = {
        "display_name": "Advancing / Receding Needle Drop",
        "icon": "sessile.svg",
        "color": "#10B981",
        "stages": ["acquisition", "compute_metrics", "overlay", "validation"],
        "calibration_params": ["needle_diameter_mm"],
        "primary_metrics": [
            "theta_advancing_deg",
            "theta_receding_deg",
            "contact_angle_hysteresis_deg",
            "base_diameter_max_mm",
        ],
    }

    def do_acquisition(self, ctx: Context) -> Context | None:
        source = ctx.sequence_path or ctx.image_path
        if source:
            path = Path(source)
            if path.is_dir():
                frames, metadata = load_image_sequence(path, fps=ctx.sequence_fps)
            elif path.suffix.lower() in VIDEO_SUFFIXES:
                frames, metadata = load_video(path)
            else:
                raise ValueError("source must be video or sequence directory")
        else:
            raw_frames = ctx.frames
            if isinstance(raw_frames, np.ndarray):
                raw_frames = [raw_frames]
            frames, metadata = frames_from_memory(list(raw_frames or []), fps=ctx.sequence_fps)

        if not frames:
            raise ValueError("No frames available for needle_hysteresis pipeline")

        ctx.frames = frames
        ctx.current_frame = frames[0]
        ctx.image = frames[0].image
        ctx.sequence_metadata = metadata
        return ctx

    def do_preprocessing(self, ctx: Context) -> Context | None:
        return ctx

    def do_feature_detection(self, ctx: Context) -> Context | None:
        return ctx

    def do_contour_extraction(self, ctx: Context) -> Context | None:
        return ctx

    def do_contour_refinement(self, ctx: Context) -> Context | None:
        return ctx

    def do_calibration(self, ctx: Context) -> Context | None:
        return ctx

    def do_geometric_features(self, ctx: Context) -> Context | None:
        return ctx

    def do_physics(self, ctx: Context) -> Context | None:
        return ctx

    def do_profile_fitting(self, ctx: Context) -> Context | None:
        return ctx

    def do_compute_metrics(self, ctx: Context) -> Context | None:
        frame_values = [
            frame if isinstance(frame, Frame) else Frame(image=frame)
            for frame in list(ctx.frames or [])
        ]
        explicit_scale = ctx.px_per_mm or (ctx.scale or {}).get("px_per_mm")
        fps = (
            ctx.sequence_fps
            or (ctx.sequence_metadata.fps if ctx.sequence_metadata else None)
            or 10.0
        )

        hyst_res = analyze_needle_hysteresis_sequence(
            frame_values,
            px_per_mm=explicit_scale,
            needle_diameter_mm=ctx.needle_diameter_mm,
            fit_method=ctx.needle_fit_method or "auto",
            fps=fps,
        )

        ctx.needle_hysteresis_result = hyst_res
        ctx.results = {
            "pipeline": "needle_hysteresis",
            "schema_version": hyst_res.schema_version,
            "accepted": hyst_res.accepted,
            "rejection_reasons": hyst_res.rejection_reasons,
            **hyst_res.summary,
            "diagnostics": {
                "needle_hysteresis": {
                    **hyst_res.diagnostics,
                    "frames": [
                        {
                            "frame_index": f.frame_index,
                            "timestamp_s": f.timestamp_s,
                            "accepted": f.accepted,
                            "state": f.state,
                            "theta_left_deg": f.theta_left_deg,
                            "theta_right_deg": f.theta_right_deg,
                            "theta_mean_deg": f.theta_mean_deg,
                            "base_diameter_mm": f.base_diameter_mm,
                            "contact_velocity_mm_s": f.contact_velocity_mm_s,
                            "rejection_reasons": f.rejection_reasons,
                        }
                        for f in hyst_res.frames
                    ],
                }
            },
        }

        ctx.qa = {
            "ok": hyst_res.accepted,
            "rejection_reasons": hyst_res.rejection_reasons,
            "checks": {
                "needle_hysteresis": {
                    "code": "hysteresis_valid" if hyst_res.accepted else "hysteresis_rejected",
                    "passed": hyst_res.accepted,
                    "severity": "error" if not hyst_res.accepted else "info",
                    "reason": "; ".join(hyst_res.rejection_reasons),
                }
            },
        }

        return ctx

    def do_overlay(self, ctx: Context) -> Context | None:
        # Simple overlay for the first accepted frame
        commands: list[dict[str, Any]] = []
        if ctx.needle_hysteresis_result and ctx.needle_hysteresis_result.frames:
            first_accepted = next(
                (f for f in ctx.needle_hysteresis_result.frames if f.accepted), None
            )
            if first_accepted:
                # Text annotation showing status
                s = ctx.needle_hysteresis_result.summary
                adv = s.get("theta_advancing_deg", "N/A")
                rec = s.get("theta_receding_deg", "N/A")
                cah = s.get("contact_angle_hysteresis_deg", "N/A")
                label = f"Adv: {adv} | Rec: {rec} | Hyst: {cah}"
                commands.append(
                    {
                        "type": "text",
                        "text": label,
                        "p": (20, 30),
                        "color": "#10B981",
                        "font_scale": 0.6,
                        "thickness": 2,
                    }
                )
        ctx.overlay_commands = commands
        return ctx

    def do_validation(self, ctx: Context) -> Context | None:
        return ctx
