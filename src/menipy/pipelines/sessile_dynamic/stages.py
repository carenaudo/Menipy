"""Phase-D dynamic sessile pipeline and sequence result promotion."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from menipy.common.sequence_acquisition import (
    VIDEO_SUFFIXES,
    frames_from_memory,
    load_image_sequence,
    load_video,
)
from menipy.common.temporal_sessile import analyze_dynamic_sessile
from menipy.models.context import Context
from menipy.models.frame import Frame
from menipy.pipelines.base import PipelineBase


class DynamicSessilePipeline(PipelineBase):
    """Track sessile geometry across a calibrated, timestamped sequence."""

    name = "sessile_dynamic"
    ui_metadata = {
        "display_name": "Dynamic Sessile",
        "icon": "sessile.svg",
        "color": "#2F80ED",
        "stages": ["acquisition", "compute_metrics", "overlay", "validation"],
        "calibration_params": ["needle_diameter_mm"],
        "primary_metrics": [
            "theta_advancing_deg",
            "theta_receding_deg",
            "contact_angle_hysteresis_deg",
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
                raise ValueError("dynamic_source_must_be_video_or_sequence_directory")
        else:
            raw_frames = ctx.frames
            if isinstance(raw_frames, np.ndarray):
                raw_frames = [raw_frames]
            frames, metadata = frames_from_memory(list(raw_frames or []), fps=ctx.sequence_fps)
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
        if ctx.sequence_metadata is None:
            raise ValueError("dynamic_sequence_metadata_missing")
        frame_values = [
            frame if isinstance(frame, Frame) else Frame(image=frame)
            for frame in list(ctx.frames or [])
        ]
        explicit_scale = ctx.px_per_mm or (ctx.scale or {}).get("px_per_mm")
        dynamic = analyze_dynamic_sessile(
            frame_values,
            ctx.sequence_metadata,
            px_per_mm=explicit_scale,
            needle_diameter_mm=ctx.needle_diameter_mm,
            contact_angle_method=ctx.contact_angle_method if ctx.contact_angle_method in {"tangent", "circle_fit", "spherical_cap", "auto_residual"} else "auto_residual",
        )
        ctx.dynamic_sessile_result = dynamic
        ctx.temporal_frame_results = dynamic.frames
        ctx.results = {
            "schema_version": dynamic.schema_version,
            **dynamic.summary,
            "series": [
                {
                    "frame_index": frame.frame_index,
                    "timestamp_s": frame.timestamp_s,
                    "accepted": frame.accepted,
                    "state": frame.state,
                    "theta_left_deg": frame.theta_left_deg,
                    "theta_right_deg": frame.theta_right_deg,
                    "half_width_mm": frame.half_width_mm,
                    "contact_velocity_mm_s": frame.contact_velocity_mm_s,
                    "rejection_reasons": frame.rejection_reasons,
                }
                for frame in dynamic.frames
            ],
            "diagnostics": {
                "dynamic_sessile": {
                    "metadata": dynamic.metadata.model_dump(mode="json"),
                    "calibration": dynamic.calibration,
                    **dynamic.diagnostics,
                    "frames": [frame.model_dump(mode="json") for frame in dynamic.frames],
                }
            },
        }
        ctx.qa = {
            "ok": dynamic.accepted,
            "rejection_reasons": dynamic.rejection_reasons,
            "checks": {
                "dynamic_sequence": {
                    "code": "dynamic_sequence_valid" if dynamic.accepted else "dynamic_sequence_rejected",
                    "passed": dynamic.accepted,
                    "severity": "error",
                    "reason": ";".join(dynamic.rejection_reasons),
                }
            },
        }
        return ctx

    def do_overlay(self, ctx: Context) -> Context | None:
        first = next((frame for frame in ctx.temporal_frame_results if frame.accepted), None)
        if first is None:
            ctx.overlay_commands = []
            return ctx
        commands: list[dict[str, Any]] = []
        if first.contour:
            commands.append({"type": "polyline", "points": first.contour, "closed": True, "color": "yellow", "thickness": 2})
        if first.baseline:
            commands.append({"type": "line", "p1": first.baseline[0], "p2": first.baseline[1], "color": "cyan", "thickness": 2})
        for contact in first.contacts or ():
            commands.append({"type": "cross", "p": contact, "color": "red", "size": 6, "thickness": 2})
        ctx.overlay_commands = commands
        return ctx

    def do_validation(self, ctx: Context) -> Context | None:
        # Sequence-level QA was computed with the temporal validity contract.
        return ctx
