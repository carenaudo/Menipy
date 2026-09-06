"""Surface free energy pipeline stages.

This is a *non-image* pipeline.  All image-processing stages are no-ops.
The heavy lifting happens in ``do_compute_metrics`` which delegates to
:mod:`menipy.math.surface_energy` for OWRK and Wu calculations.
"""

from __future__ import annotations

from menipy.models.context import Context
from menipy.pipelines.base import PipelineBase


class SurfaceEnergyPipeline(PipelineBase):
    """Compute solid surface free energy from contact angle measurements.

    This pipeline does not process images.  It expects ``ctx.sfe_liquids``
    (serialised :class:`~menipy.common.liquid_db.ProbeLiquid` dicts) and
    ``ctx.sfe_contact_angles_deg`` to be populated before execution.
    """

    name = "surface_energy"
    ui_metadata = {
        "display_name": "Surface Free Energy",
        "icon": "surface_energy.svg",
        "color": "#8B5CF6",
        "stages": ["compute_metrics", "validation"],
        "calibration_params": [],
        "primary_metrics": [
            "gamma_s_dispersive_mN_m",
            "gamma_s_polar_mN_m",
            "gamma_s_total_mN_m",
        ],
    }

    # All image stages are no-ops for this purely numerical pipeline.
    def do_acquisition(self, ctx: Context) -> Context | None:
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
        """Run OWRK and/or Wu surface energy calculation."""
        from menipy.common.liquid_db import ProbeLiquid
        from menipy.math.surface_energy import compute_surface_energy

        if not ctx.sfe_liquids or not ctx.sfe_contact_angles_deg:
            raise ValueError(
                "surface_energy pipeline requires sfe_liquids and "
                "sfe_contact_angles_deg to be populated"
            )

        # Reconstruct ProbeLiquid objects from serialised dicts
        liquids = [ProbeLiquid(**d) for d in ctx.sfe_liquids]
        angles = ctx.sfe_contact_angles_deg

        sfe_result = compute_surface_energy(
            liquids,
            angles,
            method=ctx.sfe_method,
            substrate_name=ctx.sfe_substrate_name,
        )

        ctx.results = {
            "pipeline": "surface_energy",
            "schema_version": "1.0",
            **sfe_result.to_dict(),
        }

        return ctx

    def do_overlay(self, ctx: Context) -> Context | None:
        return ctx

    def do_validation(self, ctx: Context) -> Context | None:
        """Validate SFE results and populate QA."""
        warnings = ctx.results.get("warnings", [])
        owrk = ctx.results.get("owrk", {})
        owrk_warnings = owrk.get("warnings", []) if isinstance(owrk, dict) else []
        wu = ctx.results.get("wu", {})
        wu_warnings = wu.get("warnings", []) if isinstance(wu, dict) else []

        all_warnings = warnings + owrk_warnings + wu_warnings
        has_errors = any(
            "non-physical" in w.lower() or "failed" in w.lower()
            for w in all_warnings
        )

        ctx.qa = {
            "ok": not has_errors,
            "rejection_reasons": [w for w in all_warnings if "non-physical" in w.lower() or "failed" in w.lower()],
            "checks": {
                "sfe_analysis": {
                    "code": "sfe_valid" if not has_errors else "sfe_warnings",
                    "passed": not has_errors,
                    "severity": "error" if has_errors else "warning",
                    "reason": "; ".join(all_warnings) if all_warnings else "",
                }
            },
        }

        return ctx
