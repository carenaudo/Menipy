"""Shadow-only orchestration for ONNX segmentation proposals."""

from __future__ import annotations

from importlib import import_module
from typing import Any

import cv2
import numpy as np
from scipy.spatial import cKDTree

from menipy.common.registry import SEGMENTATION_PROVIDERS
from menipy.common.segmentation_providers import (
    ModelManifestError,
    SegmentationPrompt,
    expand_box,
)


def _image_from_context(ctx: Any) -> np.ndarray | None:
    image = getattr(ctx, "image", None)
    if isinstance(image, np.ndarray):
        return image
    frames = getattr(ctx, "frames", None)
    if frames is not None:
        try:
            frame = frames[0]
            candidate = frame.image if hasattr(frame, "image") else frame
            if isinstance(candidate, np.ndarray):
                return candidate
        except (IndexError, TypeError):
            pass
    return None


def _contour_xy(ctx: Any) -> np.ndarray | None:
    for name in ("drop_contour", "detected_contour"):
        value = getattr(ctx, name, None)
        if value is not None:
            return np.asarray(value, dtype=float).reshape(-1, 2)
    contour = getattr(ctx, "contour", None)
    if contour is not None and getattr(contour, "xy", None) is not None:
        return np.asarray(contour.xy, dtype=float).reshape(-1, 2)
    return None


def _comparison_metrics(
    proposal_mask: np.ndarray | None,
    proposal_contour: np.ndarray | None,
    reference_contour: np.ndarray | None,
    image_shape: tuple[int, ...],
) -> dict[str, float]:
    if proposal_mask is None or reference_contour is None or reference_contour.size == 0:
        return {}
    reference_mask = np.zeros(image_shape[:2], dtype=np.uint8)
    cv2.drawContours(
        reference_mask,
        [np.rint(reference_contour).astype(np.int32).reshape(-1, 1, 2)],
        -1,
        1,
        -1,
    )
    observed = np.asarray(proposal_mask, dtype=bool)
    expected = reference_mask > 0
    union = int(np.logical_or(observed, expected).sum())
    intersection = int(np.logical_and(observed, expected).sum())
    metrics: dict[str, float] = {
        "classical_iou": float(intersection / union) if union else 0.0
    }
    if proposal_contour is not None and proposal_contour.shape[0] >= 2:
        distances_a = cKDTree(reference_contour).query(proposal_contour)[0]
        distances_b = cKDTree(proposal_contour).query(reference_contour)[0]
        metrics["classical_hausdorff_px"] = float(
            max(np.max(distances_a), np.max(distances_b))
        )
    return metrics


def run_shadow_segmentation(ctx: Any, pipeline: str) -> Any:
    """Attach proposal diagnostics while leaving authoritative fields untouched."""
    if getattr(ctx, "onnx_proposal_mode", "off") != "shadow":
        return ctx
    image = _image_from_context(ctx)
    diagnostics: dict[str, Any] = {
        "mode": "shadow",
        "provider": str(getattr(ctx, "segmentation_provider", "mobilesam")),
        "pipeline": str(pipeline),
        "substrate": {
            "requested": False,
            "reason": "line_geometry_required_use_classical_detector",
        },
        "proposals": {},
    }
    if image is None:
        diagnostics.update(
            {"accepted": False, "rejection_reasons": ["proposal_image_missing"]}
        )
        ctx.onnx_proposals = diagnostics
        return ctx

    provider_name = diagnostics["provider"]
    if provider_name == "mobilesam" and provider_name not in SEGMENTATION_PROVIDERS:
        import_module("menipy.common.mobilesam_provider")

    provider_factory = SEGMENTATION_PROVIDERS.get(provider_name)
    if provider_factory is None:
        diagnostics.update(
            {
                "accepted": False,
                "rejection_reasons": ["segmentation_provider_not_registered"],
            }
        )
        ctx.onnx_proposals = diagnostics
        return ctx

    classes = set(getattr(ctx, "onnx_proposal_classes", ["droplet", "needle"]))
    prompts: list[SegmentationPrompt] = []
    reference_contour = _contour_xy(ctx)
    if "droplet" in classes and reference_contour is not None and reference_contour.size:
        x, y, width, height = cv2.boundingRect(
            np.rint(reference_contour).astype(np.int32).reshape(-1, 1, 2)
        )
        prompts.append(
            SegmentationPrompt(
                "droplet",
                expand_box((x, y, x + width - 1, y + height - 1), image.shape),
            )
        )
    needle_rect = getattr(ctx, "needle_rect", None)
    if "needle" in classes and needle_rect is not None:
        x, y, width, height = needle_rect
        prompts.append(
            SegmentationPrompt(
                "needle",
                expand_box((x, y, x + width - 1, y + height - 1), image.shape),
            )
        )
    if not prompts:
        diagnostics.update(
            {"accepted": False, "rejection_reasons": ["proposal_prompts_unavailable"]}
        )
        ctx.onnx_proposals = diagnostics
        return ctx

    try:
        provider = provider_factory()
        proposals = provider.segment(image, prompts)
        for proposal in proposals:
            payload = proposal.to_diagnostics()
            if proposal.feature == "droplet":
                payload["comparison"] = _comparison_metrics(
                    proposal.mask, proposal.contour, reference_contour, image.shape
                )
            diagnostics["proposals"][proposal.feature] = payload
        diagnostics["accepted"] = bool(proposals) and all(
            proposal.accepted for proposal in proposals
        )
        diagnostics["rejection_reasons"] = [
            reason
            for proposal in proposals
            for reason in proposal.rejection_reasons
        ]
    except ModelManifestError as exc:
        diagnostics.update(
            {"accepted": False, "rejection_reasons": [exc.code], "error": str(exc)}
        )
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        diagnostics.update(
            {
                "accepted": False,
                "rejection_reasons": ["segmentation_provider_unavailable"],
                "error": str(exc),
            }
        )
    ctx.onnx_proposals = diagnostics
    return ctx


__all__ = ["run_shadow_segmentation"]
