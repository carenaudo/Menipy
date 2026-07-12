"""Contracts and validation for non-authoritative segmentation proposals."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np


class ModelManifestError(RuntimeError):
    """Raised when an ONNX model manifest or graph fails integrity checks."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class ModelManifest:
    """Versioned provenance, runtime, and integrity contract for an ONNX model."""

    identifier: str
    revision: str
    source_urls: tuple[str, ...]
    code_license: str
    weights_license: str
    distribution: str
    opset: int
    preprocessing_revision: str
    supported_domain: str
    classes: tuple[str, ...]
    files: dict[str, dict[str, Any]]
    raw: dict[str, Any] = field(repr=False)

    @classmethod
    def load(cls, path: str | Path) -> ModelManifest:
        manifest_path = Path(path)
        try:
            raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ModelManifestError("model_manifest_unreadable", str(exc)) from exc
        required = {
            "schema_version",
            "id",
            "revision",
            "source_urls",
            "licenses",
            "distribution",
            "opset",
            "preprocessing_revision",
            "supported_domain",
            "classes",
            "files",
        }
        missing = sorted(required - raw.keys())
        if missing:
            raise ModelManifestError(
                "model_manifest_missing_fields", f"Missing manifest fields: {missing}"
            )
        if str(raw["schema_version"]) != "1.0":
            raise ModelManifestError(
                "model_manifest_schema_unsupported",
                f"Unsupported model manifest schema: {raw['schema_version']}",
            )
        licenses = raw.get("licenses") or {}
        if not licenses.get("code") or not licenses.get("weights"):
            raise ModelManifestError(
                "model_manifest_license_unresolved",
                "Both code and weights license fields are required",
            )
        return cls(
            identifier=str(raw["id"]),
            revision=str(raw["revision"]),
            source_urls=tuple(str(item) for item in raw["source_urls"]),
            code_license=str(licenses["code"]),
            weights_license=str(licenses["weights"]),
            distribution=str(raw["distribution"]),
            opset=int(raw["opset"]),
            preprocessing_revision=str(raw["preprocessing_revision"]),
            supported_domain=str(raw["supported_domain"]),
            classes=tuple(str(item) for item in raw["classes"]),
            files=dict(raw["files"]),
            raw=raw,
        )

    def validate_files(self, directory: str | Path) -> None:
        """Fail closed if any graph is absent, resized, or hash-mismatched."""
        root = Path(directory)
        for name, metadata in self.files.items():
            path = (root / name).resolve()
            if not path.is_relative_to(root.resolve()):
                raise ModelManifestError(
                    "model_file_path_invalid", f"Model path escapes directory: {name}"
                )
            if not path.is_file():
                raise ModelManifestError(
                    "model_file_missing", f"Missing ONNX model file: {path}"
                )
            expected_size = int(metadata.get("bytes", -1))
            if path.stat().st_size != expected_size:
                raise ModelManifestError(
                    "model_file_size_mismatch", f"Unexpected size for {name}"
                )
            hasher = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    hasher.update(chunk)
            digest = hasher.hexdigest()
            if digest.lower() != str(metadata.get("sha256", "")).lower():
                raise ModelManifestError(
                    "model_file_hash_mismatch", f"SHA-256 mismatch for {name}"
                )


@dataclass(frozen=True)
class SegmentationPrompt:
    """Geometric prompt in original image coordinates."""

    feature: str
    box_xyxy: tuple[float, float, float, float]
    positive_points: tuple[tuple[float, float], ...] = ()
    negative_points: tuple[tuple[float, float], ...] = ()

    def clipped(self, image_shape: tuple[int, ...]) -> SegmentationPrompt:
        height, width = image_shape[:2]
        x1, y1, x2, y2 = self.box_xyxy
        clipped = (
            float(np.clip(x1, 0, max(0, width - 1))),
            float(np.clip(y1, 0, max(0, height - 1))),
            float(np.clip(x2, 0, max(0, width - 1))),
            float(np.clip(y2, 0, max(0, height - 1))),
        )
        return SegmentationPrompt(
            feature=self.feature,
            box_xyxy=clipped,
            positive_points=self.positive_points,
            negative_points=self.negative_points,
        )

    def validate(self, image_shape: tuple[int, ...]) -> None:
        height, width = image_shape[:2]
        x1, y1, x2, y2 = self.box_xyxy
        if not (0 <= x1 < x2 < width and 0 <= y1 < y2 < height):
            raise ValueError("segmentation_prompt_out_of_bounds")
        for point in (*self.positive_points, *self.negative_points):
            if not (0 <= point[0] < width and 0 <= point[1] < height):
                raise ValueError("segmentation_prompt_point_out_of_bounds")


@dataclass
class SegmentationProposal:
    """A model proposal that is never a promoted scientific measurement."""

    feature: str
    provider: str
    mask: np.ndarray | None
    contour: np.ndarray | None
    score: float
    accepted: bool
    rejection_reasons: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_diagnostics(self) -> dict[str, Any]:
        return {
            "feature": self.feature,
            "provider": self.provider,
            "score": float(self.score),
            "accepted": bool(self.accepted),
            "rejection_reasons": list(self.rejection_reasons),
            "metrics": dict(self.metrics),
            "provenance": dict(self.provenance),
            "has_mask": self.mask is not None,
            "has_contour": self.contour is not None,
        }


def expand_box(
    box_xyxy: tuple[float, float, float, float],
    image_shape: tuple[int, ...],
    fraction: float = 0.08,
) -> tuple[float, float, float, float]:
    """Expand an XYXY prompt and clip it strictly inside the image."""
    x1, y1, x2, y2 = map(float, box_xyxy)
    dx, dy = (x2 - x1) * fraction, (y2 - y1) * fraction
    height, width = image_shape[:2]
    return (
        max(0.0, x1 - dx),
        max(0.0, y1 - dy),
        min(float(width - 1), x2 + dx),
        min(float(height - 1), y2 + dy),
    )


def mask_to_proposal(
    mask: np.ndarray,
    *,
    score: float,
    prompt: SegmentationPrompt,
    provider: str,
    min_score: float = 0.5,
) -> SegmentationProposal:
    """Convert and topology-check a binary segmentation mask."""
    binary = np.asarray(mask).squeeze() > 0
    reasons: list[str] = []
    metrics: dict[str, Any] = {}
    if binary.ndim != 2 or binary.size == 0:
        return SegmentationProposal(
            prompt.feature,
            provider,
            None,
            None,
            score,
            False,
            ["proposal_mask_invalid_shape"],
        )
    if score < min_score:
        reasons.append("proposal_score_below_threshold")
    area = int(binary.sum())
    area_fraction = area / float(binary.size)
    metrics["area_px"] = area
    metrics["area_fraction"] = area_fraction
    if area == 0:
        reasons.append("proposal_mask_empty")
    if not 0.001 <= area_fraction <= 0.8:
        reasons.append("proposal_area_out_of_range")

    uint8 = binary.astype(np.uint8)
    count, _labels, stats, _ = cv2.connectedComponentsWithStats(
        uint8, connectivity=8
    )
    component_areas = stats[1:, cv2.CC_STAT_AREA] if count > 1 else np.empty(0)
    dominant_fraction = (
        float(np.max(component_areas)) / max(float(np.sum(component_areas)), 1.0)
        if component_areas.size
        else 0.0
    )
    metrics["component_count"] = int(max(count - 1, 0))
    metrics["dominant_component_fraction"] = dominant_fraction
    if dominant_fraction < 0.9:
        reasons.append("proposal_multiple_components")

    contours, hierarchy = cv2.findContours(
        uint8, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE
    )
    holes = 0
    if hierarchy is not None:
        holes = int(np.sum(hierarchy[0, :, 3] >= 0))
    metrics["holes"] = holes
    if holes:
        reasons.append("proposal_holes_detected")
    if not contours:
        reasons.append("proposal_contour_missing")
        contour = None
    else:
        external = [
            contour
            for index, contour in enumerate(contours)
            if hierarchy is None or hierarchy[0, index, 3] < 0
        ]
        selected = max(external or contours, key=cv2.contourArea)
        contour = selected.reshape(-1, 2).astype(np.float64)
        metrics["contour_points"] = int(contour.shape[0])
        if contour.shape[0] < 20:
            reasons.append("proposal_contour_too_short")

    border_contacts = int(binary[0].any()) + int(binary[-1].any())
    border_contacts += int(binary[:, 0].any()) + int(binary[:, -1].any())
    metrics["border_contacts"] = border_contacts
    if border_contacts > 2:
        reasons.append("proposal_excessive_border_contact")

    x1, y1, x2, y2 = (int(round(v)) for v in prompt.box_xyxy)
    prompt_area = binary[max(0, y1) : y2 + 1, max(0, x1) : x2 + 1]
    prompt_coverage = float(prompt_area.sum()) / max(float(area), 1.0)
    metrics["prompt_coverage"] = prompt_coverage
    if prompt_coverage < 0.8:
        reasons.append("proposal_prompt_coverage_low")

    return SegmentationProposal(
        feature=prompt.feature,
        provider=provider,
        mask=binary,
        contour=contour,
        score=float(score),
        accepted=not reasons,
        rejection_reasons=reasons,
        metrics=metrics,
    )


__all__ = [
    "ModelManifest",
    "ModelManifestError",
    "SegmentationPrompt",
    "SegmentationProposal",
    "expand_box",
    "mask_to_proposal",
]
