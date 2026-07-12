"""MobileSAM implementation of the non-authoritative segmentation boundary."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from menipy.common.mobilesam_onnx import DEFAULT_MODEL_DIR, MobileSAMOnnx
from menipy.common.registry import register_segmentation_provider
from menipy.common.segmentation_providers import (
    SegmentationPrompt,
    SegmentationProposal,
    mask_to_proposal,
)


class MobileSAMProvider:
    """Lazy ONNX-only proposal provider with one encoder pass per image."""

    name = "mobilesam"

    def __init__(
        self,
        model_dir: str | Path = DEFAULT_MODEL_DIR,
        *,
        providers: list[str] | None = None,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.providers = providers
        self._runtime: MobileSAMOnnx | None = None

    @property
    def runtime(self) -> MobileSAMOnnx:
        if self._runtime is None:
            self._runtime = MobileSAMOnnx(
                self.model_dir, providers=self.providers
            )
        return self._runtime

    def segment(
        self,
        image_bgr: np.ndarray,
        prompts: list[SegmentationPrompt],
    ) -> list[SegmentationProposal]:
        """Return topology-checked proposals without promoting any contour."""
        if not prompts:
            return []
        for prompt in prompts:
            prompt.validate(image_bgr.shape)
        started = time.perf_counter()
        embeddings = self.runtime.encode(image_bgr)
        encoder_ms = (time.perf_counter() - started) * 1000.0
        proposals: list[SegmentationProposal] = []
        for prompt in prompts:
            started = time.perf_counter()
            result = self.runtime.predict_prompt(
                image_bgr,
                prompt.box_xyxy,
                positive_points=prompt.positive_points,
                negative_points=prompt.negative_points,
                embeddings=embeddings,
            )
            decoder_ms = (time.perf_counter() - started) * 1000.0
            scores = result.scores.reshape(-1)
            index = int(np.argmax(scores))
            masks = result.masks.reshape(-1, *result.masks.shape[-2:])
            proposal = mask_to_proposal(
                masks[index],
                score=float(scores[index]),
                prompt=prompt,
                provider=self.name,
            )
            proposal.provenance.update(
                {
                    "model_id": self.runtime.manifest.identifier,
                    "model_revision": self.runtime.manifest.revision,
                    "preprocessing_revision": self.runtime.manifest.preprocessing_revision,
                    "encoder_ms": encoder_ms,
                    "decoder_ms": decoder_ms,
                    "prompt_box_xyxy": list(prompt.box_xyxy),
                }
            )
            proposals.append(proposal)
        return proposals


register_segmentation_provider("mobilesam", MobileSAMProvider)


__all__ = ["MobileSAMProvider"]
