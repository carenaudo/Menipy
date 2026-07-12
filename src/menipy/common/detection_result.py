"""Normalized detector outcomes with adapters for legacy plugin return values."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class DetectionResult:
    """Common result envelope used by calibration and detector consumers."""

    value: Any = None
    mask: np.ndarray | None = None
    confidence: float = 0.0
    accepted: bool = False
    rejection_reasons: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    parameters: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.confidence = float(np.clip(self.confidence, 0.0, 1.0))
        if self.accepted and self.value is None:
            self.accepted = False
        if not self.accepted and not self.rejection_reasons:
            self.rejection_reasons.append("not_detected")

    def to_diagnostics(self) -> dict[str, Any]:
        """Return JSON-safe metadata without embedding masks or contours."""
        return {
            "accepted": self.accepted,
            "confidence": self.confidence,
            "rejection_reasons": list(self.rejection_reasons),
            "metrics": dict(self.metrics),
            "parameters": dict(self.parameters),
            "has_mask": self.mask is not None,
        }


def normalize_detection_result(
    raw: Any,
    *,
    feature: str,
    confidence: float | None = None,
    mask: np.ndarray | None = None,
    metrics: dict[str, Any] | None = None,
    parameters: dict[str, Any] | None = None,
) -> DetectionResult:
    """Normalize typed, tuple, ndarray, scalar, and ``None`` plugin results."""
    if isinstance(raw, DetectionResult):
        return raw
    accepted = raw is not None
    return DetectionResult(
        value=raw,
        mask=mask,
        confidence=(1.0 if accepted else 0.0) if confidence is None else confidence,
        accepted=accepted,
        rejection_reasons=[] if accepted else [f"{feature}_not_detected"],
        metrics=metrics or {},
        parameters=parameters or {},
    )


__all__ = ["DetectionResult", "normalize_detection_result"]
