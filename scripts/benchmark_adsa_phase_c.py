"""Export non-blocking ONNX proposal accuracy and runtime evidence."""

from __future__ import annotations

import json
import statistics
import sys
import time
import tracemalloc
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

from adsa_conformance import generate_case  # noqa: E402
from menipy.common.mobilesam_provider import MobileSAMProvider  # noqa: E402
from menipy.common.segmentation_providers import SegmentationPrompt, expand_box  # noqa: E402


def _metrics(observed: np.ndarray, expected: np.ndarray) -> dict[str, float]:
    observed = np.asarray(observed, dtype=bool)
    expected = np.asarray(expected, dtype=bool)
    intersection = int(np.logical_and(observed, expected).sum())
    union = int(np.logical_or(observed, expected).sum())
    dice_denominator = int(observed.sum() + expected.sum())
    observed_edge = cv2.Canny(observed.astype(np.uint8) * 255, 50, 100) > 0
    expected_edge = cv2.Canny(expected.astype(np.uint8) * 255, 50, 100) > 0
    obs_yx = np.column_stack(np.where(observed_edge))
    exp_yx = np.column_stack(np.where(expected_edge))
    hausdorff = float("inf")
    boundary_f = 0.0
    if obs_yx.size and exp_yx.size:
        obs_to_exp = cKDTree(exp_yx).query(obs_yx)[0]
        exp_to_obs = cKDTree(obs_yx).query(exp_yx)[0]
        hausdorff = float(max(np.max(obs_to_exp), np.max(exp_to_obs)))
        precision = float(np.mean(obs_to_exp <= 2.0))
        recall = float(np.mean(exp_to_obs <= 2.0))
        boundary_f = 2.0 * precision * recall / max(precision + recall, 1e-12)
    return {
        "iou": float(intersection / union) if union else 0.0,
        "dice": float(2 * intersection / dice_denominator) if dice_denominator else 0.0,
        "boundary_fscore_2px": boundary_f,
        "hausdorff_px": hausdorff,
    }


def main() -> int:
    manifest = json.loads(
        (ROOT / "tests" / "data" / "adsa_detector_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    selected = [
        spec
        for spec in manifest["cases"]
        if spec["pipeline"] == "sessile"
        and spec["perturbation"] in {"clean", "blur"}
    ]
    provider = MobileSAMProvider()
    rows = []
    for spec in selected:
        case = generate_case(spec, manifest["seed"])
        target_contours, _ = cv2.findContours(
            case.target_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not target_contours:
            continue
        x, y, width, height = cv2.boundingRect(max(target_contours, key=cv2.contourArea))
        prompts = [
            SegmentationPrompt(
                "droplet",
                expand_box(
                    (x, y, x + width - 1, y + height - 1), case.image.shape
                ),
            )
        ]
        tracemalloc.start()
        started = time.perf_counter()
        proposals = provider.segment(case.image, prompts)
        runtime_ms = (time.perf_counter() - started) * 1000.0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        proposal = proposals[0]
        rows.append(
            {
                "id": spec["id"],
                "accepted": proposal.accepted,
                "rejection_reasons": proposal.rejection_reasons,
                "metrics": (
                    _metrics(proposal.mask, case.target_mask > 0)
                    if proposal.mask is not None
                    else {}
                ),
                "runtime_ms": runtime_ms,
                "peak_python_bytes": peak,
                "provider": proposal.to_diagnostics(),
            }
        )
    runtimes = [row["runtime_ms"] for row in rows]
    report = {
        "schema_version": "1.0",
        "seed": manifest["seed"],
        "evidence_semantics": "synthetic proposal benchmark; not independent scientific validation",
        "cases": rows,
        "performance_non_blocking": {
            "median_runtime_ms": statistics.median(runtimes) if runtimes else None,
            "p95_runtime_ms": (
                sorted(runtimes)[int(0.95 * (len(runtimes) - 1))]
                if runtimes
                else None
            ),
            "max_peak_python_bytes": (
                max(row["peak_python_bytes"] for row in rows) if rows else None
            ),
        },
    }
    output = ROOT / "build" / "research" / "adsa_phase_c_onnx.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        output.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    except PermissionError:
        output = ROOT / ".tmp" / "adsa_phase_c_onnx.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
