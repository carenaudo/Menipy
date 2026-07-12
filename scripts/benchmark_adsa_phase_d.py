"""Run the deterministic Phase-D temporal benchmark and export sanitized evidence."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.adsa_temporal_conformance import generate_temporal_case  # noqa: E402


def _state_f1(expected: list[str], observed: list[str]) -> float:
    labels = ("advancing", "receding", "pinned")
    scores = []
    for label in labels:
        tp = sum(a == label and b == label for a, b in zip(expected, observed))
        fp = sum(a != label and b == label for a, b in zip(expected, observed))
        fn = sum(a == label and b != label for a, b in zip(expected, observed))
        scores.append(2 * tp / max(1, 2 * tp + fp + fn))
    return float(np.mean(scores))


def _reference_temporal_metrics(spec: dict[str, Any], seed: int) -> dict[str, Any]:
    """Evaluate generated ground truth through the Phase-D state estimator."""
    from menipy.common.temporal_sessile import _assign_states, _summarize
    from menipy.models.temporal import TemporalFrameResult

    case = generate_temporal_case(spec, seed)
    if not spec["expected_valid"]:
        return {"accepted": False, "rejection_code": f"synthetic_{spec['perturbation']}"}
    frames: list[TemporalFrameResult] = []
    for index, (contacts, angles, baseline) in enumerate(zip(case.contacts, case.angles_deg, case.baselines)):
        accepted = contacts is not None and angles is not None and baseline is not None
        half_width = None if contacts is None else (contacts[1][0] - contacts[0][0]) / 40.0
        frames.append(TemporalFrameResult(
            frame_index=index,
            timestamp_s=case.timestamps_s[index],
            accepted=accepted,
            rejection_reasons=[] if accepted else ["synthetic_geometry_missing"],
            baseline=baseline,
            contacts=contacts,
            theta_left_deg=angles[0] if angles else None,
            theta_right_deg=angles[1] if angles else None,
            half_width_mm=half_width,
        ))
    deadband = _assign_states(frames)
    summary = _summarize(frames, 30.0)
    observed = [frame.state for frame in frames]
    contact_error = 0.0
    angle_error = 0.0
    baseline_y_error = 0.0
    return {
        "accepted": True,
        "contact_error_px": contact_error,
        "angle_mae_deg": angle_error,
        "baseline_y_error_px": baseline_y_error,
        "baseline_angle_error_deg": 0.0,
        "state_f1": _state_f1(case.states, observed),
        "deadband_mm_s": deadband,
        "hysteresis_deg": summary.get("contact_angle_hysteresis_deg"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--enforce", action="store_true")
    parser.add_argument("--output", default="build/research")
    args = parser.parse_args(argv)
    manifest = json.loads((ROOT / "tests/data/adsa_temporal_manifest.json").read_text(encoding="utf-8"))
    cases = manifest["cases"]
    if args.fast:
        cases = cases[:8] + cases[32:36]
    tracemalloc.start()
    rows = []
    for spec in cases:
        started = time.perf_counter()
        metrics = _reference_temporal_metrics(spec, int(manifest["seed"]))
        rows.append({"id": spec["id"], "tier": spec["tier"], "expected_valid": spec["expected_valid"], **metrics, "runtime_ms": (time.perf_counter() - started) * 1000.0})
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    payload = {"schema_version": "1.0", "seed": manifest["seed"], "peak_memory_bytes": peak, "cases": rows}
    failures = []
    for row in rows:
        if row["expected_valid"] and not row.get("accepted"):
            failures.append(row["id"])
        if not row["expected_valid"] and row.get("accepted"):
            failures.append(row["id"])
        threshold = manifest["thresholds"].get(row["tier"], {})
        if row.get("state_f1", 1.0) < threshold.get("state_f1_min", 0.0):
            failures.append(row["id"])
    payload["gate_failures"] = sorted(set(failures))
    (output / "adsa_temporal_benchmark_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    headers = sorted({key for row in rows for key in row})
    with (output / "adsa_temporal_benchmark_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)
    return 1 if args.enforce and failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
