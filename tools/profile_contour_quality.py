"""Report sessile contour quality and metrics, old selection versus new.

Runs the sessile drop detector over the bundled samples twice -- once with the
segmentation quality gate disabled, reproducing the area-only candidate
selection, and once with it active -- and prints solidity, fill ratio and the
derived measurements side by side.

This tool measures; it changes no application behavior.

Usage
-----
``uv run python tools/profile_contour_quality.py [--json report.json]``
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

# Pendant samples have no substrate; this report is about sessile segmentation.
SESSILE_SAMPLES = (
    "gota depositada 1.png",
    "prueba sesil 2.png",
    "sessile_3.jpeg",
    "sessile_clean_reference.png",
    "sessile_needle_reference.png",
)


def _load_plugins() -> None:
    """Put the bundled plugins on the path and register the detectors."""
    plugins_dir = ROOT / "plugins"
    if plugins_dir.exists() and str(plugins_dir) not in sys.path:
        sys.path.insert(0, str(plugins_dir))
    for name in ("edge_detectors", "detect_substrate", "detect_needle", "detect_drop"):
        try:
            __import__(name)
        except ImportError:  # optional plugin; the report degrades gracefully
            pass


def _substrate_y(image: np.ndarray) -> tuple[int, float]:
    """Return the detected baseline row and its confidence.

    Raises rather than guessing: a fabricated baseline yields plausible-looking
    numbers that are simply wrong, which is worse than no row in the report.
    """
    from menipy.common.sessile_detection import detect_sessile_substrate_line

    line, confidence = detect_sessile_substrate_line(image)
    return int(round((line[0][1] + line[1][1]) / 2.0)), float(confidence)


def _legacy_contour(image: np.ndarray, substrate_y: int) -> np.ndarray | None:
    """Reconstruct the pre-fix candidate selection.

    The original code pooled contours from the adaptive mask and from an Otsu
    fallback masked four rows *above* the baseline, then took the largest
    external area with no quality criterion. Reproduced here rather than kept
    behind a flag in the library, so production carries only one code path.
    """
    from menipy.common.sessile_detection import (
        _segment_sessile_otsu_fallback,
        segment_sessile_binary,
    )

    height, width = image.shape[:2]
    min_area = float(height * width) * 0.005
    pool: list[np.ndarray] = []
    masks = (
        segment_sessile_binary(image, substrate_y=substrate_y),
        _segment_sessile_otsu_fallback(
            image, substrate_y=substrate_y, needle_shaft_result=None, contact_band_px=-4
        ),
    )
    for mask in masks:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for cnt in contours:
            x, y, w, _h = cv2.boundingRect(cnt)
            if cv2.contourArea(cnt) < min_area or y < 5 or x <= 5 or (x + w) >= width - 5:
                continue
            pool.append(cnt)
    if not pool:
        return None
    return max(pool, key=cv2.contourArea).reshape(-1, 2).astype(float)


def _measure(image: np.ndarray, substrate_y: int, *, legacy: bool) -> dict[str, Any]:
    from menipy.common.sessile_detection import (
        contour_fill_ratio,
        contour_solidity,
        detect_sessile_drop_contour,
        segment_sessile_binary,
    )

    if legacy:
        contour = _legacy_contour(image, substrate_y)
        if contour is None:
            return {"detected": False}
        as_cv = contour.reshape(-1, 1, 2).astype(np.int32)
        detection = SimpleNamespace(
            contour=contour,
            solidity=contour_solidity(as_cv),
            fill_ratio=contour_fill_ratio(
                as_cv, segment_sessile_binary(image, substrate_y=substrate_y)
            ),
            contact_points=None,
        )
    else:
        detection = detect_sessile_drop_contour(image, substrate_y=substrate_y)
    if detection.contour is None:
        return {"detected": False}

    contour = np.asarray(detection.contour, dtype=float)
    x, y, w, h = cv2.boundingRect(contour.reshape(-1, 1, 2).astype(np.int32))
    height_px = float(substrate_y - contour[:, 1].min())

    contacts = detection.contact_points
    width_px = float(contacts[1][0] - contacts[0][0]) if contacts else float(w)
    # Spherical-cap contact angle: an independent geometric reference that does
    # not depend on the pipeline's tangent fit.
    theta_cap = (
        2.0 * math.degrees(math.atan2(2.0 * height_px, width_px))
        if width_px > 0
        else float("nan")
    )

    return {
        "detected": True,
        "solidity": round(float(detection.solidity or 0.0), 4),
        "fill_ratio": round(float(detection.fill_ratio or 0.0), 4),
        "bbox": [int(x), int(y), int(w), int(h)],
        "width_px": round(width_px, 1),
        "height_px": round(height_px, 1),
        "theta_cap_deg": round(theta_cap, 2),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, help="also write the report as JSON")
    args = parser.parse_args()

    sys.path.insert(0, str(ROOT))
    _load_plugins()

    samples = [ROOT / "data" / "samples" / name for name in SESSILE_SAMPLES]

    report: dict[str, Any] = {}
    header = (
        f"{'sample':<32}{'mode':<8}{'base_y':>7}{'conf':>6}"
        f"{'solidity':>9}{'fill':>7}{'w_px':>7}{'h_px':>7}{'theta':>8}"
    )
    print(header)
    print("-" * len(header))

    for sample in samples:
        image = cv2.imread(str(sample))
        if image is None:
            print(f"{sample.name:<32}missing")
            continue
        try:
            substrate_y, confidence = _substrate_y(image)
        except Exception as exc:  # noqa: BLE001 - reported, not swallowed
            print(f"{sample.name:<32}substrate detection failed: {exc}")
            report[sample.name] = {"error": str(exc)}
            continue
        entry = {
            "substrate_y": substrate_y,
            "substrate_confidence": round(confidence, 3),
            "before": _measure(image, substrate_y, legacy=True),
            "after": _measure(image, substrate_y, legacy=False),
        }
        report[sample.name] = entry

        for mode in ("before", "after"):
            row = entry[mode]
            flag = "!" if confidence < 0.75 else " "
            prefix = (
                f"{sample.name:<32}{mode:<8}{substrate_y:>7}{confidence:>5.2f}{flag}"
            )
            if not row["detected"]:
                print(f"{prefix}{'no contour':>31}")
                continue
            print(
                f"{prefix}{row['solidity']:>9.4f}{row['fill_ratio']:>7.3f}"
                f"{row['width_px']:>7.1f}{row['height_px']:>7.1f}"
                f"{row['theta_cap_deg']:>8.2f}"
            )
        print()

    print(
        "theta is the independent spherical-cap estimate 2*atan(2h/w); it does not\n"
        "use the pipeline's tangent fit, so it is a check on the contour, not on\n"
        "the solver. Physical units are omitted: without a resolved px/mm scale\n"
        "the pipeline's mm and uL fields carry pixel magnitudes.\n"
        "'!' marks a baseline detected below the 0.75 confidence threshold -- every\n"
        "number on that row is conditioned on a baseline the detector is unsure of."
    )

    if args.json:
        args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
