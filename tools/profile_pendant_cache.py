"""Alternating same-process cached/uncached fits on deterministic inputs.

uv run --extra test python tools/profile_pendant_cache.py
The uncached path changes only the local decorator, not numerical settings.
"""

import json
import platform
import sys
import time
from pathlib import Path
from statistics import median
from unittest.mock import patch
from uuid import uuid4

from check_gui_execution import isolate


def main():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    root = (
        Path(__file__).resolve().parents[1] / ".cache" / "pendant-cache" / uuid4().hex
    )
    isolate(root)
    from menipy.pipelines.pendant import strict_young_laplace as strict
    from tests.test_pendant_fit_cache import fit_input

    report = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cases": [],
    }
    original_cache = strict.lru_cache
    original_integrator = strict.integrate_young_laplace_profile_mm
    for noise, refraction in ((0, 1), (0.2, 1.33), (10, 1)):
        inputs = fit_input(noise, refraction)
        baseline = None
        runs = []
        for pair in range(4):
            for cached in (False, True) if pair % 2 == 0 else (True, False):
                with patch.object(
                    strict,
                    "lru_cache",
                    original_cache if cached else lambda **kwargs: lambda fn: fn,
                ):
                    with patch.object(
                        strict,
                        "integrate_young_laplace_profile_mm",
                        wraps=original_integrator,
                    ) as calls:
                        start = time.perf_counter()
                        result = strict.fit_pendant_young_laplace_strict(inputs)
                        elapsed = time.perf_counter() - start
                serialized = json.dumps(result, sort_keys=True)
                baseline = serialized if baseline is None else baseline
                assert serialized == baseline, "Numerical output changed"
                runs.append(
                    {
                        "cached": cached,
                        "seconds": elapsed,
                        "integrations": calls.call_count,
                        "accepted": result["strict_fit_success"],
                        "warning": result["strict_fit_warning"],
                    }
                )
        report["cases"].append(
            {
                "noise_px": noise,
                "refraction": refraction,
                "runs": runs,
                "cached_median_s": median(r["seconds"] for r in runs if r["cached"]),
                "uncached_median_s": median(
                    r["seconds"] for r in runs if not r["cached"]
                ),
                "exact_full_output_match": True,
            }
        )
    (root / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(root / "report.json")


if __name__ == "__main__":
    main()
