"""Fresh-process bootstrap timing/allocation comparison with identical results.

uv run --extra test python tools/profile_bootstrap_batches.py
"""

import argparse
import json
import os
import subprocess
import sys
import time
import tracemalloc
from pathlib import Path
from statistics import median
from uuid import uuid4


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path)
    parser.add_argument("--mode", choices=["legacy", "batched"])
    parser.add_argument("--count", type=int)
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    if args.mode:
        from check_gui_execution import isolate

        isolate(args.root / f"{args.mode}-{args.count}-state")
        import numpy as np

        from menipy.common.temporal_sessile import _bootstrap_stats
        from tests.test_bootstrap_batches import legacy_stats

        values = np.random.default_rng(82).normal(90, 12, args.count)
        function = legacy_stats if args.mode == "legacy" else _bootstrap_stats
        runs = []
        for _ in range(3):
            tracemalloc.start()
            start = time.perf_counter()
            result = function(values)
            elapsed = time.perf_counter() - start
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            runs.append({"seconds": elapsed, "traced_peak_bytes": peak})
        record = {
            "mode": args.mode,
            "count": args.count,
            "runs": runs,
            "result": result,
            "median_seconds": median(r["seconds"] for r in runs),
            "max_traced_peak_bytes": max(r["traced_peak_bytes"] for r in runs),
        }
        (args.root / f"{args.mode}-{args.count}.json").write_text(
            json.dumps(record, indent=2)
        )
        return
    root = (
        Path(__file__).resolve().parents[1]
        / ".cache"
        / "bootstrap-profile"
        / uuid4().hex
    )
    root.mkdir(parents=True)
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    report = []
    for count in (1000, 10000):
        records = []
        for mode in ("legacy", "batched"):
            subprocess.run(
                [
                    "uv",
                    "run",
                    "--offline",
                    "--extra",
                    "test",
                    "python",
                    str(Path(__file__).resolve()),
                    "--root",
                    str(root),
                    "--mode",
                    mode,
                    "--count",
                    str(count),
                ],
                check=True,
            )
            records.append(json.loads((root / f"{mode}-{count}.json").read_text()))
        assert records[0]["result"] == records[1]["result"]
        report.extend(records)
    print(root)
    print(
        json.dumps(
            [
                {k: v for k, v in r.items() if k not in ("runs", "result")}
                for r in report
            ],
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
