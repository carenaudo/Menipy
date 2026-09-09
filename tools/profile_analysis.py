"""Reproducible cold-import, warm fitting and disk-frame memory evidence.

Run: uv run --extra test python tools/profile_analysis.py --output .cache/profile
Each case runs in a fresh uv process with isolated settings/databases. Profiles
are evidence, not benchmarks for physical accuracy (scale is synthetic).
"""

import argparse
import cProfile
import hashlib
import json
import os
import platform
import pstats
import subprocess
import time
from pathlib import Path


def memory():
    """Windows native working set; no extra profiling dependency."""
    if os.name != "nt":
        return {"rss": None, "peak": None}
    import ctypes
    from ctypes import wintypes

    class Counters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
            (name, ctypes.c_size_t)
            for name in (
                "peak",
                "rss",
                "quota_peak",
                "quota",
                "nonpaged_peak",
                "nonpaged",
                "pagefile",
                "pagefile_peak",
            )
        ]

    kernel = ctypes.WinDLL("kernel32")
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    psapi = ctypes.WinDLL("psapi")
    psapi.GetProcessMemoryInfo.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(Counters),
        wintypes.DWORD,
    ]
    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    if not psapi.GetProcessMemoryInfo(
        kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb
    ):
        raise ctypes.WinError()
    return {"rss": counters.rss, "peak": counters.peak}


def worker(args):
    started = time.perf_counter()
    from check_gui_execution import isolate

    isolate(args.output / args.case / "state")
    from menipy.pipelines.discover import PIPELINE_MAP

    import_seconds = time.perf_counter() - started
    from importlib.metadata import version

    import cv2
    import numpy as np

    from menipy.common.runtime_provenance import runtime_provenance

    record = {
        "case": args.case,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu": platform.processor(),
        "import_and_isolation_s": import_seconds,
    }
    record["versions"] = {
        name: version(name) for name in ("numpy", "scipy", "PySide6", "pydantic")
    }
    record["runtime"] = runtime_provenance()
    destination = args.output / args.case
    if args.case.startswith("memory"):
        from menipy.common.sequence_acquisition import load_video

        count = int(args.case.removeprefix("memory"))
        source = destination / "clip.avi"
        writer = cv2.VideoWriter(
            str(source), cv2.VideoWriter_fourcc(*"MJPG"), 20, (640, 480)
        )
        if not writer.isOpened():
            raise RuntimeError("MJPG writer unavailable")
        try:
            for index in range(count):
                writer.write(np.full((480, 640, 3), index % 256, np.uint8))
        finally:
            writer.release()
        before = memory()["rss"]
        begin = time.perf_counter()
        store, metadata = load_video(source, disk_backed=True)
        try:
            record.update(
                frame_count=len(store),
                dimensions=[640, 480],
                decoded_pixels_retained=0,
                disk_bytes=store._file.seek(0, 2),
                acquisition_s=time.perf_counter() - begin,
                rss_before=before,
                rss_after=memory()["rss"],
                timestamps_match=bool(
                    np.allclose(metadata.timestamps_s, [i / 20 for i in range(count)])
                ),
            )
        finally:
            store.close()
        record["store_closed"] = store.closed
    elif args.case == "startup":
        from PySide6.QtWidgets import QApplication

        from menipy.gui.views.main_window import MainWindow

        app = QApplication.instance() or QApplication([])
        window = MainWindow()
        window.show()
        app.processEvents()
        record["startup_to_event_loop_s"] = time.perf_counter() - started
        window.close()
        app.processEvents()
    else:
        from menipy.common.auto_calibrator import run_auto_calibration
        from menipy.models.results import build_persisted_analysis

        filename = (
            "sessile_needle_reference.png"
            if args.case == "sessile"
            else "pendant_water_reference.png"
        )
        source = Path(__file__).resolve().parents[1] / "data" / "samples" / filename
        image = cv2.imread(str(source))
        calibration = run_auto_calibration(image, args.case)
        parameters = {
            "image": str(source),
            "scale": {"px_per_mm": 100.0},
            "needle_diameter_mm": 1.0,
            "roi": calibration.roi_rect,
            "needle_rect": calibration.needle_rect,
            "substrate_line": calibration.substrate_line,
            "physics": {"rho1": 1000.0, "rho2": 1.2, "g": 9.80665},
        }
        record["parameters"] = parameters
        record.update(
            input=str(source),
            sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            dimensions=list(image.shape),
            scale_note="Synthetic 100 px/mm; timing only",
        )
        runs = []
        baseline = None
        for index in range(args.repeats + 1):
            begin = time.perf_counter()
            if index == 0 and args.profile_first:
                first_profiler = cProfile.Profile()
                ctx = first_profiler.runcall(
                    lambda: PIPELINE_MAP[args.case]().run(**parameters)
                )
                first_profiler.dump_stats(str(destination / "first-use.prof"))
                with (destination / "first-use.txt").open(
                    "w", encoding="utf-8"
                ) as stream:
                    pstats.Stats(first_profiler, stream=stream).sort_stats(
                        "cumulative"
                    ).print_stats(45)
            else:
                ctx = PIPELINE_MAP[args.case]().run(**parameters)
            elapsed = time.perf_counter() - begin
            payload = build_persisted_analysis(ctx)
            numeric = {
                k: v
                for k, v in payload["results"].items()
                if isinstance(v, (int, float))
            }
            fingerprint = {
                "accepted": payload["accepted"],
                "rejection_reasons": payload["rejection_reasons"],
                "numeric_results": numeric,
            }
            if baseline is None:
                baseline = fingerprint
            runs.append(
                {
                    "kind": (
                        "first_use_profiled" if args.profile_first else "first_use"
                    )
                    if index == 0
                    else "warm",
                    "seconds": elapsed,
                    "timings_ms": ctx.timings_ms,
                    "same_output": fingerprint == baseline,
                    "output": fingerprint,
                }
            )
        profiler = cProfile.Profile()
        profiler.runcall(lambda: PIPELINE_MAP[args.case]().run(**parameters))
        profiler.dump_stats(str(destination / "fitting.prof"))
        with (destination / "fitting.txt").open("w", encoding="utf-8") as stream:
            pstats.Stats(profiler, stream=stream).sort_stats("cumulative").print_stats(
                45
            )
        record["runs"] = runs
        record["warm_seconds"] = {
            "median": float(np.median([r["seconds"] for r in runs[1:]])),
            "min": min(r["seconds"] for r in runs[1:]),
            "max": max(r["seconds"] for r in runs[1:]),
        }
    record["process_peak_bytes"] = memory()["peak"]
    (destination / "result.json").write_text(
        json.dumps(record, indent=2, default=str), encoding="utf-8"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--profile-first",
        action="store_true",
        help="Instrument first use separately; its timing includes profiling overhead",
    )
    parser.add_argument(
        "--case", choices=["startup", "sessile", "pendant", "memory30", "memory300"]
    )
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    args.output = args.output.resolve()
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    if args.case:
        worker(args)
        return
    args.output.mkdir(parents=True, exist_ok=True)
    for case in ("startup", "sessile", "pendant", "memory30", "memory300"):
        with (args.output / f"{case}.log").open("w", encoding="utf-8") as log:
            subprocess.run(
                [
                    "uv",
                    "run",
                    "--offline",
                    "--extra",
                    "test",
                    "python",
                    "-X",
                    "importtime",
                    str(Path(__file__).resolve()),
                    "--output",
                    str(args.output),
                    "--case",
                    case,
                    "--repeats",
                    str(args.repeats),
                ],
                stdout=log,
                stderr=log,
                check=True,
            )
        print(f"Recorded {case}", flush=True)


if __name__ == "__main__":
    main()
