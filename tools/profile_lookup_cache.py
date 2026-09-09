"""Fresh-process cold/write versus disk-hit timing with exact table comparison.

uv run --extra test python tools/profile_lookup_cache.py
"""

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from uuid import uuid4


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path)
    parser.add_argument("--case")
    args = parser.parse_args()
    if args.case:
        from check_gui_execution import isolate

        isolate(args.root / "state")
        from menipy.pipelines.pendant.approximations import _selected_plane_lookup_all

        started = time.perf_counter()
        table = _selected_plane_lookup_all()
        elapsed = time.perf_counter() - started
        digest = hashlib.sha256()
        for plane, arrays in sorted(table.items()):
            digest.update(str(plane).encode())
            for array in arrays:
                digest.update(array.tobytes())
        files = {
            p.name: p.stat().st_mtime_ns
            for p in Path.home().glob(".menipy/cache/selected_plane/*.json")
        }
        result = {
            "seconds": elapsed,
            "table_sha256": digest.hexdigest(),
            "files": files,
        }
        (args.root / f"{args.case}.json").write_text(json.dumps(result, indent=2))
        return
    root = (
        Path(__file__).resolve().parents[1] / ".cache" / "lookup-profile" / uuid4().hex
    )
    root.mkdir(parents=True)
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    for case in ("cold", "restart1", "restart2"):
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
                "--case",
                case,
            ],
            check=True,
        )
    results = [
        json.loads((root / f"{case}.json").read_text())
        for case in ("cold", "restart1", "restart2")
    ]
    assert results[0]["files"]
    assert all(r["table_sha256"] == results[0]["table_sha256"] for r in results)
    assert all(r["files"] == results[0]["files"] for r in results), (
        "Cache rebuilt across processes"
    )
    print(root)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
