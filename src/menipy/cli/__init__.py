"""Command-line interface (console-script entrypoint).

This module implements the CLI to run ADSA pipelines.
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Optional

from menipy.pipelines.base import PipelineBase, Context, PipelineError
from menipy.common import acquisition as acq

# OPTIONAL: if you use the SQLite-backed plugin system
try:
    from menipy.common.plugin_db import PluginDB
    from menipy.common.plugins import discover_into_db, load_active_plugins

    _PLUGINS_OK = True
except Exception:
    _PLUGINS_OK = False

try:
    import cv2  # type: ignore
except Exception:
    cv2 = None
try:
    from PIL import Image  # type: ignore

    _PIL_OK = True
except Exception:
    _PIL_OK = False


def _save_image_bgr(path: Path, img):
    if img is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    if cv2 is not None:
        cv2.imwrite(str(path), img)
        return
    if _PIL_OK:
        import numpy as np

        arr = img
        if arr.ndim == 2:
            Image.fromarray(arr, mode="L").save(path)
        elif arr.ndim == 3 and arr.shape[2] == 3:
            Image.fromarray(arr[..., ::-1], mode="RGB").save(path)  # BGR->RGB
        else:
            raise ValueError("Unsupported image shape for saving")
        return
    raise RuntimeError("Install opencv-python or Pillow to save images.")


def _pick_pipeline(name: str) -> PipelineBase:
    from menipy.pipelines.discover import PIPELINE_MAP

    name = name.lower()
    pipeline_cls = PIPELINE_MAP.get(name)
    if pipeline_cls is None:
        raise SystemExit(f"Unknown pipeline '{name}'")
    return pipeline_cls()


def _patch_acquisition(
    p: PipelineBase, *, image: Optional[Path], camera: Optional[int], frames: int
):
    # Replace the pipeline's acquisition hook so we can choose source via CLI.
    if image:
        img_path = str(image)

        def do_acq(ctx: Context):
            ctx.frames = acq.from_file([img_path])

            return ctx

        p.do_acquisition = do_acq  # type: ignore[attr-defined]
        return
    cam_id = 0 if camera is None else int(camera)

    def do_acq(ctx: Context):
        ctx.frames = acq.from_camera(device=cam_id, n_frames=frames)

        return ctx

    p.do_acquisition = do_acq  # type: ignore[attr-defined]


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="adsa", description="Run ADSA pipelines (CLI)")
    ap.add_argument(
        "--pipeline",
        default="sessile",
        choices=[
            "sessile",
            "oscillating",
            "capillary_rise",
            "pendant",
            "captive_bubble",
        ],
    )
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--image", type=str, help="Path to input image")
    src.add_argument("--camera", type=int, help="Camera index (e.g., 0)")
    ap.add_argument(
        "--frames", type=int, default=1, help="Number of frames when using --camera"
    )
    ap.add_argument(
        "--out", type=str, default="./out", help="Output folder (preview/results)"
    )

    # Optional plugin controls (no-ops if plugin system not present)
    ap.add_argument(
        "--plugins",
        type=str,
        default="./plugins",
        help="Plugin directories (colon/semicolon-separated)",
    )
    ap.add_argument(
        "--db", type=str, default="adsa_plugins.sqlite", help="Plugin SQLite path"
    )
    ap.add_argument(
        "--activate",
        action="append",
        default=[],
        help="Activate plugin name:kind (e.g., bezier:edge)",
    )
    ap.add_argument(
        "--deactivate", action="append", default=[], help="Deactivate plugin name:kind"
    )
    ap.add_argument(
        "--no-overlay", action="store_true", help="Skip overlay drawing stage"
    )

    args = ap.parse_args(argv)

    # Optional: SQLite plugin discovery + activation
    if _PLUGINS_OK:
        db = PluginDB(Path(args.db))
        db.init_schema()
        dirs = [Path(p) for p in str(args.plugins).replace(";", ":").split(":") if p]
        discover_into_db(db, dirs)
        for spec in args.activate:
            name, kind = spec.split(":")
            db.set_active(name, kind, True)
        for spec in args.deactivate:
            name, kind = spec.split(":")
            db.set_active(name, kind, False)
        load_active_plugins(db)

    pipeline = _pick_pipeline(args.pipeline)

    if args.no_overlay:
        pipeline.do_overlay = lambda ctx: ctx  # type: ignore[attr-defined]

    image_path = Path(args.image).expanduser().resolve() if args.image else None
    _patch_acquisition(
        pipeline, image=image_path, camera=args.camera, frames=args.frames
    )

    try:
        ctx = pipeline.run()
    except PipelineError as e:
        print(f"[adsa] error: {e}")
        return 2

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if getattr(ctx, "preview", None) is not None:
        _save_image_bgr(out_dir / "preview.png", ctx.preview)
        print(f"[adsa] wrote {out_dir/'preview.png'}")

    if getattr(ctx, "overlay", None) is not None:
        _save_image_bgr(out_dir / "overlay.png", ctx.overlay)

    with open(out_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "pipeline": pipeline.name,
                "results": ctx.results,
                "qa": ctx.qa,
                "timings_ms": ctx.timings_ms,
                "log": ctx.log,
                "error": ctx.error,
            },
            f,
            indent=2,
        )
    print(f"[adsa] wrote {out_dir/'results.json'}")
    return 0


__all__ = ["main"]
