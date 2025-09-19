from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Optional

from .pipelines.base import PipelineBase, Context, PipelineError
from .common import acquisition as acq

# OPTIONAL: if you use the SQLite-backed plugin system
try:
    from .common.plugin_db import PluginDB
    from .common.plugins import discover_into_db, load_active_plugins, discover_and_load_from_db
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
    name = name.lower()
    if name == "sessile":
        from .pipelines.sessile.stages import SessilePipeline
        return SessilePipeline()
    if name == "oscillating":
        from .pipelines.oscillating.stages import OscillatingPipeline
        return OscillatingPipeline()
    if name == "capillary_rise":
        from .pipelines.capillary_rise.stages import CapillaryRisePipeline
        return CapillaryRisePipeline()
    if name == "pendant":
        from .pipelines.pendant.stages import PendantPipeline
        return PendantPipeline()
    if name == "captive_bubble":
        from .pipelines.captive_bubble.stages import CaptiveBubblePipeline
        return CaptiveBubblePipeline()
    raise SystemExit(f"Unknown pipeline '{name}'")


def _patch_acquisition(p: PipelineBase, *, image: Optional[Path], camera: Optional[int], frames: int):
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
    ap = argparse.ArgumentParser(
        prog="adsa",
        description=(
            "Run ADSA pipelines (CLI).\n\n"
            "Plugins: when available, plugins can be managed via a SQLite DB.\n"
            "If the DB has a settings key 'plugin_dirs', discovery will use it\n"
            "(separator-separated list using ':' or ';'). You can set it via\n"
            "the subcommand: adsa plugins set-dirs \"./plugins1;./plugins2\"."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--pipeline", default="sessile",
                    choices=["sessile","oscillating","capillary_rise","pendant","captive_bubble"])
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--image", type=str, help="Path to input image")
    src.add_argument("--camera", type=int, help="Camera index (e.g., 0)")
    ap.add_argument("--frames", type=int, default=1, help="Number of frames when using --camera")
    ap.add_argument("--out", type=str, default="./out", help="Output folder (preview/results)")

    # Optional plugin controls (no-ops if plugin system not present)
    ap.add_argument(
        "--plugins",
        type=str,
        default="./plugins",
        help=(
            "Plugin directories (':'/';' separated). Note: If the DB setting "
            "'plugin_dirs' is set, it takes precedence. Use 'adsa plugins "
            "set-dirs' to store it in the DB."
        ),
    )
    ap.add_argument("--db", type=str, default="adsa_plugins.sqlite", help="Plugin SQLite path")
    ap.add_argument("--activate", action="append", default=[], help="Activate plugin name:kind (e.g., bezier:edge)")
    ap.add_argument("--deactivate", action="append", default=[], help="Deactivate plugin name:kind")
    ap.add_argument("--no-overlay", action="store_true", help="Skip overlay drawing stage")

    # Subcommands for convenience utilities (plugins, etc.)
    sub = ap.add_subparsers(dest="command")

    # adsa plugins set-dirs "dir1;dir2"
    sp_plugins = sub.add_parser("plugins", help="Plugin DB utilities")
    sp_plugins_sub = sp_plugins.add_subparsers(dest="plugins_cmd")
    sp_set_dirs = sp_plugins_sub.add_parser(
        "set-dirs",
        help=(
            "Set the DB setting 'plugin_dirs' to a separator-separated list "
            "of directories (use ':' or ';')."
        ),
    )
    sp_set_dirs.add_argument(
        "dirs",
        type=str,
        help="Plugin directories string (e.g., ./p1;./p2 or ./p1:./p2)",
    )
    sp_set_dirs.add_argument(
        "--db",
        type=str,
        default="adsa_plugins.sqlite",
        help="Plugin SQLite path (defaults to ./adsa_plugins.sqlite)",
    )

    args = ap.parse_args(argv)

    # Handle subcommands early and exit.
    if args.command == "plugins" and args.plugins_cmd == "set-dirs":
        if not _PLUGINS_OK:
            print("[adsa] plugin DB features are not available in this build.")
            return 1
        db = PluginDB(Path(getattr(args, "db", "adsa_plugins.sqlite")))
        db.init_schema()
        db.set_setting("plugin_dirs", args.dirs)
        print(
            "[adsa] set DB setting 'plugin_dirs' to:",
            args.dirs,
        )
        print(
            "[adsa] Tip: run without --plugins to use DB-configured directories."
        )
        return 0

    # Optional: SQLite plugin discovery + activation
    if _PLUGINS_OK:
        db = PluginDB(Path(args.db))
        db.init_schema()
        # prefer DB-driven discovery/load: read plugin_dirs from settings
        loaded = discover_and_load_from_db(db)
        if not loaded:
            # fallback to explicit CLI plugin dirs
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
        pipeline.do_overlay = (lambda ctx: ctx)  # type: ignore[attr-defined]

    image_path = Path(args.image).expanduser().resolve() if args.image else None
    _patch_acquisition(pipeline, image=image_path, camera=args.camera, frames=args.frames)

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
        json.dump({
            "pipeline": pipeline.name,
            "results": ctx.results,
            "qa": ctx.qa,
            "timings_ms": ctx.timings_ms,
            "log": ctx.log,
            "error": ctx.error,
        }, f, indent=2)
    print(f"[adsa] wrote {out_dir/'results.json'}")
    return 0
