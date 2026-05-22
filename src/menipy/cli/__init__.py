"""Consolidated and feature-complete Command-Line Interface package.

Provides single-image execution, camera capture, folder batch processing,
dynamic auto-calibration fallback, SOP configuration loading, and SQLite database lookups.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from menipy.common import acquisition as acq
from menipy.models.config import EdgeDetectionSettings, PreprocessingSettings
from menipy.models.context import Context
from menipy.pipelines.base import PipelineError
from menipy.pipelines.runner import PipelineRunner

# Standard log configuration
logger = logging.getLogger("menipy.cli")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[adsa] %(levelname)s: %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# Guard optional external dependencies for headless environments
try:
    import cv2
except ImportError:
    cv2 = None

try:
    from PIL import Image

    _PIL_OK = True
except ImportError:
    _PIL_OK = False

# Try loading SQLite database dependencies
try:
    from menipy.common.material_db import MaterialDB

    _MATERIAL_DB_OK = True
except ImportError:
    _MATERIAL_DB_OK = False

try:
    from menipy.common.plugin_db import PluginDB
    from menipy.common.plugins import (
        discover_and_load_from_db,
        discover_into_db,
        load_active_plugins,
    )

    _PLUGINS_OK = True
except ImportError:
    _PLUGINS_OK = False


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle numpy data types seamlessly in headless CLI."""

    def default(self, obj: Any) -> Any:
        try:
            import numpy as np

            if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
                return float(obj)
            elif isinstance(obj, (np.ndarray,)):
                return obj.tolist()
        except ImportError:
            pass
        return super().default(obj)


def _save_image_bgr(path: Path, img):
    """Save BGR image safely using OpenCV or Pillow fallback."""
    if img is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    if cv2 is not None:
        cv2.imwrite(str(path), img)
        return
    if _PIL_OK:
        arr = img
        if arr.ndim == 2:
            Image.fromarray(arr, mode="L").save(path)
        elif arr.ndim == 3 and arr.shape[2] == 3:
            Image.fromarray(arr[..., ::-1], mode="RGB").save(path)  # BGR -> RGB
        else:
            raise ValueError("Unsupported image shape for saving")
        return
    raise RuntimeError("Install opencv-python or Pillow to save images.")


def _parse_numbers(value: str, name: str, expected: int) -> tuple[float, ...]:
    """Parse comma/semicolon separated numbers."""
    cleaned = value
    for sep in (",", ";"):
        cleaned = cleaned.replace(sep, " ")
    parts = [p for p in cleaned.strip().split() if p]
    if len(parts) != expected:
        raise ValueError(f"{name} requires {expected} values")
    try:
        return tuple(float(p) for p in parts)
    except ValueError as exc:
        raise ValueError(f"{name} must contain numeric values") from exc


def _parse_rect(value: str, name: str) -> tuple[int, int, int, int]:
    """Parse rectangle coordinates (x,y,w,h)."""
    x, y, w, h = _parse_numbers(value, name, expected=4)
    if w <= 0 or h <= 0:
        raise ValueError(f"{name} width and height must be positive")
    return int(round(x)), int(round(y)), int(round(w)), int(round(h))


def _parse_line(value: str, name: str) -> tuple[tuple[int, int], tuple[int, int]]:
    """Parse line endpoints (x1,y1,x2,y2)."""
    x1, y1, x2, y2 = _parse_numbers(value, name, expected=4)
    if x1 == x2 and y1 == y2:
        raise ValueError(f"{name} endpoints must not coincide")
    return (int(round(x1)), int(round(y1))), (int(round(x2)), int(round(y2)))


def _patch_acquisition(p, *, image: Path | None, camera: int | None, frames: int):
    """Replace pipeline acquisition function for CLI mode."""
    if image:
        img_path = str(image)

        def do_acq_from_file(ctx: Context):
            ctx.frames = list(acq.from_file([img_path]))
            return ctx

        p.do_acquisition = do_acq_from_file
        return

    cam_id = 0 if camera is None else int(camera)

    def do_acq_from_camera(ctx: Context):
        ctx.frames = list(acq.from_camera(device=cam_id, n_frames=frames))
        return ctx

    p.do_acquisition = do_acq_from_camera


def main(argv: list[str] | None = None) -> int:
    """Consolidated main CLI runner."""
    ap = argparse.ArgumentParser(
        prog="adsa",
        description="Run ADSA droplet shape analysis pipelines (CLI with GUI feature parity)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
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
        help="Droplet shape analysis pipeline to run (default: sessile)",
    )

    # Input Modes (Mutually Exclusive)
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--image", type=str, help="Path to input image file")
    src.add_argument("--camera", type=int, help="Camera index (e.g., 0)")
    src.add_argument(
        "--input-dir", "-i", type=str, help="Input directory containing batch images"
    )

    ap.add_argument(
        "--glob",
        "-g",
        type=str,
        default="*.png,*.jpg,*.jpeg",
        help="Glob patterns to filter batch images, comma-separated (default: *.png,*.jpg,*.jpeg)",
    )
    ap.add_argument(
        "--frames",
        type=int,
        default=1,
        help="Number of frames to acquire when using --camera",
    )

    # Output Controls
    ap.add_argument(
        "--out",
        type=str,
        default="./out",
        help="Output directory path (default: ./out)",
    )
    ap.add_argument(
        "--output-dir",
        "-o",
        type=str,
        help="Alias for --out to define output directory path",
    )
    ap.add_argument(
        "--no-overlay", action="store_true", help="Skip overlay drawing stages"
    )

    # Manual Coordinate Coordinates (Optional if using --auto-calibrate)
    ap.add_argument("--roi", type=str, help="ROI bounding box as x,y,w,h")
    ap.add_argument("--needle", type=str, help="Needle bounding box as x,y,w,h")
    ap.add_argument(
        "--contact-line",
        "--baseline",
        type=str,
        help="Optional baseline as x1,y1,x2,y2",
    )

    # Parity: Auto-Calibration
    ap.add_argument(
        "--auto-calibrate",
        "-a",
        action="store_true",
        help="Enable automatic calibration detection (runs baseline AutoCalibrator on input)",
    )

    # Parity: SQLite Material/Needle DB Lookup
    ap.add_argument(
        "--material",
        "--fluid-name",
        type=str,
        dest="material",
        help="Fluid name to query density (rho1) from SQLite materials DB",
    )
    ap.add_argument(
        "--needle-name",
        type=str,
        help="Needle name or gauge to query outer diameter from SQLite needles DB",
    )
    ap.add_argument(
        "--needle-diameter",
        type=float,
        help="Override needle outer diameter in mm directly",
    )
    ap.add_argument(
        "--materials-db",
        type=str,
        default="menipy_materials.sqlite",
        help="Path to materials SQLite database (default: ./menipy_materials.sqlite)",
    )

    # Parity: SOP Loading
    ap.add_argument(
        "--sop",
        "-s",
        type=str,
        help="Standard Operating Procedure profile name or JSON file path",
    )

    # Preprocessing / Edge Detection Overrides
    ap.add_argument(
        "--preprocessing-method",
        type=str,
        help="Preprocessing method to override (e.g., blur, clahe)",
    )
    ap.add_argument(
        "--edge-detection-method",
        type=str,
        help="Edge detection filter to override (e.g., canny, sobel)",
    )

    # SQLite Plugins controls
    ap.add_argument(
        "--plugins",
        type=str,
        default="./plugins",
        help="Plugin scan directories, separator-separated",
    )
    ap.add_argument(
        "--db", type=str, default="adsa_plugins.sqlite", help="SQLite plugin db path"
    )
    ap.add_argument(
        "--activate",
        action="append",
        default=[],
        help="Activate plugin: name:kind",
    )
    ap.add_argument(
        "--deactivate",
        action="append",
        default=[],
        help="Deactivate plugin: name:kind",
    )

    # Subcommands
    sub = ap.add_subparsers(dest="command")
    sp_plugins = sub.add_parser("plugins", help="Plugin DB management")
    sp_plugins_sub = sp_plugins.add_subparsers(dest="plugins_cmd")
    sp_set_dirs = sp_plugins_sub.add_parser(
        "set-dirs", help="Set SQLite plugin scan directories"
    )
    sp_set_dirs.add_argument(
        "dirs", type=str, help="Directories string (colon/semicolon separated)"
    )
    sp_set_dirs.add_argument(
        "--db", type=str, default="adsa_plugins.sqlite", help="SQLite db path"
    )

    args = ap.parse_args(argv)

    # Handle subcommand set-dirs early
    if args.command == "plugins" and args.plugins_cmd == "set-dirs":
        if not _PLUGINS_OK:
            print("[adsa] SQLite plugin features are unavailable in this installation.")
            return 1
        db = PluginDB(Path(getattr(args, "db", "adsa_plugins.sqlite")))
        db.init_schema()
        db.set_setting("plugin_dirs", args.dirs)
        print(f"[adsa] Stored plugin scan directories in DB: {args.dirs}")
        return 0

    # Resolve Output Folder (supporting user-specified --output-dir and --out)
    out_dir = (
        Path(args.output_dir if args.output_dir else args.out).expanduser().resolve()
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    # Initialize SQLite Plugin system if active
    if _PLUGINS_OK:
        try:
            db = PluginDB(Path(args.db))
            db.init_schema()
            loaded = discover_and_load_from_db(db)
            if not loaded:
                dirs = [
                    Path(p) for p in str(args.plugins).replace(";", ":").split(":") if p
                ]
                discover_into_db(db, dirs)
                for spec in args.activate:
                    if ":" in spec:
                        n, k = spec.split(":", 1)
                        db.set_active(n, k, True)
                for spec in args.deactivate:
                    if ":" in spec:
                        n, k = spec.split(":", 1)
                        db.set_active(n, k, False)
                load_active_plugins(db)
        except Exception as e:
            logger.warning(f"Failed to load SQLite plugins: {e}")

    # 1. SOP Config Loading
    sop_data = None
    if args.sop:
        sop_path = Path(args.sop)
        if sop_path.is_file():
            try:
                with open(sop_path, encoding="utf-8") as f:
                    sop_data = json.load(f)
                logger.info(f"Loaded SOP configuration from JSON file: {sop_path}")
            except Exception as e:
                logger.error(f"Failed to load SOP JSON file: {e}")
                return 1
        else:
            # Attempt to query SopService database (GUI fallback)
            try:
                from menipy.gui.services.sop_service import SopService

                service = SopService()
                sop = service.get(args.pipeline, args.sop)
                if sop:
                    sop_data = {
                        "include_stages": sop.include_stages,
                        "params": sop.params,
                    }
                    logger.info(f"Loaded SOP profile '{args.sop}' from SopService")
                else:
                    logger.warning(
                        f"Sop profile '{args.sop}' not found in SopService database."
                    )
            except Exception:
                logger.warning(
                    f"Could not load SOP database in headless context for: '{args.sop}'"
                )

    # Map settings from SOP parameters if loaded
    sop_params = sop_data.get("params", {}) if sop_data else {}

    # 2. Material and Needle lookups from SQLite DB
    rho1 = 1000.0  # drop density
    rho2 = 1.2  # continuous phase density
    needle_diameter_mm = args.needle_diameter

    if _MATERIAL_DB_OK:
        try:
            mdb = MaterialDB(Path(args.materials_db))
            mdb.init_schema()  # Ensure seeded default values exist

            # Fluid lookup
            if args.material:
                mat = mdb.get_material(args.material)
                if mat:
                    rho1 = mat.get("density", 1000.0)
                    logger.info(
                        f"Retrieved fluid density for '{args.material}': {rho1} kg/m3"
                    )
                else:
                    logger.warning(
                        f"Fluid '{args.material}' not found in SQLite Materials DB. Falling back to default density."
                    )

            # Needle size lookup
            if args.needle_name:
                needles = mdb.list_needles()
                needle_match = None
                for n in needles:
                    if n["name"].lower() == args.needle_name.lower() or (
                        n.get("gauge")
                        and n["gauge"].lower() == args.needle_name.lower()
                    ):
                        needle_match = n
                        break
                if needle_match:
                    needle_diameter_mm = needle_match.get("outer_diameter")
                    logger.info(
                        f"Retrieved needle size for '{args.needle_name}': {needle_diameter_mm} mm"
                    )
                else:
                    logger.warning(
                        f"Needle specifications for '{args.needle_name}' not found in SQLite needles DB."
                    )
        except Exception as e:
            logger.warning(f"Failed to query materials SQLite DB: {e}")

    # Set up preprocessing and edge detection configs, respecting overrides
    preprocessing_settings = None
    edge_detection_settings = None

    # Preprocessing
    if args.preprocessing_method:
        preprocessing_settings = PreprocessingSettings(method=args.preprocessing_method)
    elif "preprocessing" in sop_params and sop_params["preprocessing"]:
        preprocessing_settings = PreprocessingSettings(**sop_params["preprocessing"])

    # Edge Detection
    if args.edge_detection_method:
        edge_detection_settings = EdgeDetectionSettings(
            method=args.edge_detection_method
        )
    elif "edge_detection" in sop_params and sop_params["edge_detection"]:
        edge_detection_settings = EdgeDetectionSettings(**sop_params["edge_detection"])
    else:
        # Default fallback to canny
        edge_detection_settings = EdgeDetectionSettings(method="canny")

    # Instantiate the standard modular PipelineRunner
    try:
        runner = PipelineRunner(
            pipeline_name=args.pipeline,
            preprocessing_settings=preprocessing_settings,
            edge_detection_settings=edge_detection_settings,
        )
    except PipelineError as e:
        logger.error(str(e))
        return 2

    if args.no_overlay:
        runner.pipeline.do_overlay = lambda ctx: ctx

    # Build input files queue
    files_queue: list[Path] = []
    if args.image:
        files_queue.append(Path(args.image).expanduser().resolve())
    elif args.input_dir:
        input_dir_path = Path(args.input_dir).expanduser().resolve()
        if not input_dir_path.is_dir():
            logger.error(f"Input directory does not exist: {input_dir_path}")
            return 1
        # Parse multi-pattern globs
        patterns = [p.strip() for p in args.glob.split(",") if p.strip()]
        for pat in patterns:
            files_queue.extend(input_dir_path.glob(pat))
        # Remove duplicates and sort
        files_queue = sorted(set(files_queue))
        if not files_queue:
            logger.error(
                f"No matching image files found under {input_dir_path} with glob filter: '{args.glob}'"
            )
            return 1
        logger.info(
            f"Batch mode activated. Found {len(files_queue)} images to analyze."
        )

    # 3. Main execution path
    # Parsing manual geometries early if provided
    manual_roi = None
    manual_needle = None
    manual_contact = None

    try:
        if args.roi:
            manual_roi = _parse_rect(args.roi, "ROI")
        if args.needle:
            manual_needle = _parse_rect(args.needle, "needle")
        if args.contact_line:
            manual_contact = _parse_line(args.contact_line, "contact line")
    except ValueError as e:
        ap.error(str(e))

    # Core execution block
    run_records: list[dict[str, Any]] = []

    if args.camera is not None:
        # Camera Capture Path
        logger.info(f"Opening camera stream index: {args.camera}")
        _patch_acquisition(
            runner.pipeline, image=None, camera=args.camera, frames=args.frames
        )

        # Calibration Fallback / Auto-Calibration on camera frames
        roi_rect = manual_roi
        needle_rect = manual_needle
        substrate_line = manual_contact

        if args.auto_calibrate or (not roi_rect and not needle_rect):
            logger.info("Acquiring preview frame to run Auto-Calibration...")
            try:
                temp_ctx = Context()
                _patch_acquisition(
                    runner.pipeline, image=None, camera=args.camera, frames=1
                )
                temp_ctx = runner.pipeline.do_acquisition(temp_ctx)
                if temp_ctx.frames:
                    first_frame = temp_ctx.frames[0]
                    if hasattr(first_frame, "image"):
                        first_frame = first_frame.image
                    from menipy.common.auto_calibrator import run_auto_calibration

                    cal_res = run_auto_calibration(first_frame, args.pipeline)
                    roi_rect = cal_res.roi_rect
                    needle_rect = cal_res.needle_rect
                    substrate_line = cal_res.substrate_line
                    logger.info(
                        f"Auto-Calibration completed. ROI: {roi_rect}, Needle: {needle_rect}"
                    )
            except Exception as e:
                logger.warning(
                    f"Auto-calibration failed, relying on default/fallback geometries: {e}"
                )

        # Compute calibration metrics
        px_per_mm = 100.0 / max(needle_diameter_mm or 0.72, 0.001)
        scale_dict = {"px_per_mm": px_per_mm}

        # Run pipeline
        try:
            ctx = runner.run(
                roi=roi_rect,
                needle_rect=needle_rect,
                contact_line=substrate_line,
                camera=args.camera,
                frames=args.frames,
                scale=scale_dict,
                needle_diameter_mm=needle_diameter_mm,
                physics={"rho1": rho1, "rho2": rho2, "g": 9.80665},
            )

            # Export outputs
            if getattr(ctx, "preview", None) is not None:
                _save_image_bgr(out_dir / "preview.png", ctx.preview)
            if getattr(ctx, "overlay", None) is not None:
                _save_image_bgr(out_dir / "overlay.png", ctx.overlay)

            results_out = {
                "pipeline": runner.pipeline.name,
                "results": ctx.results,
                "qa": ctx.qa.to_dict() if hasattr(ctx.qa, "to_dict") else ctx.qa,
                "timings_ms": ctx.timings_ms,
                "log": ctx.log,
                "error": ctx.error,
            }
            with open(out_dir / "results.json", "w", encoding="utf-8") as f:
                json.dump(results_out, f, indent=2, cls=NumpyEncoder)
            logger.info(
                f"Successfully processed camera capture. Outputs written to {out_dir}"
            )
            return 0
        except PipelineError as e:
            logger.error(f"Pipeline execution error: {e}")
            return 2

    # File Processing Loops (Batch or Single Image)
    for img_path in files_queue:
        logger.info(f"Analyzing: {img_path.name}")
        _patch_acquisition(runner.pipeline, image=img_path, camera=None, frames=1)

        roi_rect = manual_roi
        needle_rect = manual_needle
        substrate_line = manual_contact

        # 4. Auto-Calibration Fallback
        if args.auto_calibrate or (not roi_rect and not needle_rect):
            try:
                # Load first frame to run baseline calibrator
                temp_ctx = Context()
                temp_ctx = runner.pipeline.do_acquisition(temp_ctx)
                if temp_ctx.frames:
                    first_frame = temp_ctx.frames[0]
                    if hasattr(first_frame, "image"):
                        first_frame = first_frame.image
                    from menipy.common.auto_calibrator import run_auto_calibration

                    cal_res = run_auto_calibration(first_frame, args.pipeline)

                    # Update parameters if not explicitly provided by user
                    if not roi_rect:
                        roi_rect = cal_res.roi_rect
                    if not needle_rect:
                        needle_rect = cal_res.needle_rect
                    if not substrate_line:
                        substrate_line = cal_res.substrate_line

                    logger.debug(
                        f"Auto-Calibrated ROI: {roi_rect}, Needle: {needle_rect}, Substrate: {substrate_line}"
                    )
            except Exception as e:
                logger.warning(f"Failed to auto-calibrate image {img_path.name}: {e}")

        # Calibration computations (default to 0.72mm outer needle if DB lookup and overrides fail)
        target_needle_diam = needle_diameter_mm or 0.72
        px_per_mm = 100.0 / max(target_needle_diam, 0.001)
        scale_dict = {"px_per_mm": px_per_mm}

        try:
            ctx = runner.run(
                roi=roi_rect,
                needle_rect=needle_rect,
                contact_line=substrate_line,
                image=str(img_path),
                scale=scale_dict,
                needle_diameter_mm=target_needle_diam,
                physics={"rho1": rho1, "rho2": rho2, "g": 9.80665},
            )

            # Export individual image visuals
            base_name = img_path.stem
            if len(files_queue) == 1:
                # Single Image uses standard standard names
                preview_name = "preview.png"
                overlay_name = "overlay.png"
                json_name = "results.json"
            else:
                # Batch utilizes base names to prevent collisions
                preview_name = f"{base_name}_preview.png"
                overlay_name = f"{base_name}_overlay.png"
                json_name = f"{base_name}_results.json"

            if getattr(ctx, "preview", None) is not None:
                _save_image_bgr(out_dir / preview_name, ctx.preview)
            if getattr(ctx, "overlay", None) is not None:
                _save_image_bgr(out_dir / overlay_name, ctx.overlay)

            # Write standard results dictionary
            results_out = {
                "pipeline": runner.pipeline.name,
                "results": ctx.results,
                "qa": ctx.qa.to_dict() if hasattr(ctx.qa, "to_dict") else ctx.qa,
                "timings_ms": ctx.timings_ms,
                "log": ctx.log,
                "error": ctx.error,
            }
            with open(out_dir / json_name, "w", encoding="utf-8") as f:
                json.dump(results_out, f, indent=2, cls=NumpyEncoder)

            # Extract metrics for consolidated batch CSV
            qa_ok = (
                ctx.qa.get("ok", True)
                if isinstance(ctx.qa, dict)
                else getattr(ctx.qa, "ok", True)
            )
            metrics = ctx.results if isinstance(ctx.results, dict) else {}

            run_records.append(
                {
                    "image_path": str(img_path),
                    "pipeline": runner.pipeline.name,
                    "qa_ok": qa_ok,
                    "metrics": metrics,
                }
            )

        except PipelineError as e:
            logger.error(f"Failed to process {img_path.name}: {e}")
            if len(files_queue) == 1:
                return 2

    # 5. Generate consolidated results.csv in batch mode
    if args.input_dir and run_records:
        csv_path = out_dir / "results.csv"
        csv_headers = ["image_path", "pipeline", "qa_ok"]

        # Collect dynamic metric headers
        unique_metric_keys = set()
        for rec in run_records:
            unique_metric_keys.update(rec["metrics"].keys())
        csv_headers.extend(sorted(unique_metric_keys))

        try:
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=csv_headers)
                writer.writeheader()
                for rec in run_records:
                    row_data = {
                        "image_path": rec["image_path"],
                        "pipeline": rec["pipeline"],
                        "qa_ok": rec["qa_ok"],
                        **rec["metrics"],
                    }
                    writer.writerow(row_data)
            logger.info(
                f"Consolidated results table exported successfully to: {csv_path}"
            )
        except Exception as e:
            logger.error(f"Failed to export batch results.csv: {e}")

    logger.info(f"Analysis complete. All outputs saved in: {out_dir}")
    return 0


__all__ = ["main"]
