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
from menipy.common.sequence_acquisition import _natural_key
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


def _run_sfe(args, out_dir: Path) -> int:
    """Execute the Surface Free Energy analysis pipeline.

    Supports three input modes:
    1. ``--liquid water:65.3 --liquid diiodomethane:42.1`` (inline pairs)
    2. ``--sfe-input measurements.json`` (structured JSON)
    3. ``--sfe-csv batch.csv`` (multi-substrate batch)

    Returns 0 on success, non-zero on error.
    """
    from menipy.common.liquid_db import get_liquid, list_liquid_names
    from menipy.math.surface_energy import compute_surface_energy

    temperature = getattr(args, "sfe_temperature", 20.0)
    method = getattr(args, "sfe_method", "both")
    generate_plot = getattr(args, "plot", False)

    # ── Collect substrate measurement groups ──
    # Each entry: (substrate_name, [(liquid_name, angle), ...])
    substrate_groups: list[tuple[str | None, list[tuple[str, float]]]] = []

    # Mode 1: --liquid pairs
    if args.liquid:
        pairs: list[tuple[str, float]] = []
        for spec in args.liquid:
            if ":" not in spec:
                logger.error(
                    f"Invalid --liquid format '{spec}'. Expected 'liquid_name:angle_deg'"
                )
                return 1
            name_part, angle_part = spec.rsplit(":", 1)
            try:
                angle = float(angle_part)
            except ValueError:
                logger.error(f"Invalid angle value in --liquid '{spec}'")
                return 1
            pairs.append((name_part.strip(), angle))
        substrate_groups.append((None, pairs))

    # Mode 2: --sfe-input JSON
    elif args.sfe_input:
        sfe_path = Path(args.sfe_input).expanduser().resolve()
        if not sfe_path.is_file():
            logger.error(f"SFE input file not found: {sfe_path}")
            return 1
        try:
            with open(sfe_path, encoding="utf-8") as f:
                sfe_data = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            logger.error(f"Failed to read SFE input JSON: {exc}")
            return 1

        substrate_name = sfe_data.get("substrate")
        if sfe_data.get("temperature_c") is not None:
            temperature = sfe_data["temperature_c"]
        measurements = sfe_data.get("measurements", [])
        if not measurements:
            logger.error("SFE input JSON contains no measurements")
            return 1
        pairs = [(m["liquid"], m["contact_angle_deg"]) for m in measurements]
        substrate_groups.append((substrate_name, pairs))

    # Mode 3: --sfe-csv batch
    elif args.sfe_csv:
        csv_path = Path(args.sfe_csv).expanduser().resolve()
        if not csv_path.is_file():
            logger.error(f"SFE CSV file not found: {csv_path}")
            return 1
        try:
            with open(csv_path, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
        except (csv.Error, OSError) as exc:
            logger.error(f"Failed to read SFE CSV: {exc}")
            return 1

        # Group rows by substrate
        from collections import OrderedDict

        grouped: dict[str, list[tuple[str, float]]] = OrderedDict()
        for row in rows:
            sub = row.get("substrate", "unknown").strip()
            liq_name = row.get("liquid", "").strip()
            try:
                angle = float(row.get("contact_angle_deg", ""))
            except (ValueError, TypeError):
                logger.warning(f"Skipping invalid row: {row}")
                continue
            grouped.setdefault(sub, []).append((liq_name, angle))
        for sub_name, sub_pairs in grouped.items():
            substrate_groups.append((sub_name, sub_pairs))

    else:
        logger.error(
            "Surface energy pipeline requires one of: --liquid, --sfe-input, or --sfe-csv"
        )
        return 1

    # ── Process each substrate group ──
    from dataclasses import asdict

    from menipy.common.liquid_db import ProbeLiquid

    if generate_plot:
        try:
            import matplotlib.pyplot as plt

            from menipy.viz.owrk_plot import plot_owrk
        except ImportError:
            plot_owrk = None
            plt = None
    else:
        plot_owrk = None
        plt = None

    all_results: list[dict] = []
    available_names = list_liquid_names()

    for substrate_name, pairs in substrate_groups:
        liquids = []
        angles = []
        for liq_name, angle in pairs:
            liq = get_liquid(liq_name, temperature_c=temperature)
            if liq is None:
                logger.error(
                    f"Unknown liquid '{liq_name}'. Available: {', '.join(available_names)}"
                )
                return 1
            liquids.append(liq)
            angles.append(angle)

        label = substrate_name or "sample"
        logger.info(
            f"Computing SFE for '{label}' with {len(liquids)} liquids "
            f"({method}) at {temperature} C"
        )

        sfe_result = compute_surface_energy(
            liquids, angles, method=method, substrate_name=substrate_name
        )
        result_dict = {
            "pipeline": "surface_energy",
            "schema_version": "1.0",
            **sfe_result.to_dict(),
        }
        all_results.append(result_dict)

        # Print summary to stdout
        if sfe_result.owrk is not None:
            o = sfe_result.owrk
            r2_str = f"  R2 = {o.r_squared:.4f}" if o.r_squared is not None else ""
            print(
                f"[adsa] OWRK ({label}): gS_d = {o.gamma_s_d:.1f}, "
                f"gS_p = {o.gamma_s_p:.1f}, gS = {o.gamma_s_total:.1f} mN/m{r2_str}"
            )
        if sfe_result.wu is not None:
            w = sfe_result.wu
            print(
                f"[adsa]   Wu ({label}): gS_d = {w.gamma_s_d:.1f}, "
                f"gS_p = {w.gamma_s_p:.1f}, gS = {w.gamma_s_total:.1f} mN/m"
            )
        for warn in sfe_result.warnings:
            logger.warning(warn)

        # Generate OWRK plot if requested
        if (
            generate_plot
            and sfe_result.owrk is not None
            and plot_owrk is not None
            and plt is not None
        ):
            try:
                plot_name = (
                    f"owrk_plot_{label}.png" if substrate_name else "owrk_plot.png"
                )
                plot_path = out_dir / plot_name
                title = (
                    f"OWRK — {label}"
                    if substrate_name
                    else "OWRK Surface Energy Analysis"
                )
                fig = plot_owrk(sfe_result.owrk, output_path=plot_path, title=title)
                plt.close(fig)
                logger.info(f"OWRK plot saved: {plot_path}")
            except Exception as exc:
                logger.warning(f"Failed to generate OWRK plot: {exc}")

    # ── Write output files ──
    if len(all_results) == 1:
        # Single substrate — write results.json
        results_path = out_dir / "results.json"
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(all_results[0], f, indent=2, cls=NumpyEncoder)
        logger.info(f"SFE results written to {results_path}")
    else:
        # Multi-substrate batch — write individual JSONs + consolidated CSV
        for i, result_dict in enumerate(all_results):
            sub = result_dict.get("substrate") or f"substrate_{i + 1}"
            safe_name = sub.replace(" ", "_").replace("/", "_")
            results_path = out_dir / f"{safe_name}_results.json"
            with open(results_path, "w", encoding="utf-8") as f:
                json.dump(result_dict, f, indent=2, cls=NumpyEncoder)

        # Consolidated CSV
        csv_path = out_dir / "results.csv"
        csv_headers = [
            "substrate",
            "method",
            "gamma_s_dispersive_mN_m",
            "gamma_s_polar_mN_m",
            "gamma_s_total_mN_m",
            "r_squared",
            "warnings",
        ]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=csv_headers)
            writer.writeheader()
            for result_dict in all_results:
                sub = result_dict.get("substrate", "")
                owrk = result_dict.get("owrk")
                wu = result_dict.get("wu")
                if owrk:
                    writer.writerow(
                        {
                            "substrate": sub,
                            "method": "owrk",
                            "gamma_s_dispersive_mN_m": owrk.get(
                                "gamma_s_dispersive_mN_m"
                            ),
                            "gamma_s_polar_mN_m": owrk.get("gamma_s_polar_mN_m"),
                            "gamma_s_total_mN_m": owrk.get("gamma_s_total_mN_m"),
                            "r_squared": owrk.get("r_squared"),
                            "warnings": ";".join(owrk.get("warnings", [])),
                        }
                    )
                if wu:
                    writer.writerow(
                        {
                            "substrate": sub,
                            "method": "wu",
                            "gamma_s_dispersive_mN_m": wu.get(
                                "gamma_s_dispersive_mN_m"
                            ),
                            "gamma_s_polar_mN_m": wu.get("gamma_s_polar_mN_m"),
                            "gamma_s_total_mN_m": wu.get("gamma_s_total_mN_m"),
                            "r_squared": None,
                            "warnings": ";".join(wu.get("warnings", [])),
                        }
                    )
        logger.info(f"Batch SFE results written to {csv_path}")

    logger.info(f"SFE analysis complete. Outputs saved in: {out_dir}")
    return 0


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
            "sessile_dynamic",
            "surface_energy",
            "needle_hysteresis",
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
    src.add_argument("--video", type=str, help="Video source for sessile_dynamic")
    src.add_argument(
        "--sequence-dir", type=str, help="Ordered image sequence for sessile_dynamic"
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
    ap.add_argument("--fps", type=float, help="Required FPS for --sequence-dir")
    ap.add_argument("--px-per-mm", type=float, help="Fixed calibrated sequence scale")
    ap.add_argument(
        "--no-temporal-tracking",
        action="store_true",
        help="Disable temporal tracking and physical invariant locking in batch mode",
    )

    # Surface Free Energy arguments
    ap.add_argument(
        "--liquid",
        action="append",
        default=[],
        help="Liquid:angle pair for SFE, e.g. 'water:65.3' (repeatable)",
    )
    ap.add_argument(
        "--sfe-input",
        type=str,
        help="JSON file with SFE measurements (see docs for schema)",
    )
    ap.add_argument(
        "--sfe-csv",
        type=str,
        help="CSV with columns substrate,liquid,contact_angle_deg for batch SFE",
    )
    ap.add_argument(
        "--sfe-method",
        choices=["owrk", "wu", "both"],
        default="both",
        help="SFE calculation method (default: both)",
    )
    ap.add_argument(
        "--sfe-temperature",
        type=float,
        default=20.0,
        help="Temperature for liquid property lookup in degrees C (default: 20.0)",
    )
    ap.add_argument(
        "--plot",
        action="store_true",
        help="Generate OWRK regression plot (saved as PNG alongside results)",
    )
    ap.add_argument(
        "--list-liquids",
        action="store_true",
        help="List available probe liquids in the built-in database and exit",
    )
    ap.add_argument(
        "--needle-fit-method",
        choices=["tangent", "cdf", "auto"],
        default="auto",
        help="Contact angle fitting method for needle_hysteresis (default: auto)",
    )

    ap.add_argument(
        "--onnx-proposal-mode",
        choices=["off", "shadow"],
        default="off",
        help="Run optional ONNX proposals without promoting their outputs",
    )
    ap.add_argument(
        "--segmentation-provider",
        default="mobilesam",
        help="Registered non-authoritative segmentation provider",
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
    sp_annotate = sub.add_parser(
        "annotate", help="Generate review-required ADSA segmentation proposals"
    )
    sp_annotate.add_argument("--input", required=True, help="Input image or directory")
    sp_annotate.add_argument(
        "--pipeline",
        dest="annotation_pipeline",
        choices=["sessile", "pendant"],
        default="sessile",
    )
    sp_annotate.add_argument(
        "--provider", default="mobilesam", help="Segmentation proposal provider"
    )
    sp_annotate.add_argument(
        "--output", default=".tmp/adsa-ml/proposals", help="Ignored proposal output"
    )
    sp_annotate.add_argument(
        "--license-id",
        default="unresolved-evaluation-only",
        help="License/provenance classification recorded for source images",
    )
    sp_yolo = sub.add_parser(
        "coco-to-yolo", help="Convert approved COCO polygons to YOLO segmentation"
    )
    sp_yolo.add_argument("--input", required=True, help="COCO annotation JSON")
    sp_yolo.add_argument("--output", required=True, help="YOLO label directory")
    sp_yolo.add_argument(
        "--include-proposed",
        action="store_true",
        help="Research-only conversion including unreviewed proposals",
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
    if args.command == "annotate":
        from menipy.common.annotation_dataset import annotate_images, collect_images

        try:
            images = collect_images(args.input)
            if not images:
                print("[adsa] No supported images found for annotation.")
                return 2
            annotate_images(
                images,
                pipeline=args.annotation_pipeline,
                output_dir=args.output,
                provider_name=args.provider,
                license_id=args.license_id,
            )
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            print(f"[adsa] Annotation failed: {exc}")
            return 2
        print(f"[adsa] Proposed annotations written to: {Path(args.output).resolve()}")
        return 0
    if args.command == "coco-to-yolo":
        from menipy.common.annotation_dataset import convert_coco_to_yolo

        try:
            summary = convert_coco_to_yolo(
                args.input,
                args.output,
                approved_only=not args.include_proposed,
            )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            print(f"[adsa] COCO conversion failed: {exc}")
            return 2
        print(f"[adsa] YOLO conversion complete: {summary['label_files']} label files")
        return 0

    # --list-liquids: print the built-in probe liquid table and exit
    if getattr(args, "list_liquids", False):
        from menipy.common.liquid_db import format_liquid_table

        print(format_liquid_table(temperature_c=args.sfe_temperature))
        return 0

    # Resolve Output Folder (supporting user-specified --output-dir and --out)
    out_dir = (
        Path(args.output_dir if args.output_dir else args.out).expanduser().resolve()
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    # Surface Free Energy pipeline — bypass the standard image pipeline path
    if args.pipeline == "surface_energy":
        return _run_sfe(args, out_dir)

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

    seq_source = (
        args.video
        or args.sequence_dir
        or (
            args.input_dir
            if args.pipeline in ("sessile_dynamic", "needle_hysteresis")
            else None
        )
    )
    if seq_source:
        if args.pipeline not in ("sessile_dynamic", "needle_hysteresis"):
            ap.error(
                "--video and --sequence-dir require --pipeline sessile_dynamic or needle_hysteresis"
            )
        if (
            args.sequence_dir
            or (
                args.pipeline in ("sessile_dynamic", "needle_hysteresis")
                and args.input_dir
            )
        ) and (args.fps is None or args.fps <= 0):
            ap.error("--sequence-dir and dynamic --input-dir require a positive --fps")
        source_path = Path(seq_source).expanduser().resolve()
        try:
            if args.pipeline == "needle_hysteresis":
                ctx = runner.run(
                    sequence_path=str(source_path),
                    sequence_fps=args.fps,
                    px_per_mm=args.px_per_mm,
                    needle_diameter_mm=needle_diameter_mm,
                    needle_fit_method=args.needle_fit_method,
                )
                if ctx.needle_hysteresis_result is None:
                    raise PipelineError("needle_hysteresis_result_missing")
                from menipy.common.needle_hysteresis import (
                    export_needle_hysteresis_results,
                )

                export_needle_hysteresis_results(ctx.needle_hysteresis_result, out_dir)

                if getattr(args, "plot", False):
                    import matplotlib.pyplot as plt

                    from menipy.viz.hysteresis_plot import (
                        plot_hysteresis_loop,
                        plot_hysteresis_timeline,
                    )

                    fig1 = plot_hysteresis_timeline(
                        ctx.needle_hysteresis_result,
                        output_path=out_dir / "hysteresis_timeline.png",
                    )
                    plt.close(fig1)
                    fig2 = plot_hysteresis_loop(
                        ctx.needle_hysteresis_result,
                        output_path=out_dir / "hysteresis_loop.png",
                    )
                    plt.close(fig2)

                s = ctx.needle_hysteresis_result.summary
                adv = s.get("theta_advancing_deg", "N/A")
                rec = s.get("theta_receding_deg", "N/A")
                cah = s.get("contact_angle_hysteresis_deg", "N/A")
                print(
                    f"[adsa] Needle Hysteresis: theta_A = {adv} deg, "
                    f"theta_R = {rec} deg, CAH = {cah} deg"
                )
                logger.info(
                    f"Needle hysteresis analysis complete. Outputs written to {out_dir}"
                )
                return 0 if ctx.needle_hysteresis_result.accepted else 3

            ctx = runner.run(
                sequence_path=str(source_path),
                sequence_fps=args.fps,
                px_per_mm=args.px_per_mm,
                needle_diameter_mm=needle_diameter_mm,
                analysis_params={"contact_angle_method": "auto_residual"},
            )
            if ctx.dynamic_sessile_result is None:
                raise PipelineError("dynamic_result_missing")
            from menipy.common.temporal_sessile import export_dynamic_result

            export_dynamic_result(ctx.dynamic_sessile_result, out_dir)
            logger.info(f"Dynamic analysis complete. Outputs written to {out_dir}")
            return 0 if ctx.dynamic_sessile_result.accepted else 3
        except (PipelineError, ValueError) as exc:
            logger.error(f"Dynamic pipeline execution error: {exc}")
            return 2

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
        # Remove duplicates and sort naturally
        files_queue = sorted(set(files_queue), key=_natural_key)
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
        px_per_mm = (
            args.px_per_mm
            if args.px_per_mm is not None
            else 100.0 / max(needle_diameter_mm or 0.72, 0.001)
        )
        scale_origin = "manual" if args.px_per_mm is not None else "estimated"
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
                onnx_proposal_mode=args.onnx_proposal_mode,
                segmentation_provider=args.segmentation_provider,
            )

            # Export outputs
            if getattr(ctx, "preview", None) is not None:
                _save_image_bgr(out_dir / "preview.png", ctx.preview)
            if getattr(ctx, "overlay", None) is not None:
                _save_image_bgr(out_dir / "overlay.png", ctx.overlay)

            from menipy.models.calibration import CalibrationProvenance
            from menipy.models.results import build_persisted_analysis

            ctx.calibration_provenance = CalibrationProvenance(
                origin=scale_origin,
                px_per_mm=px_per_mm,
                warnings=[
                    "Estimated scale; physical values withheld. Supply --px-per-mm or needle calibration."
                ]
                if scale_origin == "estimated"
                else [],
            )
            persisted = build_persisted_analysis(ctx)
            results_out = {
                "pipeline": runner.pipeline.name,
                **persisted,
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
    use_tracking = (
        len(files_queue) > 1
        and not getattr(args, "no_temporal_tracking", False)
        and args.pipeline in ("sessile", "pendant")
    )
    tracker = None
    if use_tracking:
        from menipy.common.temporal_tracking import TemporalDropletTracker

        tracker = TemporalDropletTracker(pipeline=args.pipeline)

    locked_roi = manual_roi
    locked_needle = manual_needle
    locked_substrate = manual_contact
    locked_scale = args.px_per_mm
    locked_scale_origin = "manual" if locked_scale is not None else "missing"

    from menipy.common.auto_calibrator import run_auto_calibration
    from menipy.common.detection_helpers import auto_detect_features
    from menipy.models.results import build_persisted_analysis

    for img_path in files_queue:
        logger.info(f"Analyzing: {img_path.name}")
        _patch_acquisition(runner.pipeline, image=img_path, camera=None, frames=1)

        roi_rect = locked_roi
        needle_rect = locked_needle
        substrate_line = locked_substrate
        tracked_contour = None
        tracked_contacts = None
        tracked_apex = None
        is_tracked = False

        # Pre-read image if cv2 is available for tracking / auto-calibration
        img_bgr = None
        if cv2 is not None:
            img_bgr = cv2.imread(str(img_path))

        # Attempt temporal tracking if initialized
        if tracker is not None and tracker.is_tracking and img_bgr is not None:
            track_res = tracker.track_frame(img_bgr, dt=0.033)
            if track_res is not None:
                is_tracked = True
                tracked_contour = track_res.get("drop_contour")
                tracked_contacts = track_res.get("contact_points")
                tracked_apex = track_res.get("apex_point")
                roi_rect = track_res.get("roi_rect") or locked_roi
                substrate_line = track_res.get("substrate_line") or locked_substrate
                needle_rect = track_res.get("needle_rect") or locked_needle
            else:
                logger.debug(
                    f"Tracking quality gate failed for {img_path.name}, falling back to cold start"
                )

        # 4. Auto-Calibration / Feature Detection Fallback
        if not is_tracked and (
            args.auto_calibrate
            or (not roi_rect and not needle_rect)
            or (tracker is not None and not tracker.is_tracking)
        ):
            try:
                first_frame = img_bgr
                if first_frame is None:
                    temp_ctx = Context()
                    temp_ctx = runner.pipeline.do_acquisition(temp_ctx)
                    if temp_ctx.frames:
                        fr = temp_ctx.frames[0]
                        first_frame = fr.image if hasattr(fr, "image") else fr

                if first_frame is not None:
                    cal_res = run_auto_calibration(first_frame, args.pipeline)

                    # Update parameters and lock invariants
                    if not locked_roi and cal_res.roi_rect:
                        locked_roi = cal_res.roi_rect
                    if not locked_needle and cal_res.needle_rect:
                        locked_needle = cal_res.needle_rect
                    if not locked_substrate and cal_res.substrate_line:
                        locked_substrate = cal_res.substrate_line

                    roi_rect = locked_roi
                    needle_rect = locked_needle
                    substrate_line = locked_substrate

                    if tracker is not None:
                        det = auto_detect_features(
                            first_frame,
                            args.pipeline,
                            detect_needle=(locked_needle is None),
                            detect_substrate=(
                                args.pipeline == "sessile" and locked_substrate is None
                            ),
                        )
                        if locked_substrate is not None:
                            det["substrate_line"] = locked_substrate
                        if locked_needle is not None:
                            det["needle_rect"] = locked_needle
                        if "drop_contour" in det and det["drop_contour"] is not None:
                            tracker.initialize(first_frame, det, scale=locked_scale)
                            tracked_contour = det.get("drop_contour")
                            tracked_contacts = det.get("contact_points")
                            tracked_apex = det.get("apex_point")

                    logger.debug(
                        f"Locked Invariants - ROI: {locked_roi}, Needle: {locked_needle}, Substrate: {substrate_line}"
                    )
            except Exception as e:
                logger.warning(f"Failed to auto-calibrate image {img_path.name}: {e}")

        # Calibration computations (default to 0.72mm outer needle if DB lookup and overrides fail)
        target_needle_diam = needle_diameter_mm or 0.72
        if locked_scale is not None:
            px_per_mm = float(locked_scale)
            scale_origin = locked_scale_origin
        elif needle_rect and needle_rect[2] > 0 and needle_diameter_mm:
            px_per_mm = float(needle_rect[2]) / needle_diameter_mm
            locked_scale = px_per_mm
            scale_origin = "manual" if manual_needle else "measured"
            locked_scale_origin = scale_origin
        else:
            px_per_mm = 100.0 / max(target_needle_diam, 0.001)
            scale_origin = "estimated"

        scale_dict = {"px_per_mm": px_per_mm}

        try:
            run_kwargs: dict[str, Any] = {
                "roi": roi_rect,
                "needle_rect": needle_rect,
                "contact_line": substrate_line,
                "substrate_line": substrate_line,
                "image": str(img_path),
                "scale": scale_dict,
                "px_per_mm": px_per_mm,
                "needle_diameter_mm": target_needle_diam,
                "physics": {"rho1": rho1, "rho2": rho2, "g": 9.80665},
                "onnx_proposal_mode": args.onnx_proposal_mode,
                "segmentation_provider": args.segmentation_provider,
            }
            if tracked_contour is not None:
                run_kwargs["drop_contour"] = tracked_contour
            if tracked_contacts is not None:
                run_kwargs["contact_points"] = tracked_contacts
            if tracked_apex is not None:
                run_kwargs["apex_point"] = tracked_apex

            ctx = runner.run(**run_kwargs)

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
            from menipy.models.calibration import CalibrationProvenance

            ctx.calibration_provenance = CalibrationProvenance(
                origin=scale_origin,
                px_per_mm=px_per_mm,
                warnings=[
                    "Estimated scale; physical values withheld. Supply --px-per-mm or needle calibration."
                ]
                if scale_origin == "estimated"
                else [],
            )
            persisted = build_persisted_analysis(ctx)
            if not persisted["accepted"] and tracker is not None:
                tracker.reset()

            results_out = {
                "pipeline": runner.pipeline.name,
                "tracked": is_tracked,
                **persisted,
                "qa": ctx.qa.to_dict() if hasattr(ctx.qa, "to_dict") else ctx.qa,
                "timings_ms": ctx.timings_ms,
                "log": ctx.log,
                "error": ctx.error,
            }
            with open(out_dir / json_name, "w", encoding="utf-8") as f:
                json.dump(results_out, f, indent=2, cls=NumpyEncoder)

            # Extract metrics for consolidated batch CSV
            qa_ok = persisted["accepted"]
            metrics = persisted["results"]

            run_records.append(
                {
                    "image_path": str(img_path),
                    "pipeline": runner.pipeline.name,
                    "qa_ok": qa_ok,
                    "tracked": is_tracked,
                    "rejection_reasons": persisted["rejection_reasons"],
                    "diagnostics": persisted["diagnostics"],
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
        csv_headers = [
            "image_path",
            "pipeline",
            "qa_ok",
            "tracked",
            "rejection_reasons",
            "diagnostics_json",
        ]

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
                        "tracked": rec["tracked"],
                        "rejection_reasons": ";".join(rec["rejection_reasons"]),
                        "diagnostics_json": json.dumps(
                            rec["diagnostics"], cls=NumpyEncoder, separators=(",", ":")
                        ),
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
