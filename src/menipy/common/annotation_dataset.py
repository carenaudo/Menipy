"""COCO-compatible ADSA proposal generation and YOLO dataset preparation."""

from __future__ import annotations

import hashlib
import html
import json
import re
from importlib import import_module
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

from menipy.common.auto_calibrator import AutoCalibrator
from menipy.common.registry import SEGMENTATION_PROVIDERS
from menipy.common.segmentation_providers import SegmentationPrompt, expand_box

CATEGORIES: tuple[dict[str, Any], ...] = (
    {"id": 1, "name": "droplet", "supercategory": "adsa"},
    {"id": 2, "name": "needle", "supercategory": "adsa"},
    {"id": 3, "name": "substrate_band", "supercategory": "adsa"},
)


def _license_is_approved(name: str) -> bool:
    normalized = name.strip().lower()
    return bool(normalized) and not any(
        marker in normalized
        for marker in ("unresolved", "evaluation-only", "unknown", "do-not-train")
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _series_id(path: Path) -> str:
    stem = re.sub(r"[\s_-]*\d+$", "", path.stem.lower()).strip(" _-")
    return stem or path.stem.lower()


def _split_for_series(series: str) -> str:
    bucket = int(hashlib.sha256(series.encode("utf-8")).hexdigest()[:8], 16) % 10
    if bucket < 8:
        return "train"
    return "validation" if bucket == 8 else "test"


def collect_images(input_path: str | Path) -> list[Path]:
    """Collect supported images in stable order from a file or directory."""
    path = Path(input_path)
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"Annotation input does not exist: {path}")
    suffixes = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    return sorted(
        (item for item in path.rglob("*") if item.suffix.lower() in suffixes),
        key=lambda item: str(item).lower(),
    )


def _prompts_from_calibration(
    image: np.ndarray, calibration: Any
) -> tuple[list[SegmentationPrompt], list[str]]:
    prompts: list[SegmentationPrompt] = []
    ambiguities: list[str] = []
    contour = getattr(calibration, "drop_contour", None)
    if contour is not None:
        xy = np.asarray(contour, dtype=float).reshape(-1, 2)
        x, y, width, height = cv2.boundingRect(
            np.rint(xy).astype(np.int32).reshape(-1, 1, 2)
        )
        prompts.append(
            SegmentationPrompt(
                "droplet",
                expand_box((x, y, x + width - 1, y + height - 1), image.shape),
            )
        )
    else:
        ambiguities.append("drop_prompt_unavailable")
    needle = getattr(calibration, "needle_rect", None)
    if needle is not None:
        x, y, width, height = needle
        prompts.append(
            SegmentationPrompt(
                "needle",
                expand_box((x, y, x + width - 1, y + height - 1), image.shape),
            )
        )
    else:
        ambiguities.append("needle_prompt_unavailable")
    return prompts, ambiguities


def _polygon_annotation(
    annotation_id: int,
    image_id: int,
    category_id: int,
    contour: np.ndarray,
    *,
    score: float | None,
    provider: str,
) -> dict[str, Any]:
    xy = np.asarray(contour, dtype=float).reshape(-1, 2)
    x, y, width, height = cv2.boundingRect(
        np.rint(xy).astype(np.int32).reshape(-1, 1, 2)
    )
    return {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": category_id,
        "segmentation": [[float(value) for value in xy.reshape(-1)]],
        "area": float(abs(cv2.contourArea(xy.astype(np.float32)))),
        "bbox": [float(x), float(y), float(width), float(height)],
        "iscrowd": 0,
        "score": None if score is None else float(score),
        "provider": provider,
        "review_status": "proposed",
        "reviewer": None,
        "reviewed_at": None,
    }


def _substrate_annotation(
    annotation_id: int,
    image_id: int,
    line: tuple[tuple[float, float], tuple[float, float]],
) -> dict[str, Any]:
    p1, p2 = np.asarray(line[0], dtype=float), np.asarray(line[1], dtype=float)
    vec = p2 - p1
    length = float(np.linalg.norm(vec))
    normal = np.array([-vec[1], vec[0]], dtype=float) / max(length, 1e-9) * 2.0
    polygon = np.vstack([p1 - normal, p2 - normal, p2 + normal, p1 + normal])
    return _polygon_annotation(
        annotation_id,
        image_id,
        3,
        polygon,
        score=None,
        provider="classical_substrate_line",
    )


def _draw_overlay(
    image: np.ndarray, annotations: list[dict[str, Any]], categories: dict[int, str]
) -> np.ndarray:
    overlay = image.copy()
    colors = {1: (40, 220, 40), 2: (255, 160, 40), 3: (40, 220, 255)}
    for annotation in annotations:
        segment = annotation.get("segmentation", [[]])[0]
        if len(segment) < 6:
            continue
        contour = np.rint(np.asarray(segment).reshape(-1, 2)).astype(np.int32)
        color = colors.get(int(annotation["category_id"]), (255, 255, 255))
        cv2.polylines(overlay, [contour], True, color, 2)
        point = tuple(int(value) for value in contour[0])
        cv2.putText(
            overlay,
            categories[int(annotation["category_id"])],
            point,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
    return overlay


def build_readiness_report(coco: dict[str, Any]) -> dict[str, Any]:
    """Summarize proposed/approved counts without promoting proposals."""
    categories = {int(item["id"]): item["name"] for item in coco["categories"]}
    by_status: dict[str, int] = {}
    by_class: dict[str, dict[str, int]] = {}
    for annotation in coco["annotations"]:
        status = str(annotation.get("review_status", "proposed"))
        name = categories[int(annotation["category_id"])]
        by_status[status] = by_status.get(status, 0) + 1
        by_class.setdefault(name, {})[status] = (
            by_class.setdefault(name, {}).get(status, 0) + 1
        )
    by_split: dict[str, int] = {}
    by_pipeline: dict[str, int] = {}
    for image in coco["images"]:
        split = str(image["split"])
        pipeline = str(image["pipeline"])
        by_split[split] = by_split.get(split, 0) + 1
        by_pipeline[pipeline] = by_pipeline.get(pipeline, 0) + 1
    approved = by_status.get("approved", 0)
    license_names = {
        int(item["id"]): str(item.get("name", "")) for item in coco.get("licenses", [])
    }
    licenses_ok = all(
        _license_is_approved(license_names.get(int(image.get("license_id", 0)), ""))
        for image in coco["images"]
    )
    all_reviewed = approved > 0 and approved == len(coco["annotations"])
    blocking_reasons: list[str] = []
    if not all_reviewed:
        blocking_reasons.append("unreviewed_annotations_present")
    if not licenses_ok:
        blocking_reasons.append("unapproved_source_licenses")
    return {
        "schema_version": "1.0",
        "training_ready": all_reviewed and licenses_ok,
        "images": len(coco["images"]),
        "annotations": len(coco["annotations"]),
        "skipped_images": len((coco.get("menipy") or {}).get("skipped", [])),
        "by_status": by_status,
        "by_class": by_class,
        "by_split": by_split,
        "by_pipeline": by_pipeline,
        "blocking_reasons": blocking_reasons,
    }


def annotate_images(
    paths: Iterable[Path],
    *,
    pipeline: str,
    output_dir: str | Path,
    provider_name: str = "mobilesam",
    provider: Any | None = None,
    license_id: str = "unresolved-evaluation-only",
) -> dict[str, Any]:
    """Generate proposed COCO polygons, overlays, viewer, and readiness report."""
    output = Path(output_dir)
    overlays_dir = output / "overlays"
    overlays_dir.mkdir(parents=True, exist_ok=True)
    if provider is None:
        if provider_name == "mobilesam" and provider_name not in SEGMENTATION_PROVIDERS:
            import_module("menipy.common.mobilesam_provider")
        factory = SEGMENTATION_PROVIDERS.get(provider_name)
        if factory is None:
            raise RuntimeError("segmentation_provider_not_registered")
        provider = factory()

    annotations_path = output / "annotations.proposed.json"
    if annotations_path.is_file():
        coco = json.loads(annotations_path.read_text(encoding="utf-8"))
        if str((coco.get("info") or {}).get("version")) != "1.0":
            raise ValueError("annotation_resume_schema_mismatch")
        coco.setdefault("menipy", {}).setdefault("skipped", [])
    else:
        coco = {
            "info": {
                "description": "Menipy ADSA segmentation proposals",
                "version": "1.0",
                "review_status": "proposed",
            },
            "licenses": [{"id": 1, "name": license_id}],
            "categories": list(CATEGORIES),
            "images": [],
            "annotations": [],
            "menipy": {"skipped": []},
        }
    category_ids: dict[str, int] = {
        str(item["name"]): int(item["id"]) for item in CATEGORIES
    }
    next_annotation = max(
        (int(item["id"]) for item in coco["annotations"]), default=0
    ) + 1
    viewer_rows: list[str] = [
        "<figure><img src='overlays/{}' loading='lazy'><figcaption>{}</figcaption></figure>".format(
            html.escape(str(item.get("overlay_file", ""))),
            html.escape(str(item.get("file_name", ""))),
        )
        for item in coco["images"]
        if item.get("overlay_file")
    ]
    seen_hashes: set[str] = {str(item["sha256"]) for item in coco["images"]}
    for image_path in sorted((Path(item) for item in paths), key=lambda item: str(item).lower()):
        digest = _sha256(image_path)
        if digest in seen_hashes:
            continue
        seen_hashes.add(digest)
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            coco["menipy"]["skipped"].append(
                {"file_name": str(image_path), "reason": "image_decode_failed"}
            )
            continue
        image_id = len(coco["images"]) + 1
        series = _series_id(image_path)
        image_entry: dict[str, Any] = {
            "id": image_id,
            "file_name": str(image_path),
            "width": int(image.shape[1]),
            "height": int(image.shape[0]),
            "sha256": digest,
            "source": str(image_path),
            "license_id": 1,
            "pipeline": pipeline,
            "series_id": series,
            "split": _split_for_series(series),
            "ambiguity_flags": [],
            "overlay_file": f"{image_path.stem}_{digest[:10]}.png",
        }
        calibration = AutoCalibrator(image, pipeline).detect_all()
        prompts, ambiguities = _prompts_from_calibration(image, calibration)
        image_entry["ambiguity_flags"].extend(ambiguities)
        image_annotations: list[dict[str, Any]] = []
        if prompts:
            proposals = provider.segment(image, prompts)
            for proposal in proposals:
                if not proposal.accepted or proposal.contour is None:
                    image_entry["ambiguity_flags"].extend(proposal.rejection_reasons)
                    continue
                annotation = _polygon_annotation(
                    next_annotation,
                    image_id,
                    category_ids[proposal.feature],
                    proposal.contour,
                    score=proposal.score,
                    provider=provider_name,
                )
                next_annotation += 1
                image_annotations.append(annotation)
        substrate = getattr(calibration, "substrate_line", None)
        if pipeline == "sessile" and substrate is not None:
            image_annotations.append(
                _substrate_annotation(next_annotation, image_id, substrate)
            )
            next_annotation += 1
        coco["images"].append(image_entry)
        coco["annotations"].extend(image_annotations)
        overlay_name = str(image_entry["overlay_file"])
        overlay = _draw_overlay(
            image,
            image_annotations,
            {int(item["id"]): str(item["name"]) for item in CATEGORIES},
        )
        cv2.imwrite(str(overlays_dir / overlay_name), overlay)
        viewer_rows.append(
            "<figure><img src='overlays/{}' loading='lazy'><figcaption>{}<br>{}</figcaption></figure>".format(
                html.escape(overlay_name),
                html.escape(str(image_path)),
                html.escape(", ".join(image_entry["ambiguity_flags"]) or "proposal generated"),
            )
        )

    if not coco["images"]:
        raise ValueError("no_decodable_annotation_images")
    output.mkdir(parents=True, exist_ok=True)
    (output / "annotations.proposed.json").write_text(
        json.dumps(coco, indent=2, sort_keys=True), encoding="utf-8"
    )
    readiness = build_readiness_report(coco)
    (output / "readiness.json").write_text(
        json.dumps(readiness, indent=2, sort_keys=True), encoding="utf-8"
    )
    viewer = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>Menipy annotation proposals</title><style>"
        "body{font-family:sans-serif;background:#181a1b;color:#eee}"
        "main{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:16px}"
        "figure{margin:0;padding:12px;background:#242729}img{max-width:100%;height:auto}"
        "figcaption{word-break:break-all}</style></head><body>"
        "<h1>Proposed annotations — review required</h1><main>"
        + "".join(viewer_rows)
        + "</main></body></html>"
    )
    (output / "index.html").write_text(viewer, encoding="utf-8")
    return coco


def convert_coco_to_yolo(
    coco_path: str | Path,
    output_dir: str | Path,
    *,
    approved_only: bool = True,
) -> dict[str, Any]:
    """Convert reviewed COCO polygons to deterministic YOLO segmentation labels."""
    coco = json.loads(Path(coco_path).read_text(encoding="utf-8"))
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    categories = {
        int(item["id"]): index
        for index, item in enumerate(sorted(coco["categories"], key=lambda item: item["id"]))
    }
    images = {int(item["id"]): item for item in coco["images"]}
    license_names = {
        int(item["id"]): str(item.get("name", "")) for item in coco.get("licenses", [])
    }
    annotations_by_image: dict[int, list[dict[str, Any]]] = {}
    for annotation in coco["annotations"]:
        if approved_only and annotation.get("review_status") != "approved":
            continue
        image = images.get(int(annotation["image_id"]), {})
        if approved_only and not _license_is_approved(
            license_names.get(int(image.get("license_id", 0)), "")
        ):
            continue
        annotations_by_image.setdefault(int(annotation["image_id"]), []).append(annotation)
    written = 0
    for image_id, image in sorted(images.items()):
        lines: list[str] = []
        width, height = float(image["width"]), float(image["height"])
        for annotation in sorted(annotations_by_image.get(image_id, []), key=lambda item: item["id"]):
            polygon = annotation.get("segmentation", [[]])[0]
            xy = np.asarray(polygon, dtype=float).reshape(-1, 2)
            normalized = xy / np.asarray([width, height])
            coords = " ".join(f"{float(value):.8f}" for value in normalized.reshape(-1))
            lines.append(f"{categories[int(annotation['category_id'])]} {coords}")
        if lines:
            name = f"{Path(image['file_name']).stem}_{str(image['sha256'])[:10]}.txt"
            (output / name).write_text("\n".join(lines) + "\n", encoding="utf-8")
            written += 1
    summary = {
        "schema_version": "1.0",
        "approved_only": approved_only,
        "label_files": written,
        "classes": [item["name"] for item in sorted(coco["categories"], key=lambda item: item["id"])],
    }
    (output / "conversion.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    return summary


__all__ = [
    "CATEGORIES",
    "annotate_images",
    "build_readiness_report",
    "collect_images",
    "convert_coco_to_yolo",
]
