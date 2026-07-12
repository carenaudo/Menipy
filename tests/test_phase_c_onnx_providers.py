"""Phase-C integrity, topology, shadow, and annotation gates."""

from __future__ import annotations

import builtins
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from menipy.cli import main as cli_main
from menipy.common.annotation_dataset import annotate_images, convert_coco_to_yolo
from menipy.common.mobilesam_onnx import DEFAULT_MODEL_DIR, MobileSAMOnnx
from menipy.common.onnx_shadow import run_shadow_segmentation
from menipy.common.plugins import _register_from_module
from menipy.common.registry import SEGMENTATION_PROVIDERS
from menipy.common.segmentation_providers import (
    ModelManifest,
    ModelManifestError,
    SegmentationPrompt,
    SegmentationProposal,
    mask_to_proposal,
)
from menipy.common.validation import build_diagnostics
from menipy.models.context import Context


def _write_manifest(directory: Path, payload: bytes, *, digest: str | None = None) -> Path:
    graph = directory / "model.onnx"
    graph.write_bytes(payload)
    manifest = {
        "schema_version": "1.0",
        "id": "test-model",
        "revision": "test-revision",
        "source_urls": ["https://example.invalid/model"],
        "licenses": {"code": "MIT", "weights": "MIT"},
        "distribution": "test-only",
        "opset": 17,
        "preprocessing_revision": "test-v1",
        "supported_domain": "tests",
        "classes": ["droplet"],
        "files": {
            graph.name: {
                "bytes": len(payload),
                "sha256": digest or hashlib.sha256(payload).hexdigest(),
            }
        },
    }
    path = directory / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_model_manifest_fails_closed_on_missing_fields_and_hash(tmp_path: Path) -> None:
    path = _write_manifest(tmp_path, b"onnx-test")
    manifest = ModelManifest.load(path)
    manifest.validate_files(tmp_path)
    bad_path = _write_manifest(tmp_path, b"changed", digest="0" * 64)
    with pytest.raises(ModelManifestError) as error:
        ModelManifest.load(bad_path).validate_files(tmp_path)
    assert error.value.code == "model_file_hash_mismatch"
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(ModelManifestError) as error:
        ModelManifest.load(path)
    assert error.value.code == "model_manifest_missing_fields"


def test_mobilesam_provider_is_registered_and_manifest_has_distribution_gate() -> None:
    importlib.import_module("menipy.common.mobilesam_provider")
    assert "mobilesam" in SEGMENTATION_PROVIDERS
    manifest = ModelManifest.load(DEFAULT_MODEL_DIR / "manifest.json")
    assert manifest.code_license == "Apache-2.0"
    assert "release packaging" in manifest.distribution


def test_external_segmentation_provider_protocol_registers() -> None:
    def sentinel():
        return None

    previous = SEGMENTATION_PROVIDERS.get("phase-c-test")
    try:
        _register_from_module(
            SimpleNamespace(SEGMENTATION_PROVIDERS={"phase-c-test": sentinel})
        )
        assert SEGMENTATION_PROVIDERS.get("phase-c-test") is sentinel
    finally:
        if previous is None:
            SEGMENTATION_PROVIDERS._items.pop("phase-c-test", None)
        else:
            SEGMENTATION_PROVIDERS["phase-c-test"] = previous


def test_mobilesam_reports_optional_runtime_dependency(monkeypatch) -> None:
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "onnxruntime":
            raise ImportError("blocked for test")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    with pytest.raises(RuntimeError, match="optional 'onnxruntime'"):
        MobileSAMOnnx(DEFAULT_MODEL_DIR)


def test_mask_to_proposal_accepts_clean_mask_and_rejects_topology() -> None:
    prompt = SegmentationPrompt("droplet", (20.0, 20.0, 100.0, 100.0))
    mask = np.zeros((128, 128), dtype=np.uint8)
    cv2.circle(mask, (60, 60), 30, 1, -1)
    accepted = mask_to_proposal(mask, score=0.95, prompt=prompt, provider="fake")
    assert accepted.accepted
    assert accepted.contour is not None
    multiple = mask.copy()
    cv2.circle(multiple, (108, 108), 18, 1, -1)
    rejected = mask_to_proposal(
        multiple, score=0.95, prompt=prompt, provider="fake"
    )
    assert not rejected.accepted
    assert "proposal_multiple_components" in rejected.rejection_reasons
    empty = mask_to_proposal(
        np.zeros_like(mask), score=0.95, prompt=prompt, provider="fake"
    )
    assert "proposal_mask_empty" in empty.rejection_reasons


class _FakeProvider:
    def segment(self, image: np.ndarray, prompts: list[SegmentationPrompt]):
        proposals = []
        for prompt in prompts:
            mask = np.zeros(image.shape[:2], dtype=bool)
            x1, y1, x2, y2 = (int(round(value)) for value in prompt.box_xyxy)
            mask[y1 : y2 + 1, x1 : x2 + 1] = True
            contour = np.asarray(
                [[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=float
            )
            proposals.append(
                SegmentationProposal(
                    feature=prompt.feature,
                    provider="fake",
                    mask=mask,
                    contour=contour,
                    score=0.99,
                    accepted=True,
                )
            )
        return proposals


def test_shadow_is_off_by_default_and_never_promotes_fields() -> None:
    image = np.full((160, 200, 3), 220, dtype=np.uint8)
    contour = np.asarray([[50, 120], [70, 70], [130, 70], [150, 120]], dtype=float)
    ctx = Context(
        image=image,
        drop_contour=contour,
        needle_rect=(90, 0, 20, 60),
        results={"contact_angle_deg": 91.2},
    )
    before_results = dict(ctx.results)
    before_contour = np.asarray(ctx.drop_contour).copy()
    run_shadow_segmentation(ctx, "sessile")
    assert ctx.onnx_proposals == {}
    assert ctx.results == before_results

    original = SEGMENTATION_PROVIDERS.get("fake")
    SEGMENTATION_PROVIDERS["fake"] = _FakeProvider
    try:
        ctx.onnx_proposal_mode = "shadow"
        ctx.segmentation_provider = "fake"
        run_shadow_segmentation(ctx, "sessile")
    finally:
        if original is None:
            SEGMENTATION_PROVIDERS._items.pop("fake", None)
        else:
            SEGMENTATION_PROVIDERS["fake"] = original
    assert ctx.onnx_proposals["mode"] == "shadow"
    assert set(ctx.onnx_proposals["proposals"]) == {"droplet", "needle"}
    assert ctx.results == before_results
    assert np.array_equal(ctx.drop_contour, before_contour)
    assert ctx.needle_rect == (90, 0, 20, 60)
    diagnostics = build_diagnostics(ctx)
    assert diagnostics["onnx_proposals"]["mode"] == "shadow"


def test_annotation_outputs_are_proposed_deterministic_and_not_training_ready(
    tmp_path: Path,
) -> None:
    image = np.full((240, 320, 3), 210, dtype=np.uint8)
    cv2.rectangle(image, (145, 0), (175, 100), (45, 45, 45), -1)
    cv2.ellipse(image, (160, 170), (65, 55), 0, 0, 360, (45, 45, 45), -1)
    cv2.rectangle(image, (0, 215), (319, 239), (45, 45, 45), -1)
    source = tmp_path / "series_001.png"
    cv2.imwrite(str(source), image)
    output = tmp_path / "out"
    coco = annotate_images(
        [source, source],
        pipeline="sessile",
        output_dir=output,
        provider_name="fake",
        provider=_FakeProvider(),
    )
    assert len(coco["images"]) == 1
    assert coco["annotations"]
    assert all(item["review_status"] == "proposed" for item in coco["annotations"])
    assert (output / "index.html").is_file()
    readiness = json.loads((output / "readiness.json").read_text(encoding="utf-8"))
    assert readiness["training_ready"] is False
    assert readiness["blocking_reasons"] == [
        "unreviewed_annotations_present",
        "unapproved_source_licenses",
    ]
    empty_yolo = convert_coco_to_yolo(
        output / "annotations.proposed.json", tmp_path / "yolo"
    )
    assert empty_yolo["label_files"] == 0
    approved = json.loads(json.dumps(coco))
    approved["licenses"][0]["name"] = "test-approved"
    for annotation in approved["annotations"]:
        annotation["review_status"] = "approved"
        annotation["reviewer"] = "test-reviewer"
        annotation["reviewed_at"] = "2026-07-11"
    approved_path = tmp_path / "approved.json"
    approved_path.write_text(json.dumps(approved), encoding="utf-8")
    converted = convert_coco_to_yolo(approved_path, tmp_path / "approved-yolo")
    assert converted["label_files"] == 1
    resumed = annotate_images(
        [source],
        pipeline="sessile",
        output_dir=output,
        provider_name="fake",
        provider=_FakeProvider(),
    )
    assert len(resumed["images"]) == 1
    assert len(resumed["annotations"]) == len(coco["annotations"])


def test_cli_annotation_missing_input_is_rejected(tmp_path: Path) -> None:
    rc = cli_main(
        [
            "annotate",
            "--input",
            str(tmp_path / "missing"),
            "--output",
            str(tmp_path / "out"),
        ]
    )
    assert rc == 2
    assert not (tmp_path / "out").exists()


def test_corrupt_annotation_image_is_rejected(tmp_path: Path) -> None:
    corrupt = tmp_path / "broken.png"
    corrupt.write_bytes(b"not-an-image")
    with pytest.raises(ValueError, match="no_decodable_annotation_images"):
        annotate_images(
            [corrupt],
            pipeline="sessile",
            output_dir=tmp_path / "out",
            provider_name="fake",
            provider=_FakeProvider(),
        )


@pytest.mark.skipif(
    importlib.util.find_spec("onnxruntime") is None,
    reason="onnxruntime is optional",
)
def test_mobilesam_cpu_decoder_is_deterministic() -> None:
    image = cv2.imread("data/samples/gota depositada 1.png")
    assert image is not None
    runtime = MobileSAMOnnx(DEFAULT_MODEL_DIR)
    embeddings = runtime.encode(image)
    first = runtime.predict_box(image, (145, 165, 335, 240), embeddings=embeddings)
    second = runtime.predict_box(image, (145, 165, 335, 240), embeddings=embeddings)
    assert np.array_equal(first.masks > 0, second.masks > 0)
    assert np.allclose(first.scores, second.scores, atol=1e-6, rtol=0)
