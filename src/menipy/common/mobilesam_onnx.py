"""ONNX-only MobileSAM inference using separate TinyViT and mask graphs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from menipy.common.segmentation_providers import ModelManifest, ModelManifestError

IMAGE_SIZE = 1024
PIXEL_MEAN = np.asarray([123.675, 116.28, 103.53], dtype=np.float32)
PIXEL_STD = np.asarray([58.395, 57.12, 57.375], dtype=np.float32)
DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[3] / "models" / "mobilesam"


@dataclass(frozen=True)
class MobileSAMResult:
    """Masks and quality scores returned by MobileSAM."""

    masks: np.ndarray
    scores: np.ndarray

    @property
    def best_mask(self) -> np.ndarray:
        """Return the highest-scoring binary mask."""
        index = int(np.argmax(self.scores.reshape(-1)))
        return self.masks.reshape(-1, *self.masks.shape[-2:])[index] > 0.0


class MobileSAMOnnx:
    """Run MobileSAM without importing PyTorch or Ultralytics."""

    def __init__(
        self,
        model_dir: str | Path = DEFAULT_MODEL_DIR,
        *,
        providers: list[str] | None = None,
        verify_manifest: bool = True,
    ) -> None:
        try:
            import onnxruntime as ort  # type: ignore[import-untyped]
        except ImportError as exc:
            raise RuntimeError(
                "MobileSAM ONNX inference requires the optional 'onnxruntime' "
                "dependency. Install Menipy with the 'onnx' extra."
            ) from exc

        directory = Path(model_dir)
        self.manifest = ModelManifest.load(directory / "manifest.json")
        if verify_manifest:
            self.manifest.validate_files(directory)
        encoder = directory / "mobile_sam_tinyvit_encoder.onnx"
        decoder = directory / "mobile_sam_mask_decoder.onnx"
        for path in (encoder, decoder):
            if not path.is_file():
                raise FileNotFoundError(f"Missing MobileSAM ONNX model: {path}")

        selected_providers = providers or ["CPUExecutionProvider"]
        self.encoder: Any = ort.InferenceSession(
            str(encoder), providers=selected_providers
        )
        self.decoder: Any = ort.InferenceSession(
            str(decoder), providers=selected_providers
        )
        self._validate_session_contract(
            self.encoder, self.manifest.files[encoder.name], encoder.name
        )
        self._validate_session_contract(
            self.decoder, self.manifest.files[decoder.name], decoder.name
        )

    @staticmethod
    def _validate_session_contract(
        session: Any, metadata: dict[str, Any], graph_name: str
    ) -> None:
        """Verify ONNX Runtime metadata against the pinned manifest."""
        for direction, nodes in (
            ("inputs", session.get_inputs()),
            ("outputs", session.get_outputs()),
        ):
            expected = metadata.get(direction) or {}
            actual = {node.name: node for node in nodes}
            if set(actual) != set(expected):
                raise ModelManifestError(
                    "model_io_name_mismatch",
                    f"{graph_name} {direction} differ from the manifest",
                )
            for name, contract in expected.items():
                node = actual[name]
                if contract.get("dtype") and str(node.type) != contract["dtype"]:
                    raise ModelManifestError(
                        "model_io_dtype_mismatch",
                        f"{graph_name}:{name} dtype differs from the manifest",
                    )
                expected_shape = contract.get("shape")
                if expected_shape is None:
                    continue
                actual_shape = list(node.shape)
                if len(actual_shape) != len(expected_shape):
                    raise ModelManifestError(
                        "model_io_shape_mismatch",
                        f"{graph_name}:{name} rank differs from the manifest",
                    )
                for observed, wanted in zip(actual_shape, expected_shape):
                    if isinstance(wanted, int) and observed != wanted:
                        raise ModelManifestError(
                            "model_io_shape_mismatch",
                            f"{graph_name}:{name} shape differs from the manifest",
                        )

    @staticmethod
    def _scale(original_size: tuple[int, int]) -> float:
        return IMAGE_SIZE / max(original_size)

    @classmethod
    def preprocess(cls, image_bgr: np.ndarray) -> np.ndarray:
        """Convert an OpenCV BGR image to the normalized TinyViT tensor."""
        if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
            raise ValueError("MobileSAM expects a BGR image with shape HxWx3")
        height, width = image_bgr.shape[:2]
        scale = cls._scale((height, width))
        resized_height = int(height * scale + 0.5)
        resized_width = int(width * scale + 0.5)
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(
            rgb, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR
        ).astype(np.float32)
        normalized = (resized - PIXEL_MEAN) / PIXEL_STD
        padded = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.float32)
        padded[:resized_height, :resized_width] = normalized
        return np.transpose(padded, (2, 0, 1))[None]

    def encode(self, image_bgr: np.ndarray) -> np.ndarray:
        """Create reusable TinyViT image embeddings."""
        tensor = self.preprocess(image_bgr)
        return self.encoder.run(None, {"images": tensor})[0]

    def predict_box(
        self,
        image_bgr: np.ndarray,
        box: tuple[float, float, float, float],
        *,
        embeddings: np.ndarray | None = None,
    ) -> MobileSAMResult:
        """Segment the object inside an XYXY box in original image pixels."""
        return self.predict_prompt(
            image_bgr, box, embeddings=embeddings
        )

    def predict_prompt(
        self,
        image_bgr: np.ndarray,
        box: tuple[float, float, float, float],
        *,
        positive_points: tuple[tuple[float, float], ...] = (),
        negative_points: tuple[tuple[float, float], ...] = (),
        embeddings: np.ndarray | None = None,
    ) -> MobileSAMResult:
        """Segment from a box plus optional positive/negative point prompts."""
        height, width = image_bgr.shape[:2]
        scale = self._scale((height, width))
        coords_list = np.asarray(box, dtype=np.float32).reshape(2, 2).tolist()
        labels = [2.0, 3.0]
        for point in positive_points:
            coords_list.append([float(point[0]), float(point[1])])
            labels.append(1.0)
        for point in negative_points:
            coords_list.append([float(point[0]), float(point[1])])
            labels.append(0.0)
        coords = np.asarray([coords_list], dtype=np.float32) * scale
        encoded = embeddings if embeddings is not None else self.encode(image_bgr)
        masks, scores, _ = self.decoder.run(
            None,
            {
                "image_embeddings": encoded.astype(np.float32, copy=False),
                "point_coords": coords,
                "point_labels": np.asarray([labels], dtype=np.float32),
                "mask_input": np.zeros((1, 1, 256, 256), dtype=np.float32),
                "has_mask_input": np.zeros((1,), dtype=np.float32),
                "orig_im_size": np.asarray([height, width], dtype=np.float32),
            },
        )
        return MobileSAMResult(masks=masks, scores=scores)


__all__ = [
    "MobileSAMOnnx",
    "MobileSAMResult",
    "DEFAULT_MODEL_DIR",
    "ModelManifestError",
]
