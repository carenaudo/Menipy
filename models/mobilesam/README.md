# MobileSAM ONNX models

This directory contains the complete MobileSAM inference pipeline split into
two ONNX graphs:

- `mobile_sam_tinyvit_encoder.onnx`: normalized `1x3x1024x1024` RGB image to
  `1x256x64x64` image embeddings.
- `mobile_sam_mask_decoder.onnx`: image embeddings plus point/box prompts to
  masks, quality scores, and low-resolution masks.

The graphs were exported from the official MobileSAM `vit_t` checkpoint using
`scripts/export_mobilesam_onnx.py`, opset 17. Runtime inference needs NumPy,
OpenCV, and the optional `onnxruntime` dependency; it does not import PyTorch,
Ultralytics, MobileSAM, or timm.

`manifest.json` pins provenance, license observations, hashes, graph I/O,
preprocessing, and the proposal-only domain. Menipy verifies it before creating
ONNX Runtime sessions. See `NOTICE.md` for distribution constraints.

```python
import cv2

from menipy.common.mobilesam_onnx import MobileSAMOnnx

image = cv2.imread("data/samples/gota depositada 1.png")
predictor = MobileSAMOnnx()
result = predictor.predict_box(image, (145, 165, 335, 240))
mask = result.best_mask
```

Source checkpoint SHA-256:
`6dbb90523a35330fedd7f1d3dfc66f995213d81b29a5ca8108dbcdd4e37d6c2f`.
