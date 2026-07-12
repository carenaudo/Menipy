# ONNX Segmentation Proposal Contract

Schema version: `1.0`.

ONNX segmentation providers are non-authoritative proposal generators. They
must not replace calibrated contours, needle geometry, substrate lines,
acceptance state, or physical results unless a future contract explicitly
promotes a provider after scientific validation.

## Model manifest

Every provider model directory contains a `manifest.json` with a stable ID,
revision, source URLs, code and weight license observations, distribution
status, opset, preprocessing revision, supported domain, classes, and per-file
SHA-256/size/I/O contracts. Menipy verifies these fields before creating an
ONNX Runtime session and fails closed with a stable error code.

## Runtime types

- `SegmentationPrompt`: feature plus XYXY box and optional positive/negative
  points in original pixels.
- `SegmentationProposal`: binary mask, ordered contour, model score, topology
  metrics, provenance, acceptance, and rejection reasons.
- `SEGMENTATION_PROVIDERS`: registry of factories implementing
  `segment(image_bgr, prompts)`.

Proposal acceptance only means that the mask passed topology gates. It is not
scientific acceptance of an ADSA measurement.

## Shadow behavior

`onnx_proposal_mode="off"` is the default and must not load ONNX Runtime.
`shadow` writes JSON-safe metadata to `diagnostics.onnx_proposals` while leaving
all promoted pipeline fields unchanged. MobileSAM supports `droplet` and
`needle`; substrate segmentation is intentionally excluded because Menipy
requires a calibrated line geometry.

