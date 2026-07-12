# ONNX segmentation spike: YOLO26 and MobileSAM

Date: 2026-07-10

## Goal

Evaluate lightweight segmentation for droplet, needle, and substrate detection
without adding PyTorch to Menipy's runtime distribution.

## Result

### YOLO26n-seg

The official `yolo26n-seg.pt` checkpoint was exported to ONNX at 640 px with
the non-end-to-end output layout. The graph has these outputs:

- predictions: `(1, 116, 8400)`
- mask prototypes: `(1, 32, 160, 160)`

The ONNX graph passed `onnx.checker.check_model` and ran with ONNX Runtime
1.27.0 on CPU.

The pretrained checkpoint is a negative control, not a usable Menipy detector.
It is trained on COCO's 80 categories rather than Menipy's domain classes:

| Image | Result at confidence 0.10 | CPU inference |
| --- | --- | ---: |
| Sessile sample | needle classified as COCO class 9 (`traffic light`) | 225 ms |
| Pendant sample | drop classified as COCO class 64 (`potted plant`) | 499 ms |

Testing the larger `s`, `m`, `l`, or `x` COCO variants would not fix the class
mismatch. The useful YOLO experiment is to fine-tune `yolo26n-seg` with custom
classes and then export the resulting checkpoint to ONNX.

Recommended custom classes:

```yaml
names:
  0: droplet
  1: needle
  2: substrate
```

The substrate should be annotated as a narrow band around the visible solid
surface, not as the entire region below the contact line. This makes the top
mask boundary a stable substrate-line estimate.

### MobileSAM

The official MobileSAM checkpoint was tested with box prompts derived from the
current Menipy heuristics:

| Image | Box `[x1, y1, x2, y2]` | Result | CPU inference |
| --- | --- | --- | ---: |
| Sessile sample | `[145, 165, 335, 240]` | clean droplet mask | 4.71 s |
| Pendant sample | `[500, 105, 930, 650]` | clean droplet mask | 3.86 s |

MobileSAM does not accept natural-language prompts. Use geometric prompts:

- droplet: one tight box around the heuristic contour, expanded by 5-10%;
- needle: one narrow box around the top-connected component;
- substrate: MobileSAM is not the preferred detector; retain the gradient or
  Hough line detector, because the desired output is a line rather than an
  object mask;
- optional refinement: add one positive point inside the target and negative
  points in nearby needle/substrate regions when the box contains multiple
  touching objects.

The official MobileSAM ONNX exporter only exports the prompt encoder and mask
decoder. Menipy now completes that deployment with a separately exported
TinyViT image encoder. Both verified graphs live in `models/mobilesam/`, and
`src/menipy/common/mobilesam_onnx.py` runs the complete pipeline using ONNX
Runtime without importing PyTorch, Ultralytics, MobileSAM, or timm.

## Recommended Menipy design

1. Keep the existing substrate line detector as the default.
2. Use MobileSAM as an annotation/refinement tool while building the dataset.
3. Fine-tune `yolo26n-seg` for `droplet`, `needle`, and `substrate`.
4. Export training output in a separate environment; ship only the `.onnx`
   model and an optional `onnxruntime` dependency in Menipy.
5. Add the ONNX implementation through the existing `DROP_DETECTORS`,
   `NEEDLE_DETECTORS`, and `SUBSTRATE_DETECTORS` registries, with the current
   classical detectors as fallbacks.

## Annotation prompt for a human or annotation agent

> For each image, create instance-segmentation polygons for `droplet`,
> `needle`, and `substrate`. Trace the visible outer boundary precisely. Do not
> include highlights, shadows, reflections, or background inside a mask. For a
> sessile droplet, stop the droplet polygon at the liquid-solid contact line.
> For a pendant droplet, separate the droplet from the needle at their narrowest
> physical junction. Annotate the substrate as a thin continuous band whose top
> edge follows the visible contact surface. Mark ambiguous or occluded images
> for review instead of guessing.

## Runtime boundary

PyTorch and Ultralytics were installed only in an ignored temporary conversion
environment. No project dependency was changed. The intended production
runtime is OpenCV, NumPy, and ONNX Runtime only.
