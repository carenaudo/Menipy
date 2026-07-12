# MobileSAM model notice

The ONNX graphs in this directory were exported from the official MobileSAM
`vit_t` checkpoint distributed by the MobileSAM project:

- https://github.com/ChaoningZhang/MobileSAM
- source checkpoint SHA-256:
  `6dbb90523a35330fedd7f1d3dfc66f995213d81b29a5ca8108dbcdd4e37d6c2f`

The upstream repository is licensed under Apache-2.0 and includes the
checkpoint in its `weights/` directory. No separate checkpoint-specific
license was located during the Phase-C review. Release packaging must preserve
the upstream notice/license obligations and should be reviewed independently;
the graphs are not included by the current Python package-data configuration.

Menipy uses these graphs only for non-authoritative segmentation proposals and
annotation assistance.
