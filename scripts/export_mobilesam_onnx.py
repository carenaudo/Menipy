"""Export the complete MobileSAM pipeline as encoder and decoder ONNX graphs.

This script is a build-time tool. It requires PyTorch, ``mobile-sam``, and ONNX,
but the generated models only require ONNX Runtime at application runtime.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from mobile_sam import sam_model_registry
from mobile_sam.utils.onnx import SamOnnxModel


def export_models(checkpoint: Path, output_dir: Path, opset: int = 17) -> None:
    """Export TinyViT image encoder and prompt-guided mask decoder."""
    output_dir.mkdir(parents=True, exist_ok=True)
    encoder_path = output_dir / "mobile_sam_tinyvit_encoder.onnx"
    decoder_path = output_dir / "mobile_sam_mask_decoder.onnx"

    sam = sam_model_registry["vit_t"](checkpoint=str(checkpoint))
    sam.eval()

    dummy_image = torch.randn(1, 3, 1024, 1024, dtype=torch.float32)
    torch.onnx.export(
        sam.image_encoder,
        dummy_image,
        encoder_path,
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["images"],
        output_names=["image_embeddings"],
        dynamo=False,
    )

    decoder = SamOnnxModel(model=sam, return_single_mask=False)
    embed_dim = sam.prompt_encoder.embed_dim
    embed_size = sam.prompt_encoder.image_embedding_size
    mask_input_size = tuple(4 * value for value in embed_size)
    decoder_inputs = {
        "image_embeddings": torch.randn(
            1, embed_dim, *embed_size, dtype=torch.float32
        ),
        "point_coords": torch.tensor(
            [[[100.0, 100.0], [500.0, 500.0]]], dtype=torch.float32
        ),
        "point_labels": torch.tensor([[2.0, 3.0]], dtype=torch.float32),
        "mask_input": torch.zeros(1, 1, *mask_input_size, dtype=torch.float32),
        "has_mask_input": torch.zeros(1, dtype=torch.float32),
        "orig_im_size": torch.tensor([1024.0, 1024.0], dtype=torch.float32),
    }
    torch.onnx.export(
        decoder,
        tuple(decoder_inputs.values()),
        decoder_path,
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=list(decoder_inputs),
        output_names=["masks", "iou_predictions", "low_res_masks"],
        dynamic_axes={
            "point_coords": {1: "num_points"},
            "point_labels": {1: "num_points"},
        },
        dynamo=False,
    )

    print(f"Wrote {encoder_path}")
    print(f"Wrote {decoder_path}")


def main() -> None:
    """Parse command-line arguments and export both graphs."""
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()
    export_models(args.checkpoint, args.output_dir, args.opset)


if __name__ == "__main__":
    main()
