# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from functools import partial
from pathlib import Path
import urllib.request
import torch

from .modeling import (
    ImageEncoderViT,
    MaskDecoder,
    PromptEncoder,
    Sam,
    TwoWayTransformer,
)

from .modeling.image_encoder_swin import SwinTransformer

from monai.utils import ensure_tuple_rep, optional_import

def build_sam_vit_h(checkpoint=None, image_size=1024):
    return _build_sam(
        encoder_embed_dim=1280,
        encoder_depth=32,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[7, 15, 23, 31],
        checkpoint=checkpoint,
        image_size=image_size,
    )


build_sam = build_sam_vit_h


def build_sam_vit_l(checkpoint=None, image_size=1024):
    return _build_sam(
        encoder_embed_dim=1024,
        encoder_depth=24,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[5, 11, 17, 23],
        checkpoint=checkpoint,
        image_size=image_size,
    )


def build_sam_vit_b(checkpoint=None, image_size=1024):
    return _build_sam(
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        checkpoint=checkpoint,
        image_size=image_size,
    )
"""
Examples::
            # for 3D single channel input with size (96,96,96), 4-channel output and feature size of 48.
            >>> net = SwinUNETR(img_size=(96,96,96), in_channels=1, out_channels=4, feature_size=48)
            # for 3D 4-channel input with size (128,128,128), 3-channel output and (2,4,2,2) layers in each stage.
            >>> net = SwinUNETR(img_size=(128,128,128), in_channels=4, out_channels=3, depths=(2,4,2,2))
            # for 2D single channel input with size (96,96), 2-channel output and gradient checkpointing.
            >>> net = SwinUNETR(img_size=(96,96), in_channels=3, out_channels=2, use_checkpoint=True, spatial_dims=2)
"""

def build_sam_vit_swin(checkpoint=None, image_size=96):
    print('==> build_sam_vit_swin')
    return _build_sam(
        encoder_embed_dim=48,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        checkpoint=checkpoint,
        image_size=image_size,
    )

sam_model_registry = {
    "default": build_sam_vit_h,
    "vit_h": build_sam_vit_h,
    "vit_l": build_sam_vit_l,
    "vit_b": build_sam_vit_b,
    "swin_vit": build_sam_vit_swin,
}


def _build_sam(
    encoder_embed_dim,
    encoder_depth,
    encoder_num_heads,
    encoder_global_attn_indexes,
    checkpoint=None,
    image_size=None,
    spatial_dims=3,
):
    prompt_embed_dim = 768
    patch_size = ensure_tuple_rep(2, spatial_dims)
    window_size = ensure_tuple_rep(7, spatial_dims)
    image_embedding_size = [size // 32 for size in image_size]
    sam = Sam(
        image_encoder=SwinTransformer(
            in_chans=1,
            embed_dim=encoder_embed_dim,
            window_size=window_size,
            patch_size=patch_size,
            depths=(2, 2, 6, 2), #(2, 2, 6, 2),
            num_heads=(3, 6, 12, 24),
            mlp_ratio=4.0,
            qkv_bias=True,
            spatial_dims=spatial_dims,
        ),
        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=image_embedding_size,
            input_image_size=image_size,
            mask_in_chans=16,
        ),
        mask_decoder=MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        ),
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    )
    sam.eval()
    if checkpoint is not None:
        checkpoint = Path(checkpoint)
        if checkpoint.name == "sam_vit_b_01ec64.pth" and not checkpoint.exists():
            cmd = input("Download sam_vit_b_01ec64.pth from facebook AI? [y]/n: ")
            if len(cmd) == 0 or cmd.lower() == 'y':
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                print("Downloading SAM ViT-B checkpoint...")
                urllib.request.urlretrieve(
                    "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
                    checkpoint,
                )
                print(checkpoint.name, " is downloaded!")
        elif checkpoint.name == "sam_vit_h_4b8939.pth" and not checkpoint.exists():
            cmd = input("Download sam_vit_h_4b8939.pth from facebook AI? [y]/n: ")
            if len(cmd) == 0 or cmd.lower() == 'y':
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                print("Downloading SAM ViT-H checkpoint...")
                urllib.request.urlretrieve(
                    "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
                    checkpoint,
                )
                print(checkpoint.name, " is downloaded!")
        elif checkpoint.name == "sam_vit_l_0b3195.pth" and not checkpoint.exists():
            cmd = input("Download sam_vit_l_0b3195.pth from facebook AI? [y]/n: ")
            if len(cmd) == 0 or cmd.lower() == 'y':
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                print("Downloading SAM ViT-L checkpoint...")
                urllib.request.urlretrieve(
                    "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
                    checkpoint,
                )
                print(checkpoint.name, " is downloaded!")

        
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f)
        sam.load_state_dict(state_dict)
    return sam
