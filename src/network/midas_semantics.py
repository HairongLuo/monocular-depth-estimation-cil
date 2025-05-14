"""
Incorporates MiDaS with DINOv2 for semantic-aware depth estimation. 
Based on MiDaS small architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .midas_net_custom import MidasNet_small
from .dpt_depth import Dinov2Head

from loguru import logger as guru

class MidasNetSemantics(MidasNet_small):
    def __init__(self, path=None, features=64, backbone="efficientnet_lite3", non_negative=True, 
                 exportable=True, channels_last=False, align_corners=True,cfg=None, blocks={'expand': True}, dinov2_type='dinov2_vitb14'):
        super().__init__(path, features, backbone, non_negative, exportable, channels_last, align_corners, cfg, blocks)

        # Remove nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0) layer in self.scratch.output_conv
        print("Before removing:")
        print(f"{self.scratch.output_conv=}")
        guru.info("Removing nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0) layer in self.scratch.output_conv")
        self.scratch.output_conv = self.scratch.output_conv[0:4] + self.scratch.output_conv[6:]
        print("After removing:")
        print(f"{self.scratch.output_conv=}")

        # Load DINOv2
        self.dinov2 = torch.hub.load('facebookresearch/dinov2', dinov2_type)
        # Freeze DINOv2 backbone weights
        for param in self.dinov2.parameters():
            param.requires_grad = False
        print("DINOv2 backbone weights frozen")
        dim = self.dinov2.blocks[0].attn.qkv.in_features
        self.dinov2_head = Dinov2Head(1, dim, 256, use_bn=False, out_channels=[256, 512, 1024, 1024], use_clstoken=False)
        self.DINOv2_IMAGE_SIZE = (448, 560)

        # Feature fusion and depth prediction
        self.fusion_head = nn.Sequential(
            nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(features // 2, features // 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Identity(),
        )
        self.depth_head = nn.Sequential(
            nn.Conv2d(features // 2, 1, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Identity(),
        )

    def forward(self, x):
        # MiDaS branch
        if self.channels_last==True:
            x.contiguous(memory_format=torch.channels_last)

        layer_1 = self.pretrained.layer1(x)
        layer_2 = self.pretrained.layer2(layer_1)
        layer_3 = self.pretrained.layer3(layer_2)
        layer_4 = self.pretrained.layer4(layer_3)
        
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = self.scratch.refinenet4(layer_4_rn)
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn)
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn)
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)
        
        out_conv = self.scratch.output_conv(path_1)
        # (B, 32, 448, 576)
        midas_features = torch.squeeze(out_conv, dim=1)
        # guru.info(f"MiDaS feature shape: {midas_features.shape}")


        # DINOv2 branch
        x_dinov2 = F.interpolate(x, size=self.DINOv2_IMAGE_SIZE, mode="bilinear", align_corners=True)
        # guru.info(f"x_dinov2 shape: {x_dinov2.shape}")
        patch_h, patch_w = self.DINOv2_IMAGE_SIZE[0] // 14, self.DINOv2_IMAGE_SIZE[1] // 14
        # guru.info(f"patch_h: {patch_h}, patch_w: {patch_w}")
        dinov2_features = self.dinov2.get_intermediate_layers(x_dinov2, 4, return_class_token=False)
        # (B, 32, 448, 560)
        dinov2_features = self.dinov2_head(dinov2_features, patch_h, patch_w)
        # guru.info(f"DINOv2 feature shape: {dinov2_features.shape}")
        # (B, 32, 448, 576)
        dinov2_features = F.interpolate(dinov2_features, size=midas_features.shape[2:], mode="bilinear", align_corners=True)
        # guru.info(f"Interpolated DINOv2 feature shape: {dinov2_features.shape}")


        # Concatenate DINOv2 and MiDaS features
        # (B, 64, 448, 576)
        features = torch.cat([dinov2_features, midas_features], dim=1)
        # guru.info(f"Concatenated feature shape: {features.shape}")

        # Feature fusion and depth prediction
        features = self.fusion_head(features)
        depth = self.depth_head(features)
        # guru.info(f"Depth shape: {depth.shape}")

        if self.use_lb:
            depth_squeezed = torch.squeeze(depth, dim=1)
            out_for_lb = [depth, layer_4_rn, path_4, path_3, path_2, path_1]
            out_local_bins = self.local_bins(out_for_lb, depth_squeezed)
            return torch.squeeze(out_local_bins, dim=1)
        
        return torch.squeeze(depth, dim=1)
