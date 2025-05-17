"""
Incorporates MiDaS with DINOv2 for semantic-aware depth estimation. 
Based on MiDaS small architecture with enhanced feature fusion.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .midas_net_custom import MidasNet_small
from .dpt_depth import Dinov2Head

from loguru import logger as guru

class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, window_size=16):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.window_size = window_size

        # Layer normalization
        self.norm_q = nn.LayerNorm(dim)
        self.norm_k = nn.LayerNorm(dim)
        self.norm_v = nn.LayerNorm(dim)
        
        # QKV projections
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        
        # Output projection
        self.norm_out = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, dim)
        
        # Spatial reduction (8x)
        self.spatial_reduction = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )
        
        # Spatial upsampling (8x)
        self.spatial_upsample = nn.Sequential(
            nn.ConvTranspose2d(dim, dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(dim, dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(dim, dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, context):
        B, C, H, W = x.shape
        
        # Spatial reduction to reduce memory usage (8x)
        x_reduced = self.spatial_reduction(x)  # B, C, H/8, W/8
        context_reduced = self.spatial_reduction(context)  # B, C, H/8, W/8
        
        # Reshape and normalize
        x_flat = x_reduced.flatten(2).transpose(1, 2)  # B, (H/8)*(W/8), C
        context_flat = context_reduced.flatten(2).transpose(1, 2)  # B, (H/8)*(W/8), C
        
        # Apply layer normalization
        x_norm = self.norm_q(x_flat)
        context_norm_k = self.norm_k(context_flat)
        context_norm_v = self.norm_v(context_flat)
        
        # Project to Q, K, V
        q = self.q(x_norm).reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k(context_norm_k).reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v(context_norm_v).reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # Compute attention in chunks to save memory
        H_reduced, W_reduced = H // 8, W // 8  # Changed from H//4 to H//8
        num_windows_h = (H_reduced + self.window_size - 1) // self.window_size
        num_windows_w = (W_reduced + self.window_size - 1) // self.window_size
        
        # Initialize output tensor
        out = torch.zeros_like(x_flat)
        
        # Process each window
        for h in range(num_windows_h):
            for w in range(num_windows_w):
                # Get window indices
                h_start = h * self.window_size
                w_start = w * self.window_size
                h_end = min(h_start + self.window_size, H_reduced)
                w_end = min(w_start + self.window_size, W_reduced)
                
                # Get window slices
                q_window = q[:, :, h_start*W_reduced + w_start:h_end*W_reduced + w_end, :]
                k_window = k[:, :, h_start*W_reduced + w_start:h_end*W_reduced + w_end, :]
                v_window = v[:, :, h_start*W_reduced + w_start:h_end*W_reduced + w_end, :]
                
                # Compute attention for this window
                attn = (q_window @ k_window.transpose(-2, -1)) * self.scale
                attn = attn.softmax(dim=-1)
                
                # Apply attention
                window_out = (attn @ v_window).transpose(1, 2).reshape(B, -1, C)
                out[:, h_start*W_reduced + w_start:h_end*W_reduced + w_end, :] = window_out
        
        # Apply layer norm and projection
        out = self.norm_out(out)
        out = self.proj(out)
        
        # Reshape back to spatial dimensions
        out = out.transpose(1, 2).reshape(B, C, H_reduced, W_reduced)
        
        # Upsample back to original size (8x)
        out = self.spatial_upsample(out)
        
        # Add residual connection
        out = out + x

        return out

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(residual)
        out = self.relu(out)
        return out

class MidasNetSemantics(MidasNet_small):
    def __init__(self, path=None, features=32, backbone="efficientnet_lite3", non_negative=True, 
                 exportable=True, channels_last=False, align_corners=True, cfg=None, blocks={'expand': True}, 
                 dinov2_type='dinov2_vits14'):
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
        
        # Get DINOv2 feature dimensions
        dim = self.dinov2.blocks[0].attn.qkv.in_features
        self.dinov2_head = Dinov2Head(1, dim, 128, use_bn=False, out_channels=[128, 256, 512, 512], use_clstoken=False)
        # Reduced DINOv2 input size from (448, 560) to (224, 280)
        self.DINOv2_IMAGE_SIZE = (224, 280)  # Half the original size

        # Enhanced feature fusion with memory-efficient attention
        self.cross_attention = CrossAttention(features // 2, window_size=16)
        
        # Multi-scale feature fusion with sequential blocks
        self.fusion_blocks = nn.Sequential(
            ResidualBlock(features, features),
            # ResidualBlock(features, features),
            # ResidualBlock(features, features),
            # ResidualBlock(features, features)
        )
        
        # Improved fusion head with residual connections
        self.fusion_head = nn.Sequential(
            ResidualBlock(features, features // 2),
            # ResidualBlock(features // 2, features // 2),
            nn.Conv2d(features // 2, features // 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(features // 2),
            nn.ReLU(True),
        )
        
        # Depth prediction head
        self.depth_head = nn.Sequential(
            ResidualBlock(features // 2, features // 4),
            nn.Conv2d(features // 4, 1, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True) if non_negative else nn.Identity(),
        )

    def forward(self, x):
        # MiDaS branch
        if self.channels_last:
            x = x.contiguous(memory_format=torch.channels_last)

        # Get MiDaS features
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

        # DINOv2 branch with multi-scale features
        x_dinov2 = F.interpolate(x, size=self.DINOv2_IMAGE_SIZE, mode="bilinear", align_corners=True)
        patch_h, patch_w = self.DINOv2_IMAGE_SIZE[0] // 14, self.DINOv2_IMAGE_SIZE[1] // 14
        
        # Get multiple DINOv2 layers for better semantic understanding
        dinov2_features = self.dinov2.get_intermediate_layers(x_dinov2, 4, return_class_token=False)
        # (B, 32, 448, 560)
        dinov2_features = self.dinov2_head(dinov2_features, patch_h, patch_w)

        # (B, 32, 448, 576)
        # Resize DINOv2 features to match MiDaS features
        dinov2_features = F.interpolate(dinov2_features, size=midas_features.shape[2:], mode="bilinear", align_corners=True)
        
        # Enhanced feature fusion with memory-efficient attention
        # Apply cross attention
        attended_features = self.cross_attention(midas_features, dinov2_features)
        # Concatenate and fuse
        concat_features = torch.cat([attended_features, dinov2_features], dim=1)
        
        # Apply fusion blocks sequentially
        fused = self.fusion_blocks(concat_features)
        
        # Feature fusion and depth prediction
        features = self.fusion_head(fused)
        
        # Get depth prediction
        depth = self.depth_head(features)

        if self.use_lb:
            depth_squeezed = torch.squeeze(depth, dim=1)
            out_for_lb = [depth, layer_4_rn, path_4, path_3, path_2, path_1]
            out_local_bins = self.local_bins(out_for_lb, depth_squeezed)
            return torch.squeeze(out_local_bins, dim=1)
        
        return torch.squeeze(depth, dim=1)
