"""MidashNet: Network for monocular depth estimation trained by mixing several datasets.
This file contains code that is adapted from
https://github.com/thomasjpfan/pytorch_refinenet/blob/master/pytorch_refinenet/refinenet/refinenet_4cascade.py
"""
import torch
import torch.nn as nn

from .base_model import BaseModel
from .blocks import FeatureFusionBlock, FeatureFusionBlock_custom, Interpolate, _make_encoder
from .localbins_net import LocalBins_Block

# ─── Depth Gradient Refinement block ────────────────────────────────
class DGR(nn.Module):
    """
    Light edge-sharpening & channel-recalibration module.
    """
    def __init__(self, ch):
        super().__init__()
        # depth-wise Laplacian kernels (∇² and ∇³)
        self.lap2 = nn.Conv2d(ch, ch, 3, 1, 1, bias=False, groups=ch)
        self.lap3 = nn.Conv2d(ch, ch, 3, 1, 1, bias=False, groups=ch)
        with torch.no_grad():
            lap = torch.tensor([[0,1,0],[1,-4,1],[0,1,0]], dtype=torch.float32)
            self.lap2.weight.copy_(lap.repeat(ch,1,1,1))
            self.lap3.weight.copy_((lap*lap).repeat(ch,1,1,1))
        for p in self.parameters():  # keep them frozen
            p.requires_grad = False

        # channel & spatial recalibration
        self.recalib = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(3*ch, ch//8, 1), nn.GELU(),
            nn.Conv2d(ch//8, 3*ch, 1), nn.Sigmoid()
        )
        self.spatial = nn.Conv2d(3*ch, 3*ch, 3, 1, 1, groups=3*ch)

    def forward(self, x):
        l2, l3 = self.lap2(x), self.lap3(x)
        f = torch.cat([x, l2, l3], 1)
        f = f * self.recalib(f)
        f = self.spatial(f)
        return f[:, :x.shape[1]] + x      # residual

class MidasNet_small(BaseModel):
    """Network for monocular depth estimation.
    """

    def __init__(self, path=None, features=64, backbone="efficientnet_lite3", non_negative=True, exportable=True, channels_last=False, align_corners=True,cfg=None,
        blocks={'expand': True}):
        """Init.

        Args:
            path (str, optional): Path to saved model. Defaults to None.
            features (int, optional): Number of features. Defaults to 256.
            backbone (str, optional): Backbone network for encoder. Defaults to resnet50
        """
        print("Loading weights: ", path)

        super(MidasNet_small, self).__init__()

        use_pretrained = False if path else True
        self.use_lb = cfg.use_lb
                
        self.channels_last = channels_last
        self.blocks = blocks
        self.backbone = backbone

        self.groups = 1

        features1=features
        features2=features
        features3=features
        features4=features
        self.expand = False
        if "expand" in self.blocks and self.blocks['expand'] == True:
            self.expand = True
            features1=features
            features2=features*2
            features3=features*4
            features4=features*8

        self.pretrained, self.scratch = _make_encoder(self.backbone, features, use_pretrained, groups=self.groups, expand=self.expand, exportable=exportable)
  
        self.scratch.activation = nn.ReLU(False)    

        self.scratch.refinenet4 = FeatureFusionBlock_custom(features4, self.scratch.activation, deconv=False, bn=False, expand=self.expand, align_corners=align_corners)
        self.scratch.refinenet3 = FeatureFusionBlock_custom(features3, self.scratch.activation, deconv=False, bn=False, expand=self.expand, align_corners=align_corners)
        self.scratch.refinenet2 = FeatureFusionBlock_custom(features2, self.scratch.activation, deconv=False, bn=False, expand=self.expand, align_corners=align_corners)
        self.scratch.refinenet1 = FeatureFusionBlock_custom(features1, self.scratch.activation, deconv=False, bn=False, align_corners=align_corners)

        # DGR ##########
        self.dgr4 = DGR(features4)
        self.dgr3 = DGR(features3)
        self.dgr2 = DGR(features2)
        self.dgr1 = DGR(features1)
        ################

        self.scratch.output_conv = nn.Sequential(
            nn.Conv2d(features, features//2, kernel_size=3, stride=1, padding=1, groups=self.groups),
            Interpolate(scale_factor=2, mode="bilinear"),
            nn.Conv2d(features//2, 32, kernel_size=3, stride=1, padding=1),
            self.scratch.activation,
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True) if non_negative else nn.Identity(),
            nn.Identity(),
        )
        
        if self.use_lb:
            self.local_bins = LocalBins_Block(
                in_channels=features,  # Adjust based on your architecture
                n_bins=16,
                max_depth=10,
                min_depth=1e-3,
                bin_embedding_dim=128,
                n_attractors=[16, 8, 4, 1],
                attractor_alpha=300,
                attractor_gamma=2,
                attractor_kind='sum',
                attractor_type='inv',
                inverse_midas=False,
                min_temp=5,
                max_temp=50,
                model_type = "MiDaS_small",
            )

        if path:
            self.load(path)


    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input data (image)

        Returns:
            tensor: depth
        """
        if self.channels_last==True:
            print("self.channels_last = ", self.channels_last)
            x.contiguous(memory_format=torch.channels_last)


        layer_1 = self.pretrained.layer1(x)
        layer_2 = self.pretrained.layer2(layer_1)
        layer_3 = self.pretrained.layer3(layer_2)
        layer_4 = self.pretrained.layer4(layer_3)
        
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        # DGR ##########
        layer_1_rn = self.dgr1(self.scratch.layer1_rn(layer_1))
        layer_2_rn = self.dgr2(self.scratch.layer2_rn(layer_2))
        layer_3_rn = self.dgr3(self.scratch.layer3_rn(layer_3))
        layer_4_rn = self.dgr4(self.scratch.layer4_rn(layer_4))
        ################

        path_4 = self.scratch.refinenet4(layer_4_rn)
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn)
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn)
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)
        
        out_conv = self.scratch.output_conv(path_1)

        if self.use_lb:
            rel_depth = torch.squeeze(out_conv, dim=1)
            out_for_lb = [out_conv, layer_4_rn, path_4, path_3, path_2, path_1]
            out_local_bins = self.local_bins(out_for_lb, rel_depth)
            return torch.squeeze(out_local_bins, dim=1)
        
        return torch.squeeze(out_conv, dim=1)




def fuse_model(m):
    prev_previous_type = nn.Identity()
    prev_previous_name = ''
    previous_type = nn.Identity()
    previous_name = ''
    for name, module in m.named_modules():
        if prev_previous_type == nn.Conv2d and previous_type == nn.BatchNorm2d and type(module) == nn.ReLU:
            # print("FUSED ", prev_previous_name, previous_name, name)
            torch.quantization.fuse_modules(m, [prev_previous_name, previous_name, name], inplace=True)
        elif prev_previous_type == nn.Conv2d and previous_type == nn.BatchNorm2d:
            # print("FUSED ", prev_previous_name, previous_name)
            torch.quantization.fuse_modules(m, [prev_previous_name, previous_name], inplace=True)
        # elif previous_type == nn.Conv2d and type(module) == nn.ReLU:
        #    print("FUSED ", previous_name, name)
        #    torch.quantization.fuse_modules(m, [previous_name, name], inplace=True)

        prev_previous_type = previous_type
        prev_previous_name = previous_name
        previous_type = type(module)
        previous_name = name