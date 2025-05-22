import torch
import torch.nn as nn
from .midas_net_custom import MidasNet_small

class SkipBlock(nn.Module):
    """Fuse skip-connected features and adjust number of channels"""
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        return x


class MidasNetSkip(MidasNet_small):
    """Midas with skip connections to utilize low-level features"""
    def __init__(self, path=None, features=64, backbone="efficientnet_lite3", non_negative=True, 
                 exportable=True, channels_last=False, align_corners=True, cfg=None, blocks={'expand': True}):
        super().__init__(path, features, backbone, non_negative, 
                         exportable, channels_last, align_corners, cfg, blocks)
        self.low_level_extractor = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
        )

        self.skip_block_4 = SkipBlock(256 + 136, 256)
        self.skip_block_3 = SkipBlock(128 + 48, 128)
        self.skip_block_2 = SkipBlock(64 + 32, 64)
        self.skip_block_1 = SkipBlock(64 + 128, 64)

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
        
        low_level_features = self.low_level_extractor(x)

        layer_1 = self.pretrained.layer1(x)
        layer_2 = self.pretrained.layer2(layer_1)
        layer_3 = self.pretrained.layer3(layer_2)
        layer_4 = self.pretrained.layer4(layer_3)
        # guru.info(f'{x.shape=}\n{layer_1.shape=}\n{layer_2.shape=}\n{layer_3.shape=}\n{layer_4.shape=}')
        
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)
        # guru.info(f'{layer_1_rn.shape=}\n{layer_2_rn.shape=}\n{layer_3_rn.shape=}\n{layer_4_rn.shape=}')

        path_4 = self.scratch.refinenet4(layer_4_rn)
        path_4 = torch.cat((path_4, layer_3), dim=1)
        path_4 = self.skip_block_4(path_4)
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn)
        path_3 = torch.cat((path_3, layer_2), dim=1)
        path_3 = self.skip_block_3(path_3)
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn)
        path_2 = torch.cat((path_2, layer_1), dim=1)
        path_2 = self.skip_block_2(path_2)
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)
        path_1 = torch.cat((path_1, low_level_features), dim=1)
        path_1 = self.skip_block_1(path_1)
        # guru.info(f'{path_4.shape=}\n{path_3.shape=}\n{path_2.shape=}\n{path_1.shape=}')
        
        out_conv = self.scratch.output_conv(path_1)
        # guru.info(f'{out_conv.shape=}')

        if self.use_lb:
            rel_depth = torch.squeeze(out_conv, dim=1)
            out_for_lb = [out_conv, layer_4_rn, path_4, path_3, path_2, path_1]
            out_local_bins = self.local_bins(out_for_lb, rel_depth)
            return torch.squeeze(out_local_bins, dim=1)
        
        return torch.squeeze(out_conv, dim=1)