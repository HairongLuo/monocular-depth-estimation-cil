import torch
import torch.nn as nn
from .localbins_layers import (
    SeedBinRegressorUnnormed,
    SeedBinRegressor,
    AttractorLayerUnnormed,
    Projector,
    ConditionalLogBinomial,
)


class LocalBins_Block(nn.Module):
    def __init__(self, in_channels, n_bins=16, max_depth=10, min_depth=1e-3, 
                 bin_embedding_dim=128, n_attractors=[16, 8, 4, 1], attractor_alpha=300,
                 attractor_gamma=2, attractor_kind='sum', attractor_type='exp',
                 inverse_midas=False, min_temp=5, max_temp=50, model_type = "MiDaS_small"):
        super().__init__()
        # max and min need to be adjusted to the dataset
        self.max_depth = max_depth
        self.min_depth = min_depth
        self.n_bins = n_bins

        # N_MIDAS_OUT = 32
        N_MIDAS_OUT = 1 #output of midas, 1 for depth
        output_channels = MIDAS_SETTINGS[model_type]
        btlnck_features = output_channels[0]
        num_out_features = output_channels[1:]

        self.conv2 = nn.Conv2d(btlnck_features, btlnck_features,
                               kernel_size=1, stride=1, padding=0)  # btlnck conv
        SeedBinRegressorLayer = SeedBinRegressorUnnormed
        Attractor = AttractorLayerUnnormed
        self.seed_bin_regressor = SeedBinRegressorLayer(
            btlnck_features, n_bins=self.n_bins, min_depth=self.min_depth, max_depth=self.max_depth)
        self.seed_projector = Projector(btlnck_features, bin_embedding_dim)
        self.seed_projector = Projector(btlnck_features, bin_embedding_dim)
        self.projectors = nn.ModuleList([
            Projector(num_out, bin_embedding_dim)
            for num_out in num_out_features
        ])
        self.attractors = nn.ModuleList([
            Attractor(bin_embedding_dim, n_bins, n_attractors=n_attractors[i], min_depth=min_depth, max_depth=max_depth,
                      alpha=attractor_alpha, gamma=attractor_gamma, kind=attractor_kind, attractor_type=attractor_type)
            for i in range(len(num_out_features))
        ])


        last_in = N_MIDAS_OUT + 1  # +1 for relative depth

        # use log binomial instead of softmax
        self.conditional_log_binomial = ConditionalLogBinomial(
            last_in, bin_embedding_dim, n_classes=n_bins, min_temp=min_temp, max_temp=max_temp)

    def forward(self, out, rel_depth, return_probs=False, return_final_centers=False):

        outconv_activation = out[0]
        btlnck = out[1]
        x_blocks = out[2:]

        x_d0 = self.conv2(btlnck)
        x = x_d0
        _, seed_b_centers = self.seed_bin_regressor(x)

        b_prev = seed_b_centers

        prev_b_embedding = self.seed_projector(x)

        # unroll this loop for better performance
        for projector, attractor, x in zip(self.projectors, self.attractors, x_blocks):
            b_embedding = projector(x)
            b, b_centers = attractor(
                b_embedding, b_prev, prev_b_embedding, interpolate=True)
            b_prev = b.clone()
            prev_b_embedding = b_embedding.clone()

        last = outconv_activation

        # if self.inverse_midas:
        #     # invert depth followed by normalization
        #     rel_depth = 1.0 / (rel_depth + 1e-6)
        #     rel_depth = (rel_depth - rel_depth.min()) / \
        #         (rel_depth.max() - rel_depth.min())
        # concat rel depth with last. First interpolate rel depth to last size
        rel_cond = rel_depth.unsqueeze(1)
        rel_cond = nn.functional.interpolate(
            rel_cond, size=last.shape[2:], mode='bilinear', align_corners=True)
        last = torch.cat([last, rel_cond], dim=1)

        b_embedding = nn.functional.interpolate(
            b_embedding, last.shape[-2:], mode='bilinear', align_corners=True)
        
        x = self.conditional_log_binomial(last, b_embedding)

        # Now depth value is Sum px * cx , where cx are bin_centers from the last bin tensor
        # print(x.shape, b_centers.shape)
        b_centers = nn.functional.interpolate(
            b_centers, x.shape[-2:], mode='bilinear', align_corners=True)
        out = torch.sum(x * b_centers, dim=1, keepdim=True)

        # Structure output dict
        output = dict(metric_depth=out)
        if return_final_centers or return_probs:
            output['bin_centers'] = b_centers

        if return_probs:
            output['probs'] = x


        # only return the metric depth
        return output['metric_depth'].squeeze(1)
    


nchannels2models = {
    tuple([256]*5): ["DPT_BEiT_L_384", "DPT_BEiT_L_512", "DPT_BEiT_B_384", "DPT_SwinV2_L_384", "DPT_SwinV2_B_384", "DPT_SwinV2_T_256", "DPT_Large", "DPT_Hybrid"],
    (512, 256, 128, 64, 64): ["MiDaS_small"]
}

# Model name to number of output channels
MIDAS_SETTINGS = {m: k for k, v in nchannels2models.items()
                  for m in v
                  }