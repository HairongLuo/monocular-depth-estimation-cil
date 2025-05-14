# MIT License

# Copyright (c) 2022 Intelligent Systems Lab Org

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# File author: Shariq Farooq Bhat

import torch
import torch.nn as nn


class SeedBinRegressor(nn.Module):
    def __init__(self, in_features, n_bins=16, mlp_dim=256, min_depth=1e-3, max_depth=10):
        """Bin center regressor network. Bin centers are bounded on (min_depth, max_depth) interval.

        Args:
            in_features (int): input channels
            n_bins (int, optional): Number of bin centers. Defaults to 16.
            mlp_dim (int, optional): Hidden dimension. Defaults to 256.
            min_depth (float, optional): Min depth value. Defaults to 1e-3.
            max_depth (float, optional): Max depth value. Defaults to 10.
        """
        super().__init__()
        self.version = "1_1"
        self.min_depth = min_depth
        self.max_depth = max_depth

        self._net = nn.Sequential(
            nn.Conv2d(in_features, mlp_dim, 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(mlp_dim, n_bins, 1, 1, 0),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """
        Returns tensor of bin_width vectors (centers). One vector b for every pixel
        """
        B = self._net(x)
        eps = 1e-3
        B = B + eps
        B_widths_normed = B / B.sum(dim=1, keepdim=True)
        B_widths = (self.max_depth - self.min_depth) * \
            B_widths_normed  # .shape NCHW
        # pad has the form (left, right, top, bottom, front, back)
        B_widths = nn.functional.pad(
            B_widths, (0, 0, 0, 0, 1, 0), mode='constant', value=self.min_depth)
        B_edges = torch.cumsum(B_widths, dim=1)  # .shape NCHW

        B_centers = 0.5 * (B_edges[:, :-1, ...] + B_edges[:, 1:, ...])
        return B_widths_normed, B_centers


class SeedBinRegressorUnnormed(nn.Module):
    def __init__(self, in_features, n_bins=16, mlp_dim=256, min_depth=1e-3, max_depth=10):
        """Bin center regressor network. Bin centers are unbounded

        Args:
            in_features (int): input channels
            n_bins (int, optional): Number of bin centers. Defaults to 16.
            mlp_dim (int, optional): Hidden dimension. Defaults to 256.
            min_depth (float, optional): Not used. (for compatibility with SeedBinRegressor)
            max_depth (float, optional): Not used. (for compatibility with SeedBinRegressor)
        """
        super().__init__()
        self.version = "1_1"
        self._net = nn.Sequential(
            nn.Conv2d(in_features, mlp_dim, 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(mlp_dim, n_bins, 1, 1, 0),
            nn.Softplus()
        )

    def forward(self, x):
        """
        Returns tensor of bin_width vectors (centers). One vector b for every pixel
        """
        B_centers = self._net(x)
        return B_centers, B_centers


class Projector(nn.Module):
    def __init__(self, in_features, out_features, mlp_dim=128):
        """Projector MLP

        Args:
            in_features (int): input channels
            out_features (int): output channels
            mlp_dim (int, optional): hidden dimension. Defaults to 128.
        """
        super().__init__()

        self._net = nn.Sequential(
            nn.Conv2d(in_features, mlp_dim, 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(mlp_dim, out_features, 1, 1, 0),
        )

    def forward(self, x):
        return self._net(x)



class LinearSplitter(nn.Module):
    def __init__(self, in_features, prev_nbins, split_factor=2, mlp_dim=128, min_depth=1e-3, max_depth=10):
        super().__init__()

        self.prev_nbins = prev_nbins
        self.split_factor = split_factor
        self.min_depth = min_depth
        self.max_depth = max_depth

        self._net = nn.Sequential(
            nn.Conv2d(in_features, mlp_dim, 1, 1, 0),
            nn.GELU(),
            nn.Conv2d(mlp_dim, prev_nbins * split_factor, 1, 1, 0),
            nn.ReLU()
        )
    
    def forward(self, x, b_prev, prev_b_embedding=None, interpolate=True, is_for_query=False):
        """
        x : feature block; shape - n, c, h, w
        b_prev : previous bin widths normed; shape - n, prev_nbins, h, w
        """
        if prev_b_embedding is not None:
            if interpolate:
                prev_b_embedding = nn.functional.interpolate(prev_b_embedding, x.shape[-2:], mode='bilinear', align_corners=True)
            x = x + prev_b_embedding
        S = self._net(x)
        eps = 1e-3
        S = S + eps
        n, c, h, w = S.shape
        S = S.view(n, self.prev_nbins, self.split_factor, h, w)
        S_normed = S / S.sum(dim=2, keepdim=True)  # fractional splits

        b_prev = nn.functional.interpolate(b_prev, (h,w), mode='bilinear', align_corners=True)
        

        b_prev = b_prev / b_prev.sum(dim=1, keepdim=True)  # renormalize for gurantees
        # print(b_prev.shape, S_normed.shape)
        # if is_for_query:(1).expand(-1, b_prev.size(0)//n, -1, -1, -1, -1).flatten(0,1)  # TODO ? can replace all this with a single torch.repeat?
        b = b_prev.unsqueeze(2) * S_normed
        b = b.flatten(1,2)  # .shape n, prev_nbins * split_factor, h, w

        # calculate bin centers for loss calculation
        B_widths = (self.max_depth - self.min_depth) * b  # .shape N, nprev * splitfactor, H, W
        # pad has the form (left, right, top, bottom, front, back)
        B_widths = nn.functional.pad(B_widths, (0,0,0,0,1,0), mode='constant', value=self.min_depth)
        B_edges = torch.cumsum(B_widths, dim=1)  # .shape NCHW

        B_centers = 0.5 * (B_edges[:, :-1, ...] + B_edges[:,1:,...])
        return b, B_centers
    
@torch.jit.script
def exp_attractor(dx, alpha: float = 300, gamma: int = 2):
    """Exponential attractor: dc = exp(-alpha*|dx|^gamma) * dx , where dx = a - c, a = attractor point, c = bin center, dc = shift in bin centermmary for exp_attractor

    Args:
        dx (torch.Tensor): The difference tensor dx = Ai - Cj, where Ai is the attractor point and Cj is the bin center.
        alpha (float, optional): Proportional Attractor strength. Determines the absolute strength. Lower alpha = greater attraction. Defaults to 300.
        gamma (int, optional): Exponential Attractor strength. Determines the "region of influence" and indirectly number of bin centers affected. Lower gamma = farther reach. Defaults to 2.

    Returns:
        torch.Tensor : Delta shifts - dc; New bin centers = Old bin centers + dc
    """
    return torch.exp(-alpha*(torch.abs(dx)**gamma)) * (dx)


@torch.jit.script
def inv_attractor(dx, alpha: float = 300, gamma: int = 2):
    """Inverse attractor: dc = dx / (1 + alpha*dx^gamma), where dx = a - c, a = attractor point, c = bin center, dc = shift in bin center
    This is the default one according to the accompanying paper. 

    Args:
        dx (torch.Tensor): The difference tensor dx = Ai - Cj, where Ai is the attractor point and Cj is the bin center.
        alpha (float, optional): Proportional Attractor strength. Determines the absolute strength. Lower alpha = greater attraction. Defaults to 300.
        gamma (int, optional): Exponential Attractor strength. Determines the "region of influence" and indirectly number of bin centers affected. Lower gamma = farther reach. Defaults to 2.

    Returns:
        torch.Tensor: Delta shifts - dc; New bin centers = Old bin centers + dc
    """
    return dx.div(1+alpha*dx.pow(gamma))

class AttractorLayerUnnormed(nn.Module):
    def __init__(self, in_features, n_bins, n_attractors=16, mlp_dim=128, min_depth=1e-3, max_depth=10,
                 alpha=300, gamma=2, kind='sum', attractor_type='exp', memory_efficient=False):
        """
        Attractor layer for bin centers. Bin centers are unbounded
        """
        super().__init__()

        self.n_attractors = n_attractors
        self.n_bins = n_bins
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.alpha = alpha
        self.gamma = gamma
        self.kind = kind
        self.attractor_type = attractor_type
        self.memory_efficient = memory_efficient

        self._net = nn.Sequential(
            nn.Conv2d(in_features, mlp_dim, 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(mlp_dim, n_attractors, 1, 1, 0),
            nn.Softplus()
        )

    def forward(self, x, b_prev, prev_b_embedding=None, interpolate=True, is_for_query=False):
        """
        Args:
            x (torch.Tensor) : feature block; shape - n, c, h, w
            b_prev (torch.Tensor) : previous bin centers normed; shape - n, prev_nbins, h, w
        
        Returns:
            tuple(torch.Tensor,torch.Tensor) : new bin centers unbounded; shape - n, nbins, h, w. Two outputs just to keep the API consistent with the normed version
        """
        if prev_b_embedding is not None:
            if interpolate:
                prev_b_embedding = nn.functional.interpolate(
                    prev_b_embedding, x.shape[-2:], mode='bilinear', align_corners=True)
            x = x + prev_b_embedding

        A = self._net(x)
        n, c, h, w = A.shape

        b_prev = nn.functional.interpolate(
            b_prev, (h, w), mode='bilinear', align_corners=True)
        b_centers = b_prev

        if self.attractor_type == 'exp':
            dist = exp_attractor
        else:
            dist = inv_attractor

        if not self.memory_efficient:
            func = {'mean': torch.mean, 'sum': torch.sum}[self.kind]
            # .shape N, nbins, h, w
            delta_c = func(
                dist(A.unsqueeze(2) - b_centers.unsqueeze(1)), dim=1)
        else:
            delta_c = torch.zeros_like(b_centers, device=b_centers.device)
            for i in range(self.n_attractors):
                delta_c += dist(A[:, i, ...].unsqueeze(1) -
                                b_centers)  # .shape N, nbins, h, w

            if self.kind == 'mean':
                delta_c = delta_c / self.n_attractors

        b_new_centers = b_centers + delta_c
        B_centers = b_new_centers

        return b_new_centers, B_centers
    
def log_binom(n, k, eps=1e-7):
    """ log(nCk) using stirling approximation """
    n = n + eps
    k = k + eps
    return n * torch.log(n) - k * torch.log(k) - (n-k) * torch.log(n-k+eps)

class LogBinomial(nn.Module):
    def __init__(self, n_classes=256, act=torch.softmax):
        """Compute log binomial distribution for n_classes

        Args:
            n_classes (int, optional): number of output classes. Defaults to 256.
        """
        super().__init__()
        self.K = n_classes
        self.act = act
        self.register_buffer('k_idx', torch.arange(
            0, n_classes).view(1, -1, 1, 1))
        self.register_buffer('K_minus_1', torch.Tensor(
            [self.K-1]).view(1, -1, 1, 1))

    def forward(self, x, t=1., eps=1e-4):
        """Compute log binomial distribution for x

        Args:
            x (torch.Tensor - NCHW): probabilities
            t (float, torch.Tensor - NCHW, optional): Temperature of distribution. Defaults to 1..
            eps (float, optional): Small number for numerical stability. Defaults to 1e-4.

        Returns:
            torch.Tensor -NCHW: log binomial distribution logbinomial(p;t)
        """
        if x.ndim == 3:
            x = x.unsqueeze(1)  # make it nchw

        one_minus_x = torch.clamp(1 - x, eps, 1)
        x = torch.clamp(x, eps, 1)
        y = log_binom(self.K_minus_1, self.k_idx) + self.k_idx * \
            torch.log(x) + (self.K - 1 - self.k_idx) * torch.log(one_minus_x)
        return self.act(y/t, dim=1)
    
class ConditionalLogBinomial(nn.Module):
    def __init__(self, in_features, condition_dim, n_classes=256, bottleneck_factor=2, p_eps=1e-4, max_temp=50, min_temp=1e-7, act=torch.softmax):
        """Conditional Log Binomial distribution

        Args:
            in_features (int): number of input channels in main feature
            condition_dim (int): number of input channels in condition feature
            n_classes (int, optional): Number of classes. Defaults to 256.
            bottleneck_factor (int, optional): Hidden dim factor. Defaults to 2.
            p_eps (float, optional): small eps value. Defaults to 1e-4.
            max_temp (float, optional): Maximum temperature of output distribution. Defaults to 50.
            min_temp (float, optional): Minimum temperature of output distribution. Defaults to 1e-7.
        """
        super().__init__()
        self.p_eps = p_eps
        self.max_temp = max_temp
        self.min_temp = min_temp
        self.log_binomial_transform = LogBinomial(n_classes, act=act)
        bottleneck = (in_features + condition_dim) // bottleneck_factor
        self.mlp = nn.Sequential(
            nn.Conv2d(in_features + condition_dim, bottleneck,
                      kernel_size=1, stride=1, padding=0),
            nn.GELU(),
            # 2 for p linear norm, 2 for t linear norm
            nn.Conv2d(bottleneck, 2+2, kernel_size=1, stride=1, padding=0),
            nn.Softplus()
        )

    def forward(self, x, cond):
        """Forward pass

        Args:
            x (torch.Tensor - NCHW): Main feature
            cond (torch.Tensor - NCHW): condition feature

        Returns:
            torch.Tensor: Output log binomial distribution
        """
        pt = self.mlp(torch.concat((x, cond), dim=1))
        p, t = pt[:, :2, ...], pt[:, 2:, ...]

        p = p + self.p_eps
        p = p[:, 0, ...] / (p[:, 0, ...] + p[:, 1, ...])

        t = t + self.p_eps
        t = t[:, 0, ...] / (t[:, 0, ...] + t[:, 1, ...])
        t = t.unsqueeze(1)
        t = (self.max_temp - self.min_temp) * t + self.min_temp

        return self.log_binomial_transform(p, t)