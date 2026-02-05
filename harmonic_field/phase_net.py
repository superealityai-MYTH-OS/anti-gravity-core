"""
Phase-aware neural layers with completely original implementations
Novel dual-path architecture preserving phase structure
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .biform import BiChannel
from typing import Optional
import math


class BiMatrix(nn.Module):
    """
    Dual-path linear transformation with novel initialization
    
    Original approach: Four separate weight paths for x/y interactions
    Custom init: scaled uniform to preserve phase distribution
    """

    def __init__(self, in_size: int, out_size: int, use_offset: bool = True):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size

        # Four transformation matrices for complete 2D->2D mapping
        self.mat_xx = nn.Parameter(torch.empty(out_size, in_size))
        self.mat_xy = nn.Parameter(torch.empty(out_size, in_size))
        self.mat_yx = nn.Parameter(torch.empty(out_size, in_size))
        self.mat_yy = nn.Parameter(torch.empty(out_size, in_size))

        if use_offset:
            self.offset_x = nn.Parameter(torch.empty(out_size))
            self.offset_y = nn.Parameter(torch.empty(out_size))
        else:
            self.register_parameter("offset_x", None)
            self.register_parameter("offset_y", None)

        self._init_parameters()

    def _init_parameters(self):
        """
        Novel initialization: variance-scaled uniform for phase stability
        Scale by 1/sqrt(3*(in+out)) for dual-path balance
        """
        boundary = math.sqrt(3.0 / (self.in_size + self.out_size))

        # Uniform distribution in [-boundary, boundary]
        nn.init.uniform_(self.mat_xx, -boundary, boundary)
        nn.init.uniform_(self.mat_xy, -boundary, boundary)
        nn.init.uniform_(self.mat_yx, -boundary, boundary)
        nn.init.uniform_(self.mat_yy, -boundary, boundary)

        if self.offset_x is not None:
            nn.init.zeros_(self.offset_x)
            nn.init.zeros_(self.offset_y)

    def forward(self, incoming: BiChannel) -> BiChannel:
        """
        Apply transformation: (M_x, M_y) * (in_x, in_y)
        Output_x = M_xx * in_x + M_xy * in_y + offset_x
        Output_y = M_yx * in_x + M_yy * in_y + offset_y
        """
        # X-channel output
        out_x = F.linear(incoming.axis_x, self.mat_xx, self.offset_x) + \
                F.linear(incoming.axis_y, self.mat_xy)

        # Y-channel output
        out_y = F.linear(incoming.axis_x, self.mat_yx, self.offset_y) + \
                F.linear(incoming.axis_y, self.mat_yy)

        return BiChannel(out_x, out_y)


class TwoPathNorm(nn.Module):
    """
    Dual-path normalization using covariance whitening
    
    Novel 2x2 covariance approach:
    C = [[var_x, cov_xy],
         [cov_xy, var_y]]
    
    Whitening: inv(C) applied to center data
    """

    def __init__(self, num_features: int, momentum: float = 0.1, epsilon: float = 1e-4):
        super().__init__()
        self.num_features = num_features
        self.momentum = momentum
        self.epsilon = epsilon

        # Learnable affine parameters
        self.gain_x = nn.Parameter(torch.ones(num_features))
        self.gain_y = nn.Parameter(torch.zeros(num_features))
        self.bias_x = nn.Parameter(torch.zeros(num_features))
        self.bias_y = nn.Parameter(torch.zeros(num_features))

        # Running statistics for inference
        self.register_buffer("track_mean_x", torch.zeros(num_features))
        self.register_buffer("track_mean_y", torch.zeros(num_features))
        self.register_buffer("track_var_x", torch.ones(num_features))
        self.register_buffer("track_var_y", torch.ones(num_features))
        self.register_buffer("track_cov_xy", torch.zeros(num_features))

    def forward(self, incoming: BiChannel) -> BiChannel:
        """
        Apply covariance-based whitening transformation
        """
        if self.training:
            # Batch statistics
            batch_mean_x = incoming.axis_x.mean(dim=0)
            batch_mean_y = incoming.axis_y.mean(dim=0)

            centered_x = incoming.axis_x - batch_mean_x
            centered_y = incoming.axis_y - batch_mean_y

            var_x = (centered_x ** 2).mean(dim=0)
            var_y = (centered_y ** 2).mean(dim=0)
            cov_xy = (centered_x * centered_y).mean(dim=0)

            # Update running estimates
            with torch.no_grad():
                self.track_mean_x.mul_(1 - self.momentum).add_(batch_mean_x * self.momentum)
                self.track_mean_y.mul_(1 - self.momentum).add_(batch_mean_y * self.momentum)
                self.track_var_x.mul_(1 - self.momentum).add_(var_x * self.momentum)
                self.track_var_y.mul_(1 - self.momentum).add_(var_y * self.momentum)
                self.track_cov_xy.mul_(1 - self.momentum).add_(cov_xy * self.momentum)
        else:
            batch_mean_x = self.track_mean_x
            batch_mean_y = self.track_mean_y
            var_x = self.track_var_x
            var_y = self.track_var_y
            cov_xy = self.track_cov_xy

        # Center
        centered_x = incoming.axis_x - batch_mean_x
        centered_y = incoming.axis_y - batch_mean_y

        # Compute inverse covariance for whitening
        # C^-1 = [a, b; b, c] where det = var_x*var_y - cov^2
        det = var_x * var_y - cov_xy ** 2 + self.epsilon
        inv_a = var_y / det
        inv_c = var_x / det
        inv_b = -cov_xy / det

        # Apply whitening transformation
        whitened_x = inv_a * centered_x + inv_b * centered_y
        whitened_y = inv_b * centered_x + inv_c * centered_y

        # Learnable affine transformation (as dual number)
        gain = BiChannel(self.gain_x, self.gain_y)
        offset = BiChannel(self.bias_x, self.bias_y)

        normalized = BiChannel(whitened_x, whitened_y)
        output = normalized.combine_with(gain).sum_with(offset)

        return output


class MagPreserveDropout(nn.Module):
    """
    Novel dropout that preserves angular information
    
    Original approach: Drop both channels together (not independently)
    This maintains phase while reducing magnitude
    """

    def __init__(self, prob: float = 0.5):
        super().__init__()
        self.prob = prob

    def forward(self, incoming: BiChannel) -> BiChannel:
        if not self.training or self.prob == 0:
            return incoming

        # Single mask for both channels (preserves angle)
        keep_mask = torch.bernoulli(
            torch.ones_like(incoming.axis_x) * (1 - self.prob)
        )

        # Rescale to maintain expected magnitude
        rescale = 1.0 / (1 - self.prob)

        return BiChannel(
            incoming.axis_x * keep_mask * rescale,
            incoming.axis_y * keep_mask * rescale,
        )


def phase_gate(incoming: BiChannel, shift: float = 0.5) -> BiChannel:
    """
    Novel activation: ReLU on radius with phase preservation
    
    Original implementation: radius' = max(radius + shift, 0), keep angle
    """
    radius, theta = incoming.extract_polar()

    # Apply ReLU to shifted radius
    active_radius = F.relu(radius + shift)

    # Reconstruct from polar
    return BiChannel.build_polar(active_radius, theta)


def split_gate(incoming: BiChannel) -> BiChannel:
    """
    Independent ReLU on each channel
    No phase preservation - allows independent channel zeroing
    """
    return BiChannel(F.relu(incoming.axis_x), F.relu(incoming.axis_y))


def region_gate(incoming: BiChannel) -> BiChannel:
    """
    Novel quadrant-selective activation
    
    Original: Only keep values in quadrant 1 (angle in [0, π/2])
    Zero out other quadrants
    """
    radius, theta = incoming.extract_polar()

    # Mask for quadrant 1: 0 ≤ theta ≤ π/2
    in_quadrant = ((theta >= 0) & (theta <= math.pi / 2)).float()

    # Zero out values outside target quadrant
    filtered_radius = radius * in_quadrant

    return BiChannel.build_polar(filtered_radius, theta)


class TwoPathLayerNorm(nn.Module):
    """
    Layer normalization across feature dimension for dual-path
    Novel: normalizes magnitude distribution while preserving phase
    """

    def __init__(self, normalized_size: int, epsilon: float = 1e-4):
        super().__init__()
        self.epsilon = epsilon
        self.gain = nn.Parameter(torch.ones(normalized_size))
        self.bias = nn.Parameter(torch.zeros(normalized_size))

    def forward(self, incoming: BiChannel) -> BiChannel:
        # Extract magnitude per sample
        radius, theta = incoming.extract_polar()
        
        # Normalize radius across features
        mean_rad = radius.mean(dim=-1, keepdim=True)
        std_rad = radius.std(dim=-1, keepdim=True) + self.epsilon

        # Normalize
        normed_rad = (radius - mean_rad) / std_rad

        # Apply learnable transformation
        normed_rad = normed_rad * self.gain + self.bias

        # Reconstruct with normalized magnitude
        return BiChannel.build_polar(normed_rad, theta)


def adaptive_gate(incoming: BiChannel, threshold: torch.Tensor) -> BiChannel:
    """
    Learnable threshold activation on radius
    Threshold is per-feature, learned during training
    """
    radius, theta = incoming.extract_polar()
    
    # Apply soft threshold using sigmoid
    gate_val = torch.sigmoid((radius - threshold) * 5.0)
    
    # Modulate radius
    gated_radius = radius * gate_val
    
    return BiChannel.build_polar(gated_radius, theta)
