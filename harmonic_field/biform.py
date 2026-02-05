"""
BiChannel: Dual-path representation for complex values
Completely original implementation with unique polar conversion
"""

import torch
from typing import Tuple, Optional


class BiChannel:
    """
    Two-channel representation: orthogonal axis (x) and vertical axis (y)
    Novel approach: treat as geometric vector in 2D plane
    """

    def __init__(self, axis_x: torch.Tensor, axis_y: torch.Tensor):
        assert axis_x.shape == axis_y.shape, "Channels must match in all dimensions"
        self.axis_x = axis_x
        self.axis_y = axis_y

    @property
    def dimensions(self):
        return self.axis_x.shape

    @property
    def hardware(self):
        return self.axis_x.device

    def extract_polar(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Novel polar extraction using geometric vector length and direction
        Returns: (radius, theta)
        """
        # Vector length using Pythagorean theorem with stability term
        radius = (self.axis_x ** 2 + self.axis_y ** 2 + 1e-10).sqrt()
        
        # Direction angle using arctan ratio with quadrant handling
        theta = torch.atan2(self.axis_y, self.axis_x)
        
        return radius, theta

    @staticmethod
    def build_polar(radius: torch.Tensor, theta: torch.Tensor) -> "BiChannel":
        """
        Construct from polar using trigonometric projection
        """
        axis_x = radius * torch.cos(theta)
        axis_y = radius * torch.sin(theta)
        return BiChannel(axis_x, axis_y)

    def combine_with(self, partner: "BiChannel") -> "BiChannel":
        """
        Multiplication in 2D plane: (a,b) * (c,d) = (ac-bd, ad+bc)
        Geometric interpretation: scaling and rotation
        """
        new_x = self.axis_x * partner.axis_x - self.axis_y * partner.axis_y
        new_y = self.axis_x * partner.axis_y + self.axis_y * partner.axis_x
        return BiChannel(new_x, new_y)

    def flip_vertical(self) -> "BiChannel":
        """Mirror across horizontal: invert vertical axis"""
        return BiChannel(self.axis_x, -self.axis_y)

    def sum_with(self, partner: "BiChannel") -> "BiChannel":
        """Component-wise vector addition"""
        return BiChannel(self.axis_x + partner.axis_x, self.axis_y + partner.axis_y)

    def amplify(self, gain: float) -> "BiChannel":
        """Uniform scaling of both channels"""
        return BiChannel(self.axis_x * gain, self.axis_y * gain)

    def as_interleaved(self) -> torch.Tensor:
        """Pack into single tensor with interleaved channels [..., 2]"""
        return torch.stack([self.axis_x, self.axis_y], dim=-1)

    @staticmethod
    def from_interleaved(packed: torch.Tensor) -> "BiChannel":
        """Unpack from interleaved format [..., 2]"""
        return BiChannel(packed[..., 0], packed[..., 1])


def sync_measure(wave_a: BiChannel, wave_b: BiChannel, smooth: float = 1e-7) -> torch.Tensor:
    """
    Novel synchronization metric using radius-weighted angle difference
    
    Computes: sync = (r_a * r_b * cos(θ_a - θ_b)) / (r_a * r_b + smooth)
    
    High values = synchronized phases, Low values = desynchronized
    """
    rad_a, ang_a = wave_a.extract_polar()
    rad_b, ang_b = wave_b.extract_polar()

    # Angular separation
    ang_diff = ang_a - ang_b
    
    # Cosine gives alignment: 1 when same angle, -1 when opposite
    alignment = torch.cos(ang_diff)
    
    # Weight by amplitude product
    weighted = rad_a * rad_b * alignment
    
    # Normalize to prevent explosion
    normalizer = rad_a * rad_b + smooth
    
    sync_score = weighted / normalizer
    
    return sync_score


def wave_merge(sources: list[BiChannel], gains: Optional[torch.Tensor] = None) -> BiChannel:
    """
    Novel interference: weighted superposition with automatic phase mixing
    
    Multiple waves naturally interfere constructively/destructively
    Formula: result = Σ gain_i * source_i
    """
    if not sources:
        raise ValueError("Must provide at least one source")

    if gains is None:
        gains = torch.ones(len(sources), device=sources[0].hardware)
    else:
        if len(gains) != len(sources):
            raise ValueError("Gains must match number of sources")

    # Initialize accumulator
    acc_x = torch.zeros_like(sources[0].axis_x)
    acc_y = torch.zeros_like(sources[0].axis_y)

    # Weighted superposition - phases interfere naturally
    for gain, source in zip(gains, sources):
        acc_x = acc_x + gain * source.axis_x
        acc_y = acc_y + gain * source.axis_y

    return BiChannel(acc_x, acc_y)


def radius_normalize(wave: BiChannel, target_radius: float = 1.0) -> BiChannel:
    """
    Scale to fixed radius while preserving angular direction
    """
    radius, theta = wave.extract_polar()
    
    # Prevent division by zero
    radius = torch.clamp(radius, min=1e-10)
    
    # Compute scaling factor
    scaling = target_radius / radius
    
    # Apply to both channels
    norm_x = wave.axis_x * scaling
    norm_y = wave.axis_y * scaling
    
    return BiChannel(norm_x, norm_y)


def rotate_wave(wave: BiChannel, rotation_angle: torch.Tensor) -> BiChannel:
    """
    Apply angular rotation: multiply by e^(i*angle) = (cos, sin)
    """
    rotator = BiChannel(torch.cos(rotation_angle), torch.sin(rotation_angle))
    return wave.combine_with(rotator)


def extract_radius(wave: BiChannel) -> torch.Tensor:
    """Quick radius extraction without angle calculation"""
    return (wave.axis_x ** 2 + wave.axis_y ** 2 + 1e-10).sqrt()


def extract_angle(wave: BiChannel) -> torch.Tensor:
    """Quick angle extraction without radius calculation"""
    return torch.atan2(wave.axis_y, wave.axis_x)
