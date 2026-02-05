"""
Complete integrated system combining all harmonic field components
Novel pipeline with unique architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
from .biform import BiChannel, wave_merge
from .phase_net import BiMatrix, TwoPathNorm, MagPreserveDropout, phase_gate
from .phase_focus import BiPhaseScorer, GeometricPosition
from .broadcast import BroadcastHub, BroadcastLevel
from .potential import DescentPredictor


class BiEncoder(nn.Module):
    """
    Encoder that converts real input to BiChannel representation
    """

    def __init__(self, input_size: int, bichannel_size: int):
        super().__init__()
        self.prep = nn.Sequential(
            nn.Linear(input_size, bichannel_size),
            nn.LayerNorm(bichannel_size),
            nn.ReLU(),
        )
        self.to_x = nn.Linear(bichannel_size, bichannel_size)
        self.to_y = nn.Linear(bichannel_size, bichannel_size)

    def forward(self, data: torch.Tensor) -> BiChannel:
        """Convert real input to bichannel"""
        prepared = self.prep(data)
        x_channel = self.to_x(prepared)
        y_channel = self.to_y(prepared)
        return BiChannel(x_channel, y_channel)


class BiDecoder(nn.Module):
    """
    Decoder converting BiChannel to real output
    """

    def __init__(self, bichannel_size: int, output_size: int):
        super().__init__()
        # Decode both channels separately
        self.from_x = nn.Linear(bichannel_size, output_size)
        self.from_y = nn.Linear(bichannel_size, output_size)

        # Merge decoded channels
        self.merger = nn.Linear(output_size * 2, output_size)

    def forward(self, bich: BiChannel) -> torch.Tensor:
        """
        Convert bichannel to real by merging both paths
        """
        # Decode each channel
        decoded_x = self.from_x(bich.axis_x)
        decoded_y = self.from_y(bich.axis_y)

        # Concatenate and merge
        combined = torch.cat([decoded_x, decoded_y], dim=-1)
        output = self.merger(combined)

        return output


class FullHarmonicSystem(nn.Module):
    """
    Complete harmonic field architecture
    
    Novel pipeline:
    1. BiEncoder: Input → BiChannel space
    2. Phase layers: BiChannel transformations with phase awareness
    3. BiPhaseScorer: Phase-aware attention
    4. BroadcastHub: Global broadcast dynamics
    5. BiDecoder: BiChannel → Output
    
    Optional: DescentPredictor for energy-based refinement
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        phase_depth: int = 3,
        attention_heads: int = 8,
        hub_size: Optional[int] = None,
        use_energy: bool = False,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.use_energy = use_energy
        
        self.hub_size = hub_size or hidden_size

        # 1. Encoder to BiChannel
        self.encoder = BiEncoder(input_size, hidden_size)

        # 2. Phase processing layers
        self.phase_layers = nn.ModuleList()
        for _ in range(phase_depth):
            self.phase_layers.append(
                nn.ModuleDict({
                    "transform": BiMatrix(hidden_size, hidden_size),
                    "normalize": TwoPathNorm(hidden_size),
                    "dropout": MagPreserveDropout(dropout),
                })
            )

        # 3. Phase-aware attention
        self.attention = BiPhaseScorer(
            hidden_size,
            head_count=attention_heads,
            drop_rate=dropout,
        )

        # Positional encoding
        self.position_encoder = GeometricPosition(max_length=512, embed_size=hidden_size)

        # 4. Broadcast hub
        self.broadcast = BroadcastHub(
            hub_size=self.hub_size,
            specialist_count=8,
        )

        # Hub projection layers
        self.to_hub_x = nn.Linear(hidden_size, self.hub_size)
        self.to_hub_y = nn.Linear(hidden_size, self.hub_size)

        # 5. Decoder
        self.decoder = BiDecoder(hidden_size, output_size)

        # Optional energy model
        if use_energy:
            self.energy = DescentPredictor(
                input_size=hidden_size,
                output_size=output_size,
            )
        else:
            self.energy = None

    def forward(
        self,
        x: torch.Tensor,
        return_diagnostics: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        Full forward pass through harmonic system
        
        Args:
            x: Input [batch, seq_len, input_size] or [batch, input_size]
            return_diagnostics: Return detailed information
        
        Returns:
            output: Predictions [batch, output_size] or [batch, seq_len, output_size]
            diagnostics: Optional diagnostic dict
        """
        # Handle both sequence and single inputs
        is_seq = x.dim() == 3
        if not is_seq:
            x = x.unsqueeze(1)  # [batch, 1, input_size]

        batch, seq_len, _ = x.shape

        # 1. Encode to BiChannel
        bich = self.encoder(x)

        # Add positional encoding
        pos_enc = self.position_encoder(seq_len)
        bich = bich.sum_with(
            BiChannel(
                pos_enc.axis_x.unsqueeze(0).expand(batch, -1, -1),
                pos_enc.axis_y.unsqueeze(0).expand(batch, -1, -1),
            )
        )

        # 2. Phase processing with residual connections
        for layer_dict in self.phase_layers:
            # Transform
            transformed = layer_dict["transform"](bich)
            # Normalize
            normalized = layer_dict["normalize"](transformed)
            # Activate
            activated = phase_gate(normalized, shift=0.0)
            # Dropout
            activated = layer_dict["dropout"](activated)
            # Residual
            bich = bich.sum_with(activated.amplify(0.1))

        # 3. Phase-aware attention
        attended = self.attention(bich, bich, bich)

        # 4. Broadcast processing (on magnitude projection)
        # Project to hub dimension
        hub_input_x = self.to_hub_x(attended.axis_x)
        hub_input_y = self.to_hub_y(attended.axis_y)

        # Combine magnitude for hub
        hub_magnitude = torch.sqrt(hub_input_x ** 2 + hub_input_y ** 2 + 1e-8)

        # Process each position through hub
        hub_outputs = []
        broadcast_levels = []

        for t in range(seq_len):
            hub_out, level, _ = self.broadcast(
                hub_magnitude[:, t, :], return_details=return_diagnostics
            )
            hub_outputs.append(hub_out)
            broadcast_levels.append(level)

        hub_output = torch.stack(hub_outputs, dim=1)

        # Project back to hidden dimension using F.linear for efficiency
        # Reshape for batch processing: [batch, seq_len, hub_size] -> [batch*seq_len, hub_size]
        hub_flat = hub_output.reshape(-1, self.hub_size)
        hub_x_flat = F.linear(hub_flat, self.to_hub_x.weight.T)
        hub_y_flat = F.linear(hub_flat, self.to_hub_y.weight.T)
        # Reshape back: [batch*seq_len, hidden] -> [batch, seq_len, hidden]
        hub_x = hub_x_flat.reshape(batch, seq_len, self.hidden_size)
        hub_y = hub_y_flat.reshape(batch, seq_len, self.hidden_size)
        hub_bich = BiChannel(hub_x, hub_y)

        # Combine with attention output
        final_bich = attended.sum_with(hub_bich.amplify(0.3))

        # 5. Decode to output
        output = self.decoder(final_bich)

        # Optional energy refinement (expensive, eval only)
        if self.energy is not None and not self.training:
            output_flat = output.reshape(-1, self.output_size)
            refined, _ = self.energy.infer_output(output_flat, iterations=20)
            output = refined.reshape(batch, seq_len, self.output_size)

        # Remove sequence dimension if input wasn't sequence
        if not is_seq:
            output = output.squeeze(1)

        # Diagnostics
        diagnostics = None
        if return_diagnostics:
            diagnostics = {
                "broadcast_levels": broadcast_levels,
                "broadcast_count": self.broadcast.broadcast_counter.item(),
                "magnitude_mean": torch.sqrt(
                    final_bich.axis_x ** 2 + final_bich.axis_y ** 2
                ).mean().item(),
            }

        return output, diagnostics


def build_system(
    config: str = "balanced",
    input_size: int = 128,
    output_size: int = 10,
    **overrides,
) -> FullHarmonicSystem:
    """
    Factory for creating FullHarmonicSystem with presets
    
    Configs:
        - "balanced": General-purpose configuration
        - "compact": Smaller model for efficiency
        - "large": High-capacity for complex tasks
        - "energy": Includes energy-based refinement
        - "broadcast": Enhanced broadcast dynamics
    
    Args:
        config: Configuration preset name
        input_size: Input feature dimension
        output_size: Output dimension
        **overrides: Override any config parameter
    
    Returns:
        Configured FullHarmonicSystem
    """
    configs = {
        "balanced": {
            "hidden_size": 256,
            "phase_depth": 3,
            "attention_heads": 8,
            "hub_size": 256,
            "use_energy": False,
            "dropout": 0.1,
        },
        "compact": {
            "hidden_size": 128,
            "phase_depth": 2,
            "attention_heads": 4,
            "hub_size": 128,
            "use_energy": False,
            "dropout": 0.1,
        },
        "large": {
            "hidden_size": 512,
            "phase_depth": 6,
            "attention_heads": 16,
            "hub_size": 512,
            "use_energy": False,
            "dropout": 0.15,
        },
        "energy": {
            "hidden_size": 256,
            "phase_depth": 3,
            "attention_heads": 8,
            "hub_size": 256,
            "use_energy": True,
            "dropout": 0.1,
        },
        "broadcast": {
            "hidden_size": 256,
            "phase_depth": 2,
            "attention_heads": 8,
            "hub_size": 384,
            "use_energy": False,
            "dropout": 0.1,
        },
    }

    if config not in configs:
        raise ValueError(f"Unknown config: {config}. Available: {list(configs.keys())}")

    # Get config
    cfg = configs[config].copy()

    # Apply overrides
    cfg.update(overrides)

    # Build system
    system = FullHarmonicSystem(
        input_size=input_size,
        output_size=output_size,
        **cfg,
    )

    return system
