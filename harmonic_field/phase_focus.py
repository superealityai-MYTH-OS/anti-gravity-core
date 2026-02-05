"""
Phase-aware attention with novel scoring and surprise jumps
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .biform import BiChannel
from typing import Optional, Tuple
import math


class BiPhaseScorer(nn.Module):
    """
    Attention mechanism with phase-magnitude hybrid scoring
    
    Novel scoring formula: score = β·angle_match + (1-β)·magnitude_dot
    
    Balances geometric (phase) and algebraic (magnitude) properties
    """

    def __init__(
        self,
        embed_size: int,
        head_count: int = 8,
        phase_beta: float = 0.5,
        drop_rate: float = 0.1,
    ):
        super().__init__()
        self.embed_size = embed_size
        self.head_count = head_count
        self.phase_beta = phase_beta
        self.per_head_size = embed_size // head_count

        if self.per_head_size * head_count != embed_size:
            raise ValueError("embed_size must be divisible by head_count")

        # Query, Key, Value projections for dual-path
        self.query_x = nn.Linear(embed_size, embed_size)
        self.query_y = nn.Linear(embed_size, embed_size)
        self.key_x = nn.Linear(embed_size, embed_size)
        self.key_y = nn.Linear(embed_size, embed_size)
        self.value_x = nn.Linear(embed_size, embed_size)
        self.value_y = nn.Linear(embed_size, embed_size)

        # Output merger
        self.out_x = nn.Linear(embed_size, embed_size)
        self.out_y = nn.Linear(embed_size, embed_size)

        self.dropout = nn.Dropout(drop_rate)
        self.scale = math.sqrt(self.per_head_size)

    def _split_heads(self, tensor: torch.Tensor, batch: int) -> torch.Tensor:
        """
        Split into attention heads
        [batch, seq, embed] -> [batch, heads, seq, per_head]
        """
        tensor = tensor.view(batch, -1, self.head_count, self.per_head_size)
        return tensor.transpose(1, 2)

    def _hybrid_score(self, query: BiChannel, key: BiChannel) -> torch.Tensor:
        """
        Novel hybrid scoring combining phase and magnitude
        
        score = β·cos(θ_q - θ_k) + (1-β)·(|q|·|k|)/scale
        """
        # Extract polar coordinates
        query_rad, query_ang = query.extract_polar()
        key_rad, key_ang = key.extract_polar()

        # Phase alignment score
        # query: [batch, heads, seq_q, per_head] -> [batch, heads, seq_q, 1, per_head]
        # key:   [batch, heads, seq_k, per_head] -> [batch, heads, 1, seq_k, per_head]
        angle_diff = query_ang.unsqueeze(-2) - key_ang.unsqueeze(-3)
        phase_score = torch.cos(angle_diff).mean(dim=-1)  # Average over features: [batch, heads, seq_q, seq_k]

        # Magnitude dot product
        mag_score = torch.matmul(query_rad, key_rad.transpose(-2, -1)) / self.scale

        # Weighted combination
        hybrid = self.phase_beta * phase_score + (1 - self.phase_beta) * mag_score

        return hybrid

    def forward(
        self,
        query_in: BiChannel,
        key_in: BiChannel,
        value_in: BiChannel,
        mask: Optional[torch.Tensor] = None,
    ) -> BiChannel:
        """
        Apply phase-aware multi-head attention
        
        Args:
            query_in: Query bichannel [batch, seq, embed]
            key_in: Key bichannel [batch, seq, embed]
            value_in: Value bichannel [batch, seq, embed]
            mask: Optional attention mask
        
        Returns:
            Attended output as BiChannel
        """
        batch = query_in.axis_x.shape[0]

        # Project Q, K, V
        q_x = self.query_x(query_in.axis_x)
        q_y = self.query_y(query_in.axis_y)
        k_x = self.key_x(key_in.axis_x)
        k_y = self.key_y(key_in.axis_y)
        v_x = self.value_x(value_in.axis_x)
        v_y = self.value_y(value_in.axis_y)

        # Split into heads
        q_x = self._split_heads(q_x, batch)
        q_y = self._split_heads(q_y, batch)
        k_x = self._split_heads(k_x, batch)
        k_y = self._split_heads(k_y, batch)
        v_x = self._split_heads(v_x, batch)
        v_y = self._split_heads(v_y, batch)

        q = BiChannel(q_x, q_y)
        k = BiChannel(k_x, k_y)
        v = BiChannel(v_x, v_y)

        # Compute attention scores
        scores = self._hybrid_score(q, k)

        # Apply mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        # Softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply to values
        attended_x = torch.matmul(attn_weights, v.axis_x)
        attended_y = torch.matmul(attn_weights, v.axis_y)

        # Merge heads
        attended_x = attended_x.transpose(1, 2).contiguous()
        attended_y = attended_y.transpose(1, 2).contiguous()
        attended_x = attended_x.view(batch, -1, self.embed_size)
        attended_y = attended_y.view(batch, -1, self.embed_size)

        # Output projection
        out_x = self.out_x(attended_x)
        out_y = self.out_y(attended_y)

        return BiChannel(out_x, out_y)


class UnexpectedJump(nn.Module):
    """
    Attention with surprise-driven teleportation
    
    Novel mechanism: When attention entropy is high (surprise), jump to
    unexpected location instead of weighted average
    
    Enables exploration of low-probability but high-impact patterns
    """

    def __init__(
        self,
        embed_size: int,
        surprise_cutoff: float = 2.0,
        jump_chance: float = 0.3,
    ):
        super().__init__()
        self.embed_size = embed_size
        self.surprise_cutoff = surprise_cutoff
        self.jump_chance = jump_chance

        # Standard attention components
        self.query_proj = nn.Linear(embed_size, embed_size)
        self.key_proj = nn.Linear(embed_size, embed_size)
        self.value_proj = nn.Linear(embed_size, embed_size)
        self.output_proj = nn.Linear(embed_size, embed_size)

        self.scale = math.sqrt(embed_size)

    def _calc_surprise(self, attn_weights: torch.Tensor) -> torch.Tensor:
        """
        Surprise = -log(probability)
        
        High surprise = low probability = unexpected pattern
        """
        # Clip to prevent log(0)
        safe_weights = torch.clamp(attn_weights, min=1e-9)
        surprise = -torch.log(safe_weights)
        
        return surprise

    def forward(
        self,
        input_data: torch.Tensor,
        enable_jump: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply attention with optional surprise jumps
        
        Returns:
            output: Attended result
            jump_indicators: Binary mask showing jump locations
        """
        batch, seq_len, _ = input_data.shape

        # Compute Q, K, V
        queries = self.query_proj(input_data)
        keys = self.key_proj(input_data)
        values = self.value_proj(input_data)

        # Attention scores
        scores = torch.matmul(queries, keys.transpose(-2, -1)) / self.scale
        attn_weights = F.softmax(scores, dim=-1)

        # Calculate surprise
        surprise = self._calc_surprise(attn_weights)
        max_surprise, surprise_idx = surprise.max(dim=-1)

        # Determine jump locations
        jump_indicators = torch.zeros(batch, seq_len, dtype=torch.bool, device=input_data.device)

        if enable_jump and self.training:
            # Jump when: (1) surprise exceeds cutoff AND (2) random chance
            exceed_cutoff = (max_surprise > self.surprise_cutoff).float()
            random_trigger = (torch.rand(batch, seq_len, device=input_data.device) < self.jump_chance).float()
            jump_indicators = (exceed_cutoff * random_trigger).bool()

        # Standard weighted attention
        attended = torch.matmul(attn_weights, values)

        # Replace jump positions with high-surprise values
        if jump_indicators.any():
            surprise_values = values[
                torch.arange(batch).unsqueeze(1), surprise_idx
            ]
            attended = torch.where(
                jump_indicators.unsqueeze(-1), surprise_values, attended
            )

        # Output projection
        output = self.output_proj(attended)

        return output, jump_indicators


class GeometricPosition(nn.Module):
    """
    Positional encoding using phase rotation
    
    Novel approach: Position encoded as complex rotation angle
    pos -> (cos(ω·pos), sin(ω·pos)) for multiple frequencies ω
    """

    def __init__(self, max_length: int, embed_size: int):
        super().__init__()
        self.max_length = max_length
        self.embed_size = embed_size

        # Compute angle matrix
        positions = torch.arange(max_length, dtype=torch.float32).unsqueeze(1)
        dims = torch.arange(embed_size, dtype=torch.float32).unsqueeze(0)

        # Angular frequencies: 2π·pos / (10000^(2i/d))
        frequencies = 2 * math.pi * positions / (10000 ** (2 * dims / embed_size))

        # Store as cos/sin (BiChannel format)
        self.register_buffer("pos_x", torch.cos(frequencies))
        self.register_buffer("pos_y", torch.sin(frequencies))

    def forward(self, length: int) -> BiChannel:
        """
        Get positional encoding for sequence length
        
        Returns: BiChannel [length, embed_size]
        """
        if length > self.max_length:
            raise ValueError(f"Length {length} exceeds max {self.max_length}")

        return BiChannel(self.pos_x[:length], self.pos_y[:length])


class LocalWindowAttention(nn.Module):
    """
    Attention with local window restriction for efficiency
    Each position only attends to nearby positions
    """

    def __init__(self, embed_size: int, window_size: int = 64):
        super().__init__()
        self.embed_size = embed_size
        self.window_size = window_size

        self.qkv_proj = nn.Linear(embed_size, embed_size * 3)
        self.output_proj = nn.Linear(embed_size, embed_size)
        self.scale = math.sqrt(embed_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply windowed attention
        
        Args:
            x: Input [batch, seq_len, embed_size]
        
        Returns:
            Output [batch, seq_len, embed_size]
        """
        batch, seq_len, _ = x.shape

        # Project to Q, K, V
        qkv = self.qkv_proj(x)
        q, k, v = torch.chunk(qkv, 3, dim=-1)

        # For simplicity, using global attention with mask
        # In practice, would implement efficient windowed computation
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale

        # Create window mask
        positions = torch.arange(seq_len, device=x.device)
        distance = positions.unsqueeze(0) - positions.unsqueeze(1)
        window_mask = distance.abs() <= self.window_size

        # Apply mask
        scores = scores.masked_fill(~window_mask, float("-inf"))

        # Softmax and attend
        attn = F.softmax(scores, dim=-1)
        attended = torch.matmul(attn, v)

        # Output
        return self.output_proj(attended)
