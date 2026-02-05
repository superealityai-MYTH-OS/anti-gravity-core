"""
Global broadcast architecture with consciousness states
Novel implementation using prediction error monitoring
"""

import torch
import torch.nn as nn
from enum import Enum
from typing import Optional, Tuple, Dict


class BroadcastLevel(Enum):
    """Four levels of information processing"""
    BELOW = 0  # Sub-threshold
    READY = 1  # Available but not active
    ACTIVE = 2  # Globally broadcast
    ROUTINE = 3  # Automatic/habitual


class BroadcastHub(nn.Module):
    """
    Global broadcast workspace with novel prediction error dynamics
    
    Original approach:
    - Monitor prediction error (current vs expected)
    - Trigger broadcast when error exceeds threshold
    - Four processing levels based on error magnitude
    """

    def __init__(
        self,
        hub_size: int,
        specialist_count: int = 8,
        broadcast_cutoff: float = 1.5,
        fade_rate: float = 0.9,
    ):
        super().__init__()
        self.hub_size = hub_size
        self.specialist_count = specialist_count
        self.broadcast_cutoff = broadcast_cutoff
        self.fade_rate = fade_rate

        # Global broadcast buffer
        self.register_buffer("hub_content", torch.zeros(1, hub_size))

        # Specialist processors (competing modules)
        self.specialists = nn.ModuleList([
            nn.Linear(hub_size, hub_size) for _ in range(specialist_count)
        ])

        # Selection gates for specialists
        self.gates = nn.ModuleList([
            nn.Linear(hub_size, 1) for _ in range(specialist_count)
        ])

        # Level classifier network
        self.level_net = nn.Sequential(
            nn.Linear(hub_size, hub_size // 2),
            nn.ReLU(),
            nn.Linear(hub_size // 2, 4),  # 4 levels
        )

        # Prediction error calculator
        self.error_net = nn.Sequential(
            nn.Linear(hub_size, hub_size // 4),
            nn.Tanh(),
            nn.Linear(hub_size // 4, 1),
        )

        # Tracking statistics
        self.register_buffer("broadcast_counter", torch.tensor(0))
        self.register_buffer("current_error", torch.tensor(0.0))

    def _compute_pred_error(
        self, current: torch.Tensor, expected: torch.Tensor
    ) -> torch.Tensor:
        """
        Novel prediction error combining reconstruction and complexity
        
        error = ||current - expected||² + complexity_penalty
        
        Complexity = magnitude ratio (prevents trivial solutions)
        """
        # Reconstruction error
        reconstruction = torch.sum((current - expected) ** 2, dim=-1)

        # Complexity penalty based on magnitude
        cur_mag = torch.norm(current, dim=-1)
        exp_mag = torch.norm(expected, dim=-1) + 1e-7
        complexity = cur_mag / exp_mag

        # Combined error
        total_error = reconstruction + 0.5 * complexity

        return total_error

    def _classify_level(
        self, data: torch.Tensor, error: torch.Tensor
    ) -> BroadcastLevel:
        """
        Classify processing level based on error and network prediction
        
        Novel logic:
        - error > cutoff -> ACTIVE (broadcast trigger)
        - High confidence from classifier -> use that
        - Low error -> BELOW
        - Default -> ROUTINE
        """
        # Get level predictions
        level_logits = self.level_net(data)
        level_probs = torch.softmax(level_logits, dim=-1)

        # Check for broadcast trigger
        if error > self.broadcast_cutoff:
            return BroadcastLevel.ACTIVE

        # High confidence prediction
        max_prob, max_level = level_probs.max(dim=-1)
        if max_prob > 0.7:
            return BroadcastLevel(max_level.item())

        # Low error -> below threshold
        if error < 0.5:
            return BroadcastLevel.BELOW
        
        # Default to routine
        return BroadcastLevel.ROUTINE

    def _trigger_broadcast(self, content: torch.Tensor):
        """Update global hub (broadcast event)"""
        self.hub_content = content.detach()
        self.broadcast_counter += 1

    def forward(
        self, input_data: torch.Tensor, return_details: bool = False
    ) -> Tuple[torch.Tensor, BroadcastLevel, Optional[Dict]]:
        """
        Process input through broadcast hub
        
        Args:
            input_data: Input [batch, hub_size]
            return_details: Return diagnostic information
        
        Returns:
            output: Processed representation
            level: Current processing level
            details: Optional diagnostics dict
        """
        batch = input_data.shape[0]

        # Predict from current hub state
        expected = self.error_net(self.hub_content.expand(batch, -1))
        expected = expected.expand(-1, self.hub_size)

        # Compute prediction error
        error = self._compute_pred_error(input_data, expected)
        self.current_error = error.mean()

        # Specialist competition
        specialist_outs = []
        gate_scores = []

        for specialist, gate in zip(self.specialists, self.gates):
            spec_out = specialist(input_data)
            gate_score = torch.sigmoid(gate(spec_out))

            specialist_outs.append(spec_out)
            gate_scores.append(gate_score)

        # Stack and compete
        specialist_outs = torch.stack(specialist_outs, dim=1)  # [batch, specs, size]
        gate_scores = torch.stack(gate_scores, dim=1)  # [batch, specs, 1]

        # Softmax competition
        gate_weights = torch.softmax(gate_scores, dim=1)

        # Weighted combination
        combined = (specialist_outs * gate_weights).sum(dim=1)

        # Classify level
        level = self._classify_level(combined, error.mean())

        # Level-dependent processing
        if level == BroadcastLevel.ACTIVE:
            # Full broadcast
            self._trigger_broadcast(combined)
            output = combined
        elif level == BroadcastLevel.READY:
            # Partial integration
            output = 0.7 * combined + 0.3 * self.hub_content.expand(batch, -1)
        elif level == BroadcastLevel.ROUTINE:
            # No integration (automatic)
            output = combined
        else:  # BELOW
            # Fade toward baseline
            output = self.fade_rate * combined + (1 - self.fade_rate) * self.hub_content.expand(batch, -1)

        # Diagnostics
        details = None
        if return_details:
            details = {
                "error": error.mean().item(),
                "level": level,
                "gate_weights": gate_weights,
                "broadcast_count": self.broadcast_counter.item(),
                "hub_magnitude": torch.norm(self.hub_content).item(),
            }

        return output, level, details

    def reset_hub(self):
        """Clear hub state"""
        self.hub_content.zero_()
        self.broadcast_counter.zero_()

    def get_status(self) -> Dict:
        """Get current hub statistics"""
        return {
            "hub_norm": torch.norm(self.hub_content).item(),
            "error": self.current_error.item(),
            "broadcast_count": self.broadcast_counter.item(),
        }


class SpecialistRouter(nn.Module):
    """
    Routing network with competing experts
    
    Novel: Winner-take-most dynamics with soft competition
    """

    def __init__(self, in_size: int, out_size: int, expert_count: int = 4):
        super().__init__()
        self.expert_count = expert_count

        # Expert networks
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_size, out_size),
                nn.LayerNorm(out_size),
                nn.ReLU(),
                nn.Linear(out_size, out_size),
            )
            for _ in range(expert_count)
        ])

        # Router (selects experts)
        self.router = nn.Sequential(
            nn.Linear(in_size, expert_count * 2),
            nn.ReLU(),
            nn.Linear(expert_count * 2, expert_count),
        )

    def forward(
        self, input_data: torch.Tensor, temp: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Route input to experts
        
        Args:
            input_data: Input [batch, in_size]
            temp: Softmax temperature (lower = more competitive)
        
        Returns:
            output: Expert mixture [batch, out_size]
            routing_weights: Expert selection [batch, expert_count]
        """
        # Compute expert outputs
        expert_outs = [expert(input_data) for expert in self.experts]
        expert_outs = torch.stack(expert_outs, dim=1)  # [batch, experts, out_size]

        # Routing scores
        route_logits = self.router(input_data)
        route_weights = torch.softmax(route_logits / temp, dim=-1)  # [batch, experts]

        # Weighted mixture
        output = torch.einsum("beo,be->bo", expert_outs, route_weights)

        return output, route_weights


class HubIntegrator(nn.Module):
    """
    Integration layer combining hub state with input features
    """

    def __init__(self, feature_size: int, hub_size: int):
        super().__init__()
        self.feature_size = feature_size
        self.hub_size = hub_size

        # Feature -> hub projection
        self.feat_to_hub = nn.Linear(feature_size, hub_size)

        # Hub -> feature projection
        self.hub_to_feat = nn.Linear(hub_size, feature_size)

        # Gating mechanism
        self.gate_net = nn.Sequential(
            nn.Linear(feature_size + hub_size, 1),
            nn.Sigmoid(),
        )

    def forward(
        self, features: torch.Tensor, hub_state: torch.Tensor
    ) -> torch.Tensor:
        """
        Integrate features with hub state
        
        Args:
            features: Input features [batch, feature_size]
            hub_state: Hub content [1, hub_size]
        
        Returns:
            integrated: Combined representation [batch, feature_size]
        """
        batch = features.shape[0]
        hub_expanded = hub_state.expand(batch, -1)

        # Project hub to feature space
        hub_features = self.hub_to_feat(hub_expanded)

        # Compute integration gate
        concat = torch.cat([features, hub_expanded], dim=-1)
        gate = self.gate_net(concat)

        # Gated integration
        integrated = gate * features + (1 - gate) * hub_features

        return integrated


class PredictiveCache(nn.Module):
    """
    Predictive caching system for common patterns
    Learns to predict and cache frequently accessed representations
    """

    def __init__(self, size: int, capacity: int = 10):
        super().__init__()
        self.size = size
        self.capacity = capacity

        # Cache storage
        self.register_buffer("cache_keys", torch.randn(capacity, size))
        self.register_buffer("cache_values", torch.randn(capacity, size))
        self.register_buffer("access_counts", torch.zeros(capacity))

        # Query network
        self.query_net = nn.Linear(size, size)

    def forward(self, query: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """
        Query cache and return best match
        
        Returns:
            cached_value: Retrieved value
            confidence: Match quality (0-1)
        """
        # Project query
        projected_query = self.query_net(query)

        # Compute similarities to cache keys
        similarities = F.cosine_similarity(
            projected_query.unsqueeze(1),
            self.cache_keys.unsqueeze(0),
            dim=-1
        )

        # Get best match
        confidence, best_idx = similarities.max(dim=-1)

        # Update access count
        self.access_counts[best_idx] += 1

        return self.cache_values[best_idx], confidence.item()

    def update_cache(self, key: torch.Tensor, value: torch.Tensor):
        """Add or update cache entry"""
        # Find least accessed slot
        min_idx = self.access_counts.argmin()

        # Update
        self.cache_keys[min_idx] = key.detach()
        self.cache_values[min_idx] = value.detach()
        self.access_counts[min_idx] = 0
