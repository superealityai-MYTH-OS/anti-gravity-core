"""
Energy-based models with novel inference and cascade dynamics
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict


class DescentPredictor(nn.Module):
    """
    Energy model using iterative gradient descent for prediction
    
    Novel approach: Learn energy surface E(x,y), infer y by descending gradient
    
    Inference: y* = argmin_y E(x,y) via iterative gradient updates on y
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int = 256,
        depth: int = 3,
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size

        # Energy network E(x, y) -> scalar
        layers = []
        current_size = input_size + output_size

        for i in range(depth):
            layers.extend([
                nn.Linear(current_size, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.SiLU(),
            ])
            current_size = hidden_size

        layers.append(nn.Linear(hidden_size, 1))  # Scalar energy output

        self.energy_net = nn.Sequential(*layers)

    def compute_energy(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Evaluate energy E(x, y)
        
        Lower energy = better x-y pairing
        
        Args:
            x: Input [batch, input_size]
            y: Output [batch, output_size]
        
        Returns:
            energy: Scalar per sample [batch, 1]
        """
        xy = torch.cat([x, y], dim=-1)
        return self.energy_net(xy)

    def infer_output(
        self,
        x: torch.Tensor,
        iterations: int = 50,
        learning_rate: float = 0.1,
        init_y: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[float]]:
        """
        Infer y by minimizing energy via gradient descent
        
        Novel algorithm:
        1. Initialize y (random or from prior)
        2. Compute ∇_y E(x, y)
        3. Update: y ← y - lr * ∇_y E
        4. Repeat until convergence
        
        Args:
            x: Input [batch, input_size]
            iterations: Number of descent steps
            learning_rate: Step size
            init_y: Optional initialization
        
        Returns:
            y_final: Inferred output [batch, output_size]
            energy_history: Energy at each step
        """
        batch = x.shape[0]

        # Initialize y
        if init_y is None:
            y = torch.randn(batch, self.output_size, device=x.device, requires_grad=True)
        else:
            y = init_y.clone().requires_grad_(True)

        energy_history = []

        # Gradient descent on y
        opt = torch.optim.SGD([y], lr=learning_rate)

        for step in range(iterations):
            opt.zero_grad()

            # Compute energy
            energy = self.compute_energy(x, y)
            total_e = energy.mean()

            # Record
            energy_history.append(total_e.item())

            # Backprop through y
            total_e.backward()

            # Update y
            opt.step()

        return y.detach(), energy_history

    def forward(
        self, x: torch.Tensor, y: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward for training or inference
        
        If y provided: return energy (for contrastive learning)
        If y is None: perform inference
        """
        if y is not None:
            return self.compute_energy(x, y)
        else:
            y_pred, _ = self.infer_output(x)
            return y_pred


class CascadeSystem(nn.Module):
    """
    Self-organized criticality with avalanche cascades
    
    Novel algorithm: Nodes activate above threshold, triggering neighbors
    Power-law avalanche distribution emerges at criticality
    """

    def __init__(
        self,
        node_count: int = 256,
        threshold: float = 1.0,
        dissipation: float = 0.1,
    ):
        super().__init__()
        self.node_count = node_count
        self.threshold = threshold
        self.dissipation = dissipation

        # Sparse connection matrix
        self.register_buffer(
            "connections",
            self._make_sparse_network(node_count),
        )

        # Node activation states
        self.register_buffer("activations", torch.zeros(node_count))

        # Input transformation
        self.input_proj = nn.Linear(node_count, node_count)

        # Avalanche tracking
        self.avalanche_history = []

    def _make_sparse_network(self, node_count: int, sparsity: float = 0.9) -> torch.Tensor:
        """
        Create sparse random connectivity
        
        ~10% connectivity for critical dynamics
        """
        connections = torch.randn(node_count, node_count)

        # Apply sparsity
        mask = torch.rand(node_count, node_count) > sparsity
        connections = connections * mask.float()

        # Normalize rows
        row_norms = connections.abs().sum(dim=1, keepdim=True) + 1e-6
        connections = connections / row_norms

        return connections

    def _run_avalanche(
        self, initial: torch.Tensor, max_iterations: int = 100
    ) -> Tuple[torch.Tensor, int]:
        """
        Simulate avalanche dynamics
        
        Algorithm:
        1. Nodes above threshold become active
        2. Active nodes propagate through connections
        3. Apply dissipation
        4. Repeat until no new activations
        
        Returns:
            final_state: Node states after avalanche
            size: Total number of activations
        """
        state = initial.clone()
        total_activations = 0

        for iteration in range(max_iterations):
            # Find active nodes
            active = (state > self.threshold).float()

            # Count new activations
            n_active = active.sum().item()

            if n_active == 0:
                break  # Avalanche ended

            total_activations += n_active

            # Propagate through connections
            propagated = torch.matmul(self.connections, active.unsqueeze(-1)).squeeze(-1)

            # Update state with dissipation
            state = state * (1 - self.dissipation) + propagated * (1 - active)

            # Reset activated nodes
            state = state * (1 - active)

        return state, total_activations

    def forward(
        self, input_data: torch.Tensor, track: bool = True
    ) -> Tuple[torch.Tensor, Optional[int]]:
        """
        Process through avalanche dynamics
        
        Args:
            input_data: Input [batch, node_count]
            track: Whether to track avalanche statistics
        
        Returns:
            output: Final states [batch, node_count]
            max_avalanche: Size of largest avalanche
        """
        batch = input_data.shape[0]

        # Project input
        projected = self.input_proj(input_data)

        # Process each sample (avalanches are sequential)
        outputs = []
        max_size = 0

        for i in range(batch):
            initial = projected[i]

            # Run avalanche
            final, size = self._run_avalanche(initial)

            outputs.append(final)

            if track:
                max_size = max(max_size, size)
                self.avalanche_history.append(size)

        output = torch.stack(outputs, dim=0)

        return output, max_size if track else None

    def get_statistics(self) -> Dict:
        """
        Get avalanche statistics
        
        At criticality, shows power-law distribution
        """
        if len(self.avalanche_history) < 10:
            return {"mean": 0, "max": 0, "count": 0}

        recent = torch.tensor(self.avalanche_history[-1000:], dtype=torch.float32)

        return {
            "mean": recent.mean().item(),
            "max": recent.max().item(),
            "std": recent.std().item(),
            "count": len(recent),
        }

    def reset_history(self):
        """Clear avalanche tracking"""
        self.avalanche_history.clear()


class ContrastiveTrainer(nn.Module):
    """
    Training wrapper for energy models using contrastive learning
    
    Novel negative sampling via energy maximization
    """

    def __init__(self, energy_model: DescentPredictor):
        super().__init__()
        self.energy_model = energy_model

    def generate_negatives(
        self,
        x: torch.Tensor,
        n_steps: int = 20,
        step_size: float = 0.2,
    ) -> torch.Tensor:
        """
        Generate negative samples by maximizing energy
        
        Start from noise and ascend energy gradient
        """
        batch = x.shape[0]
        y_neg = torch.randn(
            batch, self.energy_model.output_size, device=x.device, requires_grad=True
        )

        opt = torch.optim.SGD([y_neg], lr=step_size)

        for _ in range(n_steps):
            opt.zero_grad()

            # Maximize energy (negative loss)
            energy = self.energy_model.compute_energy(x, y_neg)
            loss = -energy.mean()  # Negate for maximization

            loss.backward()
            opt.step()

        return y_neg.detach()

    def contrastive_loss(
        self, x: torch.Tensor, y_pos: torch.Tensor, n_negatives: int = 1
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Contrastive loss: minimize positive energy, maximize negative
        
        loss = E(x, y_pos) - E(x, y_neg)
        
        Goal: y_pos has low energy, y_neg has high energy
        """
        # Positive energy
        e_pos = self.energy_model.compute_energy(x, y_pos)

        # Generate and compute negative energies
        e_neg_total = 0
        for _ in range(n_negatives):
            y_neg = self.generate_negatives(x)
            e_neg = self.energy_model.compute_energy(x, y_neg)
            e_neg_total += e_neg

        e_neg_avg = e_neg_total / n_negatives

        # Contrastive loss
        loss = e_pos.mean() - e_neg_avg.mean()

        stats = {
            "energy_pos": e_pos.mean().item(),
            "energy_neg": e_neg_avg.mean().item(),
            "gap": (e_neg_avg - e_pos).mean().item(),
        }

        return loss, stats


class AdaptiveEnergyScheduler:
    """
    Schedule for energy-based inference
    
    Adapts iteration count based on convergence rate
    """

    def __init__(self, min_iters: int = 10, max_iters: int = 100, tolerance: float = 1e-3):
        self.min_iters = min_iters
        self.max_iters = max_iters
        self.tolerance = tolerance

    def should_stop(self, energy_history: List[float], current_iter: int) -> bool:
        """
        Decide whether to stop inference early
        
        Stop if: (1) reached min iters AND (2) energy change < tolerance
        """
        if current_iter < self.min_iters:
            return False

        if current_iter >= self.max_iters:
            return True

        # Check convergence
        if len(energy_history) >= 2:
            change = abs(energy_history[-1] - energy_history[-2])
            if change < self.tolerance:
                return True

        return False
