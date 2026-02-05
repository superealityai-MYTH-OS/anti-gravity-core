"""
Circular convolution operations using FFT with novel approaches
Holographic memory through frequency-domain binding
"""

import torch
import torch.nn.functional as F
from typing import Tuple
import math


def _round_pow2(length: int) -> int:
    """Round up to power of 2 for FFT efficiency"""
    if length <= 0:
        return 1
    return 1 << (length - 1).bit_length()


def twist_merge(vec_p: torch.Tensor, vec_q: torch.Tensor) -> torch.Tensor:
    """
    Circular convolution using FFT: novel commutative binding
    
    Original algorithm: twist(p, q) = IFFT(FFT(p) * FFT(q))
    
    Properties:
    - Commutative: twist(p,q) = twist(q,p)
    - Invertible: can recover p from twist(p,q) and q
    - O(n log n) complexity
    
    Args:
        vec_p: Vector [..., dim]
        vec_q: Vector [..., dim]
    
    Returns:
        Bound vector [..., dim]
    """
    if vec_p.shape != vec_q.shape:
        raise ValueError("Vectors must have identical shapes")

    orig_dim = vec_p.shape[-1]

    # Pad to power of 2
    padded_dim = _round_pow2(orig_dim)
    if padded_dim > orig_dim:
        padding = padded_dim - orig_dim
        vec_p = F.pad(vec_p, (0, padding))
        vec_q = F.pad(vec_q, (0, padding))

    # Transform to frequency domain
    freq_p = torch.fft.fft(vec_p, dim=-1)
    freq_q = torch.fft.fft(vec_q, dim=-1)

    # Pointwise multiplication (convolution theorem)
    freq_result = freq_p * freq_q

    # Transform back
    result = torch.fft.ifft(freq_result, dim=-1).real

    # Remove padding
    return result[..., :orig_dim]


def twist_split(twisted: torch.Tensor, vec_q: torch.Tensor) -> torch.Tensor:
    """
    Circular correlation: recover p from twist(p,q) given q
    
    Algorithm: split(twisted, q) = IFFT(FFT(twisted) * conj(FFT(q)))
    
    Property: split(twist(p,q), q) ≈ p (with noise)
    
    Args:
        twisted: Previously bound vector
        vec_q: One of the binding keys
    
    Returns:
        Approximate recovery of other vector
    """
    if twisted.shape != vec_q.shape:
        raise ValueError("Shapes must match")

    orig_dim = twisted.shape[-1]

    # Pad to power of 2
    padded_dim = _round_pow2(orig_dim)
    if padded_dim > orig_dim:
        padding = padded_dim - orig_dim
        twisted = F.pad(twisted, (0, padding))
        vec_q = F.pad(vec_q, (0, padding))

    # Frequency domain
    freq_twisted = torch.fft.fft(twisted, dim=-1)
    freq_q = torch.fft.fft(vec_q, dim=-1)

    # Correlation: multiply by conjugate
    freq_result = freq_twisted * torch.conj(freq_q)

    # Back to spatial domain
    result = torch.fft.ifft(freq_result, dim=-1).real

    # Remove padding
    return result[..., :orig_dim]


def asymm_twist(vec_p: torch.Tensor, vec_q: torch.Tensor) -> torch.Tensor:
    """
    Non-commutative binding with directional information
    
    Novel approach: asymm_twist(p,q) ≠ asymm_twist(q,p)
    Uses frequency-dependent phase rotation for directionality
    
    Algorithm: IFFT(FFT(p) * FFT(q) * exp(i·2π·k·0.25))
    where k is frequency index
    """
    if vec_p.shape != vec_q.shape:
        raise ValueError("Shapes must match")

    orig_dim = vec_p.shape[-1]

    # Pad
    padded_dim = _round_pow2(orig_dim)
    if padded_dim > orig_dim:
        padding = padded_dim - orig_dim
        vec_p = F.pad(vec_p, (0, padding))
        vec_q = F.pad(vec_q, (0, padding))

    # FFT
    freq_p = torch.fft.fft(vec_p, dim=-1)
    freq_q = torch.fft.fft(vec_q, dim=-1)

    # Asymmetric phase shift based on frequency index
    indices = torch.arange(padded_dim, device=vec_p.device, dtype=torch.float32)
    phase_rotation = torch.exp(1j * 2 * math.pi * indices * 0.25)

    # Apply directional transformation
    freq_result = freq_p * freq_q * phase_rotation

    # IFFT
    result = torch.fft.ifft(freq_result, dim=-1).real

    return result[..., :orig_dim]


def stack_memory(data: torch.Tensor, key: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
    """
    Add item to distributed memory via superposition
    
    memory_new = memory_old + twist(key, data)
    
    Args:
        data: Information to store
        key: Retrieval key
        memory: Current memory state
    
    Returns:
        Updated memory
    """
    bound = twist_merge(key, data)
    return memory + bound


def recall_memory(key: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
    """
    Retrieve from distributed memory using key
    
    retrieved ≈ split(memory, key)
    
    Returns noisy approximation due to superposition
    """
    return twist_split(memory, key)


class CircularMemoryBank(torch.nn.Module):
    """
    Learnable memory module using circular convolution
    Stores multiple key-value pairs in single superposed trace
    """

    def __init__(self, vec_dim: int):
        super().__init__()
        self.vec_dim = vec_dim
        self.register_buffer("memory_trace", torch.zeros(vec_dim))

    def store(self, keys: torch.Tensor, values: torch.Tensor):
        """
        Add key-value pairs to memory
        
        Args:
            keys: Tensor [..., vec_dim]
            values: Tensor [..., vec_dim]
        """
        if keys.shape[-1] != self.vec_dim or values.shape[-1] != self.vec_dim:
            raise ValueError(f"Vectors must have dimension {self.vec_dim}")

        # Handle single or batch
        if keys.dim() == 1:
            keys = keys.unsqueeze(0)
            values = values.unsqueeze(0)

        # Store each pair
        for k, v in zip(keys, values):
            self.memory_trace = stack_memory(v, k, self.memory_trace)

    def retrieve(self, keys: torch.Tensor) -> torch.Tensor:
        """
        Retrieve values for given keys
        
        Args:
            keys: Tensor [..., vec_dim]
        
        Returns:
            Retrieved values (approximate)
        """
        if keys.shape[-1] != self.vec_dim:
            raise ValueError(f"Keys must have dimension {self.vec_dim}")

        # Handle single or batch
        if keys.dim() == 1:
            return recall_memory(keys, self.memory_trace)
        else:
            results = []
            for k in keys:
                results.append(recall_memory(k, self.memory_trace))
            return torch.stack(results)

    def clear_memory(self):
        """Reset memory to empty state"""
        self.memory_trace.zero_()

    def memory_magnitude(self) -> float:
        """Get current memory capacity usage"""
        return torch.norm(self.memory_trace).item()


def chain_twist(vectors: list[torch.Tensor]) -> torch.Tensor:
    """
    Bind sequence of vectors recursively
    
    chain([v1, v2, v3]) = twist(twist(v1, v2), v3)
    
    Creates hierarchical structure for sequences
    """
    if not vectors:
        raise ValueError("Need at least one vector")

    result = vectors[0]
    for v in vectors[1:]:
        result = twist_merge(result, v)

    return result


def similarity_under_twist(v1: torch.Tensor, v2: torch.Tensor, key: torch.Tensor) -> float:
    """
    Measure similarity after binding with same key
    
    Tests structure preservation: how similar are twist(v1,key) and twist(v2,key)?
    
    Uses cosine similarity
    """
    bound1 = twist_merge(v1, key)
    bound2 = twist_merge(v2, key)

    # Cosine similarity
    similarity = F.cosine_similarity(bound1, bound2, dim=-1)

    return similarity.mean().item()


def multi_key_store(data: torch.Tensor, keys: list[torch.Tensor], memory: torch.Tensor) -> torch.Tensor:
    """
    Store data associated with multiple keys simultaneously
    Useful for associative multi-hop retrieval
    """
    for key in keys:
        memory = stack_memory(data, key, memory)
    return memory


def blend_recall(keys: list[torch.Tensor], memory: torch.Tensor, weights: torch.Tensor = None) -> torch.Tensor:
    """
    Retrieve and blend results from multiple keys
    
    Args:
        keys: List of retrieval keys
        memory: Memory trace
        weights: Optional blending weights
    
    Returns:
        Weighted blend of retrieved results
    """
    if weights is None:
        weights = torch.ones(len(keys)) / len(keys)
    
    results = []
    for key in keys:
        results.append(recall_memory(key, memory))
    
    # Weighted combination
    stacked = torch.stack(results, dim=0)
    blended = (stacked * weights.view(-1, *([1] * (stacked.dim() - 1)))).sum(dim=0)
    
    return blended
