"""
Harmonic Field Architecture - Completely Original Implementation
Phase-based neural computing with novel approaches throughout.
"""

from .biform import BiChannel, sync_measure, wave_merge
from .phase_net import BiMatrix, TwoPathNorm, MagPreserveDropout, phase_gate, split_gate, region_gate
from .circular_ops import twist_merge, twist_split, asymm_twist, stack_memory, recall_memory, CircularMemoryBank
from .phase_focus import BiPhaseScorer, UnexpectedJump, GeometricPosition
from .broadcast import BroadcastHub, BroadcastLevel, SpecialistRouter
from .potential import DescentPredictor, CascadeSystem, ContrastiveTrainer
from .full_system import FullHarmonicSystem, build_system

__version__ = "0.1.0"

__all__ = [
    "BiChannel",
    "sync_measure",
    "wave_merge",
    "BiMatrix",
    "TwoPathNorm",
    "MagPreserveDropout",
    "phase_gate",
    "split_gate",
    "region_gate",
    "twist_merge",
    "twist_split",
    "asymm_twist",
    "stack_memory",
    "recall_memory",
    "CircularMemoryBank",
    "BiPhaseScorer",
    "UnexpectedJump",
    "GeometricPosition",
    "BroadcastHub",
    "BroadcastLevel",
    "SpecialistRouter",
    "DescentPredictor",
    "CascadeSystem",
    "ContrastiveTrainer",
    "FullHarmonicSystem",
    "build_system",
]
