/**
 * Harmonic Field Architecture
 * 
 * A comprehensive implementation of complex-valued neural networks with:
 * - Vector Interference using CVNNs
 * - Holographic Reduced Representations (HRR)
 * - Attention Teleportation
 * - Global Workspace Theory
 * - Energy-Based Models
 * - Self-Organized Criticality
 */

// Core components
export { ComplexTensor } from './core/ComplexTensor';

// Layers
export { 
  ComplexLinear, 
  ComplexActivations, 
  ComplexBatchNorm, 
  ComplexDropout 
} from './layers/ComplexLayers';

// Holographic representations
export { HolographicReducedRepresentation } from './holographic/HolographicReducedRepresentation';

// Attention mechanisms
export { 
  HolographicAttention, 
  AttentionTeleportation,
  AttentionOutput 
} from './attention/HolographicAttention';

// Global workspace and energy models
export { 
  GlobalWorkspace, 
  EnergyBasedModel, 
  SelfOrganizedCriticality,
  AttentionState,
  WorkspaceElement,
  IgnitionEvent
} from './energy/GlobalWorkspace';

// Main model
export { 
  HarmonicFieldModel,
  PhaseAwareEncoder,
  PhaseAwareDecoder,
  createHarmonicFieldModel,
  HarmonicFieldConfig
} from './HarmonicFieldModel';
