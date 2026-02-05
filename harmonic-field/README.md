# Harmonic Field Architecture

## Overview

The Harmonic Field is a comprehensive cognitive architecture that synthesizes three major theoretical frameworks:

1. **Vector Interference** using Complex-Valued Neural Networks (CVNNs)
2. **Holography** using Holographic Reduced Representations (HRR)
3. **Attention Teleportation** governed by Global Workspace Theory and Energy-Based Models

This implementation provides a foundation for advanced AI systems with phase-aware semantic processing, efficient holographic binding/unbinding, consciousness-like ignition events, and gradient-based inference in energy landscapes.

## Architecture Components

### 1. Complex-Valued Neural Networks (CVNNs)

**Location:** `harmonic-field/core/ComplexTensor.ts`, `harmonic-field/layers/ComplexLayers.ts`

Complex-valued embeddings represent information using both **amplitude** (signal strength) and **phase** (semantic and temporal context).

**Key Features:**
- Complex tensor operations (multiplication, addition, conjugate, dot product)
- Polar coordinate representation (magnitude and phase)
- ModReLU activation function for complex numbers
- Complex batch normalization and dropout

**Example Usage:**
```typescript
import { ComplexTensor } from './harmonic-field';

// Create complex tensor from magnitude and phase
const magnitude = [1, 2, 3];
const phase = [0, Math.PI/2, Math.PI];
const tensor = ComplexTensor.fromPolar(magnitude, phase, [3]);

// Complex operations
const conj = tensor.conjugate();
const magnitude = tensor.magnitude();
const phase = tensor.phase();

// Apply activation
const activated = tensor.modReLU(0);
```

### 2. Holographic Reduced Representations (HRR)

**Location:** `harmonic-field/holographic/HolographicReducedRepresentation.ts`

HRR enables efficient binding and unbinding of concepts using FFT-based circular convolution with **O(T log T)** time complexity instead of naive **O(T²)**.

**Key Features:**
- FFT-based binding/unbinding operations
- Generalized HRR for non-commutative relationships
- Superposition for representing multiple concepts
- Similarity computation using complex inner products

**Example Usage:**
```typescript
import { HolographicReducedRepresentation as HRR } from './harmonic-field';

// Create random vectors
const concept1 = HRR.randomVector(32);
const concept2 = HRR.randomVector(32);

// Bind concepts together
const bound = HRR.bind(concept1, concept2);

// Unbind to retrieve
const retrieved = HRR.unbind(bound, concept2);

// Compute similarity
const sim = HRR.similarity(concept1, retrieved);
console.log('Similarity:', sim);

// Superpose multiple concepts
const superposed = HRR.superpose([concept1, concept2, concept3], [0.5, 0.3, 0.2]);
```

### 3. Holographic Attention Mechanism

**Location:** `harmonic-field/attention/HolographicAttention.ts`

Phase-aware attention that computes alignment between queries and keys using both magnitude and phase information.

**Key Features:**
- Phase alignment scoring
- Multi-head attention
- Cross-attention and self-attention
- Attention teleportation for jumping to high-energy positions

**Example Usage:**
```typescript
import { HolographicAttention, AttentionTeleportation } from './harmonic-field';

const attention = new HolographicAttention(128, true);

// Self-attention
const sequence = [tensor1, tensor2, tensor3];
const outputs = attention.selfAttention(sequence);

// Multi-head attention
const query = tensor1;
const result = attention.multiHeadAttention(query, sequence, sequence, 8);

// Attention teleportation
const teleporter = new AttentionTeleportation();
const energies = teleporter.computeEnergyLandscape(sequence, predictions);
const targets = teleporter.findTeleportationTargets(energies, currentPos);
```

### 4. Global Workspace Theory

**Location:** `harmonic-field/energy/GlobalWorkspace.ts`

Implements a computational model of consciousness with four attention states and ignition events triggered by prediction errors.

**Four-Way Attention Taxonomy:**
- **SUBLIMINAL**: Low amplitude, below consciousness threshold
- **PRECONSCIOUS**: Moderate amplitude, available but not attended
- **CONSCIOUS**: High amplitude with aligned phase, actively attended
- **UNCONSCIOUS**: High amplitude with misaligned phase, automatic processing

**Key Features:**
- Variational Free Energy (VFE) monitoring
- Ignition events for consciousness
- Energy-Based Models for inference
- Self-Organized Criticality for phase transitions

**Example Usage:**
```typescript
import { GlobalWorkspace, EnergyBasedModel, SelfOrganizedCriticality } from './harmonic-field';

// Create workspace
const workspace = new GlobalWorkspace(128, 0.8, 1.0);

// Add elements
workspace.addElement(tensor, energy);

// Check for ignition
const ignition = workspace.checkIgnition(prediction, target, currentEnergy);
if (ignition && ignition.triggered) {
  console.log('Ignition event!', ignition);
}

// Get workspace statistics
const stats = workspace.getStatistics();
console.log('Conscious elements:', stats.consciousCount);

// Energy-based inference
const ebm = new EnergyBasedModel(0.01, 100);
const result = ebm.infer(initial, target);

// Self-organized criticality
const soc = new SelfOrganizedCriticality(0.75);
const affected = soc.triggerAvalanche(elements, triggerIndex);
```

### 5. Complete Harmonic Field Model

**Location:** `harmonic-field/HarmonicFieldModel.ts`

The integrated model that combines all components into a complete cognitive architecture.

**Key Features:**
- Phase-aware encoding and decoding
- Multi-layer processing with attention
- Concept binding/unbinding
- Energy-based inference
- Workspace monitoring
- Attention teleportation
- Self-organized criticality

**Example Usage:**
```typescript
import { createHarmonicFieldModel } from './harmonic-field';

// Create model
const model = createHarmonicFieldModel(
  inputDim: 64,
  outputDim: 10,
  hiddenDim: 256
);

// Forward pass
const input = [1, 2, 3, ...]; // 64 dimensions
const result = model.forward(input);

console.log('Output:', result.output);
console.log('Phase:', result.phase);
console.log('Workspace stats:', result.workspaceStats);

// Bind concepts
const bound = model.bindConcepts(concept1, concept2);
const retrieved = model.unbindConcept(bound, concept2);

// Compute similarity
const sim = model.conceptSimilarity(concept1, concept2);

// Energy-based inference
const ebmResult = model.inferWithEBM(initial, target, 100);

// Process sequence with teleportation
const seqResult = model.processSequenceWithTeleportation(sequence, predictions);

// Get workspace info
const info = model.getWorkspaceInfo();
console.log('Ignition history:', info.ignitionHistory);

// Trigger criticality
const affected = model.triggerCriticality();
```

## Theoretical Advantages

### 1. Complex-Valued Embeddings
- **Phase Alignment**: Semantic coherence through phase relationships
- **Interference**: Constructive and destructive interference for filtering
- **Decision Boundary Orthogonality**: Better separation than real-valued networks

### 2. Holographic Operations
- **Efficiency**: O(T log T) vs O(T²) complexity
- **Distributed Representations**: Information spread across entire vector
- **Associative Memory**: Retrieve concepts through partial cues

### 3. Attention Mechanisms
- **Phase-Aware Scoring**: Better semantic alignment detection
- **Teleportation**: Jump to surprising/important information
- **Scalability**: Near-linear time complexity

### 4. Consciousness Modeling
- **Four-Way Taxonomy**: Models different levels of attention
- **Ignition Events**: Computational "aha!" moments
- **Energy Landscapes**: Gradient-based inference
- **Criticality**: Phase transitions for insight

## Testing

Comprehensive test suite covering all components:

```bash
npm test
```

Test files:
- `harmonic-field/tests/ComplexTensor.test.ts`
- `harmonic-field/tests/HolographicReducedRepresentation.test.ts`
- `harmonic-field/tests/HarmonicFieldModel.test.ts`

## Integration with PyTorch/TensorFlow

The current implementation is in TypeScript for compatibility with the existing codebase. For production deep learning applications, this architecture can be ported to PyTorch or TensorFlow:

**PyTorch:**
```python
import torch
import torch.nn as nn

class ComplexLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight_real = nn.Parameter(torch.randn(out_features, in_features))
        self.weight_imag = nn.Parameter(torch.randn(out_features, in_features))
        self.bias_real = nn.Parameter(torch.zeros(out_features))
        self.bias_imag = nn.Parameter(torch.zeros(out_features))
    
    def forward(self, x_real, x_imag):
        # Complex multiplication
        out_real = torch.matmul(self.weight_real, x_real) - torch.matmul(self.weight_imag, x_imag) + self.bias_real
        out_imag = torch.matmul(self.weight_real, x_imag) + torch.matmul(self.weight_imag, x_real) + self.bias_imag
        return out_real, out_imag
```

## Performance Considerations

- **Memory**: Complex tensors use 2x memory (real + imaginary)
- **Computation**: Complex operations ~2-4x slower than real operations
- **FFT Efficiency**: Use power-of-2 sizes when possible for optimal FFT performance
- **Batch Processing**: Implement batch operations for parallel processing

## Future Enhancements

1. **GPU Acceleration**: Port to PyTorch/TensorFlow for GPU support
2. **Sparse Operations**: Optimize for sparse complex tensors
3. **Hierarchical Workspace**: Multi-level global workspace
4. **Temporal Dynamics**: Add recurrent connections for sequence modeling
5. **Neuromorphic Hardware**: Adapt for neuromorphic computing platforms

## References

This implementation is based on theoretical work in:
- Complex-Valued Neural Networks (CVNNs)
- Holographic Reduced Representations (Plate, 1995)
- Global Workspace Theory (Baars, 1988; Dehaene, 2014)
- Energy-Based Models (LeCun et al., 2006)
- Self-Organized Criticality (Bak, Tang, Wiesenfeld, 1987)

## License

Part of the anti-gravity-core project.
