/**
 * Holographic Attention Mechanism
 * 
 * Implements attention using phase alignment and holographic operations.
 * Leverages complex-valued embeddings for efficient attention computation
 * with phase-aware scoring.
 * 
 * Time complexity: O(n) to O(n log n) using holographic operations
 */

import { ComplexTensor } from '../core/ComplexTensor';
import { HolographicReducedRepresentation as HRR } from '../holographic/HolographicReducedRepresentation';

export interface AttentionOutput {
  output: ComplexTensor;
  attentionWeights: number[];
  phaseAlignment: number[];
}

/**
 * Holographic Attention Layer
 * 
 * Computes attention using phase alignment between queries and keys,
 * then uses holographic binding to attend to values.
 */
export class HolographicAttention {
  private hiddenDim: number;
  private usePhaseAlignment: boolean;

  constructor(hiddenDim: number, usePhaseAlignment: boolean = true) {
    this.hiddenDim = hiddenDim;
    this.usePhaseAlignment = usePhaseAlignment;
  }

  /**
   * Compute phase alignment between query and key
   * Phase alignment indicates semantic coherence
   */
  private computePhaseAlignment(query: ComplexTensor, key: ComplexTensor): number {
    const queryPhase = query.phase();
    const keyPhase = key.phase();

    // Compute cosine similarity of phases
    let alignment = 0;
    for (let i = 0; i < queryPhase.length; i++) {
      // Phase difference
      const phaseDiff = queryPhase[i] - keyPhase[i];
      // Cosine of phase difference (1 when aligned, -1 when opposite)
      alignment += Math.cos(phaseDiff);
    }

    return alignment / queryPhase.length;
  }

  /**
   * Compute attention score using both magnitude and phase
   */
  private computeAttentionScore(
    query: ComplexTensor,
    key: ComplexTensor
  ): { score: number; phaseAlignment: number } {
    // Use conjugate transpose of key for proper complex inner product
    const keyConj = key.conjugate();
    
    // Compute dot product
    const dotProduct = query.dot(keyConj);
    
    // Magnitude-based similarity
    const magnitudeScore = Math.sqrt(
      dotProduct.real[0] * dotProduct.real[0] + 
      dotProduct.imag[0] * dotProduct.imag[0]
    );

    // Phase alignment
    const phaseAlignment = this.usePhaseAlignment 
      ? this.computePhaseAlignment(query, key)
      : 0;

    // Combined score: magnitude scaled by phase alignment
    const score = this.usePhaseAlignment
      ? magnitudeScore * (1 + phaseAlignment) / 2
      : magnitudeScore;

    return { score, phaseAlignment };
  }

  /**
   * Softmax function for attention weights
   */
  private softmax(scores: number[]): number[] {
    const maxScore = Math.max(...scores);
    const expScores = scores.map(s => Math.exp(s - maxScore));
    const sumExp = expScores.reduce((sum, exp) => sum + exp, 0);
    return expScores.map(exp => exp / sumExp);
  }

  /**
   * Forward pass for holographic attention
   * 
   * @param query - Query tensor
   * @param keys - Array of key tensors
   * @param values - Array of value tensors
   * @param mask - Optional attention mask
   */
  forward(
    query: ComplexTensor,
    keys: ComplexTensor[],
    values: ComplexTensor[],
    mask?: boolean[]
  ): AttentionOutput {
    if (keys.length !== values.length) {
      throw new Error('Number of keys must match number of values');
    }

    if (mask && mask.length !== keys.length) {
      throw new Error('Mask length must match number of keys');
    }

    // Compute attention scores
    const scores: number[] = [];
    const phaseAlignments: number[] = [];

    for (let i = 0; i < keys.length; i++) {
      if (mask && !mask[i]) {
        scores.push(-Infinity);
        phaseAlignments.push(0);
      } else {
        const { score, phaseAlignment } = this.computeAttentionScore(query, keys[i]);
        scores.push(score);
        phaseAlignments.push(phaseAlignment);
      }
    }

    // Apply softmax to get attention weights
    const attentionWeights = this.softmax(scores);

    // Compute weighted sum of values using holographic operations
    let output = values[0].scale(attentionWeights[0]);
    
    for (let i = 1; i < values.length; i++) {
      output = output.add(values[i].scale(attentionWeights[i]));
    }

    return {
      output,
      attentionWeights,
      phaseAlignment: phaseAlignments
    };
  }

  /**
   * Multi-head attention using holographic operations
   */
  multiHeadAttention(
    query: ComplexTensor,
    keys: ComplexTensor[],
    values: ComplexTensor[],
    numHeads: number,
    mask?: boolean[]
  ): AttentionOutput {
    if (this.hiddenDim % numHeads !== 0) {
      throw new Error('Hidden dimension must be divisible by number of heads');
    }

    const headDim = this.hiddenDim / numHeads;
    const allOutputs: ComplexTensor[] = [];
    const allWeights: number[][] = [];
    const allPhaseAlignments: number[][] = [];

    // Split into heads
    for (let h = 0; h < numHeads; h++) {
      const start = h * headDim;
      const end = (h + 1) * headDim;

      // Extract head-specific portions
      const queryHead = new ComplexTensor(
        query.real.slice(start, end),
        query.imag.slice(start, end),
        [headDim]
      );

      const keysHead = keys.map(k => new ComplexTensor(
        k.real.slice(start, end),
        k.imag.slice(start, end),
        [headDim]
      ));

      const valuesHead = values.map(v => new ComplexTensor(
        v.real.slice(start, end),
        v.imag.slice(start, end),
        [headDim]
      ));

      // Compute attention for this head
      const headOutput = this.forward(queryHead, keysHead, valuesHead, mask);
      allOutputs.push(headOutput.output);
      allWeights.push(headOutput.attentionWeights);
      allPhaseAlignments.push(headOutput.phaseAlignment);
    }

    // Concatenate heads
    const outputReal: number[] = [];
    const outputImag: number[] = [];
    
    for (const headOutput of allOutputs) {
      outputReal.push(...headOutput.real);
      outputImag.push(...headOutput.imag);
    }

    // Average weights and phase alignments across heads
    const avgWeights = allWeights[0].map((_, i) => 
      allWeights.reduce((sum, w) => sum + w[i], 0) / numHeads
    );
    const avgPhaseAlignment = allPhaseAlignments[0].map((_, i) => 
      allPhaseAlignments.reduce((sum, p) => sum + p[i], 0) / numHeads
    );

    return {
      output: new ComplexTensor(outputReal, outputImag, [this.hiddenDim]),
      attentionWeights: avgWeights,
      phaseAlignment: avgPhaseAlignment
    };
  }

  /**
   * Cross-attention between different sequences
   */
  crossAttention(
    querySeq: ComplexTensor[],
    keySeq: ComplexTensor[],
    valueSeq: ComplexTensor[],
    mask?: boolean[][]
  ): ComplexTensor[] {
    const outputs: ComplexTensor[] = [];

    for (let i = 0; i < querySeq.length; i++) {
      const queryMask = mask ? mask[i] : undefined;
      const result = this.forward(querySeq[i], keySeq, valueSeq, queryMask);
      outputs.push(result.output);
    }

    return outputs;
  }

  /**
   * Self-attention for a sequence
   */
  selfAttention(
    sequence: ComplexTensor[],
    mask?: boolean[][]
  ): ComplexTensor[] {
    return this.crossAttention(sequence, sequence, sequence, mask);
  }
}

/**
 * Attention Teleportation mechanism
 * 
 * Allows attention to "jump" to relevant information based on
 * energy landscapes and criticality thresholds
 */
export class AttentionTeleportation {
  private criticalityThreshold: number;
  private energyDecayRate: number;

  constructor(criticalityThreshold: number = 0.7, energyDecayRate: number = 0.95) {
    this.criticalityThreshold = criticalityThreshold;
    this.energyDecayRate = energyDecayRate;
  }

  /**
   * Compute energy landscape for a sequence
   * High energy indicates important or surprising information
   */
  computeEnergyLandscape(
    sequence: ComplexTensor[],
    predictions: ComplexTensor[]
  ): number[] {
    const energies: number[] = [];

    for (let i = 0; i < sequence.length; i++) {
      // Prediction error as energy
      const error = sequence[i].add(predictions[i].scale(-1));
      const errorMagnitude = error.magnitude();
      const energy = errorMagnitude.reduce((sum, m) => sum + m * m, 0) / errorMagnitude.length;
      energies.push(energy);
    }

    return energies;
  }

  /**
   * Determine teleportation targets based on criticality
   */
  findTeleportationTargets(
    energies: number[],
    currentPosition: number
  ): number[] {
    const targets: number[] = [];
    const avgEnergy = energies.reduce((sum, e) => sum + e, 0) / energies.length;

    for (let i = 0; i < energies.length; i++) {
      // Skip current position
      if (i === currentPosition) continue;

      // Check if energy exceeds criticality threshold
      if (energies[i] > avgEnergy * this.criticalityThreshold) {
        targets.push(i);
      }
    }

    return targets;
  }

  /**
   * Perform attention teleportation
   * "Jumps" attention to high-energy positions
   */
  teleport(
    query: ComplexTensor,
    sequence: ComplexTensor[],
    energies: number[],
    currentPosition: number,
    attentionMechanism: HolographicAttention
  ): AttentionOutput {
    // Find teleportation targets
    const targets = this.findTeleportationTargets(energies, currentPosition);

    if (targets.length === 0) {
      // No teleportation needed, use standard attention
      return attentionMechanism.forward(query, sequence, sequence);
    }

    // Create enhanced attention mask that favors high-energy positions
    const mask = sequence.map((_, i) => targets.includes(i) || i === currentPosition);

    // Compute attention with teleportation bias
    return attentionMechanism.forward(query, sequence, sequence, mask);
  }
}
