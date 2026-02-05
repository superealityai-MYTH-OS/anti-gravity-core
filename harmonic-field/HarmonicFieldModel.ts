/**
 * Harmonic Field Architecture
 * 
 * Main model that integrates:
 * - Complex-Valued Neural Networks (CVNNs)
 * - Holographic Reduced Representations (HRR)
 * - Holographic Attention with phase alignment
 * - Global Workspace Theory
 * - Energy-Based Models
 * - Self-Organized Criticality
 * 
 * This architecture enables:
 * - Phase-aware semantic processing
 * - Efficient holographic binding/unbinding
 * - Attention teleportation
 * - Consciousness-like ignition events
 * - Gradient-based inference in energy landscapes
 */

import { ComplexTensor } from './core/ComplexTensor';
import { ComplexLinear, ComplexActivations, ComplexBatchNorm, ComplexDropout } from './layers/ComplexLayers';
import { HolographicReducedRepresentation as HRR } from './holographic/HolographicReducedRepresentation';
import { HolographicAttention, AttentionTeleportation } from './attention/HolographicAttention';
import { GlobalWorkspace, EnergyBasedModel, SelfOrganizedCriticality, AttentionState } from './energy/GlobalWorkspace';

export interface HarmonicFieldConfig {
  inputDim: number;
  hiddenDim: number;
  outputDim: number;
  numLayers: number;
  numHeads: number;
  dropoutRate: number;
  usePhaseAlignment: boolean;
  ignitionThreshold: number;
  vfeThreshold: number;
  enableTeleportation: boolean;
  enableSOC: boolean;
}

/**
 * Phase-aware encoder for input processing
 */
export class PhaseAwareEncoder {
  private embedLayer: ComplexLinear;
  private normLayer: ComplexBatchNorm;
  private dropoutLayer: ComplexDropout;

  constructor(inputDim: number, hiddenDim: number, dropoutRate: number = 0.1) {
    this.embedLayer = new ComplexLinear(inputDim, hiddenDim);
    this.normLayer = new ComplexBatchNorm(hiddenDim);
    this.dropoutLayer = new ComplexDropout(dropoutRate);
  }

  /**
   * Encode input with phase information
   */
  encode(input: number[], phaseShift: number = 0): ComplexTensor {
    // Convert real input to complex with phase
    const real = input;
    const imag = input.map(x => x * Math.sin(phaseShift));
    
    const complexInput = new ComplexTensor(real, imag, [input.length]);
    
    // Apply layers
    let hidden = this.embedLayer.forward(complexInput);
    hidden = this.normLayer.forward(hidden, true);
    hidden = ComplexActivations.modReLU(hidden, 0);
    hidden = this.dropoutLayer.forward(hidden, true);
    
    return hidden;
  }
}

/**
 * Phase-aware decoder for output generation
 */
export class PhaseAwareDecoder {
  private decodeLayer: ComplexLinear;
  private normLayer: ComplexBatchNorm;

  constructor(hiddenDim: number, outputDim: number) {
    this.decodeLayer = new ComplexLinear(hiddenDim, outputDim);
    this.normLayer = new ComplexBatchNorm(outputDim);
  }

  /**
   * Decode to output, extracting magnitude for predictions
   */
  decode(hidden: ComplexTensor): { output: number[], phase: number[] } {
    let decoded = this.decodeLayer.forward(hidden);
    decoded = this.normLayer.forward(decoded, false);
    
    return {
      output: decoded.magnitude(),
      phase: decoded.phase()
    };
  }

  /**
   * Reconstruct input (for dual-headed architecture)
   */
  reconstruct(hidden: ComplexTensor): ComplexTensor {
    const decoded = this.decodeLayer.forward(hidden);
    return this.normLayer.forward(decoded, false);
  }
}

/**
 * Main Harmonic Field Model
 */
export class HarmonicFieldModel {
  private config: HarmonicFieldConfig;
  private encoder: PhaseAwareEncoder;
  private decoder: PhaseAwareDecoder;
  private layers: ComplexLinear[];
  private attentionLayers: HolographicAttention[];
  private globalWorkspace: GlobalWorkspace;
  private energyModel: EnergyBasedModel;
  private socModel?: SelfOrganizedCriticality;
  private teleportation?: AttentionTeleportation;

  constructor(config: HarmonicFieldConfig) {
    this.config = config;
    
    // Initialize encoder/decoder
    this.encoder = new PhaseAwareEncoder(config.inputDim, config.hiddenDim, config.dropoutRate);
    this.decoder = new PhaseAwareDecoder(config.hiddenDim, config.outputDim);

    // Initialize layers
    this.layers = [];
    for (let i = 0; i < config.numLayers; i++) {
      this.layers.push(new ComplexLinear(config.hiddenDim, config.hiddenDim));
    }

    // Initialize attention layers
    this.attentionLayers = [];
    for (let i = 0; i < config.numLayers; i++) {
      this.attentionLayers.push(new HolographicAttention(config.hiddenDim, config.usePhaseAlignment));
    }

    // Initialize global workspace
    this.globalWorkspace = new GlobalWorkspace(
      config.hiddenDim,
      config.ignitionThreshold,
      config.vfeThreshold
    );

    // Initialize energy-based model
    this.energyModel = new EnergyBasedModel();

    // Optional components
    if (config.enableSOC) {
      this.socModel = new SelfOrganizedCriticality();
    }

    if (config.enableTeleportation) {
      this.teleportation = new AttentionTeleportation();
    }
  }

  /**
   * Forward pass through the model
   */
  forward(
    input: number[],
    context?: ComplexTensor[],
    phaseShift: number = 0
  ): {
    output: number[];
    phase: number[];
    hidden: ComplexTensor;
    attentionWeights?: number[][];
    ignitionEvent?: any;
    workspaceStats?: any;
  } {
    // Encode input with phase
    let hidden = this.encoder.encode(input, phaseShift);

    // Add to global workspace
    this.globalWorkspace.addElement(hidden, 0);

    // Process through layers with attention
    const allAttentionWeights: number[][] = [];

    for (let i = 0; i < this.config.numLayers; i++) {
      // Apply linear transformation
      hidden = this.layers[i].forward(hidden);
      hidden = ComplexActivations.modReLU(hidden, 0);

      // Apply self-attention if no context, otherwise cross-attention
      if (!context || context.length === 0) {
        const selfAttResult = this.attentionLayers[i].forward(
          hidden,
          [hidden],
          [hidden]
        );
        hidden = selfAttResult.output;
        allAttentionWeights.push(selfAttResult.attentionWeights);
      } else {
        const crossAttResult = this.attentionLayers[i].forward(
          hidden,
          context,
          context
        );
        hidden = crossAttResult.output;
        allAttentionWeights.push(crossAttResult.attentionWeights);
      }

      // Update workspace
      this.globalWorkspace.addElement(hidden, i);
    }

    // Decode output
    const decoded = this.decoder.decode(hidden);

    // Check for ignition events
    let ignitionEvent = null;
    if (context && context.length > 0) {
      const prediction = hidden;
      const target = context[0]; // Use first context element as target
      ignitionEvent = this.globalWorkspace.checkIgnition(prediction, target, allAttentionWeights.length);
    }

    // Get workspace statistics
    const workspaceStats = this.globalWorkspace.getStatistics();

    return {
      output: decoded.output,
      phase: decoded.phase,
      hidden,
      attentionWeights: allAttentionWeights,
      ignitionEvent,
      workspaceStats
    };
  }

  /**
   * Perform inference using energy-based optimization
   */
  inferWithEBM(
    initial: number[],
    target: number[],
    maxIterations: number = 100
  ): {
    finalOutput: number[];
    energy: number;
    iterations: number;
  } {
    // Encode initial state and target
    const initialState = this.encoder.encode(initial);
    const targetState = this.encoder.encode(target);

    // Perform energy-based inference
    const result = this.energyModel.infer(initialState, targetState);

    // Decode final state
    const decoded = this.decoder.decode(result.finalState);

    return {
      finalOutput: decoded.output,
      energy: result.finalEnergy,
      iterations: result.iterations
    };
  }

  /**
   * Bind two concepts using HRR
   */
  bindConcepts(concept1: number[], concept2: number[]): ComplexTensor {
    const c1 = this.encoder.encode(concept1);
    const c2 = this.encoder.encode(concept2);
    return HRR.bind(c1, c2);
  }

  /**
   * Unbind to retrieve a concept
   */
  unbindConcept(bound: ComplexTensor, key: number[]): number[] {
    const keyTensor = this.encoder.encode(key);
    const unbound = HRR.unbind(bound, keyTensor);
    const decoded = this.decoder.decode(unbound);
    return decoded.output;
  }

  /**
   * Compute similarity between two concepts
   */
  conceptSimilarity(concept1: number[], concept2: number[]): number {
    const c1 = this.encoder.encode(concept1);
    const c2 = this.encoder.encode(concept2);
    return HRR.similarity(c1, c2);
  }

  /**
   * Get workspace information for consciousness monitoring
   */
  getWorkspaceInfo(): {
    elements: any[];
    statistics: any;
    ignitionHistory: any[];
    avalancheStats?: any;
  } {
    return {
      elements: this.globalWorkspace.getConsciousElements(),
      statistics: this.globalWorkspace.getStatistics(),
      ignitionHistory: this.globalWorkspace.getIgnitionHistory(),
      avalancheStats: this.socModel?.getAvalancheStats()
    };
  }

  /**
   * Reset the model state
   */
  reset(): void {
    this.globalWorkspace.clear();
  }

  /**
   * Process a sequence with attention teleportation
   */
  processSequenceWithTeleportation(
    sequence: number[][],
    predictions: number[][]
  ): {
    outputs: number[][];
    teleportations: number[][];
  } {
    if (!this.teleportation) {
      throw new Error('Teleportation not enabled in config');
    }

    const encodedSeq = sequence.map(s => this.encoder.encode(s));
    const encodedPred = predictions.map(p => this.encoder.encode(p));

    // Compute energy landscape
    const energies = this.teleportation.computeEnergyLandscape(encodedSeq, encodedPred);

    const outputs: number[][] = [];
    const teleportations: number[][] = [];

    for (let i = 0; i < sequence.length; i++) {
      // Find teleportation targets
      const targets = this.teleportation.findTeleportationTargets(energies, i);
      teleportations.push(targets);

      // Perform attention with teleportation
      const result = this.teleportation.teleport(
        encodedSeq[i],
        encodedSeq,
        energies,
        i,
        this.attentionLayers[0]
      );

      // Decode output
      const decoded = this.decoder.decode(result.output);
      outputs.push(decoded.output);
    }

    return { outputs, teleportations };
  }

  /**
   * Trigger self-organized criticality
   */
  triggerCriticality(): number[] {
    if (!this.socModel) {
      throw new Error('SOC not enabled in config');
    }

    const elements = this.globalWorkspace.getConsciousElements();
    
    if (elements.length === 0) {
      return [];
    }

    // Find highest energy element
    const maxEnergyIdx = elements.reduce(
      (maxIdx, elem, idx, arr) => elem.energy > arr[maxIdx].energy ? idx : maxIdx,
      0
    );

    // Trigger avalanche
    return this.socModel.triggerAvalanche(elements, maxEnergyIdx);
  }
}

/**
 * Factory function to create a Harmonic Field model with default config
 */
export function createHarmonicFieldModel(
  inputDim: number,
  outputDim: number,
  hiddenDim: number = 256
): HarmonicFieldModel {
  const config: HarmonicFieldConfig = {
    inputDim,
    hiddenDim,
    outputDim,
    numLayers: 4,
    numHeads: 8,
    dropoutRate: 0.1,
    usePhaseAlignment: true,
    ignitionThreshold: 0.8,
    vfeThreshold: 1.0,
    enableTeleportation: true,
    enableSOC: true
  };

  return new HarmonicFieldModel(config);
}
