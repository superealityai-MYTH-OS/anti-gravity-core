/**
 * Global Workspace Theory Implementation
 * 
 * Implements the Global Workspace with:
 * - Four-way taxonomy of attention (subliminal, preconscious, conscious, unconscious)
 * - Ignition events triggered by prediction errors
 * - Energy-Based Models for inference
 * - Self-Organized Criticality (SOC) for phase transitions
 * - Variational Free Energy monitoring
 */

import { ComplexTensor } from '../core/ComplexTensor';
import { HolographicAttention, AttentionOutput } from '../attention/HolographicAttention';

/**
 * Attention states in the four-way taxonomy
 */
export enum AttentionState {
  SUBLIMINAL = 'SUBLIMINAL',       // Below conscious threshold, low amplitude
  PRECONSCIOUS = 'PRECONSCIOUS',   // Available but not attended, moderate amplitude
  CONSCIOUS = 'CONSCIOUS',         // Actively attended, high amplitude, aligned phase
  UNCONSCIOUS = 'UNCONSCIOUS'      // Automatic processing, high amplitude, misaligned phase
}

/**
 * State of a cognitive element in the workspace
 */
export interface WorkspaceElement {
  content: ComplexTensor;
  amplitude: number;
  phase: number;
  energy: number;
  state: AttentionState;
}

/**
 * Ignition event that brings content to consciousness
 */
export interface IgnitionEvent {
  timestamp: number;
  element: WorkspaceElement;
  predictionError: number;
  energySpike: number;
  triggered: boolean;
}

/**
 * Global Workspace implementation
 */
export class GlobalWorkspace {
  private workspace: WorkspaceElement[];
  private attentionMechanism: HolographicAttention;
  private ignitionThreshold: number;
  private vfeThreshold: number;
  private historySize: number;
  private history: IgnitionEvent[];

  constructor(
    hiddenDim: number,
    ignitionThreshold: number = 0.8,
    vfeThreshold: number = 1.0,
    historySize: number = 100
  ) {
    this.workspace = [];
    this.attentionMechanism = new HolographicAttention(hiddenDim);
    this.ignitionThreshold = ignitionThreshold;
    this.vfeThreshold = vfeThreshold;
    this.historySize = historySize;
    this.history = [];
  }

  /**
   * Classify attention state based on amplitude and phase
   */
  private classifyAttentionState(
    amplitude: number,
    phase: number,
    phaseAlignment: number
  ): AttentionState {
    const amplitudeThreshold = 0.5;
    const phaseAlignmentThreshold = 0.6;

    if (amplitude < amplitudeThreshold) {
      return AttentionState.SUBLIMINAL;
    }

    if (amplitude >= amplitudeThreshold && phaseAlignment >= phaseAlignmentThreshold) {
      return AttentionState.CONSCIOUS;
    }

    if (amplitude >= amplitudeThreshold && phaseAlignment < phaseAlignmentThreshold) {
      return AttentionState.UNCONSCIOUS;
    }

    return AttentionState.PRECONSCIOUS;
  }

  /**
   * Add an element to the workspace
   */
  addElement(content: ComplexTensor, energy: number = 0): void {
    const magnitudes = content.magnitude();
    const phases = content.phase();
    
    const amplitude = magnitudes.reduce((sum, m) => sum + m, 0) / magnitudes.length;
    const phase = phases.reduce((sum, p) => sum + p, 0) / phases.length;

    // Determine attention state
    const state = this.classifyAttentionState(amplitude, phase, 0);

    const element: WorkspaceElement = {
      content,
      amplitude,
      phase,
      energy,
      state
    };

    this.workspace.push(element);
  }

  /**
   * Compute Variational Free Energy (VFE)
   * VFE = Complexity - Accuracy
   * Lower VFE indicates better predictive model
   */
  computeVariationalFreeEnergy(
    prediction: ComplexTensor,
    target: ComplexTensor
  ): number {
    // Prediction error (negative log likelihood)
    const error = prediction.add(target.scale(-1));
    const errorMagnitudes = error.magnitude();
    const accuracy = -errorMagnitudes.reduce((sum, m) => sum + m * m, 0) / errorMagnitudes.length;

    // Complexity (KL divergence approximation using magnitude variance)
    const magnitudes = prediction.magnitude();
    const meanMag = magnitudes.reduce((sum, m) => sum + m, 0) / magnitudes.length;
    const variance = magnitudes.reduce((sum, m) => sum + (m - meanMag) ** 2, 0) / magnitudes.length;
    const complexity = Math.log(variance + 1e-6);

    return complexity - accuracy;
  }

  /**
   * Check for ignition event
   * Ignition occurs when prediction error crosses critical threshold
   */
  checkIgnition(
    prediction: ComplexTensor,
    target: ComplexTensor,
    currentEnergy: number
  ): IgnitionEvent | null {
    const vfe = this.computeVariationalFreeEnergy(prediction, target);
    
    // Compute prediction error magnitude
    const error = prediction.add(target.scale(-1));
    const errorMags = error.magnitude();
    const predictionError = errorMags.reduce((sum, m) => sum + m * m, 0) / errorMags.length;

    // Check if threshold is crossed
    const triggered = predictionError > this.ignitionThreshold || vfe > this.vfeThreshold;

    if (triggered) {
      const element: WorkspaceElement = {
        content: target,
        amplitude: target.magnitude().reduce((sum, m) => sum + m, 0) / target.size(),
        phase: target.phase().reduce((sum, p) => sum + p, 0) / target.size(),
        energy: currentEnergy,
        state: AttentionState.CONSCIOUS
      };

      const event: IgnitionEvent = {
        timestamp: Date.now(),
        element,
        predictionError,
        energySpike: currentEnergy,
        triggered: true
      };

      // Add to history
      this.history.push(event);
      if (this.history.length > this.historySize) {
        this.history.shift();
      }

      return event;
    }

    return null;
  }

  /**
   * Broadcast to global workspace
   * Simulates the "broadcasting" of conscious content
   */
  broadcast(element: WorkspaceElement): AttentionOutput {
    if (this.workspace.length === 0) {
      throw new Error('Workspace is empty');
    }

    // Use attention mechanism to broadcast
    const query = element.content;
    const keys = this.workspace.map(e => e.content);
    const values = this.workspace.map(e => e.content);

    return this.attentionMechanism.forward(query, keys, values);
  }

  /**
   * Update workspace states based on attention
   */
  updateStates(attentionOutput: AttentionOutput): void {
    for (let i = 0; i < this.workspace.length; i++) {
      const element = this.workspace[i];
      const attentionWeight = attentionOutput.attentionWeights[i];
      const phaseAlignment = attentionOutput.phaseAlignment[i];

      // Update amplitude based on attention
      element.amplitude = element.amplitude * 0.9 + attentionWeight * 0.1;

      // Reclassify state
      element.state = this.classifyAttentionState(
        element.amplitude,
        element.phase,
        phaseAlignment
      );
    }
  }

  /**
   * Get elements in conscious state
   */
  getConsciousElements(): WorkspaceElement[] {
    return this.workspace.filter(e => e.state === AttentionState.CONSCIOUS);
  }

  /**
   * Get workspace statistics
   */
  getStatistics(): {
    totalElements: number;
    consciousCount: number;
    preconsciousCount: number;
    subliminalCount: number;
    unconsciousCount: number;
    averageEnergy: number;
    averageAmplitude: number;
  } {
    const counts = {
      [AttentionState.CONSCIOUS]: 0,
      [AttentionState.PRECONSCIOUS]: 0,
      [AttentionState.SUBLIMINAL]: 0,
      [AttentionState.UNCONSCIOUS]: 0
    };

    let totalEnergy = 0;
    let totalAmplitude = 0;

    for (const element of this.workspace) {
      counts[element.state]++;
      totalEnergy += element.energy;
      totalAmplitude += element.amplitude;
    }

    return {
      totalElements: this.workspace.length,
      consciousCount: counts[AttentionState.CONSCIOUS],
      preconsciousCount: counts[AttentionState.PRECONSCIOUS],
      subliminalCount: counts[AttentionState.SUBLIMINAL],
      unconsciousCount: counts[AttentionState.UNCONSCIOUS],
      averageEnergy: this.workspace.length > 0 ? totalEnergy / this.workspace.length : 0,
      averageAmplitude: this.workspace.length > 0 ? totalAmplitude / this.workspace.length : 0
    };
  }

  /**
   * Clear workspace
   */
  clear(): void {
    this.workspace = [];
  }

  /**
   * Get ignition history
   */
  getIgnitionHistory(): IgnitionEvent[] {
    return [...this.history];
  }
}

/**
 * Energy-Based Model for inference optimization
 */
export class EnergyBasedModel {
  private learningRate: number;
  private numIterations: number;

  constructor(learningRate: number = 0.01, numIterations: number = 100) {
    this.learningRate = learningRate;
    this.numIterations = numIterations;
  }

  /**
   * Energy function: E(x) = -log P(x)
   * Lower energy = higher probability
   */
  energy(state: ComplexTensor, target: ComplexTensor): number {
    const diff = state.add(target.scale(-1));
    const mags = diff.magnitude();
    return mags.reduce((sum, m) => sum + m * m, 0) / mags.length;
  }

  /**
   * Compute energy gradient (simplified)
   */
  private computeGradient(state: ComplexTensor, target: ComplexTensor): ComplexTensor {
    // Gradient = 2 * (state - target) / n
    const diff = state.add(target.scale(-1));
    return diff.scale(2.0 / state.size());
  }

  /**
   * Perform inference via gradient descent on energy landscape
   * Finds low-energy states that match the target
   */
  infer(initial: ComplexTensor, target: ComplexTensor, tolerance: number = 1e-4): {
    finalState: ComplexTensor;
    finalEnergy: number;
    iterations: number;
  } {
    let state = initial.clone();
    let prevEnergy = this.energy(state, target);

    for (let iter = 0; iter < this.numIterations; iter++) {
      // Compute gradient
      const gradient = this.computeGradient(state, target);

      // Update state: state = state - learning_rate * gradient
      state = state.add(gradient.scale(-this.learningRate));

      // Compute new energy
      const currentEnergy = this.energy(state, target);

      // Check convergence
      if (Math.abs(currentEnergy - prevEnergy) < tolerance) {
        return {
          finalState: state,
          finalEnergy: currentEnergy,
          iterations: iter + 1
        };
      }

      prevEnergy = currentEnergy;
    }

    return {
      finalState: state,
      finalEnergy: prevEnergy,
      iterations: this.numIterations
    };
  }
}

/**
 * Self-Organized Criticality for phase transitions
 * Models "aha!" moments and sudden insight
 */
export class SelfOrganizedCriticality {
  private criticalThreshold: number;
  private avalancheHistory: number[];

  constructor(criticalThreshold: number = 0.75) {
    this.criticalThreshold = criticalThreshold;
    this.avalancheHistory = [];
  }

  /**
   * Check if system is at critical point
   */
  isAtCriticality(energy: number, averageEnergy: number): boolean {
    return energy > averageEnergy * this.criticalThreshold;
  }

  /**
   * Trigger avalanche (cascade of activation)
   * Simulates sudden phase transition
   */
  triggerAvalanche(
    elements: WorkspaceElement[],
    triggerIndex: number
  ): number[] {
    const affected: number[] = [triggerIndex];
    const visited = new Set<number>([triggerIndex]);
    const queue = [triggerIndex];

    // Breadth-first propagation of activation
    while (queue.length > 0) {
      const current = queue.shift()!;
      const currentElement = elements[current];

      // Find neighbors with similar phase
      for (let i = 0; i < elements.length; i++) {
        if (visited.has(i)) continue;

        const phaseDiff = Math.abs(currentElement.phase - elements[i].phase);
        
        // Propagate if phases are aligned (within π/4)
        if (phaseDiff < Math.PI / 4 || phaseDiff > 7 * Math.PI / 4) {
          affected.push(i);
          visited.add(i);
          queue.push(i);
        }
      }
    }

    // Record avalanche size
    this.avalancheHistory.push(affected.length);

    return affected;
  }

  /**
   * Get avalanche statistics
   */
  getAvalancheStats(): {
    totalAvalanches: number;
    averageSize: number;
    maxSize: number;
  } {
    if (this.avalancheHistory.length === 0) {
      return { totalAvalanches: 0, averageSize: 0, maxSize: 0 };
    }

    return {
      totalAvalanches: this.avalancheHistory.length,
      averageSize: this.avalancheHistory.reduce((sum, size) => sum + size, 0) / this.avalancheHistory.length,
      maxSize: Math.max(...this.avalancheHistory)
    };
  }
}
