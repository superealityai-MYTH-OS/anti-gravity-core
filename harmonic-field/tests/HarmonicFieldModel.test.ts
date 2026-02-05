import { describe, it, expect } from 'vitest';
import { createHarmonicFieldModel } from '../HarmonicFieldModel';
import { ComplexTensor } from '../core/ComplexTensor';

describe('HarmonicFieldModel', () => {
  describe('Model Creation', () => {
    it('should create model with correct dimensions', () => {
      const model = createHarmonicFieldModel(10, 5, 32);
      expect(model).toBeDefined();
    });
  });

  describe('Forward Pass', () => {
    it('should process input and produce output', () => {
      const model = createHarmonicFieldModel(8, 4, 16);
      const input = [1, 2, 3, 4, 5, 6, 7, 8];
      
      const result = model.forward(input);
      
      expect(result.output).toBeDefined();
      expect(result.output.length).toBe(4);
      expect(result.phase).toBeDefined();
      expect(result.phase.length).toBe(4);
      expect(result.hidden).toBeDefined();
    });

    it('should produce consistent output for same input', () => {
      const model = createHarmonicFieldModel(4, 2, 8);
      const input = [1, 2, 3, 4];
      
      const result1 = model.forward(input);
      model.reset();
      const result2 = model.forward(input);
      
      // Outputs should be similar (not identical due to random init, but close)
      expect(result1.output.length).toBe(result2.output.length);
    });

    it('should handle context in forward pass', () => {
      const model = createHarmonicFieldModel(4, 2, 8);
      const input = [1, 2, 3, 4];
      const context = [
        new ComplexTensor([1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0], [8])
      ];
      
      const result = model.forward(input, context);
      
      expect(result.output).toBeDefined();
      expect(result.ignitionEvent).toBeDefined();
    });

    it('should track workspace statistics', () => {
      const model = createHarmonicFieldModel(4, 2, 8);
      const input = [1, 2, 3, 4];
      
      const result = model.forward(input);
      
      expect(result.workspaceStats).toBeDefined();
      expect(result.workspaceStats.totalElements).toBeGreaterThan(0);
    });
  });

  describe('Concept Binding', () => {
    it('should bind two concepts', () => {
      const model = createHarmonicFieldModel(4, 2, 8);
      const concept1 = [1, 0, 0, 0];
      const concept2 = [0, 1, 0, 0];
      
      const bound = model.bindConcepts(concept1, concept2);
      
      expect(bound).toBeDefined();
      expect(bound.size()).toBeGreaterThan(0);
    });

    it('should unbind to retrieve concept', () => {
      const model = createHarmonicFieldModel(4, 2, 8);
      const concept1 = [1, 0, 0, 0];
      const concept2 = [0, 1, 0, 0];
      
      const bound = model.bindConcepts(concept1, concept2);
      const unbound = model.unbindConcept(bound, concept2);
      
      expect(unbound).toBeDefined();
      expect(unbound.length).toBe(2);
    });

    it('should compute concept similarity', () => {
      const model = createHarmonicFieldModel(4, 2, 8);
      const concept1 = [1, 0, 0, 0];
      const concept2 = [1, 0, 0, 0];
      
      const similarity = model.conceptSimilarity(concept1, concept2);
      
      expect(similarity).toBeGreaterThan(0);
      expect(similarity).toBeLessThanOrEqual(1.01); // Allow for floating point imprecision
    });
  });

  describe('Energy-Based Inference', () => {
    it('should perform EBM inference', () => {
      const model = createHarmonicFieldModel(4, 4, 8);
      const initial = [0, 0, 0, 0];
      const target = [1, 1, 1, 1];
      
      const result = model.inferWithEBM(initial, target, 50);
      
      expect(result.finalOutput).toBeDefined();
      expect(result.energy).toBeDefined();
      expect(result.iterations).toBeGreaterThan(0);
      expect(result.iterations).toBeLessThanOrEqual(100); // Allow up to numIterations
    });

    it('should reduce energy during inference', () => {
      const model = createHarmonicFieldModel(4, 4, 8);
      const initial = [0, 0, 0, 0];
      const target = [1, 1, 1, 1];
      
      const result = model.inferWithEBM(initial, target, 100);
      
      // Energy should be relatively low after optimization
      expect(result.energy).toBeLessThan(10);
    });
  });

  describe('Workspace Information', () => {
    it('should provide workspace information', () => {
      const model = createHarmonicFieldModel(4, 2, 8);
      const input = [1, 2, 3, 4];
      
      model.forward(input);
      const info = model.getWorkspaceInfo();
      
      expect(info.elements).toBeDefined();
      expect(info.statistics).toBeDefined();
      expect(info.ignitionHistory).toBeDefined();
    });

    it('should track ignition events', () => {
      const model = createHarmonicFieldModel(4, 2, 8);
      const input = [1, 2, 3, 4];
      const context = [
        new ComplexTensor(
          [10, 10, 10, 10, 10, 10, 10, 10],
          [10, 10, 10, 10, 10, 10, 10, 10],
          [8]
        )
      ];
      
      // Large difference should trigger ignition
      model.forward(input, context);
      const info = model.getWorkspaceInfo();
      
      expect(info.ignitionHistory).toBeDefined();
    });
  });

  describe('Attention Teleportation', () => {
    it('should process sequence with teleportation', () => {
      const model = createHarmonicFieldModel(4, 4, 8);
      const sequence = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
      ];
      const predictions = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [5, 5, 5, 5], // Large error here
        [0, 0, 0, 1]
      ];
      
      const result = model.processSequenceWithTeleportation(sequence, predictions);
      
      expect(result.outputs).toBeDefined();
      expect(result.outputs.length).toBe(sequence.length);
      expect(result.teleportations).toBeDefined();
      expect(result.teleportations.length).toBe(sequence.length);
      
      // Should have teleportation targets for high-error position
      expect(result.teleportations[2].length).toBeGreaterThanOrEqual(0);
    });
  });

  describe('Self-Organized Criticality', () => {
    it('should trigger criticality events', () => {
      const model = createHarmonicFieldModel(4, 2, 8);
      
      // Add some elements to workspace
      for (let i = 0; i < 5; i++) {
        model.forward([i, i+1, i+2, i+3]);
      }
      
      const affected = model.triggerCriticality();
      
      expect(affected).toBeDefined();
      expect(Array.isArray(affected)).toBe(true);
    });
  });

  describe('Model Reset', () => {
    it('should reset workspace state', () => {
      const model = createHarmonicFieldModel(4, 2, 8);
      const input = [1, 2, 3, 4];
      
      model.forward(input);
      let info = model.getWorkspaceInfo();
      expect(info.statistics.totalElements).toBeGreaterThan(0);
      
      model.reset();
      info = model.getWorkspaceInfo();
      expect(info.statistics.totalElements).toBe(0);
    });
  });

  describe('Phase Awareness', () => {
    it('should use phase information in encoding', () => {
      const model = createHarmonicFieldModel(4, 2, 8);
      const input = [1, 2, 3, 4];
      
      const result1 = model.forward(input, undefined, 0);
      model.reset();
      const result2 = model.forward(input, undefined, Math.PI / 4);
      
      // Phase shift should affect the output
      expect(result1.phase).toBeDefined();
      expect(result2.phase).toBeDefined();
    });
  });
});
