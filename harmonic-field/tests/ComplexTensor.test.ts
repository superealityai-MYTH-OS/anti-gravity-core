import { describe, it, expect } from 'vitest';
import { ComplexTensor } from '../core/ComplexTensor';

describe('ComplexTensor', () => {
  describe('Construction', () => {
    it('should create a complex tensor with real and imaginary parts', () => {
      const real = [1, 2, 3];
      const imag = [4, 5, 6];
      const tensor = new ComplexTensor(real, imag, [3]);
      
      expect(tensor.real).toEqual(real);
      expect(tensor.imag).toEqual(imag);
      expect(tensor.shape).toEqual([3]);
    });

    it('should throw error if real and imaginary parts have different lengths', () => {
      expect(() => {
        new ComplexTensor([1, 2], [3], [2]);
      }).toThrow('Real and imaginary parts must have the same length');
    });

    it('should create tensor from polar coordinates', () => {
      const magnitude = [1, 2];
      const phase = [0, Math.PI / 2];
      const tensor = ComplexTensor.fromPolar(magnitude, phase, [2]);
      
      expect(tensor.real[0]).toBeCloseTo(1, 5);
      expect(tensor.real[1]).toBeCloseTo(0, 5);
      expect(tensor.imag[0]).toBeCloseTo(0, 5);
      expect(tensor.imag[1]).toBeCloseTo(2, 5);
    });

    it('should create zero tensor', () => {
      const tensor = ComplexTensor.zeros([3]);
      
      expect(tensor.real).toEqual([0, 0, 0]);
      expect(tensor.imag).toEqual([0, 0, 0]);
    });

    it('should create random tensor with correct shape', () => {
      const tensor = ComplexTensor.randn([5]);
      
      expect(tensor.size()).toBe(5);
      expect(tensor.real.length).toBe(5);
      expect(tensor.imag.length).toBe(5);
    });
  });

  describe('Basic Operations', () => {
    it('should compute magnitude correctly', () => {
      const tensor = new ComplexTensor([3], [4], [1]);
      const magnitude = tensor.magnitude();
      
      expect(magnitude[0]).toBeCloseTo(5, 5);
    });

    it('should compute phase correctly', () => {
      const tensor = new ComplexTensor([1], [1], [1]);
      const phase = tensor.phase();
      
      expect(phase[0]).toBeCloseTo(Math.PI / 4, 5);
    });

    it('should multiply complex tensors correctly', () => {
      const t1 = new ComplexTensor([1, 2], [2, 3], [2]);
      const t2 = new ComplexTensor([3, 4], [4, 5], [2]);
      const result = t1.multiply(t2);
      
      // (1+2i)*(3+4i) = 3+4i+6i+8i^2 = 3+10i-8 = -5+10i
      expect(result.real[0]).toBeCloseTo(-5, 5);
      expect(result.imag[0]).toBeCloseTo(10, 5);
      
      // (2+3i)*(4+5i) = 8+10i+12i+15i^2 = 8+22i-15 = -7+22i
      expect(result.real[1]).toBeCloseTo(-7, 5);
      expect(result.imag[1]).toBeCloseTo(22, 5);
    });

    it('should add complex tensors correctly', () => {
      const t1 = new ComplexTensor([1, 2], [3, 4], [2]);
      const t2 = new ComplexTensor([5, 6], [7, 8], [2]);
      const result = t1.add(t2);
      
      expect(result.real).toEqual([6, 8]);
      expect(result.imag).toEqual([10, 12]);
    });

    it('should compute conjugate correctly', () => {
      const tensor = new ComplexTensor([1, 2], [3, 4], [2]);
      const conj = tensor.conjugate();
      
      expect(conj.real).toEqual([1, 2]);
      expect(conj.imag).toEqual([-3, -4]);
    });

    it('should scale by real number correctly', () => {
      const tensor = new ComplexTensor([1, 2], [3, 4], [2]);
      const scaled = tensor.scale(2);
      
      expect(scaled.real).toEqual([2, 4]);
      expect(scaled.imag).toEqual([6, 8]);
    });
  });

  describe('Dot Product', () => {
    it('should compute dot product correctly', () => {
      const t1 = new ComplexTensor([1, 0], [0, 1], [2]);
      const t2 = new ComplexTensor([0, 1], [1, 0], [2]);
      const result = t1.dot(t2);
      
      expect(result.size()).toBe(1);
      // conj(t1) · t2 = (1-0i)*(0+1i) + (0-1i)*(1+0i) = 1i + (-1i) = 0
      expect(result.real[0]).toBeCloseTo(0, 5);
      expect(result.imag[0]).toBeCloseTo(0, 5);
    });
  });

  describe('ModReLU Activation', () => {
    it('should apply modReLU correctly for positive magnitude', () => {
      const tensor = new ComplexTensor([3], [4], [1]);
      const activated = tensor.modReLU(0);
      
      const mag = Math.sqrt(3*3 + 4*4); // = 5
      expect(activated.magnitude()[0]).toBeCloseTo(5, 5);
    });

    it('should apply modReLU with bias correctly', () => {
      const tensor = new ComplexTensor([3], [4], [1]);
      const activated = tensor.modReLU(-2);
      
      // magnitude = 5, 5 + (-2) = 3
      expect(activated.magnitude()[0]).toBeCloseTo(3, 5);
    });

    it('should zero out values below threshold', () => {
      const tensor = new ComplexTensor([0.1], [0.1], [1]);
      const activated = tensor.modReLU(-0.5);
      
      // magnitude ≈ 0.141, 0.141 + (-0.5) < 0, so should be 0
      expect(activated.magnitude()[0]).toBeCloseTo(0, 5);
    });
  });

  describe('Shape Operations', () => {
    it('should clone tensor correctly', () => {
      const original = new ComplexTensor([1, 2], [3, 4], [2]);
      const cloned = original.clone();
      
      expect(cloned.real).toEqual(original.real);
      expect(cloned.imag).toEqual(original.imag);
      expect(cloned.shape).toEqual(original.shape);
      
      // Ensure it's a deep copy
      cloned.real[0] = 999;
      expect(original.real[0]).toBe(1);
    });

    it('should reshape tensor correctly', () => {
      const tensor = new ComplexTensor([1, 2, 3, 4], [5, 6, 7, 8], [4]);
      const reshaped = tensor.reshape([2, 2]);
      
      expect(reshaped.shape).toEqual([2, 2]);
      expect(reshaped.size()).toBe(4);
    });

    it('should throw error when reshaping to incompatible size', () => {
      const tensor = new ComplexTensor([1, 2, 3], [4, 5, 6], [3]);
      
      expect(() => {
        tensor.reshape([2, 2]);
      }).toThrow('New shape must have the same total size');
    });
  });
});
