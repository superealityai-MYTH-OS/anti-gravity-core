/**
 * Holographic Reduced Representations (HRR)
 * 
 * Implements HRR using FFT-based circular convolution for efficient binding operations.
 * Time complexity: O(T log T) using FFT instead of O(T^2) for naive convolution.
 * 
 * HRR allows us to bind vectors together in a way that preserves information and
 * supports unbinding operations. This is key for representing structured relationships.
 */

import { ComplexTensor } from '../core/ComplexTensor';

export class HolographicReducedRepresentation {
  /**
   * FFT implementation using Cooley-Tukey algorithm
   * Converts time-domain signal to frequency domain
   */
  private static fft(real: number[], imag: number[]): { real: number[], imag: number[] } {
    const n = real.length;
    
    if (n <= 1) {
      return { real: [...real], imag: [...imag] };
    }

    // Ensure power of 2 for efficiency
    if ((n & (n - 1)) !== 0) {
      // Pad to next power of 2
      const nextPow2 = Math.pow(2, Math.ceil(Math.log2(n)));
      real = [...real, ...new Array(nextPow2 - n).fill(0)];
      imag = [...imag, ...new Array(nextPow2 - n).fill(0)];
      return this.fft(real, imag);
    }

    // Divide: split into even and odd indices
    const evenReal: number[] = [];
    const evenImag: number[] = [];
    const oddReal: number[] = [];
    const oddImag: number[] = [];

    for (let i = 0; i < n; i++) {
      if (i % 2 === 0) {
        evenReal.push(real[i]);
        evenImag.push(imag[i]);
      } else {
        oddReal.push(real[i]);
        oddImag.push(imag[i]);
      }
    }

    // Conquer: recursively compute FFT of even and odd parts
    const evenFFT = this.fft(evenReal, evenImag);
    const oddFFT = this.fft(oddReal, oddImag);

    // Combine
    const resultReal = new Array(n);
    const resultImag = new Array(n);

    for (let k = 0; k < n / 2; k++) {
      const angle = -2 * Math.PI * k / n;
      const twiddleReal = Math.cos(angle);
      const twiddleImag = Math.sin(angle);

      // Complex multiplication: twiddle * odd[k]
      const tOddReal = twiddleReal * oddFFT.real[k] - twiddleImag * oddFFT.imag[k];
      const tOddImag = twiddleReal * oddFFT.imag[k] + twiddleImag * oddFFT.real[k];

      resultReal[k] = evenFFT.real[k] + tOddReal;
      resultImag[k] = evenFFT.imag[k] + tOddImag;
      resultReal[k + n / 2] = evenFFT.real[k] - tOddReal;
      resultImag[k + n / 2] = evenFFT.imag[k] - tOddImag;
    }

    return { real: resultReal, imag: resultImag };
  }

  /**
   * Inverse FFT - converts frequency domain back to time domain
   */
  private static ifft(real: number[], imag: number[]): { real: number[], imag: number[] } {
    const n = real.length;
    
    // Take complex conjugate
    const conjImag = imag.map(x => -x);
    
    // Apply FFT
    const result = this.fft(real, conjImag);
    
    // Take complex conjugate again and scale by 1/n
    return {
      real: result.real.map(x => x / n),
      imag: result.imag.map(x => -x / n)
    };
  }

  /**
   * Circular convolution using FFT
   * This is the binding operation in HRR
   * 
   * bind(A, B) = IFFT(FFT(A) * FFT(B))
   * 
   * Time complexity: O(n log n)
   */
  static bind(a: ComplexTensor, b: ComplexTensor): ComplexTensor {
    if (a.size() !== b.size()) {
      throw new Error('Tensors must have the same size for binding');
    }

    // Apply FFT to both tensors
    const aFFT = this.fft(a.real, a.imag);
    const bFFT = this.fft(b.real, b.imag);

    // Element-wise complex multiplication in frequency domain
    const productReal: number[] = [];
    const productImag: number[] = [];

    for (let i = 0; i < aFFT.real.length; i++) {
      productReal.push(
        aFFT.real[i] * bFFT.real[i] - aFFT.imag[i] * bFFT.imag[i]
      );
      productImag.push(
        aFFT.real[i] * bFFT.imag[i] + aFFT.imag[i] * bFFT.real[i]
      );
    }

    // Apply inverse FFT to get back to time domain
    const result = this.ifft(productReal, productImag);

    // Trim back to original size if padded
    const originalSize = a.size();
    return new ComplexTensor(
      result.real.slice(0, originalSize),
      result.imag.slice(0, originalSize),
      a.shape
    );
  }

  /**
   * Unbinding operation (approximate inverse)
   * 
   * unbind(C, B) ≈ A  where C = bind(A, B)
   * 
   * This is done by binding with the complex conjugate:
   * unbind(C, B) = bind(C, conj(B))
   */
  static unbind(bound: ComplexTensor, key: ComplexTensor): ComplexTensor {
    return this.bind(bound, key.conjugate());
  }

  /**
   * Generalized binding for non-commutative relationships
   * Uses different operations for left and right binding to preserve order
   * 
   * This implements GHRR (Generalized HRR) for structured asymmetric relationships
   */
  static bindNonCommutative(
    left: ComplexTensor,
    right: ComplexTensor,
    leftOp: 'bind' | 'permute',
    rightOp: 'bind' | 'permute'
  ): ComplexTensor {
    let result = left.clone();

    // Apply left operation
    if (leftOp === 'permute') {
      result = this.permute(result);
    }

    // Bind with right
    result = this.bind(result, right);

    // Apply right operation
    if (rightOp === 'permute') {
      result = this.permute(result);
    }

    return result;
  }

  /**
   * Permutation operation for non-commutative binding
   * Implements a circular shift to break commutativity
   */
  private static permute(tensor: ComplexTensor, shift: number = 1): ComplexTensor {
    const n = tensor.size();
    const real = new Array(n);
    const imag = new Array(n);

    for (let i = 0; i < n; i++) {
      const newIdx = (i + shift) % n;
      real[newIdx] = tensor.real[i];
      imag[newIdx] = tensor.imag[i];
    }

    return new ComplexTensor(real, imag, tensor.shape);
  }

  /**
   * Compute similarity between two tensors using cosine similarity
   * in the complex domain (using magnitude and phase alignment)
   */
  static similarity(a: ComplexTensor, b: ComplexTensor): number {
    if (a.size() !== b.size()) {
      throw new Error('Tensors must have the same size for similarity');
    }

    // Compute inner product
    const dotProduct = a.dot(b);
    const dotMag = Math.sqrt(
      dotProduct.real[0] * dotProduct.real[0] + 
      dotProduct.imag[0] * dotProduct.imag[0]
    );

    // Compute norms
    const aMag = Math.sqrt(
      a.real.reduce((sum, r, i) => sum + r * r + a.imag[i] * a.imag[i], 0)
    );
    const bMag = Math.sqrt(
      b.real.reduce((sum, r, i) => sum + r * r + b.imag[i] * b.imag[i], 0)
    );

    if (aMag === 0 || bMag === 0) return 0;

    return dotMag / (aMag * bMag);
  }

  /**
   * Create a normalized random vector suitable for HRR operations
   * This ensures vectors have unit magnitude for stable binding/unbinding
   */
  static randomVector(size: number): ComplexTensor {
    const tensor = ComplexTensor.randn([size], 1.0 / Math.sqrt(size));
    
    // Normalize to unit magnitude
    const mags = tensor.magnitude();
    const norm = Math.sqrt(mags.reduce((sum, m) => sum + m * m, 0));
    
    return tensor.scale(1.0 / norm);
  }

  /**
   * Superposition of multiple vectors (weighted sum)
   * This allows representing multiple concepts simultaneously
   */
  static superpose(tensors: ComplexTensor[], weights?: number[]): ComplexTensor {
    if (tensors.length === 0) {
      throw new Error('Need at least one tensor for superposition');
    }

    if (!weights) {
      weights = new Array(tensors.length).fill(1.0 / tensors.length);
    }

    let result = tensors[0].scale(weights[0]);
    
    for (let i = 1; i < tensors.length; i++) {
      result = result.add(tensors[i].scale(weights[i]));
    }

    return result;
  }
}
