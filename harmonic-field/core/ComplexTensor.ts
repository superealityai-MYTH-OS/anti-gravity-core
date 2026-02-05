/**
 * ComplexTensor: Core data structure for complex-valued neural networks
 * 
 * Represents complex numbers with real and imaginary components for use in CVNNs.
 * Provides basic complex arithmetic operations needed for neural network computations.
 */

export class ComplexTensor {
  public real: number[];
  public imag: number[];
  public shape: number[];

  constructor(real: number[], imag: number[], shape: number[]) {
    if (real.length !== imag.length) {
      throw new Error('Real and imaginary parts must have the same length');
    }
    this.real = real;
    this.imag = imag;
    this.shape = shape;
  }

  /**
   * Create a ComplexTensor from magnitude and phase
   * z = r * e^(iθ) = r * (cos(θ) + i*sin(θ))
   */
  static fromPolar(magnitude: number[], phase: number[], shape: number[]): ComplexTensor {
    const real = magnitude.map((r, i) => r * Math.cos(phase[i]));
    const imag = magnitude.map((r, i) => r * Math.sin(phase[i]));
    return new ComplexTensor(real, imag, shape);
  }

  /**
   * Create a ComplexTensor initialized with zeros
   */
  static zeros(shape: number[]): ComplexTensor {
    const size = shape.reduce((a, b) => a * b, 1);
    return new ComplexTensor(
      new Array(size).fill(0),
      new Array(size).fill(0),
      shape
    );
  }

  /**
   * Create a ComplexTensor with random values (Gaussian distribution)
   */
  static randn(shape: number[], stddev: number = 1.0): ComplexTensor {
    const size = shape.reduce((a, b) => a * b, 1);
    const real = new Array(size).fill(0).map(() => this.randomGaussian() * stddev);
    const imag = new Array(size).fill(0).map(() => this.randomGaussian() * stddev);
    return new ComplexTensor(real, imag, shape);
  }

  /**
   * Box-Muller transform for Gaussian random numbers
   */
  private static randomGaussian(): number {
    const u1 = Math.random();
    const u2 = Math.random();
    return Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
  }

  /**
   * Get magnitude (amplitude) of complex numbers
   * |z| = sqrt(real^2 + imag^2)
   */
  magnitude(): number[] {
    return this.real.map((r, i) => Math.sqrt(r * r + this.imag[i] * this.imag[i]));
  }

  /**
   * Get phase (angle) of complex numbers
   * θ = atan2(imag, real)
   */
  phase(): number[] {
    return this.real.map((r, i) => Math.atan2(this.imag[i], r));
  }

  /**
   * Complex multiplication: (a + bi) * (c + di) = (ac - bd) + (ad + bc)i
   */
  multiply(other: ComplexTensor): ComplexTensor {
    if (this.real.length !== other.real.length) {
      throw new Error('Tensors must have the same size for element-wise multiplication');
    }
    
    const real = this.real.map((r, i) => 
      r * other.real[i] - this.imag[i] * other.imag[i]
    );
    const imag = this.real.map((r, i) => 
      r * other.imag[i] + this.imag[i] * other.real[i]
    );
    
    return new ComplexTensor(real, imag, this.shape);
  }

  /**
   * Complex addition: (a + bi) + (c + di) = (a + c) + (b + d)i
   */
  add(other: ComplexTensor): ComplexTensor {
    if (this.real.length !== other.real.length) {
      throw new Error('Tensors must have the same size for addition');
    }
    
    const real = this.real.map((r, i) => r + other.real[i]);
    const imag = this.imag.map((im, i) => im + other.imag[i]);
    
    return new ComplexTensor(real, imag, this.shape);
  }

  /**
   * Complex conjugate: conj(a + bi) = a - bi
   */
  conjugate(): ComplexTensor {
    return new ComplexTensor(
      [...this.real],
      this.imag.map(im => -im),
      this.shape
    );
  }

  /**
   * Scale by a real number
   */
  scale(scalar: number): ComplexTensor {
    return new ComplexTensor(
      this.real.map(r => r * scalar),
      this.imag.map(im => im * scalar),
      this.shape
    );
  }

  /**
   * Compute dot product with another ComplexTensor
   * Includes complex conjugate for proper inner product in Hilbert space
   */
  dot(other: ComplexTensor): ComplexTensor {
    if (this.real.length !== other.real.length) {
      throw new Error('Tensors must have the same size for dot product');
    }

    // Conjugate transpose of this tensor
    const conjThis = this.conjugate();
    
    // Element-wise multiplication and sum
    let realSum = 0;
    let imagSum = 0;
    
    for (let i = 0; i < this.real.length; i++) {
      const prodReal = conjThis.real[i] * other.real[i] - conjThis.imag[i] * other.imag[i];
      const prodImag = conjThis.real[i] * other.imag[i] + conjThis.imag[i] * other.real[i];
      realSum += prodReal;
      imagSum += prodImag;
    }
    
    return new ComplexTensor([realSum], [imagSum], [1]);
  }

  /**
   * Apply a complex activation function (e.g., modReLU)
   * ModReLU: z' = ReLU(|z| + b) * (z / |z|)
   */
  modReLU(bias: number = 0): ComplexTensor {
    const mags = this.magnitude();
    const real: number[] = [];
    const imag: number[] = [];
    
    for (let i = 0; i < this.real.length; i++) {
      const mag = mags[i];
      const activated = Math.max(0, mag + bias);
      
      if (mag > 0) {
        const scale = activated / mag;
        real.push(this.real[i] * scale);
        imag.push(this.imag[i] * scale);
      } else {
        real.push(0);
        imag.push(0);
      }
    }
    
    return new ComplexTensor(real, imag, this.shape);
  }

  /**
   * Get the size of the tensor
   */
  size(): number {
    return this.real.length;
  }

  /**
   * Clone the tensor
   */
  clone(): ComplexTensor {
    return new ComplexTensor([...this.real], [...this.imag], [...this.shape]);
  }

  /**
   * Reshape the tensor
   */
  reshape(newShape: number[]): ComplexTensor {
    const newSize = newShape.reduce((a, b) => a * b, 1);
    if (newSize !== this.size()) {
      throw new Error('New shape must have the same total size');
    }
    return new ComplexTensor(this.real, this.imag, newShape);
  }
}
