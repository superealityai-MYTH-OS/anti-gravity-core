/**
 * Complex-Valued Neural Network (CVNN) Layers
 * 
 * Implements neural network layers that operate on complex-valued tensors.
 * These layers use phase information for semantic and temporal context,
 * enabling better representation learning and decision boundary orthogonality.
 */

import { ComplexTensor } from '../core/ComplexTensor';

/**
 * Complex-valued linear layer
 * Performs: output = W * input + b
 * where W is a complex weight matrix and b is a complex bias
 */
export class ComplexLinear {
  private weights: ComplexTensor;
  private bias: ComplexTensor;
  private inputDim: number;
  private outputDim: number;

  constructor(inputDim: number, outputDim: number, initScale: number = 0.1) {
    this.inputDim = inputDim;
    this.outputDim = outputDim;
    
    // Initialize weights using He initialization adapted for complex numbers
    const scale = initScale * Math.sqrt(2.0 / inputDim);
    this.weights = ComplexTensor.randn([outputDim, inputDim], scale);
    this.bias = ComplexTensor.zeros([outputDim]);
  }

  /**
   * Forward pass
   */
  forward(input: ComplexTensor): ComplexTensor {
    if (input.size() !== this.inputDim) {
      throw new Error(`Input size ${input.size()} doesn't match layer input dim ${this.inputDim}`);
    }

    // Matrix-vector multiplication for complex numbers
    const outputReal: number[] = [];
    const outputImag: number[] = [];

    for (let i = 0; i < this.outputDim; i++) {
      let sumReal = this.bias.real[i];
      let sumImag = this.bias.imag[i];

      for (let j = 0; j < this.inputDim; j++) {
        const wIdx = i * this.inputDim + j;
        const wReal = this.weights.real[wIdx];
        const wImag = this.weights.imag[wIdx];
        const inReal = input.real[j];
        const inImag = input.imag[j];

        // Complex multiplication: (w_r + w_i*i) * (in_r + in_i*i)
        sumReal += wReal * inReal - wImag * inImag;
        sumImag += wReal * inImag + wImag * inReal;
      }

      outputReal.push(sumReal);
      outputImag.push(sumImag);
    }

    return new ComplexTensor(outputReal, outputImag, [this.outputDim]);
  }

  /**
   * Get parameters for training
   */
  getParameters(): { weights: ComplexTensor; bias: ComplexTensor } {
    return {
      weights: this.weights,
      bias: this.bias
    };
  }

  /**
   * Set parameters (for loading trained weights)
   */
  setParameters(weights: ComplexTensor, bias: ComplexTensor): void {
    if (weights.size() !== this.outputDim * this.inputDim) {
      throw new Error('Invalid weights size');
    }
    if (bias.size() !== this.outputDim) {
      throw new Error('Invalid bias size');
    }
    this.weights = weights;
    this.bias = bias;
  }
}

/**
 * Complex-valued activation functions
 */
export class ComplexActivations {
  /**
   * ModReLU activation function
   * Applies ReLU to the magnitude while preserving phase
   * 
   * ModReLU(z) = ReLU(|z| + b) * (z / |z|)
   */
  static modReLU(input: ComplexTensor, bias: number = 0): ComplexTensor {
    return input.modReLU(bias);
  }

  /**
   * CReLU (Complex ReLU) - applies ReLU separately to real and imaginary parts
   */
  static cReLU(input: ComplexTensor): ComplexTensor {
    const real = input.real.map(r => Math.max(0, r));
    const imag = input.imag.map(im => Math.max(0, im));
    return new ComplexTensor(real, imag, input.shape);
  }

  /**
   * zReLU - phase-aware activation
   * Only activates if both magnitude and phase are in the first quadrant
   */
  static zReLU(input: ComplexTensor): ComplexTensor {
    const real: number[] = [];
    const imag: number[] = [];

    for (let i = 0; i < input.size(); i++) {
      const r = input.real[i];
      const im = input.imag[i];
      
      // Only activate if both real and imaginary parts are positive
      if (r >= 0 && im >= 0) {
        real.push(r);
        imag.push(im);
      } else {
        real.push(0);
        imag.push(0);
      }
    }

    return new ComplexTensor(real, imag, input.shape);
  }

  /**
   * Complex sigmoid activation using the split approach
   */
  static complexSigmoid(input: ComplexTensor): ComplexTensor {
    const sigmoid = (x: number) => 1 / (1 + Math.exp(-x));
    
    const real = input.real.map(sigmoid);
    const imag = input.imag.map(sigmoid);
    
    return new ComplexTensor(real, imag, input.shape);
  }

  /**
   * Complex tanh activation
   */
  static complexTanh(input: ComplexTensor): ComplexTensor {
    const real = input.real.map(Math.tanh);
    const imag = input.imag.map(Math.tanh);
    
    return new ComplexTensor(real, imag, input.shape);
  }
}

/**
 * Complex Batch Normalization
 * Normalizes both amplitude and phase separately
 */
export class ComplexBatchNorm {
  private runningMeanReal: number[];
  private runningMeanImag: number[];
  private runningVarReal: number[];
  private runningVarImag: number[];
  private epsilon: number;
  private momentum: number;

  constructor(numFeatures: number, epsilon: number = 1e-5, momentum: number = 0.1) {
    this.epsilon = epsilon;
    this.momentum = momentum;
    this.runningMeanReal = new Array(numFeatures).fill(0);
    this.runningMeanImag = new Array(numFeatures).fill(0);
    this.runningVarReal = new Array(numFeatures).fill(1);
    this.runningVarImag = new Array(numFeatures).fill(1);
  }

  /**
   * Forward pass with batch normalization
   */
  forward(input: ComplexTensor, training: boolean = true): ComplexTensor {
    const n = input.size();
    
    if (training) {
      // Compute batch statistics
      const meanReal = input.real.reduce((sum, r) => sum + r, 0) / n;
      const meanImag = input.imag.reduce((sum, im) => sum + im, 0) / n;
      
      const varReal = input.real.reduce((sum, r) => sum + (r - meanReal) ** 2, 0) / n;
      const varImag = input.imag.reduce((sum, im) => sum + (im - meanImag) ** 2, 0) / n;
      
      // Update running statistics
      for (let i = 0; i < this.runningMeanReal.length && i < n; i++) {
        this.runningMeanReal[i] = (1 - this.momentum) * this.runningMeanReal[i] + this.momentum * meanReal;
        this.runningMeanImag[i] = (1 - this.momentum) * this.runningMeanImag[i] + this.momentum * meanImag;
        this.runningVarReal[i] = (1 - this.momentum) * this.runningVarReal[i] + this.momentum * varReal;
        this.runningVarImag[i] = (1 - this.momentum) * this.runningVarImag[i] + this.momentum * varImag;
      }
      
      // Normalize
      const real = input.real.map(r => (r - meanReal) / Math.sqrt(varReal + this.epsilon));
      const imag = input.imag.map(im => (im - meanImag) / Math.sqrt(varImag + this.epsilon));
      
      return new ComplexTensor(real, imag, input.shape);
    } else {
      // Use running statistics
      const real = input.real.map((r, i) => 
        (r - this.runningMeanReal[Math.min(i, this.runningMeanReal.length - 1)]) / 
        Math.sqrt(this.runningVarReal[Math.min(i, this.runningVarReal.length - 1)] + this.epsilon)
      );
      const imag = input.imag.map((im, i) => 
        (im - this.runningMeanImag[Math.min(i, this.runningMeanImag.length - 1)]) / 
        Math.sqrt(this.runningVarImag[Math.min(i, this.runningVarImag.length - 1)] + this.epsilon)
      );
      
      return new ComplexTensor(real, imag, input.shape);
    }
  }
}

/**
 * Dropout for complex-valued networks
 * Drops entire complex values (both real and imaginary parts together)
 */
export class ComplexDropout {
  private dropoutRate: number;

  constructor(dropoutRate: number = 0.5) {
    this.dropoutRate = dropoutRate;
  }

  forward(input: ComplexTensor, training: boolean = true): ComplexTensor {
    if (!training || this.dropoutRate === 0) {
      return input.clone();
    }

    const scale = 1.0 / (1.0 - this.dropoutRate);
    const real: number[] = [];
    const imag: number[] = [];

    for (let i = 0; i < input.size(); i++) {
      if (Math.random() > this.dropoutRate) {
        real.push(input.real[i] * scale);
        imag.push(input.imag[i] * scale);
      } else {
        real.push(0);
        imag.push(0);
      }
    }

    return new ComplexTensor(real, imag, input.shape);
  }
}
