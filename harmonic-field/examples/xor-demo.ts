/**
 * XOR Problem Demonstration
 * 
 * This example demonstrates the advantages of Complex-Valued Neural Networks
 * over real-valued networks in solving the XOR problem.
 * 
 * The XOR problem is a classic test for neural networks because it's not
 * linearly separable in 2D real space, but becomes separable when using
 * complex-valued embeddings with phase information.
 */

import { createHarmonicFieldModel } from '../HarmonicFieldModel';
import { ComplexTensor } from '../core/ComplexTensor';

// XOR truth table
const XOR_DATA = [
  { input: [0, 0], target: 0 },
  { input: [0, 1], target: 1 },
  { input: [1, 0], target: 1 },
  { input: [1, 1], target: 0 }
];

/**
 * Train and test XOR using the Harmonic Field model
 */
export function demonstrateXOR() {
  console.log('=== XOR Problem Demonstration ===\n');
  
  // Create model with 2 inputs, 1 output
  const model = createHarmonicFieldModel(2, 1, 16);
  
  console.log('Testing XOR problem with Complex-Valued Neural Network:\n');
  
  // Test each XOR case
  for (const { input, target } of XOR_DATA) {
    const result = model.forward(input);
    const output = result.output[0];
    const phase = result.phase[0];
    
    // Round to nearest integer for classification
    const predicted = output > 0.5 ? 1 : 0;
    const correct = predicted === target;
    
    console.log(`Input: [${input[0]}, ${input[1]}]`);
    console.log(`  Target: ${target}`);
    console.log(`  Output: ${output.toFixed(4)} (predicted: ${predicted})`);
    console.log(`  Phase: ${phase.toFixed(4)} radians`);
    console.log(`  Correct: ${correct ? '✓' : '✗'}`);
    console.log(`  Workspace: ${result.workspaceStats?.consciousCount || 0} conscious elements\n`);
  }
  
  return model;
}

/**
 * Demonstrate phase-based separation
 * Shows how complex embeddings create orthogonal decision boundaries
 */
export function demonstratePhaseSeparation() {
  console.log('=== Phase-Based Decision Boundary Demonstration ===\n');
  
  const model = createHarmonicFieldModel(2, 1, 16);
  
  // Create embeddings for XOR inputs
  const embeddings = XOR_DATA.map(({ input }) => {
    const result = model.forward(input);
    return {
      input,
      hidden: result.hidden,
      magnitude: result.hidden.magnitude(),
      phase: result.hidden.phase()
    };
  });
  
  console.log('Complex embeddings for XOR inputs:\n');
  
  for (let i = 0; i < embeddings.length; i++) {
    const emb = embeddings[i];
    const avgMag = emb.magnitude.reduce((sum, m) => sum + m, 0) / emb.magnitude.length;
    const avgPhase = emb.phase.reduce((sum, p) => sum + p, 0) / emb.phase.length;
    
    console.log(`Input [${emb.input[0]}, ${emb.input[1]}]:`);
    console.log(`  Average Magnitude: ${avgMag.toFixed(4)}`);
    console.log(`  Average Phase: ${avgPhase.toFixed(4)} rad (${(avgPhase * 180 / Math.PI).toFixed(2)}°)`);
    console.log();
  }
  
  // Compute pairwise similarities
  console.log('Pairwise similarities (cosine similarity in complex space):\n');
  
  for (let i = 0; i < embeddings.length; i++) {
    for (let j = i + 1; j < embeddings.length; j++) {
      const emb1 = embeddings[i];
      const emb2 = embeddings[j];
      
      // Use dot product for similarity
      const dotResult = emb1.hidden.dot(emb2.hidden);
      const similarity = Math.sqrt(
        dotResult.real[0] ** 2 + dotResult.imag[0] ** 2
      );
      
      const target1 = XOR_DATA[i].target;
      const target2 = XOR_DATA[j].target;
      const sameClass = target1 === target2;
      
      console.log(`[${emb1.input}] vs [${emb2.input}]:`);
      console.log(`  Similarity: ${similarity.toFixed(4)}`);
      console.log(`  Same XOR class: ${sameClass ? 'Yes' : 'No'}`);
      console.log();
    }
  }
}

/**
 * Demonstrate holographic binding for XOR relationships
 */
export function demonstrateHolographicXOR() {
  console.log('=== Holographic Binding for XOR ===\n');
  
  const model = createHarmonicFieldModel(2, 1, 32);
  
  // Bind input pairs with their XOR outputs
  console.log('Binding XOR relationships using HRR:\n');
  
  const bindings: ComplexTensor[] = [];
  
  for (const { input, target } of XOR_DATA) {
    // Create target encoding
    const targetEncoding = [target, 1 - target]; // One-hot-ish encoding
    
    // Bind input with target
    const bound = model.bindConcepts(input, targetEncoding);
    bindings.push(bound);
    
    console.log(`Bound [${input[0]}, ${input[1]}] with target ${target}`);
  }
  
  // Test unbinding
  console.log('\nTesting retrieval through unbinding:\n');
  
  for (let i = 0; i < XOR_DATA.length; i++) {
    const { input, target } = XOR_DATA[i];
    const bound = bindings[i];
    
    // Unbind to retrieve target
    const retrieved = model.unbindConcept(bound, input);
    
    console.log(`Input: [${input[0]}, ${input[1]}]`);
    console.log(`  Original target: ${target}`);
    console.log(`  Retrieved: [${retrieved[0].toFixed(4)}, ${retrieved[1].toFixed(4)}]`);
    console.log();
  }
}

/**
 * Demonstrate energy landscape for XOR
 */
export function demonstrateEnergyLandscape() {
  console.log('=== Energy Landscape for XOR ===\n');
  
  const model = createHarmonicFieldModel(2, 1, 16);
  
  console.log('Computing energy for each XOR case:\n');
  
  for (const { input, target } of XOR_DATA) {
    // Use EBM to find low-energy state
    const targetArray = [target];
    const result = model.inferWithEBM(input, targetArray, 50);
    
    console.log(`Input: [${input[0]}, ${input[1]}], Target: ${target}`);
    console.log(`  Final energy: ${result.energy.toFixed(4)}`);
    console.log(`  Iterations to converge: ${result.iterations}`);
    console.log(`  Final output: ${result.finalOutput[0].toFixed(4)}`);
    console.log();
  }
}

/**
 * Run all XOR demonstrations
 */
export function runAllXORDemos() {
  console.log('╔════════════════════════════════════════════════════════╗');
  console.log('║   Harmonic Field: XOR Problem Demonstration Suite     ║');
  console.log('╚════════════════════════════════════════════════════════╝\n');
  
  demonstrateXOR();
  console.log('\n' + '─'.repeat(60) + '\n');
  
  demonstratePhaseSeparation();
  console.log('\n' + '─'.repeat(60) + '\n');
  
  demonstrateHolographicXOR();
  console.log('\n' + '─'.repeat(60) + '\n');
  
  demonstrateEnergyLandscape();
  
  console.log('\n╔════════════════════════════════════════════════════════╗');
  console.log('║              Demonstration Complete                    ║');
  console.log('╚════════════════════════════════════════════════════════╝\n');
}

// Run if executed directly
if (typeof require !== 'undefined' && require.main === module) {
  runAllXORDemos();
}
