/**
 * Basic Usage Examples for Harmonic Field Architecture
 * 
 * This file provides simple, runnable examples demonstrating the main
 * features of the Harmonic Field architecture.
 */

import {
  ComplexTensor,
  HolographicReducedRepresentation as HRR,
  createHarmonicFieldModel,
  GlobalWorkspace,
  EnergyBasedModel
} from '../index';

/**
 * Example 1: Basic Complex Tensor Operations
 */
export function example1_ComplexTensors() {
  console.log('Example 1: Complex Tensor Operations\n');
  
  // Create complex tensors
  const t1 = new ComplexTensor([1, 2, 3], [4, 5, 6], [3]);
  const t2 = new ComplexTensor([7, 8, 9], [10, 11, 12], [3]);
  
  console.log('Tensor 1 magnitude:', t1.magnitude());
  console.log('Tensor 1 phase:', t1.phase());
  
  // Complex multiplication
  const product = t1.multiply(t2);
  console.log('Product magnitude:', product.magnitude());
  
  // Complex addition
  const sum = t1.add(t2);
  console.log('Sum:', sum.real, sum.imag);
  
  // Conjugate
  const conj = t1.conjugate();
  console.log('Conjugate imag:', conj.imag);
  
  // Polar representation
  const polar = ComplexTensor.fromPolar([1, 2], [0, Math.PI/2], [2]);
  console.log('Polar tensor:', polar.real, polar.imag);
  
  console.log();
}

/**
 * Example 2: Holographic Binding and Unbinding
 */
export function example2_HolographicBinding() {
  console.log('Example 2: Holographic Binding\n');
  
  // Create random vectors
  const dog = HRR.randomVector(32);
  const cat = HRR.randomVector(32);
  const animal = HRR.randomVector(32);
  
  console.log('Created three random concept vectors (dog, cat, animal)');
  
  // Bind dog with animal
  const dogIsAnimal = HRR.bind(dog, animal);
  console.log('Bound: dog + animal');
  
  // Unbind to retrieve
  const retrievedAnimal = HRR.unbind(dogIsAnimal, dog);
  console.log('Unbound with dog to retrieve animal');
  
  // Check similarity
  const similarity = HRR.similarity(animal, retrievedAnimal);
  console.log(`Similarity with original: ${similarity.toFixed(4)}`);
  
  // Superposition
  const animals = HRR.superpose([dog, cat], [0.6, 0.4]);
  console.log('Created superposition: 60% dog + 40% cat');
  
  const simToDog = HRR.similarity(dog, animals);
  const simToCat = HRR.similarity(cat, animals);
  console.log(`Similarity to dog: ${simToDog.toFixed(4)}`);
  console.log(`Similarity to cat: ${simToCat.toFixed(4)}`);
  
  console.log();
}

/**
 * Example 3: Harmonic Field Model
 */
export function example3_HarmonicFieldModel() {
  console.log('Example 3: Harmonic Field Model\n');
  
  // Create model
  const model = createHarmonicFieldModel(10, 5, 64);
  console.log('Created model: 10 input dims, 5 output dims, 64 hidden dims');
  
  // Forward pass
  const input = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
  const result = model.forward(input);
  
  console.log('Output:', result.output);
  console.log('Phase:', result.phase);
  console.log('Workspace stats:', result.workspaceStats);
  
  // Concept similarity
  const concept1 = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0];
  const concept2 = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0];
  const similarity = model.conceptSimilarity(concept1, concept2);
  console.log(`Concept similarity: ${similarity.toFixed(4)}`);
  
  // Bind and unbind
  const bound = model.bindConcepts(concept1, concept2);
  const retrieved = model.unbindConcept(bound, concept1);
  console.log('Retrieved concept:', retrieved);
  
  console.log();
}

/**
 * Example 4: Global Workspace and Consciousness
 */
export function example4_GlobalWorkspace() {
  console.log('Example 4: Global Workspace\n');
  
  const workspace = new GlobalWorkspace(64, 0.7, 1.0);
  console.log('Created global workspace');
  
  // Add elements with varying energies
  const elem1 = ComplexTensor.randn([64], 0.1);
  const elem2 = ComplexTensor.randn([64], 0.5);
  const elem3 = ComplexTensor.randn([64], 1.0);
  
  workspace.addElement(elem1, 0.2);
  workspace.addElement(elem2, 0.6);
  workspace.addElement(elem3, 1.5);
  
  console.log('Added 3 elements with different energies');
  
  // Get statistics
  const stats = workspace.getStatistics();
  console.log('Workspace statistics:');
  console.log(`  Total elements: ${stats.totalElements}`);
  console.log(`  Conscious: ${stats.consciousCount}`);
  console.log(`  Preconscious: ${stats.preconsciousCount}`);
  console.log(`  Subliminal: ${stats.subliminalCount}`);
  console.log(`  Average energy: ${stats.averageEnergy.toFixed(4)}`);
  
  // Check for ignition
  const prediction = ComplexTensor.randn([64], 0.1);
  const target = ComplexTensor.randn([64], 2.0); // Very different
  const ignition = workspace.checkIgnition(prediction, target, 1.5);
  
  if (ignition && ignition.triggered) {
    console.log('\nIgnition event detected!');
    console.log(`  Prediction error: ${ignition.predictionError.toFixed(4)}`);
    console.log(`  Energy spike: ${ignition.energySpike.toFixed(4)}`);
  }
  
  console.log();
}

/**
 * Example 5: Energy-Based Inference
 */
export function example5_EnergyBasedInference() {
  console.log('Example 5: Energy-Based Inference\n');
  
  const ebm = new EnergyBasedModel(0.05, 100);
  console.log('Created energy-based model');
  
  // Initial state and target
  const initial = ComplexTensor.zeros([32]);
  const target = ComplexTensor.randn([32], 1.0);
  
  console.log('Starting inference from zero state to random target');
  
  // Perform inference
  const result = ebm.infer(initial, target, 0.001);
  
  console.log(`Converged in ${result.iterations} iterations`);
  console.log(`Final energy: ${result.finalEnergy.toFixed(6)}`);
  console.log(`Final state magnitude: ${result.finalState.magnitude().slice(0, 5)}`);
  
  console.log();
}

/**
 * Example 6: Attention and Workspace Together
 */
export function example6_IntegratedExample() {
  console.log('Example 6: Integrated Example\n');
  
  // Create a full model
  const model = createHarmonicFieldModel(8, 4, 32);
  
  // Process multiple inputs
  const inputs = [
    [1, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0]
  ];
  
  console.log('Processing sequence of 4 inputs...\n');
  
  for (let i = 0; i < inputs.length; i++) {
    const result = model.forward(inputs[i]);
    
    console.log(`Step ${i + 1}:`);
    console.log(`  Output: [${result.output.map(o => o.toFixed(3)).join(', ')}]`);
    console.log(`  Conscious elements: ${result.workspaceStats?.consciousCount}`);
    console.log(`  Average amplitude: ${result.workspaceStats?.averageAmplitude.toFixed(4)}`);
  }
  
  // Get final workspace info
  const info = model.getWorkspaceInfo();
  console.log(`\nFinal workspace state:`);
  console.log(`  Total conscious elements: ${info.elements.length}`);
  console.log(`  Ignition events: ${info.ignitionHistory.length}`);
  
  console.log();
}

/**
 * Run all examples
 */
export function runAllExamples() {
  console.log('╔════════════════════════════════════════════════════════╗');
  console.log('║       Harmonic Field: Basic Usage Examples            ║');
  console.log('╚════════════════════════════════════════════════════════╝\n');
  
  example1_ComplexTensors();
  example2_HolographicBinding();
  example3_HarmonicFieldModel();
  example4_GlobalWorkspace();
  example5_EnergyBasedInference();
  example6_IntegratedExample();
  
  console.log('╔════════════════════════════════════════════════════════╗');
  console.log('║            All Examples Completed                      ║');
  console.log('╚════════════════════════════════════════════════════════╝\n');
}

// Run if executed directly
if (typeof require !== 'undefined' && require.main === module) {
  runAllExamples();
}
