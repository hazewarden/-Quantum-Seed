#!/usr/bin/env python3
"""
Quantum Seed: Simple Quantum-Inspired Learning System
Author: Luke Spookwalker
Date: 2025
License: MIT
"""

import numpy as np
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt

class QuantumSeed:
    """
    A quantum-inspired learning system that learns from single examples
    using quantum state collapse dynamics.
    
    Key Features:
    - Single-example learning capability
    - Quantum state collapse instead of gradient descent
    - Phase-encoded memory
    - Quantum interference generalization
    """
    
    def __init__(self, size: int = 8, collapse_rate: float = 0.5, 
                 learning_rate: float = 0.3):
        """
        Initialize a quantum learning seed.
        
        Parameters:
        -----------
        size : int
            Number of quantum dimensions (like qubits)
        collapse_rate : float
            How strongly patterns cause quantum collapse (0-1)
        learning_rate : float
            Speed of quantum state rotation during learning
        """
        self.size = size
        self.collapse_rate = collapse_rate
        self.learning_rate = learning_rate
        
        # Initialize in maximal superposition
        # Equal amplitudes, random phases
        self.amplitudes = np.ones(size, dtype=complex) / np.sqrt(size)
        phases = np.random.uniform(0, 2*np.pi, size)
        self.amplitudes *= np.exp(1j * phases)
        
        # Quantum memory for interference patterns
        self.interference_patterns = []
        self.learning_history = []
        
        print(f"⚛️ Quantum Seed initialized with {size} dimensions")
        print(f"   Collapse rate: {collapse_rate}, Learning rate: {learning_rate}")
    
    def collapse_learn(self, pattern: np.ndarray, label: int, 
                      visualize: bool = False) -> bool:
        """
        Learn from a single example using quantum collapse dynamics.
        
        Parameters:
        -----------
        pattern : np.ndarray
            Input pattern (array of +1/-1)
        label : int
            Target label (+1 or -1)
        visualize : bool
            Whether to show learning visualization
        
        Returns:
        --------
        bool : True if prediction was correct after learning
        """
        # Store initial state
        initial_state = self.amplitudes.copy()
        
        # Make prediction before learning
        pred_before = self.predict(pattern)
        correct_before = (pred_before == label)
        
        if not correct_before:
            # QUANTUM COLLAPSE LEARNING
            error = label - pred_before
            
            # Apply quantum rotation to each dimension
            rotation = np.exp(1j * self.learning_rate * error)
            
            for i in range(self.size):
                if pattern[i] != 0:
                    # Rotate amplitude based on pattern and error
                    self.amplitudes[i] *= rotation * pattern[i]
            
            # Apply quantum interference
            self._apply_interference()
            
            # Normalize quantum state (preserve probability)
            self._normalize_state()
            
            # Record learning event
            self.learning_history.append({
                'pattern': pattern.copy(),
                'label': label,
                'error': error,
                'state_change': np.linalg.norm(self.amplitudes - initial_state),
                'entropy_change': self.quantum_entropy() - self._state_entropy(initial_state)
            })
        
        # Store interference pattern for future learning
        self.interference_patterns.append(np.abs(self.amplitudes))
        
        # Make prediction after learning
        pred_after = self.predict(pattern)
        correct_after = (pred_after == label)
        
        if visualize:
            self._visualize_learning(initial_state, pattern, label, 
                                   pred_before, pred_after, correct_before)
        
        return correct_after
    
    def predict(self, pattern: np.ndarray) -> int:
        """
        Make a quantum measurement prediction.
        
        Parameters:
        -----------
        pattern : np.ndarray
            Input pattern to classify
        
        Returns:
        --------
        int : Prediction (+1 or -1)
        """
        # Calculate quantum overlap
        overlap = np.abs(np.vdot(self.amplitudes, pattern))**2
        
        # Also consider similarity to learned patterns (quantum tunneling)
        similarity = self._pattern_similarity(pattern)
        
        # Combined quantum score
        quantum_score = 0.7 * overlap + 0.3 * similarity
        
        # Threshold decision
        return 1 if quantum_score > 0.25 else -1
    
    def _apply_interference(self):
        """Apply quantum interference to amplify correct patterns."""
        # Fourier transform for interference
        freq_domain = np.fft.fft(self.amplitudes)
        
        # Amplify constructive interference frequencies
        magnitude = np.abs(freq_domain)
        phase = np.angle(freq_domain)
        
        # Boost frequencies that match learned patterns
        if self.interference_patterns:
            for pattern in self.interference_patterns[-3:]:
                pattern_fft = np.fft.fft(pattern)
                # Align phases for constructive interference
                phase_correction = np.angle(pattern_fft) - phase
                phase += 0.1 * phase_correction
        
        # Reconstruct with modified interference
        self.amplitudes = np.fft.ifft(magnitude * np.exp(1j * phase))
        self._normalize_state()
    
    def _pattern_similarity(self, pattern: np.ndarray) -> float:
        """Calculate quantum similarity to stored patterns."""
        if not self.interference_patterns:
            return 0.0
        
        similarities = []
        for stored_pattern in self.interference_patterns[-5:]:
            sim = np.abs(np.vdot(stored_pattern, pattern))
            similarities.append(sim)
        
        return np.mean(similarities) if similarities else 0.0
    
    def _normalize_state(self):
        """Normalize quantum state to preserve total probability = 1."""
        norm = np.linalg.norm(self.amplitudes)
        if norm > 0:
            self.amplitudes /= norm
    
    def quantum_entropy(self) -> float:
        """Calculate quantum entropy of the current state."""
        probabilities = np.abs(self.amplitudes)**2
        probabilities = probabilities[probabilities > 0]
        probabilities = probabilities / probabilities.sum()
        return -np.sum(probabilities * np.log(probabilities))
    
    def _state_entropy(self, state: np.ndarray) -> float:
        """Calculate entropy of a given state."""
        probabilities = np.abs(state)**2
        probabilities = probabilities[probabilities > 0]
        probabilities = probabilities / probabilities.sum()
        return -np.sum(probabilities * np.log(probabilities))
    
    def get_state_info(self) -> dict:
        """Get information about current quantum state."""
        return {
            'amplitudes': np.abs(self.amplitudes),
            'phases': np.angle(self.amplitudes),
            'entropy': self.quantum_entropy(),
            'coherence': np.sum(np.abs(self.amplitudes))**2 / self.size,
            'num_patterns_learned': len(self.interference_patterns)
        }
    
    def _visualize_learning(self, initial_state: np.ndarray, 
                          pattern: np.ndarray, label: int,
                          pred_before: int, pred_after: int, 
                          correct_before: bool):
        """Visualize the quantum learning process."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot 1: Amplitude comparison
        ax = axes[0, 0]
        x = np.arange(self.size)
        width = 0.35
        ax.bar(x - width/2, np.abs(initial_state), width, 
               label='Before', alpha=0.7)
        ax.bar(x + width/2, np.abs(self.amplitudes), width, 
               label='After', alpha=0.7)
        ax.set_xlabel('Dimension')
        ax.set_ylabel('Amplitude')
        ax.set_title('Quantum State Amplitudes')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Phase change
        ax = axes[0, 1]
        phase_change = np.angle(self.amplitudes) - np.angle(initial_state)
        ax.bar(x, phase_change, color='purple', alpha=0.7)
        ax.set_xlabel('Dimension')
        ax.set_ylabel('Phase Change (radians)')
        ax.set_title('Quantum Phase Rotation')
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Pattern visualization
        ax = axes[1, 0]
        colors = ['green' if p > 0 else 'red' for p in pattern]
        ax.bar(x, pattern, color=colors, alpha=0.7)
        ax.set_xlabel('Dimension')
        ax.set_ylabel('Value')
        ax.set_title(f'Pattern (Label: {label})')
        ax.set_ylim(-1.5, 1.5)
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Learning outcome
        ax = axes[1, 1]
        outcomes = ['Before', 'After']
        predictions = [pred_before, pred_after]
        correct = [1 if correct_before else 0, 1]
        
        x_pos = np.arange(len(outcomes))
        ax.bar(x_pos - 0.2, predictions, 0.4, label='Prediction', alpha=0.7)
        ax.bar(x_pos + 0.2, correct, 0.4, label='Correct', alpha=0.7)
        ax.axhline(y=label, color='g', linestyle='--', label='True Label')
        
        ax.set_xlabel('Learning Stage')
        ax.set_ylabel('Value')
        ax.set_title(f'Learning Outcome: {"Success" if pred_after == label else "Learning"}')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(outcomes)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.suptitle('Quantum Collapse Learning Visualization', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    def run_experiment(self, patterns: List[np.ndarray], 
                      labels: List[int]) -> dict:
        """
        Run a complete learning experiment.
        
        Returns:
        --------
        dict : Experiment results and metrics
        """
        print("🔬 Running Quantum Learning Experiment")
        print("=" * 50)
        
        results = {
            'total_patterns': len(patterns),
            'correct_after_learning': 0,
            'collapses_needed': 0,
            'accuracy_history': [],
            'entropy_history': []
        }
        
        for i, (pattern, label) in enumerate(zip(patterns, labels)):
            print(f"\nPattern {i+1}: {pattern} → Label: {label}")
            
            # Make initial prediction
            pred_initial = self.predict(pattern)
            print(f"  Initial prediction: {pred_initial}")
            
            # Learn from pattern
            success = self.collapse_learn(pattern, label)
            
            if success:
                results['correct_after_learning'] += 1
                if pred_initial != label:
                    results['collapses_needed'] += 1
                    print(f"  ✅ Learned via quantum collapse")
                else:
                    print(f"  ✅ Already correct (quantum interference)")
            else:
                print(f"  ❌ Still learning...")
            
            # Record metrics
            results['accuracy_history'].append(
                results['correct_after_learning'] / (i + 1)
            )
            results['entropy_history'].append(self.quantum_entropy())
            
            print(f"  Current entropy: {self.quantum_entropy():.3f}")
        
        # Calculate final metrics
        results['final_accuracy'] = results['correct_after_learning'] / len(patterns)
        results['collapse_efficiency'] = (
            results['correct_after_learning'] / 
            max(1, results['collapses_needed'])
        )
        results['final_entropy'] = self.quantum_entropy()
        
        print("\n" + "=" * 50)
        print("EXPERIMENT RESULTS:")
        print(f"  Patterns: {results['total_patterns']}")
        print(f"  Correct after learning: {results['correct_after_learning']}")
        print(f"  Final accuracy: {results['final_accuracy']:.1%}")
        print(f"  Collapses needed: {results['collapses_needed']}")
        print(f"  Collapse efficiency: {results['collapse_efficiency']:.2f}")
        print(f"  Final quantum entropy: {results['final_entropy']:.3f}")
        
        return results