#!/usr/bin/env python3
"""
Advanced Quantum Collapse Learner
Author: Luke Spookwalker
Date: 2025
License: MIT
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft
from dataclasses import dataclass
from enum import Enum

class CollapseMode(Enum):
    """Modes of quantum collapse learning."""
    GRADUAL = "gradual"      # Small adjustments
    SUDDEN = "sudden"        # Complete reconfiguration
    RESONANT = "resonant"    # Match existing patterns
    CREATIVE = "creative"    # Explore new solutions

@dataclass
class QuantumMetrics:
    """Container for quantum performance metrics."""
    entropy: float
    coherence: float
    phase_diversity: float
    interference_strength: float
    collapse_count: int

class QuantumCollapseLearner:
    """
    Advanced quantum collapse learning system with multiple collapse modes,
    quantum memory, and sophisticated interference patterns.
    """
    
    def __init__(self, num_qubits: int = 6, 
                 collapse_strength: float = 0.7,
                 interference_depth: int = 3,
                 mode: CollapseMode = CollapseMode.RESONANT):
        """
        Initialize advanced quantum learner.
        
        Parameters:
        -----------
        num_qubits : int
            Number of simulated qubits (2^num_qubits states)
        collapse_strength : float
            How strongly patterns cause quantum collapse
        interference_depth : int
            How many past patterns influence interference
        mode : CollapseMode
            Type of collapse learning to use
        """
        self.num_qubits = num_qubits
        self.num_states = 2 ** num_qubits
        self.collapse_strength = collapse_strength
        self.interference_depth = interference_depth
        self.mode = mode
        
        # Initialize quantum state in uniform superposition
        self.state_vector = np.ones(self.num_states, dtype=complex)
        self.state_vector /= np.linalg.norm(self.state_vector)
        
        # Add random phases for diversity
        random_phases = np.random.uniform(0, 2*np.pi, self.num_states)
        self.state_vector *= np.exp(1j * random_phases)
        
        # Quantum memory systems
        self.pattern_memory = []           # Stored patterns
        self.interference_field = None      # Cumulative interference
        self.collapse_history = []          # History of collapses
        
        # Performance tracking
        self.metrics = QuantumMetrics(
            entropy=self._calculate_entropy(),
            coherence=self._calculate_coherence(),
            phase_diversity=self._calculate_phase_diversity(),
            interference_strength=0.0,
            collapse_count=0
        )
        
        print(f"🌌 Advanced Quantum Collapse Learner initialized")
        print(f"   Qubits: {num_qubits} → {self.num_states} states")
        print(f"   Mode: {mode.value}, Strength: {collapse_strength}")
        print(f"   Interference depth: {interference_depth}")
    
    def encode_pattern(self, pattern: np.ndarray) -> int:
        """Encode binary pattern to quantum state index."""
        # Convert pattern to binary string
        binary = ''.join(['1' if x > 0 else '0' for x in pattern[:self.num_qubits]])
        return int(binary, 2)
    
    def collapse_learn(self, pattern: np.ndarray, label: int, 
                      learning_rate: float = 0.5) -> Dict:
        """
        Advanced quantum collapse learning with multiple modes.
        
        Returns:
        --------
        Dict with learning results and metrics
        """
        initial_state = self.state_vector.copy()
        pattern_idx = self.encode_pattern(pattern)
        
        # Make prediction before learning
        pred_before = self.predict(pattern)
        quantum_confidence = self._calculate_confidence(pattern)
        
        learning_result = {
            'pattern_idx': pattern_idx,
            'label': label,
            'prediction_before': pred_before,
            'confidence_before': quantum_confidence,
            'correct_before': pred_before == label,
            'collapse_applied': False,
            'collapse_strength': 0.0,
            'mode_used': None
        }
        
        if pred_before != label:
            # Apply quantum collapse based on selected mode
            self._apply_collapse(pattern, label, pattern_idx, learning_rate)
            learning_result['collapse_applied'] = True
            learning_result['collapse_strength'] = self.collapse_strength
            learning_result['mode_used'] = self.mode.value
            
            self.collapse_history.append({
                'pattern': pattern.copy(),
                'label': label,
                'mode': self.mode.value,
                'state_change': np.linalg.norm(self.state_vector - initial_state)
            })
            self.metrics.collapse_count += 1
        
        # Update quantum memory
        self._update_memory(pattern, label)
        
        # Apply quantum interference
        self._apply_advanced_interference()
        
        # Update metrics
        self._update_metrics()
        
        # Get final prediction
        pred_after = self.predict(pattern)
        learning_result.update({
            'prediction_after': pred_after,
            'correct_after': pred_after == label,
            'state_entropy': self.metrics.entropy,
            'quantum_coherence': self.metrics.coherence,
            'final_confidence': self._calculate_confidence(pattern)
        })
        
        return learning_result
    
    def _apply_collapse(self, pattern: np.ndarray, label: int, 
                       pattern_idx: int, learning_rate: float):
        """Apply collapse based on selected mode."""
        if self.mode == CollapseMode.GRADUAL:
            self._gradual_collapse(pattern, label, learning_rate)
        elif self.mode == CollapseMode.SUDDEN:
            self._sudden_collapse(pattern_idx, label)
        elif self.mode == CollapseMode.RESONANT:
            self._resonant_collapse(pattern, label)
        elif self.mode == CollapseMode.CREATIVE:
            self._creative_collapse(pattern, label)
    
    def _gradual_collapse(self, pattern: np.ndarray, label: int, 
                         learning_rate: float):
        """Gradual quantum state adjustment."""
        for i in range(self.num_qubits):
            if i < len(pattern) and pattern[i] != 0:
                # Create rotation gate
                angle = learning_rate * self.collapse_strength * label * pattern[i]
                rotation = np.exp(1j * angle)
                
                # Apply to all states where qubit i matters
                for state in range(self.num_states):
                    binary = format(state, f'0{self.num_qubits}b')
                    if binary[i] == '1':
                        self.state_vector[state] *= rotation
    
    def _sudden_collapse(self, pattern_idx: int, label: int):
        """Sudden collapse to target state."""
        # Amplify target state probability
        probabilities = np.abs(self.state_vector)**2
        
        if label > 0:
            # Amplify target state
            probabilities[pattern_idx] *= (1 + self.collapse_strength)
        else:
            # Suppress target state
            probabilities[pattern_idx] *= (1 - self.collapse_strength)
        
        # Renormalize and apply
        probabilities = np.maximum(probabilities, 0)
        probabilities /= probabilities.sum()
        self.state_vector = np.sqrt(probabilities) * np.exp(1j * np.angle(self.state_vector))
    
    def _resonant_collapse(self, pattern: np.ndarray, label: int):
        """Resonant collapse matching existing patterns."""
        if not self.pattern_memory:
            self._gradual_collapse(pattern, label, 0.5)
            return
        
        # Find most similar stored pattern
        similarities = []
        for memory in self.pattern_memory[-self.interference_depth:]:
            sim = self._pattern_similarity(pattern, memory['pattern'])
            similarities.append((sim, memory))
        
        if similarities:
            max_sim, best_match = max(similarities, key=lambda x: x[0])
            
            if max_sim > 0.5:  # Significant similarity
                # Resonate with similar pattern
                resonance_strength = self.collapse_strength * max_sim
                
                # Adjust state toward resonance
                target_idx = self.encode_pattern(best_match['pattern'])
                current_amp = self.state_vector[target_idx]
                
                if label == best_match['label']:
                    # Constructive resonance
                    new_amp = current_amp * (1 + resonance_strength)
                else:
                    # Destructive resonance
                    new_amp = current_amp * (1 - resonance_strength)
                
                self.state_vector[target_idx] = new_amp
                self._normalize_state()
    
    def _creative_collapse(self, pattern: np.ndarray, label: int):
        """Creative collapse exploring new solutions."""
        # Add quantum noise for exploration
        noise = np.random.normal(0, 0.1, self.state_vector.shape) + \
                1j * np.random.normal(0, 0.1, self.state_vector.shape)
        
        # Scale noise by collapse strength
        exploration = self.collapse_strength * 0.3
        self.state_vector += exploration * noise
        self._normalize_state()
        
        # Then apply slight correction toward target
        self._gradual_collapse(pattern, label, 0.2)
    
    def _update_memory(self, pattern: np.ndarray, label: int):
        """Update quantum memory with new pattern."""
        memory_entry = {
            'pattern': pattern.copy(),
            'label': label,
            'quantum_state': self.state_vector.copy(),
            'entropy': self.metrics.entropy
        }
        
        self.pattern_memory.append(memory_entry)
        
        # Keep only recent memories (FIFO)
        if len(self.pattern_memory) > self.interference_depth * 2:
            self.pattern_memory.pop(0)
    
    def _apply_advanced_interference(self):
        """Apply sophisticated quantum interference."""
        if len(self.pattern_memory) < 2:
            return
        
        # Build interference field from memories
        interference_field = np.zeros(self.num_states, dtype=complex)
        
        for memory in self.pattern_memory[-self.interference_depth:]:
            # Weight by recency and confidence
            weight = 1.0 / (len(self.pattern_memory) - self.pattern_memory.index(memory))
            interference_field += weight * memory['quantum_state']
        
        # Normalize interference field
        norm = np.linalg.norm(interference_field)
        if norm > 0:
            interference_field /= norm
        
        # Apply interference to current state
        interference_strength = 0.3  # How much interference affects state
        self.state_vector = (1 - interference_strength) * self.state_vector + \
                           interference_strength * interference_field
        
        self._normalize_state()
        self.metrics.interference_strength = interference_strength
    
    def predict(self, pattern: np.ndarray) -> int:
        """Make quantum prediction with confidence estimation."""
        pattern_idx = self.encode_pattern(pattern)
        
        # Direct state amplitude
        direct_confidence = np.abs(self.state_vector[pattern_idx])**2
        
        # Quantum tunneling to similar states
        similar_states = self._find_similar_states(pattern_idx, radius=2)
        tunneling_confidence = 0.0
        if similar_states:
            for state_idx in similar_states:
                tunneling_confidence += np.abs(self.state_vector[state_idx])**2
            tunneling_confidence /= len(similar_states)
        
        # Memory-based prediction
        memory_confidence = 0.0
        if self.pattern_memory:
            for memory in self.pattern_memory[-3:]:
                if self._pattern_similarity(pattern, memory['pattern']) > 0.6:
                    memory_confidence += 1 if memory['label'] > 0 else -1
            memory_confidence = max(0, memory_confidence / 3)
        
        # Combined quantum confidence
        total_confidence = (
            0.5 * direct_confidence +
            0.3 * tunneling_confidence +
            0.2 * memory_confidence
        )
        
        return 1 if total_confidence > 0.25 else -1
    
    def _calculate_confidence(self, pattern: np.ndarray) -> float:
        """Calculate quantum confidence score for prediction."""
        pred = self.predict(pattern)
        pattern_idx = self.encode_pattern(pattern)
        
        direct = np.abs(self.state_vector[pattern_idx])**2
        
        # Find similar states for tunneling
        similar = self._find_similar_states(pattern_idx, radius=1)
        tunneling = np.mean([np.abs(self.state_vector[i])**2 for i in similar]) if similar else 0
        
        return 0.7 * direct + 0.3 * tunneling
    
    def _find_similar_states(self, state_idx: int, radius: int = 1) -> List[int]:
        """Find quantum states within Hamming distance radius."""
        similar = []
        target_binary = format(state_idx, f'0{self.num_qubits}b')
        
        for i in range(self.num_states):
            binary = format(i, f'0{self.num_qubits}b')
            hamming_dist = sum(1 for a, b in zip(target_binary, binary) if a != b)
            if hamming_dist <= radius:
                similar.append(i)
        
        return similar
    
    def _pattern_similarity(self, pattern1: np.ndarray, pattern2: np.ndarray) -> float:
        """Calculate similarity between two patterns."""
        if len(pattern1) != len(pattern2):
            return 0.0
        matches = sum(1 for p1, p2 in zip(pattern1, pattern2) if p1 == p2)
        return matches / len(pattern1)
    
    def _normalize_state(self):
        """Normalize quantum state vector."""
        norm = np.linalg.norm(self.state_vector)
        if norm > 0:
            self.state_vector /= norm
    
    def _calculate_entropy(self) -> float:
        """Calculate quantum entropy."""
        probabilities = np.abs(self.state_vector)**2
        probabilities = probabilities[probabilities > 0]
        probabilities = probabilities / probabilities.sum()
        return -np.sum(probabilities * np.log(probabilities))
    
    def _calculate_coherence(self) -> float:
        """Calculate quantum coherence."""
        return np.sum(np.abs(self.state_vector))**2 / self.num_states
    
    def _calculate_phase_diversity(self) -> float:
        """Calculate phase diversity metric."""
        phases = np.angle(self.state_vector)
        unique_phases = len(set(np.round(phases, 2)))
        return unique_phases / self.num_states
    
    def _update_metrics(self):
        """Update all quantum metrics."""
        self.metrics.entropy = self._calculate_entropy()
        self.metrics.coherence = self._calculate_coherence()
        self.metrics.phase_diversity = self._calculate_phase_diversity()
    
    def get_detailed_metrics(self) -> Dict:
        """Get comprehensive quantum metrics."""
        return {
            'entropy': self.metrics.entropy,
            'coherence': self.metrics.coherence,
            'phase_diversity': self.metrics.phase_diversity,
            'interference_strength': self.metrics.interference_strength,
            'collapse_count': self.metrics.collapse_count,
            'memory_size': len(self.pattern_memory),
            'state_norm': np.linalg.norm(self.state_vector),
            'max_amplitude': np.max(np.abs(self.state_vector)),
            'min_amplitude': np.min(np.abs(self.state_vector))
        }
    
    def visualize_quantum_state(self, max_states: int = 16):
        """Visualize the quantum state."""
        if self.num_states > max_states:
            indices = np.argsort(np.abs(self.state_vector))[-max_states:]
            states = [self.state_vector[i] for i in indices]
            labels = [format(i, f'0{self.num_qubits}b') for i in indices]
        else:
            states = self.state_vector
            labels = [format(i, f'0{self.num_qubits}b') for i in range(self.num_states)]
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot amplitudes
        ax = axes[0]
        amplitudes = np.abs(states)
        x_pos = np.arange(len(amplitudes))
        ax.bar(x_pos, amplitudes, alpha=0.7, color='blue')
        ax.set_xlabel('Quantum State')
        ax.set_ylabel('Amplitude')
        ax.set_title('Quantum State Amplitudes')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
        
        # Plot phases on unit circle
        ax = axes[1]
        phases = np.angle(states)
        amplitudes = np.abs(states)
        
        # Draw unit circle
        theta = np.linspace(0, 2*np.pi, 100)
        ax.plot(np.cos(theta), np.sin(theta), 'k-', alpha=0.3)
        ax.axhline(y=0, color='k', alpha=0.3, linestyle='--')
        ax.axvline(x=0, color='k', alpha=0.3, linestyle='--')
        
        # Plot each state
        for i, (amp, phase) in enumerate(zip(amplitudes, phases)):
            x = amp * np.cos(phase)
            y = amp * np.sin(phase)
            
            # Color by amplitude
            color = plt.cm.viridis(amp / np.max(amplitudes))
            
            ax.plot([0, x], [0, y], color=color, alpha=0.6, linewidth=2)
            ax.scatter(x, y, color=color, s=100*amp, alpha=0.8)
            
            # Add state label
            if amp > 0.1:  # Only label significant states
                ax.annotate(labels[i], (x, y), xytext=(5, 5),
                          textcoords='offset points', fontsize=8)
        
        ax.set_xlabel('Real Part')
        ax.set_ylabel('Imaginary Part')
        ax.set_title('Quantum Phases on Complex Plane')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        plt.suptitle(f'Quantum State Visualization ({self.num_qubits} qubits)', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()