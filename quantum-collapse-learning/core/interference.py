#!/usr/bin/env python3
"""
Quantum Interference System
Author: Luke Spookwalker
Date: 2025
License: MIT

Advanced quantum interference patterns for learning acceleration.
Implements constructive/destructive interference, quantum beating,
and interference-based memory recall.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Union
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftfreq
from scipy.signal import convolve
from dataclasses import dataclass
from enum import Enum
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning)

class InterferenceType(Enum):
    """Types of quantum interference."""
    CONSTRUCTIVE = "constructive"      # Waves add up
    DESTRUCTIVE = "destructive"        # Waves cancel
    BEATING = "beating"                # Quantum beats
    STATIONARY = "stationary"          # Standing waves
    ENTANGLED = "entangled"            # Non-local interference

@dataclass
class InterferencePattern:
    """Container for quantum interference pattern."""
    pattern: np.ndarray
    amplitude: float
    phase: np.ndarray
    frequency: float
    type: InterferenceType
    created_at: float
    strength: float = 1.0

class QuantumInterference:
    """
    Advanced quantum interference system for learning acceleration.
    
    Key features:
    1. Constructive interference amplifies correct patterns
    2. Destructive interference suppresses incorrect patterns  
    3. Quantum beating creates rhythmic learning patterns
    4. Stationary waves create stable memory formations
    5. Entangled interference for non-local pattern correlation
    """
    
    def __init__(self, 
                 num_dimensions: int = 8,
                 interference_strength: float = 0.5,
                 memory_capacity: int = 10,
                 beat_frequency: float = 0.1):
        """
        Initialize quantum interference system.
        
        Parameters:
        -----------
        num_dimensions : int
            Number of quantum dimensions
        interference_strength : float
            How strongly patterns interfere (0-1)
        memory_capacity : int
            Number of patterns to remember for interference
        beat_frequency : float
            Frequency of quantum beats for rhythmic learning
        """
        self.num_dimensions = num_dimensions
        self.interference_strength = interference_strength
        self.memory_capacity = memory_capacity
        self.beat_frequency = beat_frequency
        
        # Interference pattern memory
        self.pattern_memory: List[InterferencePattern] = []
        self.interference_field = np.zeros(num_dimensions, dtype=complex)
        
        # Quantum beat tracking
        self.beat_phase = 0.0
        self.beat_history = []
        
        # Performance metrics
        self.interference_count = 0
        self.constructive_count = 0
        self.destructive_count = 0
        
        print(f"🌀 Quantum Interference System initialized")
        print(f"   Dimensions: {num_dimensions}")
        print(f"   Strength: {interference_strength}")
        print(f"   Memory: {memory_capacity} patterns")
        print(f"   Beat frequency: {beat_frequency}")
    
    def create_interference(self, 
                          pattern: np.ndarray,
                          label: int,
                          current_state: np.ndarray) -> np.ndarray:
        """
        Create quantum interference between pattern and current state.
        
        Parameters:
        -----------
        pattern : np.ndarray
            Input pattern (+1/-1)
        label : int
            Pattern label (+1/-1)
        current_state : np.ndarray
            Current quantum state
        
        Returns:
        --------
        np.ndarray: Interfered state
        """
        # Store pattern for future interference
        self._store_pattern(pattern, label, current_state)
        
        # Update interference field
        self._update_interference_field()
        
        # Apply interference based on type
        interfered_state = self._apply_interference(current_state, pattern, label)
        
        # Apply quantum beats if enabled
        if self.beat_frequency > 0:
            interfered_state = self._apply_quantum_beats(interfered_state)
        
        # Update metrics
        self.interference_count += 1
        
        return interfered_state
    
    def _store_pattern(self, 
                      pattern: np.ndarray, 
                      label: int,
                      state: np.ndarray):
        """Store pattern in quantum memory for interference."""
        # Calculate pattern properties
        amplitude = np.abs(np.vdot(state, pattern))
        phase = np.angle(np.vdot(state, pattern))
        
        # Determine interference type
        if label > 0:
            interference_type = InterferenceType.CONSTRUCTIVE
            self.constructive_count += 1
        else:
            interference_type = InterferenceType.DESTRUCTIVE
            self.destructive_count += 1
        
        # Create interference pattern
        interference_pattern = InterferencePattern(
            pattern=pattern.copy(),
            amplitude=amplitude,
            phase=phase * np.ones(self.num_dimensions),
            frequency=self._calculate_frequency(pattern),
            type=interference_type,
            created_at=self.interference_count,
            strength=self.interference_strength
        )
        
        # Add to memory
        self.pattern_memory.append(interference_pattern)
        
        # Maintain memory capacity
        if len(self.pattern_memory) > self.memory_capacity:
            self.pattern_memory.pop(0)
    
    def _update_interference_field(self):
        """Update the cumulative interference field."""
        if not self.pattern_memory:
            self.interference_field = np.zeros(self.num_dimensions, dtype=complex)
            return
        
        # Reset field
        self.interference_field = np.zeros(self.num_dimensions, dtype=complex)
        
        # Sum all patterns with recency weighting
        for i, pattern in enumerate(self.pattern_memory):
            # Recent patterns have higher weight
            recency_weight = 1.0 / (len(self.pattern_memory) - i + 1)
            
            if pattern.type == InterferenceType.CONSTRUCTIVE:
                # Add constructively
                contribution = pattern.pattern.astype(complex) * \
                             pattern.amplitude * \
                             np.exp(1j * pattern.phase) * \
                             recency_weight
                self.interference_field += contribution
            else:
                # Subtract destructively
                contribution = pattern.pattern.astype(complex) * \
                             pattern.amplitude * \
                             np.exp(1j * pattern.phase) * \
                             recency_weight
                self.interference_field -= contribution
    
    def _apply_interference(self, 
                          state: np.ndarray,
                          pattern: np.ndarray,
                          label: int) -> np.ndarray:
        """
        Apply interference to quantum state.
        
        Returns:
        --------
        np.ndarray: Interfered state
        """
        if not self.pattern_memory:
            return state
        
        # Create interference from memory
        memory_interference = self._create_memory_interference(state, pattern, label)
        
        # Create frequency domain interference
        frequency_interference = self._create_frequency_interference(state)
        
        # Combine interferences
        total_interference = (
            self.interference_strength * memory_interference +
            (1 - self.interference_strength) * frequency_interference
        )
        
        # Apply to state
        interfered_state = state + total_interference
        
        # Normalize
        norm = np.linalg.norm(interfered_state)
        if norm > 0:
            interfered_state /= norm
        
        return interfered_state
    
    def _create_memory_interference(self,
                                  state: np.ndarray,
                                  current_pattern: np.ndarray,
                                  current_label: int) -> np.ndarray:
        """Create interference based on stored memories."""
        interference = np.zeros_like(state, dtype=complex)
        
        for stored_pattern in self.pattern_memory:
            # Calculate similarity
            similarity = self._pattern_similarity(
                current_pattern, 
                stored_pattern.pattern
            )
            
            if similarity > 0.3:  # Significant similarity
                if (current_label > 0 and stored_pattern.type == InterferenceType.CONSTRUCTIVE) or \
                   (current_label < 0 and stored_pattern.type == InterferenceType.DESTRUCTIVE):
                    # Constructive interference
                    interference += similarity * stored_pattern.strength * \
                                   stored_pattern.pattern.astype(complex) * \
                                   np.exp(1j * stored_pattern.phase)
                else:
                    # Destructive interference  
                    interference -= similarity * stored_pattern.strength * \
                                   stored_pattern.pattern.astype(complex) * \
                                   np.exp(1j * stored_pattern.phase)
        
        return interference
    
    def _create_frequency_interference(self, state: np.ndarray) -> np.ndarray:
        """Create frequency-domain interference patterns."""
        # Fourier transform
        state_fft = fft(state)
        frequencies = fftfreq(len(state))
        
        # Find dominant frequencies in memory
        dominant_freqs = []
        for pattern in self.pattern_memory[-3:]:  # Recent patterns
            pattern_fft = fft(pattern.pattern.astype(complex))
            dominant_idx = np.argmax(np.abs(pattern_fft))
            dominant_freqs.append(frequencies[dominant_idx])
        
        if dominant_freqs:
            # Create filter that amplifies dominant frequencies
            filter_fft = np.zeros_like(state_fft)
            for freq in dominant_freqs:
                idx = np.argmin(np.abs(frequencies - freq))
                filter_fft[idx] = 1.0
            
            # Apply filter
            filtered_fft = state_fft * filter_fft
            
            # Inverse transform
            interference = ifft(filtered_fft)
            
            # Normalize
            norm = np.linalg.norm(interference)
            if norm > 0:
                interference /= norm
            
            return interference
        
        return np.zeros_like(state, dtype=complex)
    
    def _apply_quantum_beats(self, state: np.ndarray) -> np.ndarray:
        """Apply quantum beats (rhythmic interference)."""
        # Update beat phase
        self.beat_phase += 2 * np.pi * self.beat_frequency
        self.beat_phase %= 2 * np.pi
        
        # Record beat
        self.beat_history.append(self.beat_phase)
        if len(self.beat_history) > 100:
            self.beat_history.pop(0)
        
        # Create beat modulation
        beat_modulation = np.cos(self.beat_phase) + 1j * np.sin(self.beat_phase)
        
        # Apply beat to state
        beat_state = state * (1 + 0.1 * beat_modulation)
        
        # Normalize
        norm = np.linalg.norm(beat_state)
        if norm > 0:
            beat_state /= norm
        
        return beat_state
    
    def _calculate_frequency(self, pattern: np.ndarray) -> float:
        """Calculate frequency signature of pattern."""
        # Use Fourier transform to find dominant frequency
        pattern_fft = fft(pattern.astype(complex))
        frequencies = fftfreq(len(pattern))
        
        dominant_idx = np.argmax(np.abs(pattern_fft))
        return np.abs(frequencies[dominant_idx])
    
    def _pattern_similarity(self, pattern1: np.ndarray, pattern2: np.ndarray) -> float:
        """Calculate similarity between patterns."""
        if len(pattern1) != len(pattern2):
            return 0.0
        
        # Cosine similarity for +1/-1 patterns
        dot_product = np.dot(pattern1, pattern2)
        norm1 = np.linalg.norm(pattern1)
        norm2 = np.linalg.norm(pattern2)
        
        if norm1 > 0 and norm2 > 0:
            similarity = dot_product / (norm1 * norm2)
            # Convert from [-1, 1] to [0, 1]
            return (similarity + 1) / 2
        
        return 0.0
    
    def clear_memory(self):
        """Clear interference memory."""
        self.pattern_memory = []
        self.interference_field = np.zeros(self.num_dimensions, dtype=complex)
        self.beat_history = []
        print("🧹 Interference memory cleared")
    
    def get_interference_metrics(self) -> Dict:
        """Get interference performance metrics."""
        return {
            'total_interferences': self.interference_count,
            'constructive_count': self.constructive_count,
            'destructive_count': self.destructive_count,
            'memory_size': len(self.pattern_memory),
            'constructive_ratio': (
                self.constructive_count / max(1, self.interference_count)
            ),
            'beat_phase': self.beat_phase,
            'field_strength': np.linalg.norm(self.interference_field),
            'pattern_diversity': self._calculate_pattern_diversity()
        }
    
    def _calculate_pattern_diversity(self) -> float:
        """Calculate diversity of stored patterns."""
        if len(self.pattern_memory) < 2:
            return 1.0  # Maximum diversity with single pattern
        
        similarities = []
        for i in range(len(self.pattern_memory)):
            for j in range(i + 1, len(self.pattern_memory)):
                sim = self._pattern_similarity(
                    self.pattern_memory[i].pattern,
                    self.pattern_memory[j].pattern
                )
                similarities.append(sim)
        
        if similarities:
            avg_similarity = np.mean(similarities)
            return 1.0 - avg_similarity  # Diversity = 1 - similarity
        
        return 1.0
    
    def visualize_interference(self, current_state: np.ndarray):
        """Visualize quantum interference patterns."""
        if not self.pattern_memory:
            print("No interference patterns to visualize")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Plot 1: Interference field
        ax = axes[0, 0]
        field_real = np.real(self.interference_field)
        field_imag = np.imag(self.interference_field)
        
        x = np.arange(self.num_dimensions)
        width = 0.35
        ax.bar(x - width/2, field_real, width, label='Real', alpha=0.7)
        ax.bar(x + width/2, field_imag, width, label='Imaginary', alpha=0.7)
        
        ax.set_xlabel('Dimension')
        ax.set_ylabel('Field Strength')
        ax.set_title('Interference Field')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Pattern similarity matrix
        ax = axes[0, 1]
        n_patterns = min(8, len(self.pattern_memory))
        
        if n_patterns > 1:
            similarity_matrix = np.zeros((n_patterns, n_patterns))
            for i in range(n_patterns):
                for j in range(n_patterns):
                    similarity_matrix[i, j] = self._pattern_similarity(
                        self.pattern_memory[-n_patterns+i].pattern,
                        self.pattern_memory[-n_patterns+j].pattern
                    )
            
            im = ax.imshow(similarity_matrix, cmap='viridis', vmin=0, vmax=1)
            ax.set_xlabel('Pattern Index')
            ax.set_ylabel('Pattern Index')
            ax.set_title('Pattern Similarity Matrix')
            plt.colorbar(im, ax=ax)
        else:
            ax.text(0.5, 0.5, 'Need at least 2 patterns', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Pattern Similarity')
        
        # Plot 3: Current state vs interference
        ax = axes[0, 2]
        state_amplitude = np.abs(current_state)
        state_phase = np.angle(current_state)
        
        # Show amplitude and phase
        ax.plot(state_amplitude, 'b-', label='Amplitude', linewidth=2)
        ax2 = ax.twinx()
        ax2.plot(state_phase, 'r--', label='Phase', alpha=0.7)
        
        ax.set_xlabel('Dimension')
        ax.set_ylabel('Amplitude', color='b')
        ax2.set_ylabel('Phase (radians)', color='r')
        ax.set_title('Current Quantum State')
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Beat pattern history
        ax = axes[1, 0]
        if self.beat_history:
            ax.plot(self.beat_history, 'g-', linewidth=2)
            ax.set_xlabel('Time Step')
            ax.set_ylabel('Beat Phase')
            ax.set_title('Quantum Beat Pattern')
            ax.grid(True, alpha=0.3)
        
        # Plot 5: Interference type distribution
        ax = axes[1, 1]
        types = ['Constructive', 'Destructive']
        counts = [self.constructive_count, self.destructive_count]
        
        colors = ['green', 'red']
        bars = ax.bar(types, counts, color=colors, alpha=0.7)
        ax.set_ylabel('Count')
        ax.set_title('Interference Type Distribution')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   f'{count}', ha='center', va='bottom')
        
        # Plot 6: Memory pattern strengths
        ax = axes[1, 2]
        if self.pattern_memory:
            strengths = [p.strength for p in self.pattern_memory]
            ages = [self.interference_count - p.created_at 
                   for p in self.pattern_memory]
            
            colors = ['green' if p.type == InterferenceType.CONSTRUCTIVE else 'red'
                     for p in self.pattern_memory]
            
            ax.scatter(ages, strengths, c=colors, s=100, alpha=0.7)
            ax.set_xlabel('Age (interference steps)')
            ax.set_ylabel('Strength')
            ax.set_title('Pattern Strength vs Age')
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('Quantum Interference System Visualization', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()

    def demonstrate_interference_effects(self):
        """Demonstrate different interference effects."""
        print("\n" + "=" * 70)
        print("🌀 DEMONSTRATING QUANTUM INTERFERENCE EFFECTS")
        print("=" * 70)
        
        # Create test patterns
        pattern1 = np.array([1, 1, 1, 1, -1, -1, -1, -1])
        pattern2 = np.array([1, -1, 1, -1, 1, -1, 1, -1])
        
        # Start with uniform state
        initial_state = np.ones(self.num_dimensions, dtype=complex) / np.sqrt(self.num_dimensions)
        
        print("\n1. Constructive Interference Test:")
        print("   Adding similar patterns should amplify the signal")
        
        # First pattern (constructive)
        state_after1 = self.create_interference(pattern1, 1, initial_state)
        overlap1 = np.abs(np.vdot(state_after1, pattern1))**2
        print(f"   After pattern 1: Overlap = {overlap1:.3f}")
        
        # Second similar pattern (should constructively interfere)
        similar_pattern = pattern1 * 0.9  # Slightly different
        state_after2 = self.create_interference(similar_pattern, 1, state_after1)
        overlap2 = np.abs(np.vdot(state_after2, pattern1))**2
        print(f"   After similar pattern: Overlap = {overlap2:.3f}")
        
        if overlap2 > overlap1:
            print(f"   ✅ CONSTRUCTIVE INTERFERENCE: Overlap increased by {overlap2-overlap1:.3f}")
        else:
            print(f"   ❌ No constructive interference")
        
        print("\n2. Destructive Interference Test:")
        print("   Adding opposite patterns should cancel the signal")
        
        # Clear memory for clean test
        self.clear_memory()
        
        # Start with pattern1 learned
        state = self.create_interference(pattern1, 1, initial_state)
        overlap_before = np.abs(np.vdot(state, pattern1))**2
        print(f"   After learning pattern 1: Overlap = {overlap_before:.3f}")
        
        # Add opposite pattern
        opposite_pattern = -pattern1
        state_after = self.create_interference(opposite_pattern, -1, state)
        overlap_after = np.abs(np.vdot(state_after, pattern1))**2
        print(f"   After opposite pattern: Overlap = {overlap_after:.3f}")
        
        if overlap_after < overlap_before:
            reduction = overlap_before - overlap_after
            print(f"   ✅ DESTRUCTIVE INTERFERENCE: Overlap reduced by {reduction:.3f}")
        else:
            print(f"   ❌ No destructive interference")
        
        print("\n3. Quantum Beats Demonstration:")
        print("   Showing rhythmic interference patterns")
        
        # Track beat effects
        beat_states = []
        current_state = initial_state.copy()
        
        for i in range(20):
            current_state = self._apply_quantum_beats(current_state)
            beat_states.append(np.abs(current_state[0]))  # Track first dimension
        
        print(f"   Beat amplitude varies between {min(beat_states):.3f} and {max(beat_states):.3f}")
        print(f"   Beat frequency: {self.beat_frequency:.3f} Hz")
        
        return {
            'constructive_gain': overlap2 - overlap1 if overlap2 > overlap1 else 0,
            'destructive_reduction': overlap_before - overlap_after if overlap_after < overlap_before else 0,
            'beat_amplitude_variation': max(beat_states) - min(beat_states)
        }