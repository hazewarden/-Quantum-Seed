#!/usr/bin/env python3
"""
Quantum Dynamics: Core quantum operations and utilities
Author: Luke Spookwalker
Date: 2025
License: MIT
"""

import numpy as np
from typing import Tuple, List, Optional
from scipy.linalg import expm

class QuantumDynamics:
    """Core quantum operations for quantum learning."""
    
    @staticmethod
    def hadamard_gate() -> np.ndarray:
        """Create Hadamard gate for superposition."""
        return np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    
    @staticmethod
    def pauli_x() -> np.ndarray:
        """Pauli-X (NOT) gate."""
        return np.array([[0, 1], [1, 0]])
    
    @staticmethod
    def pauli_y() -> np.ndarray:
        """Pauli-Y gate."""
        return np.array([[0, -1j], [1j, 0]])
    
    @staticmethod
    def pauli_z() -> np.ndarray:
        """Pauli-Z gate."""
        return np.array([[1, 0], [0, -1]])
    
    @staticmethod
    def rotation_gate(theta: float, axis: str = 'z') -> np.ndarray:
        """
        Create rotation gate around specified axis.
        
        Parameters:
        -----------
        theta : float
            Rotation angle in radians
        axis : str
            Rotation axis ('x', 'y', or 'z')
        
        Returns:
        --------
        np.ndarray: 2x2 rotation matrix
        """
        if axis == 'x':
            return np.array([
                [np.cos(theta/2), -1j*np.sin(theta/2)],
                [-1j*np.sin(theta/2), np.cos(theta/2)]
            ])
        elif axis == 'y':
            return np.array([
                [np.cos(theta/2), -np.sin(theta/2)],
                [np.sin(theta/2), np.cos(theta/2)]
            ])
        elif axis == 'z':
            return np.array([
                [np.exp(-1j*theta/2), 0],
                [0, np.exp(1j*theta/2)]
            ])
        else:
            raise ValueError(f"Unknown axis: {axis}")
    
    @staticmethod
    def cnot_gate() -> np.ndarray:
        """Create CNOT (controlled-NOT) gate."""
        return np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0]
        ])
    
    @staticmethod
    def create_superposition(num_qubits: int) -> np.ndarray:
        """
        Create uniform superposition state.
        
        Parameters:
        -----------
        num_qubits : int
            Number of qubits
        
        Returns:
        --------
        np.ndarray: State vector in uniform superposition
        """
        num_states = 2 ** num_qubits
        state = np.ones(num_states, dtype=complex) / np.sqrt(num_states)
        
        # Add random phases for diversity
        phases = np.random.uniform(0, 2*np.pi, num_states)
        state *= np.exp(1j * phases)
        
        return state
    
    @staticmethod
    def apply_quantum_force(state: np.ndarray, force_vector: np.ndarray, 
                           strength: float = 1.0) -> np.ndarray:
        """
        Apply quantum force to state (phase rotation).
        
        Parameters:
        -----------
        state : np.ndarray
            Quantum state vector
        force_vector : np.ndarray
            Force to apply (same shape as state)
        strength : float
            Strength of force application
        
        Returns:
        --------
        np.ndarray: Updated state vector
        """
        # Apply phase rotation: exp(i * strength * force)
        phase_shift = np.exp(1j * strength * force_vector)
        new_state = state * phase_shift
        
        # Normalize
        norm = np.linalg.norm(new_state)
        if norm > 0:
            new_state /= norm
        
        return new_state
    
    @staticmethod
    def quantum_interference(state: np.ndarray, 
                            interference_pattern: np.ndarray,
                            strength: float = 0.5) -> np.ndarray:
        """
        Apply quantum interference to state.
        
        Parameters:
        -----------
        state : np.ndarray
            Current quantum state
        interference_pattern : np.ndarray
            Pattern to interfere with
        strength : float
            Interference strength (0-1)
        
        Returns:
        --------
        np.ndarray: Interfered state
        """
        # Fourier domain interference
        state_fft = np.fft.fft(state)
        pattern_fft = np.fft.fft(interference_pattern)
        
        # Blend in Fourier domain
        magnitude = np.abs(state_fft)
        phase = np.angle(state_fft)
        
        # Adjust phase toward interference pattern
        pattern_phase = np.angle(pattern_fft)
        phase_difference = pattern_phase - phase
        
        # Apply interference
        new_phase = phase + strength * phase_difference
        new_state_fft = magnitude * np.exp(1j * new_phase)
        
        # Transform back
        new_state = np.fft.ifft(new_state_fft)
        
        # Normalize
        norm = np.linalg.norm(new_state)
        if norm > 0:
            new_state /= norm
        
        return new_state
    
    @staticmethod
    def partial_collapse(state: np.ndarray, target_indices: List[int],
                        amplification: float = 1.5) -> np.ndarray:
        """
        Partially collapse quantum state toward target states.
        
        Parameters:
        -----------
        state : np.ndarray
            Current quantum state
        target_indices : List[int]
            Indices of states to amplify
        amplification : float
            How much to amplify target states
        
        Returns:
        --------
        np.ndarray: Partially collapsed state
        """
        probabilities = np.abs(state)**2
        
        # Amplify target states
        for idx in target_indices:
            if 0 <= idx < len(probabilities):
                probabilities[idx] *= amplification
        
        # Renormalize
        probabilities = np.maximum(probabilities, 0)
        probabilities /= probabilities.sum()
        
        # Preserve phases, update amplitudes
        new_state = np.sqrt(probabilities) * np.exp(1j * np.angle(state))
        
        # Final normalization
        norm = np.linalg.norm(new_state)
        if norm > 0:
            new_state /= norm
        
        return new_state
    
    @staticmethod
    def calculate_entanglement(state: np.ndarray, num_qubits: int) -> float:
        """
        Calculate entanglement measure of quantum state.
        
        Parameters:
        -----------
        state : np.ndarray
            Quantum state vector
        num_qubits : int
            Number of qubits
        
        Returns:
        --------
        float: Entanglement measure (0-1)
        """
        if num_qubits == 1:
            return 0.0  # Single qubit can't be entangled
        
        # Reshape to matrix for Schmidt decomposition
        split = num_qubits // 2
        dim1 = 2 ** split
        dim2 = 2 ** (num_qubits - split)
        
        if len(state) != dim1 * dim2:
            # Pad with zeros if needed
            padded_state = np.zeros(dim1 * dim2, dtype=complex)
            padded_state[:len(state)] = state
            state = padded_state
        
        # Reshape to matrix
        matrix = state.reshape(dim1, dim2)
        
        # Singular value decomposition
        U, S, Vh = np.linalg.svd(matrix)
        
        # Entanglement entropy
        S = S[S > 0]
        probabilities = S**2
        probabilities /= probabilities.sum()
        
        entanglement = -np.sum(probabilities * np.log(probabilities))
        
        # Normalize to [0, 1]
        max_entanglement = np.log(min(dim1, dim2))
        if max_entanglement > 0:
            entanglement /= max_entanglement
        
        return entanglement
    
    @staticmethod
    def measure_state(state: np.ndarray, basis: Optional[np.ndarray] = None) -> int:
        """
        Perform quantum measurement on state.
        
        Parameters:
        -----------
        state : np.ndarray
            Quantum state to measure
        basis : np.ndarray, optional
            Measurement basis states
        
        Returns:
        --------
        int: Measurement outcome (state index)
        """
        if basis is None:
            # Standard computational basis measurement
            probabilities = np.abs(state)**2
            outcome = np.random.choice(len(state), p=probabilities)
            return outcome
        
        else:
            # Project onto basis states
            projections = np.abs(np.dot(basis.conj(), state))**2
            probabilities = projections / projections.sum()
            outcome = np.random.choice(len(basis), p=probabilities)
            return outcome
    
    @staticmethod
    def quantum_fidelity(state1: np.ndarray, state2: np.ndarray) -> float:
        """
        Calculate quantum fidelity between two states.
        
        Parameters:
        -----------
        state1, state2 : np.ndarray
            Quantum states to compare
        
        Returns:
        --------
        float: Fidelity measure (0-1)
        """
        # Ensure both are normalized
        norm1 = np.linalg.norm(state1)
        norm2 = np.linalg.norm(state2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        state1_normalized = state1 / norm1
        state2_normalized = state2 / norm2
        
        # Fidelity = |⟨ψ1|ψ2⟩|^2
        overlap = np.abs(np.vdot(state1_normalized, state2_normalized))**2
        return overlap
    
    @staticmethod
    def create_entangled_state(num_qubits: int, 
                              entanglement_type: str = 'bell') -> np.ndarray:
        """
        Create entangled quantum states.
        
        Parameters:
        -----------
        num_qubits : int
            Number of qubits
        entanglement_type : str
            Type of entanglement ('bell', 'ghz', 'w')
        
        Returns:
        --------
        np.ndarray: Entangled state vector
        """
        num_states = 2 ** num_qubits
        
        if entanglement_type == 'bell' and num_qubits == 2:
            # Bell state: (|00⟩ + |11⟩)/√2
            state = np.zeros(num_states, dtype=complex)
            state[0] = 1/np.sqrt(2)  # |00⟩
            state[3] = 1/np.sqrt(2)  # |11⟩
            
        elif entanglement_type == 'ghz':
            # GHZ state: (|0...0⟩ + |1...1⟩)/√2
            state = np.zeros(num_states, dtype=complex)
            state[0] = 1/np.sqrt(2)  # |0...0⟩
            state[-1] = 1/np.sqrt(2)  # |1...1⟩
            
        elif entanglement_type == 'w' and num_qubits == 3:
            # W state: (|001⟩ + |010⟩ + |100⟩)/√3
            state = np.zeros(num_states, dtype=complex)
            state[1] = 1/np.sqrt(3)  # |001⟩
            state[2] = 1/np.sqrt(3)  # |010⟩
            state[4] = 1/np.sqrt(3)  # |100⟩
            
        else:
            # Default: uniform superposition
            state = np.ones(num_states, dtype=complex) / np.sqrt(num_states)
        
        return state
    
    @staticmethod
    def quantum_tomography(state: np.ndarray, num_qubits: int) -> dict:
        """
        Perform basic quantum state tomography.
        
        Parameters:
        -----------
        state : np.ndarray
            Quantum state to analyze
        num_qubits : int
        
        Returns:
        --------
        dict: Tomography results
        """
        probabilities = np.abs(state)**2
        
        return {
            'probabilities': probabilities,
            'phases': np.angle(state),
            'entropy': -np.sum(probabilities[probabilities > 0] * 
                              np.log(probabilities[probabilities > 0])),
            'purity': np.sum(probabilities**2),
            'max_probability': np.max(probabilities),
            'min_probability': np.min(probabilities[probabilities > 0]),
            'coherence': np.sum(np.abs(state))**2 / len(state)
        }