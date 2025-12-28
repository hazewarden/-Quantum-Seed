#!/usr/bin/env python3
"""
Unit tests for QuantumDynamics class
Author: Luke Spookwalker
Date: 2025
License: MIT
"""

import numpy as np
import pytest
from core.quantum_dynamics import QuantumDynamics

class TestQuantumDynamics:
    """Test suite for QuantumDynamics class."""
    
    def setup_method(self):
        """Setup test fixture."""
        self.dynamics = QuantumDynamics()
    
    def test_quantum_gates(self):
        """Test quantum gate creation."""
        # Hadamard gate
        H = self.dynamics.hadamard_gate()
        assert H.shape == (2, 2)
        assert np.allclose(H @ H.conj().T, np.eye(2), atol=1e-10)  # Unitary
        assert np.allclose(H, H.T, atol=1e-10)  # Symmetric
        
        # Pauli gates
        X = self.dynamics.pauli_x()
        Y = self.dynamics.pauli_y()
        Z = self.dynamics.pauli_z()
        
        for gate in [X, Y, Z]:
            assert gate.shape == (2, 2)
            assert np.allclose(gate @ gate.conj().T, np.eye(2), atol=1e-10)
        
        # Specific properties
        assert np.allclose(X @ X, np.eye(2), atol=1e-10)  # X^2 = I
        assert np.allclose(Y @ Y, np.eye(2), atol=1e-10)  # Y^2 = I
        assert np.allclose(Z @ Z, np.eye(2), atol=1e-10)  # Z^2 = I
    
    def test_rotation_gates(self):
        """Test rotation gate creation."""
        # Test rotation around different axes
        for axis in ['x', 'y', 'z']:
            for angle in [0, np.pi/4, np.pi/2, np.pi]:
                R = self.dynamics.rotation_gate(angle, axis)
                
                # Should be unitary
                assert R.shape == (2, 2)
                assert np.allclose(R @ R.conj().T, np.eye(2), atol=1e-10)
                
                # Rotation by 0 should be identity
                if angle == 0:
                    assert np.allclose(R, np.eye(2), atol=1e-10)
                
                # Rotation by 2π should be identity (up to global phase)
                R_2pi = self.dynamics.rotation_gate(2*np.pi, axis)
                # For spin-1/2, rotation by 2π gives -I
                assert np.allclose(R_2pi @ R_2pi, np.eye(2), atol=1e-10)
        
        # Test invalid axis
        with pytest.raises(ValueError):
            self.dynamics.rotation_gate(np.pi/2, 'invalid')
    
    def test_cnot_gate(self):
        """Test CNOT gate creation."""
        CNOT = self.dynamics.cnot_gate()
        
        assert CNOT.shape == (4, 4)
        assert np.allclose(CNOT @ CNOT.conj().T, np.eye(4), atol=1e-10)  # Unitary
        
        # Test CNOT action
        # |00⟩ -> |00⟩
        assert np.allclose(CNOT @ np.array([1, 0, 0, 0]), 
                          np.array([1, 0, 0, 0]), atol=1e-10)
        # |01⟩ -> |01⟩
        assert np.allclose(CNOT @ np.array([0, 1, 0, 0]), 
                          np.array([0, 1, 0, 0]), atol=1e-10)
        # |10⟩ -> |11⟩
        assert np.allclose(CNOT @ np.array([0, 0, 1, 0]), 
                          np.array([0, 0, 0, 1]), atol=1e-10)
        # |11⟩ -> |10⟩
        assert np.allclose(CNOT @ np.array([0, 0, 0, 1]), 
                          np.array([0, 0, 1, 0]), atol=1e-10)
    
    def test_create_superposition(self):
        """Test superposition state creation."""
        for num_qubits in [1, 2, 3, 4]:
            state = self.dynamics.create_superposition(num_qubits)
            num_states = 2 ** num_qubits
            
            assert len(state) == num_states
            assert np.isclose(np.linalg.norm(state), 1.0, atol=1e-10)  # Normalized
            
            # All amplitudes should have magnitude 1/√N
            expected_amplitude = 1 / np.sqrt(num_states)
            assert np.allclose(np.abs(state), expected_amplitude, atol=1e-10)
    
    def test_apply_quantum_force(self):
        """Test quantum force application."""
        # Create test state
        state = np.array([1, 0, 0, 0], dtype=complex)  # |00⟩
        state = state / np.linalg.norm(state)
        
        # Create force vector
        force = np.array([0, 0.1, 0.2, 0.3], dtype=complex)
        
        # Apply force
        new_state = self.dynamics.apply_quantum_force(state, force, strength=0.5)
        
        # Should be normalized
        assert np.isclose(np.linalg.norm(new_state), 1.0, atol=1e-10)
        
        # Should be different from original
        assert not np.allclose(new_state, state, atol=1e-5)
        
        # Test with zero force
        new_state_zero = self.dynamics.apply_quantum_force(state, np.zeros_like(force))
        assert np.allclose(new_state_zero, state, atol=1e-10)
    
    def test_quantum_interference(self):
        """Test quantum interference."""
        # Create two states
        state1 = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)  # (|00⟩ + |11⟩)/√2
        state2 = np.array([1, 0, 0, -1], dtype=complex) / np.sqrt(2)  # (|00⟩ - |11⟩)/√2
        
        # Apply interference
        interfered = self.dynamics.quantum_interference(state1, state2, strength=0.5)
        
        # Should be normalized
        assert np.isclose(np.linalg.norm(interfered), 1.0, atol=1e-10)
        
        # With strength 0, should return original state
        same_state = self.dynamics.quantum_interference(state1, state2, strength=0)
        assert np.allclose(same_state, state1, atol=1e-10)
        
        # With strength 1, should approach interference pattern
        strong_interference = self.dynamics.quantum_interference(state1, state2, strength=1)
        assert not np.allclose(strong_interference, state1, atol=0.1)
    
    def test_partial_collapse(self):
        """Test partial quantum state collapse."""
        # Start with uniform superposition
        state = np.ones(8, dtype=complex) / np.sqrt(8)
        
        # Collapse toward states 0 and 1
        target_indices = [0, 1]
        collapsed = self.dynamics.partial_collapse(state, target_indices, amplification=2.0)
        
        # Should be normalized
        assert np.isclose(np.linalg.norm(collapsed), 1.0, atol=1e-10)
        
        # Target states should have higher probability
        probs_before = np.abs(state)**2
        probs_after = np.abs(collapsed)**2
        
        for idx in target_indices:
            assert probs_after[idx] > probs_before[idx]
        
        # Non-target states should have lower or equal probability
        for idx in range(8):
            if idx not in target_indices:
                assert probs_after[idx] <= probs_before[idx] + 1e-10
        
        # Test with amplification = 1 (no change)
        unchanged = self.dynamics.partial_collapse(state, target_indices, amplification=1.0)
        assert np.allclose(unchanged, state, atol=1e-10)
    
    def test_calculate_entanglement(self):
        """Test entanglement calculation."""
        # Product state (not entangled)
        product_state = np.array([1, 0, 0, 0], dtype=complex)  # |00⟩
        product_state = product_state / np.linalg.norm(product_state)
        
        entanglement1 = self.dynamics.calculate_entanglement(product_state, num_qubits=2)
        assert np.isclose(entanglement1, 0.0, atol=1e-5)
        
        # Bell state (maximally entangled)
        bell_state = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)  # (|00⟩ + |11⟩)/√2
        entanglement2 = self.dynamics.calculate_entanglement(bell_state, num_qubits=2)
        assert np.isclose(entanglement2, 1.0, atol=1e-5)
        
        # Test with single qubit (should return 0)
        single_qubit_state = np.array([1, 0], dtype=complex)
        entanglement3 = self.dynamics.calculate_entanglement(single_qubit_state, num_qubits=1)
        assert np.isclose(entanglement3, 0.0, atol=1e-5)
    
    def test_measure_state(self):
        """Test quantum state measurement."""
        # Create superposition state
        state = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)  # (|00⟩ + |11⟩)/√2
        
        # Measure multiple times (should get |00⟩ or |11⟩)
        outcomes = []
        for _ in range(100):
            outcome = self.dynamics.measure_state(state)
            outcomes.append(outcome)
        
        # Should only get 0 or 3 (|00⟩ or |11⟩)
        assert set(outcomes) <= {0, 3}
        
        # With probabilities ~0.5 each
        p0 = outcomes.count(0) / len(outcomes)
        p3 = outcomes.count(3) / len(outcomes)
        assert np.isclose(p0, 0.5, atol=0.1)
        assert np.isclose(p3, 0.5, atol=0.1)
        
        # Test with custom basis
        basis = np.eye(4, dtype=complex)  # Computational basis
        outcome = self.dynamics.measure_state(state, basis)
        assert outcome in [0, 1, 2, 3]
    
    def test_quantum_fidelity(self):
        """Test quantum fidelity calculation."""
        # Same state should have fidelity 1
        state1 = np.array([1, 0, 0, 0], dtype=complex)
        state1 = state1 / np.linalg.norm(state1)
        
        fidelity1 = self.dynamics.quantum_fidelity(state1, state1)
        assert np.isclose(fidelity1, 1.0, atol=1e-10)
        
        # Orthogonal states should have fidelity 0
        state2 = np.array([0, 1, 0, 0], dtype=complex)
        state2 = state2 / np.linalg.norm(state2)
        
        fidelity2 = self.dynamics.quantum_fidelity(state1, state2)
        assert np.isclose(fidelity2, 0.0, atol=1e-10)
        
        # Test with unnormalized states
        state3 = np.array([2, 0, 0, 0], dtype=complex)  # Unnormalized
        state4 = np.array([3, 0, 0, 0], dtype=complex)  # Unnormalized
        
        fidelity3 = self.dynamics.quantum_fidelity(state3, state4)
        assert np.isclose(fidelity3, 1.0, atol=1e-10)  # Same direction
        
        # Test with zero states
        zero_state = np.zeros(4, dtype=complex)
        fidelity4 = self.dynamics.quantum_fidelity(state1, zero_state)
        assert np.isclose(fidelity4, 0.0, atol=1e-10)
    
    def test_create_entangled_state(self):
        """Test entangled state creation."""
        # Bell state (2 qubits)
        bell = self.dynamics.create_entangled_state(2, 'bell')
        assert len(bell) == 4
        assert np.isclose(np.linalg.norm(bell), 1.0, atol=1e-10)
        
        # Should be (|00⟩ + |11⟩)/√2
        expected = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)], dtype=complex)
        assert np.allclose(bell, expected, atol=1e-10)
        
        # GHZ state (3 qubits)
        ghz = self.dynamics.create_entangled_state(3, 'ghz')
        assert len(ghz) == 8
        assert np.isclose(np.linalg.norm(ghz), 1.0, atol=1e-10)
        
        # Should be (|000⟩ + |111⟩)/√2
        expected_ghz = np.zeros(8, dtype=complex)
        expected_ghz[0] = 1/np.sqrt(2)  # |000⟩
        expected_ghz[7] = 1/np.sqrt(2)  # |111⟩
        assert np.allclose(ghz, expected_ghz, atol=1e-10)
        
        # W state (3 qubits)
        w = self.dynamics.create_entangled_state(3, 'w')
        assert len(w) == 8
        assert np.isclose(np.linalg.norm(w), 1.0, atol=1e-10)
        
        # Should be (|001⟩ + |010⟩ + |100⟩)/√3
        expected_w = np.zeros(8, dtype=complex)
        expected_w[1] = 1/np.sqrt(3)  # |001⟩
        expected_w[2] = 1/np.sqrt(3)  # |010⟩
        expected_w[4] = 1/np.sqrt(3)  # |100⟩
        assert np.allclose(w, expected_w, atol=1e-10)
        
        # Default (uniform superposition)
        default = self.dynamics.create_entangled_state(3, 'unknown')
        assert len(default) == 8
        assert np.isclose(np.linalg.norm(default), 1.0, atol=1e-10)
    
    def test_quantum_tomography(self):
        """Test quantum state tomography."""
        # Create a test state
        state = np.array([0.5, 0.5, 0.5, 0.5], dtype=complex)
        state = state / np.linalg.norm(state)
        
        # Perform tomography
        tomography = self.dynamics.quantum_tomography(state, num_qubits=2)
        
        # Check structure
        expected_keys = [
            'probabilities',
            'phases',
            'entropy',
            'purity',
            'max_probability',
            'min_probability',
            'coherence'
        ]
        
        for key in expected_keys:
            assert key in tomography
        
        # Validate values
        probs = tomography['probabilities']
        assert len(probs) == len(state)
        assert np.isclose(np.sum(probs), 1.0, atol=1e-10)
        assert np.all(probs >= 0)
        
        phases = tomography['phases']
        assert len(phases) == len(state)
        
        entropy = tomography['entropy']
        assert entropy >= 0
        
        purity = tomography['purity']
        assert 0 <= purity <= 1
        
        max_prob = tomography['max_probability']
        min_prob = tomography['min_probability']
        assert max_prob >= min_prob
        
        coherence = tomography['coherence']
        assert coherence >= 0
    
    def test_edge_cases(self):
        """Test edge cases and error handling."""
        # Zero state
        zero_state = np.zeros(4, dtype=complex)
        
        # These should handle zero states gracefully
        fidelity = self.dynamics.quantum_fidelity(zero_state, zero_state)
        assert np.isclose(fidelity, 0.0, atol=1e-10)
        
        # Very small state
        small_state = np.ones(4, dtype=complex) * 1e-10
        small_state = small_state / np.linalg.norm(small_state)
        
        tomography = self.dynamics.quantum_tomography(small_state, num_qubits=2)
        assert tomography is not None
        
        # Large number of qubits (within reason)
        large_state = self.dynamics.create_superposition(6)  # 64 states
        assert len(large_state) == 64
        assert np.isclose(np.linalg.norm(large_state), 1.0, atol=1e-10)
    
    def test_reproducibility(self):
        """Test that same inputs produce same results."""
        # Test gate creation is deterministic
        H1 = self.dynamics.hadamard_gate()
        H2 = self.dynamics.hadamard_gate()
        assert np.allclose(H1, H2, atol=1e-10)
        
        # Test rotation gates
        R1 = self.dynamics.rotation_gate(np.pi/3, 'x')
        R2 = self.dynamics.rotation_gate(np.pi/3, 'x')
        assert np.allclose(R1, R2, atol=1e-10)
        
        # Test superposition creation (deterministic except for random phases)
        state1 = self.dynamics.create_superposition(3)
        state2 = self.dynamics.create_superposition(3)
        
        # Amplitudes should be same, phases might differ randomly
        assert np.allclose(np.abs(state1), np.abs(state2), atol=1e-10)
    
    def test_mathematical_properties(self):
        """Test mathematical properties of quantum operations."""
        # Test unitarity of gates
        gates = [
            self.dynamics.hadamard_gate(),
            self.dynamics.pauli_x(),
            self.dynamics.pauli_y(),
            self.dynamics.pauli_z(),
            self.dynamics.rotation_gate(np.pi/4, 'x'),
            self.dynamics.rotation_gate(np.pi/4, 'y'),
            self.dynamics.rotation_gate(np.pi/4, 'z'),
            self.dynamics.cnot_gate(),
        ]
        
        for gate in gates:
            # U†U = I
            product = gate.conj().T @ gate
            assert np.allclose(product, np.eye(gate.shape[0]), atol=1e-10)
            
            # UU† = I
            product2 = gate @ gate.conj().T
            assert np.allclose(product2, np.eye(gate.shape[0]), atol=1e-10)
        
        # Test that interference preserves normalization
        state = np.array([1, 2, 3, 4], dtype=complex)
        state = state / np.linalg.norm(state)
        pattern = np.array([1, -1, 1, -1], dtype=complex)
        
        for strength in [0, 0.3, 0.7, 1.0]:
            interfered = self.dynamics.quantum_interference(state, pattern, strength)
            assert np.isclose(np.linalg.norm(interfered), 1.0, atol=1e-10)

if __name__ == "__main__":
    # Run tests directly
    test_suite = TestQuantumDynamics()
    test_suite.setup_method()
    
    print("Running QuantumDynamics tests...")
    methods = [m for m in dir(test_suite) if m.startswith('test_')]
    
    for method_name in methods:
        print(f"  Testing {method_name}...", end="")
        method = getattr(test_suite, method_name)
        try:
            method()
            print(" ✓ PASSED")
        except AssertionError as e:
            print(f" ✗ FAILED: {e}")
        except Exception as e:
            print(f" ✗ ERROR: {e}")
    
    print("\nAll tests completed!")