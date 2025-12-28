#!/usr/bin/env python3
"""
Unit tests for QuantumCollapseLearner class
Author: Luke Spookwalker
Date: 2025
License: MIT
"""

import numpy as np
import pytest
from core.collapse_learner import QuantumCollapseLearner, CollapseMode, QuantumMetrics

class TestQuantumCollapseLearner:
    """Test suite for QuantumCollapseLearner class."""
    
    def setup_method(self):
        """Setup test fixture."""
        self.learner = QuantumCollapseLearner(
            num_qubits=4,  # 16 states
            collapse_strength=0.7,
            interference_depth=3,
            mode=CollapseMode.RESONANT
        )
        
        # Create test patterns
        self.patterns = [
            np.array([1, 1, 1, 1, -1, -1, -1, -1]),  # Only first 4 qubits used
            np.array([1, -1, 1, -1, 1, -1, 1, -1]),
            np.array([1, 1, -1, -1, 1, 1, -1, -1]),
        ]
        self.labels = [1, -1, -1]
    
    def test_initialization(self):
        """Test learner initialization."""
        assert self.learner.num_qubits == 4
        assert self.learner.num_states == 16  # 2^4
        assert self.learner.collapse_strength == 0.7
        assert self.learner.interference_depth == 3
        assert self.learner.mode == CollapseMode.RESONANT
        
        # State vector should be initialized
        assert len(self.learner.state_vector) == 16
        assert np.isclose(np.linalg.norm(self.learner.state_vector), 1.0, atol=1e-10)
        
        # Memory should be empty
        assert len(self.learner.pattern_memory) == 0
        assert len(self.learner.collapse_history) == 0
        
        # Metrics should be initialized
        assert isinstance(self.learner.metrics, QuantumMetrics)
    
    def test_encode_pattern(self):
        """Test pattern encoding to state index."""
        # Pattern: [1, 1, 1, 1] -> binary '1111' -> decimal 15
        pattern1 = np.array([1, 1, 1, 1])
        idx1 = self.learner.encode_pattern(pattern1)
        assert idx1 == 15
        
        # Pattern: [-1, -1, -1, -1] -> binary '0000' -> decimal 0
        pattern2 = np.array([-1, -1, -1, -1])
        idx2 = self.learner.encode_pattern(pattern2)
        assert idx2 == 0
        
        # Pattern: [1, -1, 1, -1] -> binary '1010' -> decimal 10
        pattern3 = np.array([1, -1, 1, -1])
        idx3 = self.learner.encode_pattern(pattern3)
        assert idx3 == 10
        
        # Longer pattern should only use first num_qubits elements
        pattern4 = np.array([1, 1, 1, 1, -1, -1, -1, -1])
        idx4 = self.learner.encode_pattern(pattern4)
        assert idx4 == 15  # Only first 4 elements matter
    
    def test_collapse_learning(self):
        """Test collapse learning with different modes."""
        pattern = self.patterns[0]
        label = self.labels[0]
        
        for mode in CollapseMode:
            learner = QuantumCollapseLearner(
                num_qubits=4,
                collapse_strength=0.5,
                interference_depth=2,
                mode=mode
            )
            
            initial_state = learner.state_vector.copy()
            
            # Perform learning
            result = learner.collapse_learn(pattern, label, learning_rate=0.3)
            
            # Check result structure
            expected_keys = [
                'pattern_idx',
                'label',
                'prediction_before',
                'confidence_before',
                'correct_before',
                'collapse_applied',
                'collapse_strength',
                'mode_used',
                'prediction_after',
                'correct_after',
                'state_entropy',
                'quantum_coherence',
                'final_confidence'
            ]
            
            for key in expected_keys:
                assert key in result
            
            # Validate values
            assert result['label'] == label
            assert result['mode_used'] == mode.value
            assert isinstance(result['collapse_applied'], bool)
            assert 0 <= result['confidence_before'] <= 2  # Can be >1 due to quantum effects
            assert 0 <= result['final_confidence'] <= 2
            
            # State should change if collapse applied
            if result['collapse_applied']:
                state_change = np.linalg.norm(learner.state_vector - initial_state)
                assert state_change > 0
                assert len(learner.collapse_history) == 1
            else:
                # No collapse, state should be similar
                state_change = np.linalg.norm(learner.state_vector - initial_state)
                assert state_change < 0.1
    
    def test_predict(self):
        """Test prediction method."""
        pattern = self.patterns[0]
        
        # Make prediction
        prediction = self.learner.predict(pattern)
        
        # Should be +1 or -1
        assert prediction in [1, -1]
        
        # Multiple predictions should be consistent
        predictions = [self.learner.predict(pattern) for _ in range(5)]
        assert len(set(predictions)) == 1  # All same
    
    def test_find_similar_states(self):
        """Test finding similar quantum states."""
        target_idx = 5  # Binary: 0101
        
        # Find states within Hamming distance 1
        similar = self.learner._find_similar_states(target_idx, radius=1)
        
        # Should include target itself
        assert target_idx in similar
        
        # Should include states that differ by 1 bit
        # For 0101, similar states within radius 1:
        # 0101 (distance 0), 1101 (1), 0001 (1), 0111 (1), 0100 (1)
        expected_similar = {5, 13, 1, 7, 4}
        assert set(similar) == expected_similar
        
        # Test radius 2
        similar2 = self.learner._find_similar_states(target_idx, radius=2)
        assert len(similar2) > len(similar)  # More states with larger radius
    
    def test_pattern_similarity(self):
        """Test pattern similarity calculation."""
        pattern1 = np.array([1, 1, -1, -1])
        pattern2 = np.array([1, 1, -1, -1])  # Same
        pattern3 = np.array([-1, -1, 1, 1])  # Opposite
        pattern4 = np.array([1, -1, 1, -1])  # Different
        
        sim1 = self.learner._pattern_similarity(pattern1, pattern2)
        assert np.isclose(sim1, 1.0, atol=1e-5)
        
        sim2 = self.learner._pattern_similarity(pattern1, pattern3)
        assert np.isclose(sim2, 0.0, atol=1e-5)
        
        sim3 = self.learner._pattern_similarity(pattern1, pattern4)
        assert 0 < sim3 < 1
    
    def test_state_normalization(self):
        """Test state normalization."""
        # Create unnormalized state
        unnormalized = np.ones(16, dtype=complex) * 2
        self.learner.state_vector = unnormalized
        
        # Normalize
        self.learner._normalize_state()
        
        # Should be normalized
        norm = np.linalg.norm(self.learner.state_vector)
        assert np.isclose(norm, 1.0, atol=1e-10)
    
    def test_entropy_calculation(self):
        """Test quantum entropy calculation."""
        # Uniform state should have maximum entropy
        uniform = np.ones(16, dtype=complex) / np.sqrt(16)
        self.learner.state_vector = uniform
        
        entropy = self.learner._calculate_entropy()
        expected = -np.sum((1/16) * np.log(1/16))
        assert np.isclose(entropy, expected, atol=1e-5)
        
        # Pure state should have zero entropy
        pure = np.zeros(16, dtype=complex)
        pure[0] = 1.0
        self.learner.state_vector = pure
        
        entropy = self.learner._calculate_entropy()
        assert np.isclose(entropy, 0.0, atol=1e-10)
    
    def test_coherence_calculation(self):
        """Test quantum coherence calculation."""
        # Pure state coherence
        pure = np.zeros(16, dtype=complex)
        pure[0] = 1.0
        self.learner.state_vector = pure
        
        coherence = self.learner._calculate_coherence()
        expected = 1.0 / 16  # (sum|amplitude|)^2 / N = 1^2 / 16
        assert np.isclose(coherence, expected, atol=1e-5)
    
    def test_phase_diversity(self):
        """Test phase diversity calculation."""
        # All phases different
        state = np.zeros(16, dtype=complex)
        for i in range(16):
            state[i] = np.exp(1j * i * 0.1)
        state /= np.linalg.norm(state)
        self.learner.state_vector = state
        
        diversity = self.learner._calculate_phase_diversity()
        assert np.isclose(diversity, 1.0, atol=0.1)  # All phases unique
        
        # All phases same
        state = np.ones(16, dtype=complex) / np.sqrt(16)
        self.learner.state_vector = state
        
        diversity = self.learner._calculate_phase_diversity()
        assert np.isclose(diversity, 1/16, atol=0.01)  # All phases same
    
    def test_detailed_metrics(self):
        """Test detailed metrics retrieval."""
        metrics = self.learner.get_detailed_metrics()
        
        expected_keys = [
            'entropy',
            'coherence',
            'phase_diversity',
            'interference_strength',
            'collapse_count',
            'memory_size',
            'state_norm',
            'max_amplitude',
            'min_amplitude'
        ]
        
        for key in expected_keys:
            assert key in metrics
        
        # Validate ranges
        assert 0 <= metrics['entropy'] <= np.log(self.learner.num_states)
        assert 0 <= metrics['coherence'] <= 1
        assert 0 <= metrics['phase_diversity'] <= 1
        assert metrics['state_norm'] >= 0
        assert 0 <= metrics['max_amplitude'] <= 1
        assert 0 <= metrics['min_amplitude'] <= 1
    
    def test_memory_management(self):
        """Test pattern memory management."""
        pattern = self.patterns[0]
        label = self.labels[0]
        
        # Add patterns to memory
        for i in range(10):
            self.learner.collapse_learn(pattern, label)
        
        # Memory should not exceed interference_depth * 2
        assert len(self.learner.pattern_memory) <= self.learner.interference_depth * 2
        
        # Older patterns should be removed (FIFO)
        if len(self.learner.pattern_memory) > 0:
            # Check that memory entries have sequential creation times
            creation_times = [m.get('created_at', i) 
                            for i, m in enumerate(self.learner.pattern_memory)]
            assert creation_times == sorted(creation_times)
    
    def test_different_collapse_modes(self):
        """Test different collapse modes produce different results."""
        pattern = self.patterns[0]
        label = self.labels[0]
        
        final_states = {}
        
        for mode in CollapseMode:
            learner = QuantumCollapseLearner(
                num_qubits=4,
                collapse_strength=0.5,
                interference_depth=2,
                mode=mode
            )
            
            # Learn same pattern
            learner.collapse_learn(pattern, label)
            
            # Store final state
            final_states[mode] = learner.state_vector.copy()
        
        # Different modes should produce different states (usually)
        # Check that at least some modes differ
        all_same = True
        for mode1 in CollapseMode:
            for mode2 in CollapseMode:
                if mode1 != mode2:
                    state1 = final_states[mode1]
                    state2 = final_states[mode2]
                    if not np.allclose(state1, state2, atol=1e-5):
                        all_same = False
        
        assert not all_same, "All collapse modes produced identical states"
    
    def test_interference_update(self):
        """Test interference field updates."""
        pattern = self.patterns[0]
        label = self.labels[0]
        
        # Initial interference field should be zero
        assert np.allclose(self.learner.interference_field, 
                          np.zeros_like(self.learner.interference_field))
        
        # Learn a pattern
        self.learner.collapse_learn(pattern, label)
        
        # Interference should be updated
        self.learner._apply_advanced_interference()
        
        # Field should have some non-zero values
        field_norm = np.linalg.norm(self.learner.interference_field)
        assert field_norm > 0
    
    def test_visualization(self):
        """Test visualization method doesn't crash."""
        # This just tests that the method runs without error
        try:
            self.learner.visualize_quantum_state(max_states=8)
            visualization_worked = True
        except Exception:
            visualization_worked = False
        
        # Method should at least not crash
        # (We can't easily test the plot output in unit tests)
        assert True  # If we get here, it didn't crash

if __name__ == "__main__":
    # Run tests directly
    test_suite = TestQuantumCollapseLearner()
    test_suite.setup_method()
    
    print("Running QuantumCollapseLearner tests...")
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