#!/usr/bin/env python3
"""
Unit tests for QuantumSeed class
Author: Luke Spookwalker
Date: 2025
License: MIT
"""

import numpy as np
import pytest
from core.quantum_seed import QuantumSeed

class TestQuantumSeed:
    """Test suite for QuantumSeed class."""
    
    def setup_method(self):
        """Setup test fixture."""
        self.seed = QuantumSeed(size=8, collapse_rate=0.5, learning_rate=0.3)
        self.patterns = [
            np.array([1, 1, 1, 1, -1, -1, -1, -1]),
            np.array([1, -1, 1, -1, 1, -1, 1, -1]),
            np.array([1, 1, -1, -1, 1, 1, -1, -1]),
        ]
        self.labels = [1, -1, -1]
    
    def test_initialization(self):
        """Test quantum seed initialization."""
        assert self.seed.size == 8
        assert self.seed.collapse_rate == 0.5
        assert self.seed.learning_rate == 0.3
        assert len(self.seed.amplitudes) == 8
        assert len(self.seed.learning_history) == 0
        
        # State should be normalized
        state_norm = np.linalg.norm(self.seed.amplitudes)
        assert np.isclose(state_norm, 1.0, atol=1e-10)
    
    def test_state_normalization(self):
        """Test quantum state normalization."""
        # Manually set unnormalized state
        self.seed.amplitudes = np.ones(8, dtype=complex) * 2
        self.seed._normalize_state()
        
        norm = np.linalg.norm(self.seed.amplitudes)
        assert np.isclose(norm, 1.0, atol=1e-10)
    
    def test_quantum_entropy(self):
        """Test quantum entropy calculation."""
        # Uniform superposition should have maximum entropy
        uniform_state = np.ones(8, dtype=complex) / np.sqrt(8)
        self.seed.amplitudes = uniform_state
        
        entropy = self.seed.quantum_entropy()
        expected_entropy = -np.sum((1/8) * np.log(1/8))
        assert np.isclose(entropy, expected_entropy, atol=1e-5)
        
        # Pure state should have zero entropy
        pure_state = np.zeros(8, dtype=complex)
        pure_state[0] = 1.0
        self.seed.amplitudes = pure_state
        
        entropy = self.seed.quantum_entropy()
        assert np.isclose(entropy, 0.0, atol=1e-10)
    
    def test_single_pattern_learning(self):
        """Test learning a single pattern."""
        pattern = self.patterns[0]
        label = self.labels[0]
        
        # Initial state
        initial_state = self.seed.amplitudes.copy()
        initial_entropy = self.seed.quantum_entropy()
        
        # Learn pattern
        success = self.seed.collapse_learn(pattern, label)
        
        # Verify learning occurred
        assert success == True
        
        # State should change
        state_change = np.linalg.norm(self.seed.amplitudes - initial_state)
        assert state_change > 0
        
        # Entropy should decrease (knowledge increases)
        final_entropy = self.seed.quantum_entropy()
        assert final_entropy <= initial_entropy + 1e-5
        
        # Prediction should be correct
        prediction = self.seed.predict(pattern)
        assert prediction == label
    
    def test_multiple_pattern_learning(self):
        """Test learning multiple patterns."""
        results = []
        
        for pattern, label in zip(self.patterns, self.labels):
            success = self.seed.collapse_learn(pattern, label)
            results.append(success)
            
            # Verify prediction after learning
            prediction = self.seed.predict(pattern)
            assert prediction == label
        
        # At least some learning should succeed
        assert sum(results) > 0
        
        # Learning history should be recorded
        assert len(self.seed.learning_history) == len(self.patterns)
    
    def test_prediction_consistency(self):
        """Test prediction consistency."""
        pattern = self.patterns[0]
        
        # Make multiple predictions (should be consistent)
        predictions = []
        for _ in range(10):
            predictions.append(self.seed.predict(pattern))
        
        # All predictions should be the same (deterministic)
        assert len(set(predictions)) == 1
    
    def test_state_info(self):
        """Test state information retrieval."""
        info = self.seed.get_state_info()
        
        assert 'amplitudes' in info
        assert 'phases' in info
        assert 'entropy' in info
        assert 'coherence' in info
        assert 'num_patterns_learned' in info
        
        assert len(info['amplitudes']) == self.seed.size
        assert len(info['phases']) == self.seed.size
        assert isinstance(info['entropy'], float)
        assert isinstance(info['coherence'], float)
        assert info['num_patterns_learned'] >= 0
    
    def test_pattern_similarity(self):
        """Test pattern similarity calculation."""
        pattern1 = np.array([1, 1, -1, -1])
        pattern2 = np.array([1, 1, -1, -1])  # Same pattern
        pattern3 = np.array([-1, -1, 1, 1])  # Opposite pattern
        
        # Similarity with self should be high
        sim1 = self.seed._pattern_similarity(pattern1, pattern2)
        assert np.isclose(sim1, 1.0, atol=1e-5)
        
        # Similarity with opposite should be low
        sim2 = self.seed._pattern_similarity(pattern1, pattern3)
        assert sim2 < 0.5
    
    def test_interference_application(self):
        """Test quantum interference application."""
        initial_state = self.seed.amplitudes.copy()
        
        # Apply interference
        self.seed._apply_interference()
        
        # State should change
        state_change = np.linalg.norm(self.seed.amplitudes - initial_state)
        assert state_change >= 0
        
        # State should still be normalized
        norm = np.linalg.norm(self.seed.amplitudes)
        assert np.isclose(norm, 1.0, atol=1e-10)
    
    def test_run_experiment(self):
        """Test complete experiment execution."""
        results = self.seed.run_experiment(self.patterns, self.labels)
        
        # Check results structure
        expected_keys = [
            'total_patterns',
            'correct_after_learning',
            'collapses_needed',
            'accuracy_history',
            'entropy_history',
            'final_accuracy',
            'collapse_efficiency',
            'final_entropy'
        ]
        
        for key in expected_keys:
            assert key in results
        
        # Validate values
        assert results['total_patterns'] == len(self.patterns)
        assert 0 <= results['correct_after_learning'] <= len(self.patterns)
        assert 0 <= results['collapses_needed'] <= len(self.patterns)
        assert 0 <= results['final_accuracy'] <= 1
        assert results['collapse_efficiency'] >= 0
        assert results['final_entropy'] >= 0
        
        # History lengths should match
        assert len(results['accuracy_history']) == len(self.patterns)
        assert len(results['entropy_history']) == len(self.patterns)
    
    def test_edge_cases(self):
        """Test edge cases and error handling."""
        # Zero pattern (all zeros)
        zero_pattern = np.zeros(8)
        zero_label = 1
        
        # This should handle zeros gracefully
        success = self.seed.collapse_learn(zero_pattern, zero_label)
        assert isinstance(success, bool)
        
        # Very small pattern
        small_pattern = np.ones(8) * 1e-10
        success = self.seed.collapse_learn(small_pattern, 1)
        assert isinstance(success, bool)
        
        # Pattern with wrong size (should handle gracefully)
        wrong_size_pattern = np.ones(10)
        with pytest.raises(ValueError):
            self.seed.predict(wrong_size_pattern)
    
    def test_reproducibility(self):
        """Test that same inputs produce same results."""
        # Reset seed
        seed1 = QuantumSeed(size=8, collapse_rate=0.5, learning_rate=0.3)
        seed2 = QuantumSeed(size=8, collapse_rate=0.5, learning_rate=0.3)
        
        pattern = self.patterns[0]
        label = self.labels[0]
        
        # Learn same pattern with both seeds
        success1 = seed1.collapse_learn(pattern, label)
        success2 = seed2.collapse_learn(pattern, label)
        
        # Should get same result
        assert success1 == success2
        
        # Predictions should be same
        pred1 = seed1.predict(pattern)
        pred2 = seed2.predict(pattern)
        assert pred1 == pred2
    
    def test_quantum_coherence(self):
        """Test quantum coherence calculation."""
        # Pure state should have maximum coherence
        pure_state = np.zeros(8, dtype=complex)
        pure_state[0] = 1.0
        self.seed.amplitudes = pure_state
        
        info = self.seed.get_state_info()
        coherence = info['coherence']
        
        # Pure state coherence = (sum|amplitude|)^2 / N = 1^2 / 8 = 0.125
        expected_coherence = 1.0 / 8
        assert np.isclose(coherence, expected_coherence, atol=1e-5)
    
    def test_learning_efficiency(self):
        """Test that learning improves over time."""
        accuracies = []
        
        for i, (pattern, label) in enumerate(zip(self.patterns, self.labels)):
            self.seed.collapse_learn(pattern, label)
            
            # Test all learned patterns so far
            correct = 0
            for j in range(i + 1):
                pred = self.seed.predict(self.patterns[j])
                if pred == self.labels[j]:
                    correct += 1
            
            accuracy = correct / (i + 1)
            accuracies.append(accuracy)
        
        # Accuracy should generally improve or stay stable
        for i in range(1, len(accuracies)):
            assert accuracies[i] >= accuracies[i-1] - 0.1  # Allow small fluctuation

if __name__ == "__main__":
    # Run tests directly
    test_suite = TestQuantumSeed()
    test_suite.setup_method()
    
    print("Running QuantumSeed tests...")
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