"""
Tests for Phase 8: Axiom Analysis

Tests for GPT framework and axiom candidate analysis.
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from axioms.gpt_framework import (
    StateSpace,
    StateSpaceType,
    Effect,
    Measurement,
    LinearTransformation,
    GPT,
    create_classical_bit_gpt,
    create_qubit_gpt,
    StateSpaceComparison,
    analyze_simplex_to_ball_gap,
)

from axioms.axiom_candidates import (
    AxiomID,
    AxiomStatus,
    StateExtensionChecker,
    BornRuleChecker,
    ReversibilityChecker,
    NonCommutativityChecker,
    NoCloningChecker,
    ContextualityChecker,
    AxiomImplicationGraph,
    compare_axiom_status,
    find_minimal_axiom_set_for_quantum,
)


# =============================================================================
# GPT Framework Tests
# =============================================================================

class TestStateSpace:
    """Tests for StateSpace class."""
    
    def test_simplex_creation(self):
        """Test creating a simplex state space."""
        # 2-state simplex (1-simplex = line segment)
        extreme_points = np.array([[1, 0], [0, 1]])
        ss = StateSpace(
            dimension=2,
            extreme_points=extreme_points,
            space_type=StateSpaceType.SIMPLEX
        )
        
        assert ss.dimension == 2
        assert ss.n_extreme_points == 2
        assert ss.space_type == StateSpaceType.SIMPLEX
    
    def test_mixed_state(self):
        """Test creating mixed states from weights."""
        extreme_points = np.array([[1, 0], [0, 1]])
        ss = StateSpace(
            dimension=2,
            extreme_points=extreme_points,
            space_type=StateSpaceType.SIMPLEX
        )
        
        # 50-50 mixture
        mixed = ss.mixed_state(np.array([0.5, 0.5]))
        np.testing.assert_array_almost_equal(mixed, [0.5, 0.5])
        
        # Pure state
        pure = ss.mixed_state(np.array([1.0, 0.0]))
        np.testing.assert_array_almost_equal(pure, [1.0, 0.0])
    
    def test_invalid_weights(self):
        """Test that invalid weights raise errors."""
        extreme_points = np.array([[1, 0], [0, 1]])
        ss = StateSpace(
            dimension=2,
            extreme_points=extreme_points,
            space_type=StateSpaceType.SIMPLEX
        )
        
        # Weights don't sum to 1
        with pytest.raises(ValueError):
            ss.mixed_state(np.array([0.3, 0.3]))
        
        # Negative weights
        with pytest.raises(ValueError):
            ss.mixed_state(np.array([1.5, -0.5]))


class TestEffect:
    """Tests for Effect class."""
    
    def test_probability_calculation(self):
        """Test probability calculation."""
        effect = Effect(np.array([1, 0]))
        
        # Pure state |0⟩
        prob0 = effect.probability(np.array([1, 0]))
        assert np.isclose(prob0, 1.0)
        
        # Pure state |1⟩
        prob1 = effect.probability(np.array([0, 1]))
        assert np.isclose(prob1, 0.0)
        
        # Mixed state
        prob_mixed = effect.probability(np.array([0.5, 0.5]))
        assert np.isclose(prob_mixed, 0.5)


class TestMeasurement:
    """Tests for Measurement class."""
    
    def test_probability_distribution(self):
        """Test that measurement gives valid probability distribution."""
        effects = [
            Effect(np.array([1, 0]), name="|0⟩"),
            Effect(np.array([0, 1]), name="|1⟩")
        ]
        measurement = Measurement(effects, name="Z")
        
        # Test on pure state
        probs = measurement.probabilities(np.array([1, 0]))
        assert np.isclose(np.sum(probs), 1.0)
        assert probs[0] > probs[1]
        
        # Test on mixed state
        probs_mixed = measurement.probabilities(np.array([0.5, 0.5]))
        np.testing.assert_array_almost_equal(probs_mixed, [0.5, 0.5])


class TestLinearTransformation:
    """Tests for LinearTransformation class."""
    
    def test_apply(self):
        """Test applying transformation."""
        # NOT gate (swap)
        matrix = np.array([[0, 1], [1, 0]])
        not_gate = LinearTransformation(matrix, name="NOT")
        
        result = not_gate.apply(np.array([1, 0]))
        np.testing.assert_array_almost_equal(result, [0, 1])
    
    def test_reversibility(self):
        """Test reversibility check."""
        # Reversible (permutation)
        perm = LinearTransformation(np.array([[0, 1], [1, 0]]))
        assert perm.is_reversible()
        
        # Non-reversible (singular)
        singular = LinearTransformation(np.array([[1, 1], [1, 1]]))
        assert not singular.is_reversible()
    
    def test_composition(self):
        """Test composing transformations."""
        t1 = LinearTransformation(np.array([[0, 1], [1, 0]]), name="A")
        t2 = LinearTransformation(np.array([[1, 0], [0, -1]]), name="B")
        
        composed = t1.compose(t2)
        expected = t1.matrix @ t2.matrix
        np.testing.assert_array_almost_equal(composed.matrix, expected)


class TestGPTCreation:
    """Tests for GPT creation functions."""
    
    def test_classical_1bit(self):
        """Test creating 1-bit classical GPT."""
        gpt = create_classical_bit_gpt(1)
        
        assert gpt.name == "Classical_1bit"
        assert gpt.state_space.dimension == 2
        assert gpt.state_space.n_extreme_points == 2
        assert gpt.state_space.space_type == StateSpaceType.SIMPLEX
        assert len(gpt.measurements) == 1
        assert len(gpt.transformations) == 2  # 2! permutations
    
    def test_classical_2bit(self):
        """Test creating 2-bit classical GPT."""
        gpt = create_classical_bit_gpt(2)
        
        assert gpt.name == "Classical_2bit"
        assert gpt.state_space.dimension == 4
        assert gpt.state_space.n_extreme_points == 4
        assert len(gpt.transformations) == 24  # 4! permutations
    
    def test_qubit(self):
        """Test creating qubit GPT."""
        gpt = create_qubit_gpt()
        
        assert gpt.name == "Qubit"
        assert gpt.state_space.dimension == 4  # Bloch representation
        assert gpt.state_space.space_type == StateSpaceType.BALL
        assert len(gpt.measurements) == 3  # X, Y, Z


class TestStateSpaceComparison:
    """Tests for state space comparison."""
    
    def test_comparison(self):
        """Test comparing classical and quantum."""
        classical = create_classical_bit_gpt(1)
        quantum = create_qubit_gpt()
        
        comparison = StateSpaceComparison(classical, quantum)
        diff = comparison.structural_difference()
        
        assert diff['gpt1_is_simplex'] == True
        assert diff['gpt2_is_simplex'] == False
        assert diff['gpt2_is_ball'] == True
    
    def test_gap_analysis(self):
        """Test simplex to ball gap analysis."""
        gap = analyze_simplex_to_ball_gap(1)
        
        assert 'gap_analysis' in gap
        assert 'missing_structure' in gap['gap_analysis']
        assert 'continuous amplitude' in gap['gap_analysis']['missing_structure']


# =============================================================================
# Axiom Checker Tests
# =============================================================================

class TestStateExtensionChecker:
    """Tests for A1: State Extension."""
    
    def test_classical_fails(self):
        """Classical theories do not satisfy state extension."""
        classical = create_classical_bit_gpt(1)
        checker = StateExtensionChecker()
        
        status = checker.check(classical)
        assert status.satisfied == False
    
    def test_quantum_satisfies(self):
        """Quantum theory satisfies state extension."""
        quantum = create_qubit_gpt()
        checker = StateExtensionChecker()
        
        status = checker.check(quantum)
        assert status.satisfied == True


class TestReversibilityChecker:
    """Tests for A3: Reversibility."""
    
    def test_classical_reversible(self):
        """Classical permutation transformations are reversible."""
        classical = create_classical_bit_gpt(1)
        checker = ReversibilityChecker()
        
        status = checker.check(classical)
        assert status.satisfied == True
    
    def test_quantum_reversible(self):
        """Quantum unitary transformations are reversible."""
        quantum = create_qubit_gpt()
        checker = ReversibilityChecker()
        
        status = checker.check(quantum)
        assert status.satisfied == True


class TestNoCloningChecker:
    """Tests for A5: No-Cloning."""
    
    def test_classical_violates(self):
        """Classical theories violate no-cloning (can copy)."""
        classical = create_classical_bit_gpt(1)
        checker = NoCloningChecker()
        
        status = checker.check(classical)
        assert status.satisfied == False
    
    def test_quantum_satisfies(self):
        """Quantum theory satisfies no-cloning."""
        quantum = create_qubit_gpt()
        checker = NoCloningChecker()
        
        status = checker.check(quantum)
        assert status.satisfied == True


class TestContextualityChecker:
    """Tests for A6: Contextuality."""
    
    def test_classical_non_contextual(self):
        """Classical theories are non-contextual."""
        classical = create_classical_bit_gpt(1)
        checker = ContextualityChecker()
        
        status = checker.check(classical)
        assert status.satisfied == False
    
    def test_quantum_contextual(self):
        """Quantum theory is contextual."""
        quantum = create_qubit_gpt()
        checker = ContextualityChecker()
        
        status = checker.check(quantum)
        assert status.satisfied == True


# =============================================================================
# Integration Tests
# =============================================================================

class TestAxiomImplicationGraph:
    """Tests for axiom implication analysis."""
    
    def test_check_all(self):
        """Test checking all axioms."""
        graph = AxiomImplicationGraph()
        classical = create_classical_bit_gpt(1)
        
        statuses = graph.check_all(classical)
        
        assert len(statuses) == 6  # 6 axioms
        assert all(isinstance(s, AxiomStatus) for s in statuses.values())
    
    def test_analyze_implications(self):
        """Test implication analysis."""
        graph = AxiomImplicationGraph()
        quantum = create_qubit_gpt()
        
        analysis = graph.analyze_implications(quantum)
        
        assert 'satisfied' in analysis
        assert 'not_satisfied' in analysis
        assert analysis['n_satisfied'] > 0


class TestCompareAxiomStatus:
    """Tests for comparing axiom status across GPTs."""
    
    def test_comparison(self):
        """Test comparing classical and quantum."""
        classical = create_classical_bit_gpt(1)
        quantum = create_qubit_gpt()
        
        comparison = compare_axiom_status([classical, quantum])
        
        assert 'comparison_table' in comparison
        assert len(comparison['comparison_table']) == 6
        
        # A1 should differ between classical and quantum
        a1_status = comparison['comparison_table']['A1_STATE_EXTENSION']
        assert a1_status['Classical_1bit'] == False
        assert a1_status['Qubit'] == True


class TestMinimalAxiomSet:
    """Tests for minimal axiom set search."""
    
    def test_find_minimal(self):
        """Test finding minimal axiom sets."""
        minimal = find_minimal_axiom_set_for_quantum()
        
        assert 'minimal_sets' in minimal
        assert len(minimal['minimal_sets']) >= 3
        assert 'key_insight' in minimal
        
        # A1 should appear in the insight
        assert 'A1' in minimal['key_insight']


# =============================================================================
# Hypothesis Tests (H8.x)
# =============================================================================

class TestHypothesisH8:
    """Tests related to Phase 8 hypotheses."""
    
    def test_h8_1_superposition_primitive(self):
        """
        H8.1: A1 (state extension/superposition) cannot be derived from others.
        
        Evidence: Classical computation satisfies A2, A3, but not A1.
        """
        classical = create_classical_bit_gpt(1)
        graph = AxiomImplicationGraph()
        
        statuses = graph.check_all(classical)
        
        # Classical satisfies A2 (Born rule), A3 (reversibility)
        assert statuses[AxiomID.A2_BORN_RULE].satisfied == True
        assert statuses[AxiomID.A3_REVERSIBILITY].satisfied == True
        
        # But NOT A1 (state extension)
        assert statuses[AxiomID.A1_STATE_EXTENSION].satisfied == False
        
        # This supports H8.1: A1 is not derivable from A2, A3
    
    def test_h8_2_no_cloning_derivable(self):
        """
        H8.2: No-cloning is derivable from A1 + A3.
        
        Check: Quantum (A1+A3) → A5, Classical (no A1) → no A5
        """
        quantum = create_qubit_gpt()
        classical = create_classical_bit_gpt(1)
        
        graph = AxiomImplicationGraph()
        
        q_status = graph.check_all(quantum)
        c_status = graph.check_all(classical)
        
        # Quantum has A1, A3, and A5
        assert q_status[AxiomID.A1_STATE_EXTENSION].satisfied == True
        assert q_status[AxiomID.A3_REVERSIBILITY].satisfied == True
        assert q_status[AxiomID.A5_NO_CLONING].satisfied == True
        
        # Classical has A3 but not A1, and violates A5
        assert c_status[AxiomID.A3_REVERSIBILITY].satisfied == True
        assert c_status[AxiomID.A1_STATE_EXTENSION].satisfied == False
        assert c_status[AxiomID.A5_NO_CLONING].satisfied == False
        
        # This supports H8.2: A5 is derivable from A1 + A3
    
    def test_h8_3_non_commutativity_insufficient(self):
        """
        H8.3: Non-commutativity (A4) alone doesn't generate quantum structure.
        
        This is harder to test directly, but we can verify that
        classical theories lack both A1 and A4.
        """
        classical = create_classical_bit_gpt(1)
        graph = AxiomImplicationGraph()
        
        statuses = graph.check_all(classical)
        
        # Classical has only one measurement, so A4 check is limited
        # But the key point is: even if we added non-commuting measurements,
        # without A1, we wouldn't get quantum structure
        
        assert statuses[AxiomID.A1_STATE_EXTENSION].satisfied == False
        # A4 may or may not be satisfied depending on measurements


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

