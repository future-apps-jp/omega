"""
Tests for Phase 9: Resource Theory of Coherence

Tests for coherence measures, resource states, operations, and
information-theoretic principles.
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from information.resource_theory import (
    # Density matrix
    is_density_matrix,
    pure_state_density,
    mixed_state_density,
    
    # States
    computational_basis,
    plus_state,
    minus_state,
    ghz_state,
    
    # Coherence
    L1Coherence,
    RelativeEntropyCoherence,
    RobustnessCoherence,
    ResourceState,
    
    # Operations
    PermutationOperation,
    DephasingOperation,
    MeasurementOperation,
    HadamardOperation,
    PhaseGateOperation,
    
    # Experiments
    ResourceInjectionExperiment,
    
    # Information
    check_no_cloning,
    check_no_deleting,
    analyze_information_principles,
)


# =============================================================================
# Density Matrix Tests
# =============================================================================

class TestDensityMatrix:
    """Tests for density matrix operations."""
    
    def test_pure_state_density(self):
        """Test creating density matrix from pure state."""
        psi = np.array([1, 0])
        rho = pure_state_density(psi)
        
        assert is_density_matrix(rho)
        assert np.isclose(rho[0, 0], 1.0)
        assert np.isclose(rho[1, 1], 0.0)
    
    def test_mixed_state_density(self):
        """Test creating mixed state density matrix."""
        probs = np.array([0.5, 0.5])
        states = [np.array([1, 0]), np.array([0, 1])]
        
        rho = mixed_state_density(probs, states)
        
        assert is_density_matrix(rho)
        assert np.isclose(rho[0, 0], 0.5)
        assert np.isclose(rho[1, 1], 0.5)
    
    def test_coherent_state_density(self):
        """Test density matrix for coherent state |+⟩."""
        psi = plus_state(1)
        rho = pure_state_density(psi)
        
        assert is_density_matrix(rho)
        # Off-diagonal elements should be non-zero
        assert np.abs(rho[0, 1]) > 0.1
    
    def test_invalid_density_matrix(self):
        """Test detection of invalid density matrix."""
        # Not trace 1
        rho_bad = np.array([[0.5, 0], [0, 0.3]])
        assert not is_density_matrix(rho_bad)
        
        # Not positive semi-definite
        rho_bad2 = np.array([[2, 0], [0, -1]])
        assert not is_density_matrix(rho_bad2)


# =============================================================================
# Standard States Tests
# =============================================================================

class TestStandardStates:
    """Tests for standard quantum states."""
    
    def test_computational_basis(self):
        """Test computational basis states."""
        basis = computational_basis(1)
        
        assert len(basis) == 2
        np.testing.assert_array_equal(basis[0], [1, 0])
        np.testing.assert_array_equal(basis[1], [0, 1])
    
    def test_plus_state(self):
        """Test |+⟩ state."""
        plus = plus_state(1)
        
        assert len(plus) == 2
        assert np.isclose(np.linalg.norm(plus), 1.0)
        assert np.isclose(plus[0], plus[1])
    
    def test_minus_state(self):
        """Test |−⟩ state."""
        minus = minus_state()
        
        assert len(minus) == 2
        assert np.isclose(np.linalg.norm(minus), 1.0)
        assert np.isclose(plus_state(1) @ minus, 0.0)  # Orthogonal to |+⟩
    
    def test_ghz_state(self):
        """Test GHZ state."""
        ghz = ghz_state(2)
        
        assert len(ghz) == 4
        assert np.isclose(np.linalg.norm(ghz), 1.0)
        assert np.isclose(np.abs(ghz[0]), 1/np.sqrt(2))
        assert np.isclose(np.abs(ghz[-1]), 1/np.sqrt(2))


# =============================================================================
# Coherence Measure Tests
# =============================================================================

class TestCoherenceMeasures:
    """Tests for coherence measures."""
    
    def test_l1_coherence_incoherent(self):
        """L1 coherence of incoherent state should be 0."""
        rho = pure_state_density(np.array([1, 0]))
        measure = L1Coherence()
        
        coherence = measure.measure(rho)
        assert np.isclose(coherence, 0.0)
    
    def test_l1_coherence_coherent(self):
        """L1 coherence of |+⟩ should be maximal (for qubit)."""
        rho = pure_state_density(plus_state(1))
        measure = L1Coherence()
        
        coherence = measure.measure(rho)
        assert coherence > 0.9  # Should be ~1 for |+⟩
    
    def test_l1_coherence_mixed(self):
        """L1 coherence of classical mixture should be 0."""
        rho = mixed_state_density(
            np.array([0.5, 0.5]),
            [np.array([1, 0]), np.array([0, 1])]
        )
        measure = L1Coherence()
        
        coherence = measure.measure(rho)
        assert np.isclose(coherence, 0.0)
    
    def test_relative_entropy_coherence(self):
        """Test relative entropy of coherence."""
        measure = RelativeEntropyCoherence()
        
        # Incoherent state
        rho_inc = pure_state_density(np.array([1, 0]))
        assert np.isclose(measure.measure(rho_inc), 0.0)
        
        # Coherent state
        rho_coh = pure_state_density(plus_state(1))
        assert measure.measure(rho_coh) > 0.5


# =============================================================================
# Resource State Tests
# =============================================================================

class TestResourceState:
    """Tests for ResourceState class."""
    
    def test_incoherent_is_free(self):
        """Incoherent states should be classified as free."""
        state = ResourceState.from_pure_state(np.array([1, 0]))
        assert state.is_free
    
    def test_coherent_is_not_free(self):
        """Coherent states should NOT be classified as free."""
        state = ResourceState.from_pure_state(plus_state(1))
        assert not state.is_free
    
    def test_coherence_values(self):
        """Test that coherence values are computed correctly."""
        state = ResourceState.from_pure_state(plus_state(1))
        
        assert 'L1_coherence' in state.coherence
        assert 'relative_entropy_coherence' in state.coherence
        assert state.coherence['L1_coherence'] > 0


# =============================================================================
# Operation Tests
# =============================================================================

class TestOperations:
    """Tests for quantum operations."""
    
    def test_permutation_preserves_density(self):
        """Permutation operation preserves density matrix properties."""
        rho = pure_state_density(plus_state(1))
        NOT = np.array([[0, 1], [1, 0]])
        op = PermutationOperation(NOT)
        
        rho_new = op.apply(rho)
        assert is_density_matrix(rho_new)
    
    def test_permutation_is_free(self):
        """Permutation operations are free."""
        NOT = np.array([[0, 1], [1, 0]])
        op = PermutationOperation(NOT)
        
        assert op.is_coherence_non_generating()
    
    def test_dephasing_removes_coherence(self):
        """Dephasing should remove all coherence."""
        rho = pure_state_density(plus_state(1))
        op = DephasingOperation()
        measure = L1Coherence()
        
        rho_dephased = op.apply(rho)
        
        assert is_density_matrix(rho_dephased)
        assert np.isclose(measure.measure(rho_dephased), 0.0)
    
    def test_hadamard_creates_coherence(self):
        """Hadamard creates coherence from |0⟩."""
        rho = pure_state_density(np.array([1, 0]))
        op = HadamardOperation()
        measure = L1Coherence()
        
        initial_coherence = measure.measure(rho)
        rho_new = op.apply(rho)
        final_coherence = measure.measure(rho_new)
        
        assert initial_coherence < 0.1
        assert final_coherence > 0.9
    
    def test_hadamard_is_not_free(self):
        """Hadamard is NOT a free operation."""
        op = HadamardOperation()
        assert not op.is_coherence_non_generating()


# =============================================================================
# Resource Injection Experiment Tests
# =============================================================================

class TestResourceInjectionExperiment:
    """Tests for resource injection experiments."""
    
    def test_classical_cannot_amplify(self):
        """Classical computation cannot amplify coherence."""
        experiment = ResourceInjectionExperiment()
        results = experiment.run_classical_computation_test()
        
        assert results['supports_H9_3']
        assert 'CANNOT amplify' in results['conclusion']
    
    def test_permutation_preserves_coherence(self):
        """Permutation should preserve coherence amount."""
        experiment = ResourceInjectionExperiment()
        
        resource = ResourceState.from_pure_state(plus_state(1))
        NOT = np.array([[0, 1], [1, 0]])
        op = PermutationOperation(NOT)
        
        result = experiment.inject_and_evolve(resource, op)
        
        # Coherence should be approximately preserved
        assert result.is_coherence_preserved or result.is_coherence_destroyed == False
    
    def test_dephasing_destroys_coherence(self):
        """Dephasing should destroy coherence."""
        experiment = ResourceInjectionExperiment()
        
        resource = ResourceState.from_pure_state(plus_state(1))
        op = DephasingOperation()
        
        result = experiment.inject_and_evolve(resource, op)
        
        assert result.is_coherence_destroyed


# =============================================================================
# Information Principles Tests
# =============================================================================

class TestInformationPrinciples:
    """Tests for information-theoretic principles."""
    
    def test_no_cloning_non_orthogonal(self):
        """No-cloning applies to non-orthogonal states."""
        result = check_no_cloning(None)
        
        assert not result['perfect_cloning_possible']
        assert 'CONSEQUENCE' in result['conclusion']
    
    def test_no_deleting_reversible(self):
        """Reversible computation satisfies no-deleting."""
        result = check_no_deleting("reversible")
        
        assert result['satisfies_no_deleting']
    
    def test_no_deleting_irreversible(self):
        """Irreversible computation violates no-deleting."""
        result = check_no_deleting("irreversible")
        
        assert not result['satisfies_no_deleting']
    
    def test_information_principles_analysis(self):
        """Test comprehensive information principles analysis."""
        result = analyze_information_principles()
        
        # No-cloning should be unique to quantum
        assert 'no_cloning' in result['unique_to_quantum']
        
        # Information conservation shared
        assert 'information_conservation' in result['shared_with_reversible']
        
        # Conclusion should support H9.2
        assert 'RESULT' in result['conclusion'] or 'result' in result['conclusion'].lower()


# =============================================================================
# Hypothesis Tests (H9.x)
# =============================================================================

class TestHypothesisH9:
    """Tests related to Phase 9 hypotheses."""
    
    def test_h9_1_information_conservation_insufficient(self):
        """
        H9.1: Information conservation alone doesn't give quantum structure.
        
        Evidence: Reversible classical computation conserves information
        but is not quantum.
        """
        analysis = analyze_information_principles()
        
        # Reversible classical satisfies information conservation
        assert analysis['results']['classical_reversible']['information_conservation']
        
        # But NOT no-cloning (which requires superposition)
        assert not analysis['results']['classical_reversible']['no_cloning']
        
        # Conclusion: Information conservation is insufficient
    
    def test_h9_2_no_cloning_is_result(self):
        """
        H9.2: No-cloning is a result of quantum structure, not a cause.
        
        Evidence: No-cloning requires non-orthogonal states, which
        require superposition (A1).
        """
        result = check_no_cloning(None)
        
        # Non-orthogonal states exist only with superposition
        assert 'CONSEQUENCE' in result['conclusion']
        
        # Cannot derive superposition from no-cloning
        # (circular: no-cloning requires non-orthogonal states which require superposition)
    
    def test_h9_3_classical_cannot_generate_coherence(self):
        """
        H9.3: Classical computation cannot generate coherence.
        
        Evidence: Permutations and measurements are free operations.
        """
        experiment = ResourceInjectionExperiment()
        results = experiment.run_classical_computation_test()
        
        assert results['supports_H9_3']
    
    def test_h9_4_coherence_as_quantumness_indicator(self):
        """
        H9.4: Coherence generation ability is an indicator of quantumness.
        
        Evidence: Only Hadamard (quantum gate) can create coherence.
        Classical operations cannot.
        """
        measure = L1Coherence()
        rho_0 = pure_state_density(np.array([1, 0]))
        
        # Classical operations
        NOT = PermutationOperation(np.array([[0, 1], [1, 0]]))
        dephase = DephasingOperation()
        
        # Quantum operation
        hadamard = HadamardOperation()
        
        # Classical cannot create coherence from |0⟩
        rho_not = NOT.apply(rho_0)
        rho_dephase = dephase.apply(rho_0)
        
        assert measure.measure(rho_not) < 0.1
        assert measure.measure(rho_dephase) < 0.1
        
        # Hadamard CAN create coherence
        rho_h = hadamard.apply(rho_0)
        assert measure.measure(rho_h) > 0.9
        
        # Coherence generation distinguishes quantum from classical


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

