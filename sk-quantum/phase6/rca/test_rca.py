"""
Tests for RCA implementation.

Tests:
1. RCA correctness and reversibility
2. Graph construction
3. Hamiltonian and quantum walk
4. Group structure analysis
"""

import pytest
import numpy as np

from .automata import (
    ReversibleCellularAutomaton,
    Rule90,
    Rule150,
    RCAState,
    analyze_rca_group,
)
from .graph import RCAGraph
from .hamiltonian import RCAHamiltonian, build_rca_hamiltonian


class TestRCAState:
    """Tests for RCA state."""
    
    def test_state_creation(self):
        """Test RCA state creation."""
        state = RCAState((1, 0, 1, 0))
        assert state.cells == (1, 0, 1, 0)
        assert str(state) == "1010"
    
    def test_state_equality(self):
        """Test state equality."""
        s1 = RCAState((1, 0, 0))
        s2 = RCAState((1, 0, 0))
        s3 = RCAState((0, 1, 0))
        
        assert s1 == s2
        assert s1 != s3
    
    def test_state_hash(self):
        """Test state hashing for use in dict/set."""
        s1 = RCAState((1, 0, 0))
        s2 = RCAState((1, 0, 0))
        
        states = {s1}
        assert s2 in states


class TestReversibleCA:
    """Tests for reversible cellular automaton."""
    
    def test_rule90_evolution(self):
        """Test Rule 90 evolution."""
        rca = Rule90(4)
        state = RCAState((1, 0, 0, 0))
        
        new_state = rca.apply_rule(state)
        # Rule 90: left XOR right
        # Position 0: cells[3] XOR cells[1] = 0 XOR 0 = 0
        # Position 1: cells[0] XOR cells[2] = 1 XOR 0 = 1
        # Position 2: cells[1] XOR cells[3] = 0 XOR 0 = 0
        # Position 3: cells[2] XOR cells[0] = 0 XOR 1 = 1
        assert new_state.cells == (0, 1, 0, 1)
    
    def test_reversibility(self):
        """Test that second-order RCA is reversible."""
        rca = Rule90(4)
        initial = RCAState((1, 0, 0, 0))
        previous = RCAState((0, 0, 0, 0))
        
        # Evolve forward
        new, new_prev = rca.evolve(initial, previous)
        
        # Reverse
        rev_curr, rev_prev = rca.reverse(new_prev, new)
        
        assert rev_curr == initial
        assert rev_prev == previous
    
    def test_orbit_closure(self):
        """Test that orbits eventually close."""
        rca = Rule90(3)
        initial = RCAState((1, 0, 0))
        previous = RCAState((0, 0, 0))
        
        orbit = rca.generate_orbit(initial, previous, max_steps=100)
        
        # Orbit should close (return to initial)
        assert len(orbit) < 100, "Orbit should close within 100 steps"
    
    def test_transition_matrix_is_permutation(self):
        """Test that transition matrix is a permutation."""
        rca = Rule90(2)
        matrix = rca.get_transition_matrix(include_history=True)
        
        # Permutation: exactly one 1 per row and column
        assert np.allclose(np.sum(matrix, axis=0), 1)
        assert np.allclose(np.sum(matrix, axis=1), 1)
        
        # Orthogonal
        assert np.allclose(matrix @ matrix.T, np.eye(len(matrix)))


class TestRCAGraph:
    """Tests for RCA graph construction."""
    
    def test_graph_from_orbit(self):
        """Test graph construction from orbit."""
        rca = Rule90(3)
        graph = RCAGraph(rca)
        
        initial = RCAState((1, 0, 0))
        previous = RCAState((0, 0, 0))
        graph.build_from_initial(initial, previous)
        
        assert graph.num_nodes > 0
        assert graph.num_edges > 0
    
    def test_adjacency_matrix_symmetric(self):
        """Test that undirected adjacency is symmetric."""
        rca = Rule90(3)
        graph = RCAGraph(rca)
        
        initial = RCAState((1, 0, 0))
        previous = RCAState((0, 0, 0))
        graph.build_from_initial(initial, previous)
        
        adj = graph.get_adjacency_matrix()
        assert np.allclose(adj, adj.T)
    
    def test_directed_adjacency_not_symmetric(self):
        """Test that directed adjacency is not necessarily symmetric."""
        rca = Rule90(3)
        graph = RCAGraph(rca)
        
        initial = RCAState((1, 0, 0))
        previous = RCAState((0, 0, 0))
        graph.build_from_initial(initial, previous)
        
        directed = graph.get_directed_adjacency()
        # Should be a valid transition matrix
        assert np.allclose(np.sum(directed, axis=0), 1)  # Column stochastic


class TestRCAHamiltonian:
    """Tests for RCA Hamiltonian."""
    
    def test_hamiltonian_hermitian(self):
        """Test that Hamiltonian is Hermitian."""
        H = build_rca_hamiltonian(3, rule=90, full_space=False)
        
        # Adjacency-based Hamiltonian should be real symmetric
        assert np.allclose(H.H, H.H.T)
    
    def test_quantum_walk_probability_conservation(self):
        """Test that quantum walk preserves total probability."""
        H = build_rca_hamiltonian(3, rule=90, full_space=False)
        times = np.linspace(0, 5, 20)
        
        probs = H.quantum_walk(0, times)
        
        # Total probability should be 1 at each time
        for i in range(len(times)):
            assert np.isclose(np.sum(probs[i]), 1.0, rtol=1e-5)
    
    def test_classical_walk_probability_conservation(self):
        """Test that classical walk preserves total probability."""
        H = build_rca_hamiltonian(3, rule=90, full_space=False)
        times = np.linspace(0, 5, 20)
        
        probs = H.classical_walk(0, times)
        
        # Total probability should be 1 at each time
        for i in range(len(times)):
            assert np.isclose(np.sum(probs[i]), 1.0, rtol=1e-5)
    
    def test_spectral_analysis(self):
        """Test spectral analysis."""
        H = build_rca_hamiltonian(3, rule=90, full_space=False)
        spec = H.spectral_analysis()
        
        assert len(spec.eigenvalues) > 0
        # Eigenvalues of real symmetric matrix are real
        assert np.allclose(spec.eigenvalues.imag, 0)


class TestGroupAnalysis:
    """Tests for group structure analysis."""
    
    def test_rca_group_is_finite(self):
        """Test that RCA generates finite group."""
        rca = Rule90(2)
        props = analyze_rca_group(rca)
        
        assert props['order'] is not None
        assert props['order'] > 0
    
    def test_rca_is_permutation_group(self):
        """Test that RCA transition is permutation."""
        rca = Rule90(2)
        props = analyze_rca_group(rca)
        
        assert props['is_permutation'] is True
        assert props['is_orthogonal'] is True
    
    def test_eigenvalues_are_roots_of_unity(self):
        """Test that eigenvalues are roots of unity (|λ| = 1)."""
        rca = Rule90(2)
        props = analyze_rca_group(rca)
        
        # All eigenvalues should have magnitude 1
        magnitudes = np.abs(props['eigenvalues'])
        assert np.allclose(magnitudes, 1.0)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

