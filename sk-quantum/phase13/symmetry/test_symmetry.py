"""
Tests for Phase 13.1: Group Structure and Spinor Analysis
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from group_structure import (
    PermutationGroup, SpinorAnalysis, SO3Embedding,
    analyze_toffoli_fredkin_group, analyze_symmetric_group
)

from group_structure import toffoli_matrix, fredkin_matrix


class TestPermutationGroup:
    """Test PermutationGroup class."""
    
    def test_identity_group(self):
        """Identity generates trivial group."""
        I = np.eye(4)
        group = PermutationGroup([I], name="trivial")
        assert group.order == 1
    
    def test_swap_group(self):
        """SWAP generates Z_2."""
        SWAP = np.array([
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1]
        ], dtype=float)
        group = PermutationGroup([SWAP], name="Z2")
        assert group.order == 2
    
    def test_toffoli_group_generation(self):
        """Toffoli generates a finite group."""
        T = toffoli_matrix()
        group = PermutationGroup([T], name="Toffoli")
        # Toffoli has order 2 (self-inverse)
        assert group.order == 2
    
    def test_toffoli_fredkin_group(self):
        """Toffoli + Fredkin generate larger group."""
        T = toffoli_matrix()
        F = fredkin_matrix()
        group = PermutationGroup([T, F], name="TF")
        assert group.order > 2


class TestSpinorAnalysis:
    """Test SpinorAnalysis class."""
    
    def test_permutation_eigenvalues_roots_of_unity(self):
        """Permutation matrices have eigenvalues that are roots of unity."""
        T = toffoli_matrix()
        F = fredkin_matrix()
        group = PermutationGroup([T, F])
        
        # All eigenvalues should be on the unit circle (roots of unity)
        for M in group.get_elements_as_matrices():
            eigenvalues = np.linalg.eigvals(M)
            for ev in eigenvalues:
                assert abs(abs(ev) - 1) < 1e-10, f"Eigenvalue {ev} not on unit circle"
    
    def test_no_pm_i_eigenvalues(self):
        """Toffoli/Fredkin group should not have ±i eigenvalues."""
        T = toffoli_matrix()
        F = fredkin_matrix()
        group = PermutationGroup([T, F])
        
        spinor = SpinorAnalysis(group)
        has_pm_i, pm_i_elements = spinor.has_pm_i_eigenvalues()
        
        # The TF group has order 6, so elements have orders 1, 2, 3
        # None of these give ±i eigenvalues (need order 4)
        assert not has_pm_i, f"Found ±i eigenvalues in {len(pm_i_elements)} elements"
    
    def test_no_j_squared_minus_i(self):
        """No J² = -I in permutation group."""
        T = toffoli_matrix()
        F = fredkin_matrix()
        group = PermutationGroup([T, F])
        
        spinor = SpinorAnalysis(group)
        has_j2, J = spinor.find_j_squared_minus_i()
        
        assert not has_j2, "Should not find J² = -I in permutation group"
    
    def test_spinor_obstruction_exists(self):
        """Spinor obstruction should exist for permutation groups."""
        T = toffoli_matrix()
        F = fredkin_matrix()
        group = PermutationGroup([T, F])
        
        so3 = SO3Embedding(group)
        obstruction = so3.spinor_obstruction()
        
        assert obstruction['spinor_obstruction'], "Should have spinor obstruction"


class TestSO3Embedding:
    """Test SO3Embedding analysis."""
    
    def test_element_orders(self):
        """Analyze element orders in group."""
        T = toffoli_matrix()
        F = fredkin_matrix()
        group = PermutationGroup([T, F])
        
        so3 = SO3Embedding(group)
        rotation = so3.check_rotation_subgroup()
        
        assert 'max_element_order' in rotation
        assert rotation['max_element_order'] >= 2  # At least identity and Toffoli


class TestHypothesisH13_1:
    """Test Hypothesis H13.1: Discrete symmetries are limited to permutation groups."""
    
    def test_toffoli_fredkin_is_permutation_group(self):
        """Toffoli/Fredkin group is a subgroup of S_8."""
        T = toffoli_matrix()
        F = fredkin_matrix()
        
        # Verify they are permutation matrices
        for M in [T, F]:
            # Each row/column has exactly one 1
            assert np.allclose(M.sum(axis=0), np.ones(8))
            assert np.allclose(M.sum(axis=1), np.ones(8))
            # All entries are 0 or 1
            assert np.all((M == 0) | (M == 1))
    
    def test_eigenvalues_are_roots_of_unity(self):
        """All eigenvalues are roots of unity."""
        T = toffoli_matrix()
        F = fredkin_matrix()
        group = PermutationGroup([T, F])
        
        for M in group.get_elements_as_matrices():
            eigenvalues = np.linalg.eigvals(M)
            for ev in eigenvalues:
                # Roots of unity have |λ| = 1
                assert abs(abs(ev) - 1) < 1e-10, f"Eigenvalue {ev} is not on unit circle"


class TestHypothesisH13_2:
    """Test Hypothesis H13.2: SO(3) symmetry requires A1."""
    
    def test_no_continuous_approximation(self):
        """Finite permutation groups cannot approximate continuous rotations."""
        T = toffoli_matrix()
        F = fredkin_matrix()
        group = PermutationGroup([T, F])
        
        so3 = SO3Embedding(group)
        rotation = so3.check_rotation_subgroup()
        
        # Finite groups have finite max order
        assert rotation['max_element_order'] < 100
        
        # Note: SO(3) has elements of arbitrarily high order (rotations by 2π/n)
        # A finite permutation group cannot have this property
    
    def test_spinor_requires_complex_structure(self):
        """Spinors require J² = -I which permutations don't have."""
        T = toffoli_matrix()
        F = fredkin_matrix()
        group = PermutationGroup([T, F])
        
        spinor = SpinorAnalysis(group)
        has_j2, _ = spinor.find_j_squared_minus_i()
        
        # This is the KEY result: permutation groups cannot have J² = -I
        assert not has_j2, "Permutation groups should not have J² = -I"
        
        # Therefore, to get spinor representations (needed for spin-1/2),
        # we must extend beyond permutations — this is A1


class TestMainTheorem:
    """Test the main theorem about spinor obstruction."""
    
    def test_full_analysis(self):
        """Run full analysis and verify key results."""
        results = analyze_toffoli_fredkin_group(n_bits=3)
        
        # Key assertions
        assert not results['has_j_squared_minus_i'], "Should not have J² = -I"
        assert results['spinor_obstruction'], "Should have spinor obstruction"
    
    def test_symmetric_group_also_obstructed(self):
        """Even full symmetric group has spinor obstruction in permutation rep."""
        results = analyze_symmetric_group(n=4)
        
        assert not results['has_j_squared_minus_i']
        assert results['spinor_obstruction']


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

