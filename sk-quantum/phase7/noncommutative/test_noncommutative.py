"""
Tests for Phase 7: Non-commutativity Analysis
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'phase0'))

from .operators import (
    ExpressionBasis,
    SKAlgebra,
    SOperator,
    KOperator,
    create_sk_algebra,
)
from .commutator import CommutatorAnalysis, run_commutator_analysis
from .superposition import SuperpositionAnalysis, run_superposition_analysis


class TestExpressionBasis:
    """Tests for expression basis."""
    
    def test_basis_creation(self):
        """Test basis creation."""
        basis = ExpressionBasis(max_depth=1, variables=['a'])
        assert basis.dimension > 0
    
    def test_basis_contains_primitives(self):
        """Test that basis contains S, K, and variables."""
        basis = ExpressionBasis(max_depth=1, variables=['a'])
        # Should have at least S, K, a
        assert basis.dimension >= 3
    
    def test_index_lookup(self):
        """Test index lookup."""
        basis = ExpressionBasis(max_depth=1, variables=['a'])
        for i in range(basis.dimension):
            expr = basis.get_expression(i)
            idx = basis.get_index(expr)
            assert idx == i


class TestSKOperators:
    """Tests for SK operators."""
    
    def test_s_operator_creates_matrix(self):
        """Test S operator matrix creation."""
        algebra = create_sk_algebra(max_depth=1)
        S_matrix = algebra.S_op.get_matrix()
        assert S_matrix.shape[0] == algebra.basis.dimension
        assert S_matrix.shape[1] == algebra.basis.dimension
    
    def test_k_operator_creates_matrix(self):
        """Test K operator matrix creation."""
        algebra = create_sk_algebra(max_depth=1)
        K_matrix = algebra.K_op.get_matrix()
        assert K_matrix.shape[0] == algebra.basis.dimension
        assert K_matrix.shape[1] == algebra.basis.dimension
    
    def test_operators_have_nonzero_elements(self):
        """Test that operators are not zero matrices."""
        algebra = create_sk_algebra(max_depth=1)
        S_matrix = algebra.S_op.get_matrix()
        K_matrix = algebra.K_op.get_matrix()
        
        assert np.any(S_matrix != 0)
        assert np.any(K_matrix != 0)


class TestCommutator:
    """Tests for commutator analysis."""
    
    def test_commutator_antisymmetric(self):
        """Test that [S,K] = -[K,S]."""
        algebra = create_sk_algebra(max_depth=2)
        S = algebra.S_op.get_matrix()
        K = algebra.K_op.get_matrix()
        
        SK = S @ K
        KS = K @ S
        
        comm_SK = SK - KS
        comm_KS = KS - SK
        
        assert np.allclose(comm_SK, -comm_KS)
    
    def test_commutator_analysis_structure(self):
        """Test commutator analysis returns expected structure."""
        algebra = create_sk_algebra(max_depth=2)
        analyzer = CommutatorAnalysis(algebra)
        
        result = analyzer.analyze_structure()
        
        assert 'dimension' in result
        assert 'is_nonzero' in result
        assert 'frobenius_norm' in result
        assert 'rank' in result
    
    def test_lie_structure_analysis(self):
        """Test Lie structure analysis."""
        algebra = create_sk_algebra(max_depth=2)
        analyzer = CommutatorAnalysis(algebra)
        
        result = analyzer.analyze_lie_structure()
        
        assert 'antisymmetric' in result
        assert 'jacobi_identity' in result
        assert result['antisymmetric'] is True  # Should always be true


class TestSuperposition:
    """Tests for superposition analysis."""
    
    def test_basis_state_evolution(self):
        """Test basis state evolution analysis."""
        algebra = create_sk_algebra(max_depth=2)
        analyzer = SuperpositionAnalysis(algebra)
        
        result = analyzer.analyze_basis_state_evolution()
        
        assert 'individual_results' in result
        assert 'overall_conclusion' in result
    
    def test_coherence_structure(self):
        """Test coherence structure analysis."""
        algebra = create_sk_algebra(max_depth=2)
        analyzer = SuperpositionAnalysis(algebra)
        
        result = analyzer.analyze_coherence_structure()
        
        assert 'has_coherence_structure' in result
        assert 'interpretation' in result
    
    def test_superposition_requirements(self):
        """Test superposition requirements analysis."""
        result = run_superposition_analysis(max_depth=2)
        
        assert 'requirements' in result
        assert 'conclusion' in result
        assert len(result['requirements']) >= 4


class TestIntegration:
    """Integration tests."""
    
    def test_full_commutator_analysis(self):
        """Test full commutator analysis pipeline."""
        result = run_commutator_analysis(max_depth=2)
        
        assert 'analysis' in result
        assert 'lie_structure' in result
        assert 'interpretations' in result
        assert 'superposition_relation' in result
    
    def test_full_superposition_analysis(self):
        """Test full superposition analysis pipeline."""
        result = run_superposition_analysis(max_depth=2)
        
        assert 'requirements' in result
        assert 'satisfied_count' in result
        assert 'detailed_analysis' in result
    
    def test_sk_algebra_structure(self):
        """Test SK algebra structure analysis."""
        algebra = create_sk_algebra(max_depth=2)
        
        result = algebra.analyze_algebra_structure()
        
        assert 'S' in result
        assert 'K' in result
        assert 'SK_equals_KS' in result
        assert 'commutator_analysis' in result


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

