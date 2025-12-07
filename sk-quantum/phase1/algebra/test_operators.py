"""
Tests for SK Reduction Operators Algebra
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'phase0'))

import pytest
import numpy as np

from sk_parser import parse, to_string, to_canonical
from operators import (
    IdentityOp, SReductionOp, KReductionOp, PathReductionOp,
    CompositeOp, SumOp, OperatorAlgebra, MatrixRepresentation,
    CliffordAnalysis, run_algebraic_analysis
)


# =============================================================================
# Basic Operator Tests
# =============================================================================

class TestIdentityOp:
    
    def test_identity_returns_same(self):
        op = IdentityOp()
        expr = parse("S K K a")
        assert to_canonical(op(expr)) == to_canonical(expr)
    
    def test_identity_repr(self):
        op = IdentityOp()
        assert repr(op) == "Î"


class TestSReductionOp:
    
    def test_s_reduction_basic(self):
        op = SReductionOp()
        expr = parse("S a b c")  # S a b c -> (a c) (b c)
        result = op(expr)
        assert result is not None
        assert to_canonical(result) == to_canonical(parse("(a c) (b c)"))
    
    def test_s_reduction_no_redex(self):
        op = SReductionOp()
        expr = parse("K a")  # S-redex がない
        result = op(expr)
        assert result is None
    
    def test_s_reduction_nested(self):
        op = SReductionOp()
        expr = parse("(K a) (S b c d)")  # 内側に S-redex
        result = op(expr)
        assert result is not None


class TestKReductionOp:
    
    def test_k_reduction_basic(self):
        op = KReductionOp()
        expr = parse("K a b")  # K a b -> a
        result = op(expr)
        assert result is not None
        assert to_canonical(result) == "a"
    
    def test_k_reduction_no_redex(self):
        op = KReductionOp()
        expr = parse("S a b")  # K-redex がない
        result = op(expr)
        assert result is None


# =============================================================================
# Composite Operator Tests
# =============================================================================

class TestCompositeOp:
    
    def test_composition(self):
        S_op = SReductionOp()
        K_op = KReductionOp()
        
        # K を先に適用
        composite = CompositeOp(S_op, K_op)
        
        expr = parse("K a b")
        result = composite(expr)
        # K a b -> a, then S に適用しようとするが S-redex がない
        # なので None になる可能性
    
    def test_composition_repr(self):
        S_op = SReductionOp()
        K_op = KReductionOp()
        composite = CompositeOp(S_op, K_op)
        assert "∘" in repr(composite)


class TestSumOp:
    
    def test_sum_first_applicable(self):
        S_op = SReductionOp()
        K_op = KReductionOp()
        sum_op = SumOp(S_op, K_op)
        
        expr = parse("K a b")  # K-redex のみ
        result = sum_op(expr)
        assert result is not None
        assert to_canonical(result) == "a"
    
    def test_sum_apply_all(self):
        S_op = SReductionOp()
        K_op = KReductionOp()
        sum_op = SumOp(S_op, K_op, coefficients=[1.0, 1.0])
        
        expr = parse("K a b")
        results = sum_op.apply_all(expr)
        # S は適用不可、K のみ適用可能
        assert len(results) == 1
        assert results[0][0] == 1.0


# =============================================================================
# Operator Algebra Tests
# =============================================================================

class TestOperatorAlgebra:
    
    def test_generators(self):
        algebra = OperatorAlgebra()
        assert 'I' in algebra.generators
        assert 'S' in algebra.generators
        assert 'K' in algebra.generators
    
    def test_test_relation(self):
        algebra = OperatorAlgebra()
        
        # I と I は同じ
        expr = parse("S K K a")
        assert algebra.test_relation(expr, algebra.I, algebra.I)
        
        # S と K は異なる（S-redex と K-redex の違い）
        # このテストは式依存
    
    def test_find_relations(self):
        algebra = OperatorAlgebra()
        test_exprs = [parse("S a b c"), parse("K a b")]
        relations = algebra.find_relations(test_exprs)
        # 関係式があればリストに含まれる
        assert isinstance(relations, list)


# =============================================================================
# Matrix Representation Tests
# =============================================================================

class TestMatrixRepresentation:
    
    def test_basis_construction(self):
        basis = [parse("S"), parse("K"), parse("a")]
        matrix_rep = MatrixRepresentation(basis)
        assert matrix_rep.dim == 3
    
    def test_identity_matrix(self):
        basis = [parse("S"), parse("K"), parse("a")]
        matrix_rep = MatrixRepresentation(basis)
        
        I_mat = matrix_rep.operator_matrix(IdentityOp())
        # 恒等演算子は単位行列に近い
        expected = np.eye(3)
        assert np.allclose(I_mat, expected)
    
    def test_operator_matrix_shape(self):
        basis = [parse("S"), parse("K"), parse("a"), parse("S K")]
        matrix_rep = MatrixRepresentation(basis)
        
        S_mat = matrix_rep.operator_matrix(SReductionOp())
        assert S_mat.shape == (4, 4)


# =============================================================================
# Clifford Analysis Tests
# =============================================================================

class TestCliffordAnalysis:
    
    def test_anticommutator(self):
        basis = [parse("S"), parse("K"), parse("a")]
        matrix_rep = MatrixRepresentation(basis)
        clifford = CliffordAnalysis(matrix_rep)
        
        A = np.array([[1, 0], [0, -1]])
        B = np.array([[0, 1], [1, 0]])
        anticomm = clifford.anticommutator(A, B)
        expected = A @ B + B @ A
        assert np.allclose(anticomm, expected)
    
    def test_commutator(self):
        basis = [parse("S"), parse("K"), parse("a")]
        matrix_rep = MatrixRepresentation(basis)
        clifford = CliffordAnalysis(matrix_rep)
        
        A = np.array([[1, 0], [0, -1]])
        B = np.array([[0, 1], [1, 0]])
        comm = clifford.commutator(A, B)
        expected = A @ B - B @ A
        assert np.allclose(comm, expected)
    
    def test_check_clifford_relations(self):
        basis = [parse("S"), parse("K"), parse("a"), parse("S K")]
        matrix_rep = MatrixRepresentation(basis)
        clifford = CliffordAnalysis(matrix_rep)
        
        results = clifford.check_clifford_relations()
        assert 'S² = I?' in results
        assert 'K² = I?' in results
        assert '{S, K} = 0?' in results


# =============================================================================
# Integration Test
# =============================================================================

class TestAlgebraicAnalysis:
    
    def test_run_analysis(self):
        results = run_algebraic_analysis(verbose=False)
        assert 'relations' in results
        assert 'imaginary_analysis' in results
        assert 'clifford_analysis' in results
        assert 'has_imaginary_structure' in results
        assert 'is_clifford_like' in results


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])



