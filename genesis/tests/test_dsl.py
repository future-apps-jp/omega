"""
Tests for DSL implementations.
"""

import pytest
import numpy as np

from genesis.core.dsl import ValueGene, VariableGene
from genesis.dsl.scalar import (
    ScalarOp, ScalarVal, ScalarVar,
    generate_scalar_program, ScalarDSL
)
from genesis.dsl.matrix import (
    MatrixOp, MatMulOp, MatPowerOp, GetElementOp, MatrixVal,
    generate_matrix_program, MatrixDSL, create_optimal_solution
)


class TestScalarDSL:
    """Tests for Scalar DSL."""
    
    def test_scalar_val(self):
        """Test scalar constant."""
        gene = ScalarVal(42)
        assert gene.evaluate({}) == 42
        assert gene.size() == 1
    
    def test_scalar_var(self):
        """Test scalar variable."""
        gene = ScalarVar("x")
        assert gene.evaluate({"x": 10}) == 10
        assert gene.evaluate({}) == 0  # Default
    
    def test_scalar_add(self):
        """Test scalar addition."""
        left = ScalarVal(3)
        right = ScalarVal(5)
        add = ScalarOp("ADD", [left, right])
        assert add.evaluate({}) == 8
        assert add.size() == 3
    
    def test_scalar_mul(self):
        """Test scalar multiplication."""
        left = ScalarVal(3)
        right = ScalarVal(5)
        mul = ScalarOp("MUL", [left, right])
        assert mul.evaluate({}) == 15
    
    def test_scalar_nested(self):
        """Test nested scalar expression: (3 + 5) * 2 = 16"""
        add = ScalarOp("ADD", [ScalarVal(3), ScalarVal(5)])
        mul = ScalarOp("MUL", [add, ScalarVal(2)])
        assert mul.evaluate({}) == 16
        assert mul.size() == 5  # MUL + ADD + 3 + 5 + 2
    
    def test_generate_scalar_program(self):
        """Test random scalar program generation."""
        program = generate_scalar_program(max_depth=3)
        assert program is not None
        assert program.size() >= 1
        # Should be evaluable
        result = program.evaluate({"x": 5})
        assert isinstance(result, (int, float))
    
    def test_scalar_copy(self):
        """Test program copying."""
        original = ScalarOp("ADD", [ScalarVal(1), ScalarVal(2)])
        copy = original.copy()
        assert str(original) == str(copy)
        assert original.id != copy.id


class TestMatrixDSL:
    """Tests for Matrix DSL."""
    
    def test_matrix_op(self):
        """Test matrix constant."""
        mat = np.array([[1, 2], [3, 4]])
        gene = MatrixOp(mat)
        result = gene.evaluate({})
        np.testing.assert_array_equal(result, mat)
        assert gene.size() == 1  # Matrix is K=1
    
    def test_matrix_var(self):
        """Test matrix variable reference."""
        mat = np.array([[1, 0], [0, 1]])
        gene = MatrixOp(var_name="A")
        result = gene.evaluate({"A": mat})
        np.testing.assert_array_equal(result, mat)
    
    def test_matmul(self):
        """Test matrix multiplication."""
        mat = np.array([[1, 2], [3, 4]])
        left = MatrixOp(mat)
        right = MatrixOp(mat)
        mul = MatMulOp(left, right)
        
        expected = np.dot(mat, mat)
        result = mul.evaluate({})
        np.testing.assert_array_equal(result, expected)
    
    def test_matpow(self):
        """Test matrix power."""
        mat = np.array([[1, 1], [1, 0]])  # Fibonacci matrix
        matrix_gene = MatrixOp(mat)
        power_gene = MatrixVal(5)
        pow_op = MatPowerOp(matrix_gene, power_gene)
        
        expected = np.linalg.matrix_power(mat, 5)
        result = pow_op.evaluate({})
        np.testing.assert_array_equal(result, expected)
    
    def test_get_element(self):
        """Test element extraction."""
        mat = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        matrix_gene = MatrixOp(mat)
        get = GetElementOp(matrix_gene, MatrixVal(1), MatrixVal(2))
        
        assert get.evaluate({}) == 6  # mat[1, 2]
    
    def test_optimal_solution(self):
        """Test the optimal solution for graph walk."""
        adj = np.array([
            [0, 1, 1, 0, 0],
            [1, 0, 1, 1, 0],
            [1, 1, 0, 1, 0],
            [0, 1, 1, 0, 1],
            [0, 0, 0, 1, 0]
        ])
        
        optimal = create_optimal_solution(steps=3)
        result = optimal.evaluate({"A": adj})
        
        # Verify against ground truth
        expected = np.linalg.matrix_power(adj, 3)[0, 4]
        assert result == expected
        
        # K should be small (GET + MATPOW + M + steps + row + col = 6)
        assert optimal.size() <= 7
    
    def test_generate_matrix_program(self):
        """Test random matrix program generation."""
        adj = np.eye(5)
        program = generate_matrix_program(adj, max_depth=3)
        assert program is not None
        assert program.size() >= 1


class TestDSLComparison:
    """Compare Scalar vs Matrix DSL description lengths."""
    
    def test_description_length_comparison(self):
        """
        Compare K for solving the graph walk task.
        
        Matrix DSL should have significantly smaller K.
        """
        # The optimal Matrix solution
        optimal_matrix = create_optimal_solution(steps=3)
        k_matrix = optimal_matrix.size()
        
        # A typical Scalar solution would be much longer
        # This is a placeholder - actual scalar solutions
        # would need to enumerate paths
        scalar_program = generate_scalar_program(max_depth=5)
        k_scalar = scalar_program.size()
        
        # Matrix K should be small (around 5-7)
        assert k_matrix <= 7, f"Matrix K too large: {k_matrix}"
        
        # Note: We can't guarantee scalar is longer since it's random,
        # but in practice it should be on average
        print(f"Matrix K: {k_matrix}, Scalar K: {k_scalar}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

