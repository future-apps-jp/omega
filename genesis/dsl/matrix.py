"""
Species B: Matrix DSL (The Quantum/A1 Mutants)

A DSL with matrix operations: MATMUL, TRANSPOSE, GET.
These are the "quantum beings" that can compute in superposition.

For the graph walk task, they can express A^k in O(1) description length,
giving them a massive advantage over scalar DSLs.
"""

from typing import Any, Dict, List, Optional
import random
import numpy as np

from genesis.core.dsl import Gene, ValueGene, VariableGene, OperatorGene, DSL


class MatrixOp(Gene):
    """
    A gene holding a matrix value.
    
    This is the key primitive that enables "superposition" -
    representing multiple states/paths simultaneously.
    """
    
    def __init__(self, matrix: Optional[np.ndarray] = None, var_name: Optional[str] = None):
        super().__init__()
        self.matrix = matrix
        self.var_name = var_name  # If set, retrieve from context
    
    def evaluate(self, context: Dict[str, Any]) -> Any:
        if self.var_name and self.var_name in context:
            return context[self.var_name]
        return self.matrix
    
    def size(self) -> int:
        # Matrix as a primitive has K=1 (A1's strength)
        return 1
    
    def copy(self) -> 'MatrixOp':
        matrix_copy = self.matrix.copy() if self.matrix is not None else None
        return MatrixOp(matrix_copy, self.var_name)
    
    def __str__(self) -> str:
        if self.var_name:
            return self.var_name
        return "M"


class MatMulOp(OperatorGene):
    """
    Matrix multiplication operation.
    
    This is the quantum evolution operator - it applies a transformation
    to a state, or in graph terms, computes all k-step paths at once.
    """
    
    def __init__(self, left: Gene, right: Gene):
        super().__init__("MATMUL", [left, right])
    
    def evaluate(self, context: Dict[str, Any]) -> Any:
        left_val = self.children[0].evaluate(context)
        right_val = self.children[1].evaluate(context)
        
        try:
            # Handle different types
            if isinstance(left_val, np.ndarray) and isinstance(right_val, np.ndarray):
                return np.dot(left_val, right_val)
            elif isinstance(left_val, np.ndarray):
                return left_val * float(right_val)
            elif isinstance(right_val, np.ndarray):
                return float(left_val) * right_val
            else:
                return float(left_val) * float(right_val)
        except Exception:
            return 0
    
    def copy(self) -> 'MatMulOp':
        return MatMulOp(self.children[0].copy(), self.children[1].copy())
    
    def __str__(self) -> str:
        return f"(MATMUL {self.children[0]} {self.children[1]})"


class MatPowerOp(OperatorGene):
    """
    Matrix power operation: M^k
    
    This is the key insight - for graph walks, A^k gives all k-step paths.
    """
    
    def __init__(self, matrix: Gene, power: Gene):
        super().__init__("MATPOW", [matrix, power])
    
    def evaluate(self, context: Dict[str, Any]) -> Any:
        mat = self.children[0].evaluate(context)
        power = self.children[1].evaluate(context)
        
        try:
            if isinstance(mat, np.ndarray):
                return np.linalg.matrix_power(mat, int(power))
            return mat ** int(power)
        except Exception:
            return 0
    
    def copy(self) -> 'MatPowerOp':
        return MatPowerOp(self.children[0].copy(), self.children[1].copy())
    
    def __str__(self) -> str:
        return f"(MATPOW {self.children[0]} {self.children[1]})"


class GetElementOp(OperatorGene):
    """
    Get element from matrix: M[i, j]
    
    This is the "measurement" operation - extracting a classical
    value from the quantum superposition.
    """
    
    def __init__(self, matrix: Gene, row: Gene, col: Gene):
        super().__init__("GET", [matrix, row, col])
    
    def evaluate(self, context: Dict[str, Any]) -> Any:
        mat = self.children[0].evaluate(context)
        row = self.children[1].evaluate(context)
        col = self.children[2].evaluate(context)
        
        try:
            if isinstance(mat, np.ndarray):
                return mat[int(row), int(col)]
            return mat
        except (IndexError, TypeError):
            return 0
    
    def size(self) -> int:
        # GET has fixed overhead plus children
        return 1 + sum(c.size() for c in self.children)
    
    def copy(self) -> 'GetElementOp':
        return GetElementOp(
            self.children[0].copy(),
            self.children[1].copy(),
            self.children[2].copy()
        )
    
    def __str__(self) -> str:
        return f"(GET {self.children[0]} {self.children[1]} {self.children[2]})"


class MatrixVal(ValueGene):
    """A constant scalar value in matrix context."""
    
    def copy(self) -> 'MatrixVal':
        return MatrixVal(self.value)


# DSL Definition
MatrixDSL = DSL(
    name="Matrix",
    primitives=['MATMUL', 'MATPOW', 'GET', 'MATRIX', 'CONST', 'VAR']
)


def generate_matrix_program(
    adjacency_matrix: Optional[np.ndarray] = None,
    max_depth: int = 3,
    current_depth: int = 0
) -> Gene:
    """
    Generate a random matrix program.
    
    Args:
        adjacency_matrix: The graph adjacency matrix (if available)
        max_depth: Maximum tree depth
        current_depth: Current depth (for recursion)
    
    Returns:
        A random Gene tree
    """
    # Terminal condition
    if current_depth >= max_depth or (current_depth > 0 and random.random() < 0.4):
        choice = random.random()
        if choice < 0.4 and adjacency_matrix is not None:
            return MatrixOp(adjacency_matrix.copy())
        elif choice < 0.6:
            return MatrixOp(var_name="A")  # Reference to context
        else:
            return MatrixVal(random.randint(0, 5))
    
    # Generate operation
    op_type = random.random()
    
    if op_type < 0.3:
        # Matrix multiplication
        return MatMulOp(
            generate_matrix_program(adjacency_matrix, max_depth, current_depth + 1),
            generate_matrix_program(adjacency_matrix, max_depth, current_depth + 1)
        )
    elif op_type < 0.5:
        # Matrix power (the key insight!)
        return MatPowerOp(
            generate_matrix_program(adjacency_matrix, max_depth, current_depth + 1),
            MatrixVal(random.randint(1, 5))
        )
    elif op_type < 0.8:
        # Get element (measurement)
        return GetElementOp(
            generate_matrix_program(adjacency_matrix, max_depth, current_depth + 1),
            MatrixVal(0),  # start node
            MatrixVal(4)   # end node
        )
    else:
        # Scalar value
        return MatrixVal(random.randint(0, 5))


def generate_matrix_individual(
    adjacency_matrix: Optional[np.ndarray] = None,
    depth: int = 3
) -> Gene:
    """Generate a matrix DSL individual for evolution."""
    return generate_matrix_program(adjacency_matrix, max_depth=depth)


def create_optimal_solution(steps: int = 3) -> Gene:
    """
    Create the optimal solution: (GET (MATPOW A steps) 0 4)
    
    This is what evolution should discover.
    """
    return GetElementOp(
        MatPowerOp(
            MatrixOp(var_name="A"),
            MatrixVal(steps)
        ),
        MatrixVal(0),
        MatrixVal(4)
    )

