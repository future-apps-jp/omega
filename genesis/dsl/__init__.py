"""
DSL Implementations

Contains different "species" of DSLs:
- Scalar: Classical scalar operations (ADD, MUL, LOOP)
- Matrix: Quantum-like matrix operations (MATMUL, VECTOR)
"""

from genesis.dsl.scalar import ScalarDSL, ScalarOp, ScalarVal
from genesis.dsl.matrix import MatrixDSL, MatrixOp, MatMulOp, GetElementOp

__all__ = [
    "ScalarDSL",
    "ScalarOp", 
    "ScalarVal",
    "MatrixDSL",
    "MatrixOp",
    "MatMulOp",
    "GetElementOp",
]

