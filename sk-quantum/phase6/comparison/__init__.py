# Phase 6: Three Model Comparison
"""
Comparison framework for:
1. SK computation (irreversible)
2. RCA (reversible, discrete)
3. Quantum circuits (reversible, continuous)
"""

from .model_comparison import ModelComparison, ComparisonResult
from .quantum_circuit import SimpleQuantumCircuit

__all__ = [
    'ModelComparison',
    'ComparisonResult',
    'SimpleQuantumCircuit',
]

