"""
Phase 4: Reversible Computation Algebra
=======================================

可逆計算の代数構造を解析するモジュール
"""

from .gates import (
    ReversibleGate,
    NOTGate, CNOTGate, ToffoliGate, FredkinGate, SWAPGate,
    IdentityGate, CompositeGate, EmbeddedGate,
    GateGroup,
    NOT, CNOT, TOFFOLI, FREDKIN, SWAP,
    verify_reversibility, verify_self_inverse, matrix_properties
)

from .group_analysis import (
    GroupAnalyzer, GroupAnalysisResult,
    analyze_toffoli_group, analyze_parity_structure, parity,
    run_group_analysis
)

from .symplectic import (
    SymplecticAnalyzer, SymplecticEmbeddingResult,
    symplectic_form, is_symplectic, symplectic_eigenvalues,
    compare_structures, run_symplectic_analysis
)

__all__ = [
    # Gates
    'ReversibleGate',
    'NOTGate', 'CNOTGate', 'ToffoliGate', 'FredkinGate', 'SWAPGate',
    'IdentityGate', 'CompositeGate', 'EmbeddedGate',
    'GateGroup',
    'NOT', 'CNOT', 'TOFFOLI', 'FREDKIN', 'SWAP',
    'verify_reversibility', 'verify_self_inverse', 'matrix_properties',
    
    # Group Analysis
    'GroupAnalyzer', 'GroupAnalysisResult',
    'analyze_toffoli_group', 'analyze_parity_structure', 'parity',
    'run_group_analysis',
    
    # Symplectic Analysis
    'SymplecticAnalyzer', 'SymplecticEmbeddingResult',
    'symplectic_form', 'is_symplectic', 'symplectic_eigenvalues',
    'compare_structures', 'run_symplectic_analysis',
]

