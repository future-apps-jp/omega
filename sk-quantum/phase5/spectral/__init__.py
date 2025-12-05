"""
Phase 5: Hamiltonian and Interference
=====================================

計算グラフのスペクトル解析と量子ウォーク
"""

from .hamiltonian import (
    ComputationHamiltonian,
    SpectralAnalysis,
    ContinuousTimeQuantumWalk,
    ClassicalRandomWalk,
    InterferenceAnalysis,
    build_hamiltonian_from_expression,
    analyze_expression,
    run_phase5_analysis,
)

__all__ = [
    'ComputationHamiltonian',
    'SpectralAnalysis',
    'ContinuousTimeQuantumWalk',
    'ClassicalRandomWalk',
    'InterferenceAnalysis',
    'build_hamiltonian_from_expression',
    'analyze_expression',
    'run_phase5_analysis',
]

