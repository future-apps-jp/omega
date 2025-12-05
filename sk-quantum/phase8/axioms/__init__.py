"""
Phase 8: Minimal Axiom Set Exploration

This package provides tools for analyzing the minimal set of axioms
needed to derive quantum structure from computational substrates.

Key modules:
- gpt_framework: Generalized Probabilistic Theories framework
- axiom_candidates: Axiom candidate formalization and analysis
"""

from .gpt_framework import (
    GPT,
    StateSpace,
    StateSpaceType,
    Effect,
    Measurement,
    Transformation,
    LinearTransformation,
    StateSpaceComparison,
    create_classical_bit_gpt,
    create_qubit_gpt,
    analyze_simplex_to_ball_gap,
)

from .axiom_candidates import (
    AxiomID,
    AxiomStatus,
    AxiomImplicationGraph,
    compare_axiom_status,
    find_minimal_axiom_set_for_quantum,
)

__all__ = [
    # GPT Framework
    'GPT',
    'StateSpace',
    'StateSpaceType',
    'Effect',
    'Measurement',
    'Transformation',
    'LinearTransformation',
    'StateSpaceComparison',
    'create_classical_bit_gpt',
    'create_qubit_gpt',
    'analyze_simplex_to_ball_gap',
    
    # Axiom Analysis
    'AxiomID',
    'AxiomStatus',
    'AxiomImplicationGraph',
    'compare_axiom_status',
    'find_minimal_axiom_set_for_quantum',
]

