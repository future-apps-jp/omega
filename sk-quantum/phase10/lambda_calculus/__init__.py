"""
Phase 10: Lambda Calculus Analysis

This package provides tools for analyzing lambda calculus
to verify the universality of our quantum structure results.

Key modules:
- lambda_core: Core lambda calculus implementation
- lambda_analysis: Algebraic and GPT analysis
"""

from .lambda_core import (
    # AST
    Term, Var, Abs, App,
    
    # Parsing
    parse,
    
    # Reduction
    beta_reduce, beta_reduce_step, is_normal_form,
    ReductionStrategy,
    
    # Standard combinators
    I, K, S, TRUE, FALSE,
    church_numeral,
    
    # Multiway graph
    LambdaMultiwayGraph, LambdaNode,
)

from .lambda_analysis import (
    # Algebraic analysis
    LambdaAlgebraicProperties,
    analyze_reduction_algebra,
    check_symplectic_embedding,
    
    # GPT analysis
    create_lambda_gpt,
    analyze_lambda_axioms,
    
    # Resource theory
    analyze_lambda_coherence,
    
    # Comparison
    compare_lambda_sk,
    
    # Main analysis
    run_full_lambda_analysis,
)

__all__ = [
    # Core
    'Term', 'Var', 'Abs', 'App',
    'parse',
    'beta_reduce', 'beta_reduce_step', 'is_normal_form',
    'ReductionStrategy',
    'I', 'K', 'S', 'TRUE', 'FALSE',
    'church_numeral',
    'LambdaMultiwayGraph', 'LambdaNode',
    
    # Analysis
    'LambdaAlgebraicProperties',
    'analyze_reduction_algebra',
    'check_symplectic_embedding',
    'create_lambda_gpt',
    'analyze_lambda_axioms',
    'analyze_lambda_coherence',
    'compare_lambda_sk',
    'run_full_lambda_analysis',
]



