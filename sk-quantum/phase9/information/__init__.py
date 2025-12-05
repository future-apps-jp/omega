"""
Phase 9: Information-Theoretic Approach

This package provides tools for analyzing the relationship between
information-theoretic principles and quantum structure using
Resource Theory of Coherence.

Key modules:
- resource_theory: Coherence as a resource framework
"""

from .resource_theory import (
    # Density matrix operations
    is_density_matrix,
    pure_state_density,
    mixed_state_density,
    
    # Standard states
    computational_basis,
    plus_state,
    minus_state,
    ghz_state,
    
    # Coherence measures
    CoherenceMeasure,
    L1Coherence,
    RelativeEntropyCoherence,
    RobustnessCoherence,
    
    # Resource states
    ResourceState,
    
    # Operations
    FreeOperation,
    PermutationOperation,
    DephasingOperation,
    MeasurementOperation,
    HadamardOperation,
    PhaseGateOperation,
    
    # Experiments
    ResourceInjectionResult,
    ResourceInjectionExperiment,
    
    # Information principles
    check_no_cloning,
    check_no_deleting,
    analyze_information_principles,
)

__all__ = [
    # Density matrix
    'is_density_matrix',
    'pure_state_density',
    'mixed_state_density',
    
    # States
    'computational_basis',
    'plus_state',
    'minus_state',
    'ghz_state',
    
    # Coherence
    'CoherenceMeasure',
    'L1Coherence',
    'RelativeEntropyCoherence',
    'RobustnessCoherence',
    'ResourceState',
    
    # Operations
    'FreeOperation',
    'PermutationOperation',
    'DephasingOperation',
    'MeasurementOperation',
    'HadamardOperation',
    'PhaseGateOperation',
    
    # Experiments
    'ResourceInjectionResult',
    'ResourceInjectionExperiment',
    
    # Information
    'check_no_cloning',
    'check_no_deleting',
    'analyze_information_principles',
]

