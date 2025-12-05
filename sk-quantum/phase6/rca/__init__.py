# RCA - Reversible Cellular Automata
"""
Phase 6: Reversible Cellular Automata implementation for model comparison.

This module implements Rule 90 (reversible) and related RCA variants
to compare with SK computation and quantum circuits.
"""

from .automata import ReversibleCellularAutomaton, Rule90, Rule150
from .graph import RCAGraph
from .hamiltonian import RCAHamiltonian

__all__ = [
    'ReversibleCellularAutomaton',
    'Rule90',
    'Rule150',
    'RCAGraph',
    'RCAHamiltonian',
]

