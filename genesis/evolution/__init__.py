"""
Evolution Engine

Provides mechanisms for evolutionary computation:
- Mutation: AST-based structural mutation
- Selection: Fitness-based selection
- Population: Population management
"""

from genesis.evolution.mutation import Mutator
from genesis.evolution.selection import Selection
from genesis.evolution.population import Population

__all__ = ["Mutator", "Selection", "Population"]

