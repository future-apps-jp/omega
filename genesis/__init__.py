"""
Genesis-Matrix: Evolutionary Emergence of Quantum Structures

A simulation framework for investigating the evolutionary emergence of
matrix-based DSLs (quantum-like structures) under resource constraints.

Part of Research Plan v5: Artificial Physics
"""

__version__ = "0.1.0"
__author__ = "Hiroshi Kohashiguchi"

from genesis.core.localhost import Localhost
from genesis.core.container import Container
from genesis.core.dsl import DSL, Gene
from genesis.core.fitness import FitnessEvaluator

__all__ = [
    "Localhost",
    "Container", 
    "DSL",
    "Gene",
    "FitnessEvaluator",
]

