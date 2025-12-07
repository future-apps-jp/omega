"""
Genesis Core Module

Provides the fundamental abstractions for the Genesis simulation:
- Localhost: The parent universe providing computational resources
- Container: Child universes with their own DSLs and phenomena
- DSL: Domain-specific language abstraction
- Fitness: Evaluation and selection mechanisms
"""

from genesis.core.localhost import Localhost
from genesis.core.container import Container
from genesis.core.dsl import DSL, Gene
from genesis.core.fitness import FitnessEvaluator

__all__ = ["Localhost", "Container", "DSL", "Gene", "FitnessEvaluator"]

