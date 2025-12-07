"""
Container: Child Universe

A Container represents a child universe with its own DSL (physical laws)
and phenomena (programs/code running under those laws).

Each container competes for resources based on its efficiency:
- Lower description length K = better
- Faster execution time T = better
- Higher accuracy = better
"""

from typing import Any, Dict, Optional, Callable
from dataclasses import dataclass, field
import uuid
import time

from genesis.core.dsl import Gene


@dataclass
class ContainerStats:
    """Statistics for a container."""
    evaluations: int = 0
    total_time_ms: float = 0.0
    last_result: Any = None
    last_error: Optional[str] = None


class Container:
    """
    A child universe with its own DSL and phenomena.
    
    Attributes:
        id: Unique identifier
        program: The Gene tree representing the "physical law"
        species: Name of the DSL species (e.g., "Scalar", "Matrix")
        fitness: Current fitness score
        is_alive: Whether this container is still active
    """
    
    def __init__(
        self,
        program: Gene,
        species: str = "Unknown",
        task: Optional[Callable] = None,
    ):
        self.id = str(uuid.uuid4())[:8]
        self.program = program
        self.species = species
        self.task = task
        
        self.fitness = 0.0
        self.is_alive = True
        self.death_reason: Optional[str] = None
        
        self.stats = ContainerStats()
        self._cached_k: Optional[int] = None
        self._last_execution_time_ms: float = 0.0
    
    def description_length(self) -> int:
        """Return the Kolmogorov complexity proxy (K)."""
        if self._cached_k is None:
            self._cached_k = self.program.size()
        return self._cached_k
    
    def invalidate_cache(self):
        """Invalidate cached values after mutation."""
        self._cached_k = None
    
    def evaluate(self, context: Optional[Dict[str, Any]] = None) -> Any:
        """
        Execute the program and return the result.
        
        Also updates execution statistics.
        """
        if not self.is_alive:
            return None
        
        context = context or {}
        
        start_time = time.time()
        try:
            result = self.program.evaluate(context)
            self._last_execution_time_ms = (time.time() - start_time) * 1000
            
            self.stats.evaluations += 1
            self.stats.total_time_ms += self._last_execution_time_ms
            self.stats.last_result = result
            self.stats.last_error = None
            
            return result
        except Exception as e:
            self._last_execution_time_ms = (time.time() - start_time) * 1000
            self.stats.last_error = str(e)
            raise
    
    def execution_time_ms(self) -> float:
        """Return the last execution time in milliseconds."""
        return self._last_execution_time_ms
    
    def inflation_rate(self) -> float:
        """
        Calculate the inflation rate: V_inf ∝ 1/(K × T)
        
        Higher inflation rate = more efficient = gets more resources.
        """
        k = self.description_length()
        t = max(self._last_execution_time_ms, 0.001)  # Avoid division by zero
        
        # Add a small constant to avoid extreme values
        return 1.0 / (k * t + 0.01)
    
    def mark_dead(self, reason: str):
        """Mark this container as dead."""
        self.is_alive = False
        self.death_reason = reason
        self.fitness = 0.0
    
    def copy(self) -> 'Container':
        """Create a copy of this container with a new ID."""
        new_container = Container(
            program=self.program.copy(),
            species=self.species,
            task=self.task,
        )
        return new_container
    
    def __str__(self) -> str:
        status = "alive" if self.is_alive else f"dead({self.death_reason})"
        return f"Container({self.id}, {self.species}, K={self.description_length()}, fitness={self.fitness:.2f}, {status})"
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def code(self) -> str:
        """Return the program as code."""
        return str(self.program)

