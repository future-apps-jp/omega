"""
Localhost: The Parent Universe

The Localhost represents the meta-universe that provides computational
resources to child universes (Containers). It acts as a resource manager
without teleological purpose - it simply allocates resources to efficient
processes.

Key concepts:
- Memory bound (Holographic constraint): K <= A
- Inflation rate: V_inf ∝ 1/(K × T)
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import time


@dataclass
class ResourceConfig:
    """Configuration for localhost resources."""
    max_memory: int = 10000  # Maximum total memory units
    max_containers: int = 100  # Maximum number of containers
    timeout_ms: float = 1000.0  # Execution timeout in milliseconds
    holographic_bound: int = 100  # Max description length K


class Localhost:
    """
    The parent universe providing computational resources.
    
    Implements the "resource manager" that allocates resources based on
    efficiency (minimal K, fast execution).
    """
    
    def __init__(self, config: Optional[ResourceConfig] = None):
        self.config = config or ResourceConfig()
        self.containers: List['Container'] = []
        self.generation = 0
        self.total_memory_used = 0
        self.history: List[Dict[str, Any]] = []
    
    def register_container(self, container: 'Container') -> bool:
        """
        Register a new container (child universe).
        
        Returns False if registration fails (e.g., resource limits exceeded).
        """
        if len(self.containers) >= self.config.max_containers:
            return False
        
        # Check holographic bound
        if container.description_length() > self.config.holographic_bound:
            container.mark_dead("Exceeded holographic bound")
            return False
        
        self.containers.append(container)
        return True
    
    def allocate_resources(self) -> Dict[str, float]:
        """
        Allocate resources to containers based on their inflation rates.
        
        Returns a dict mapping container IDs to allocated memory fractions.
        """
        allocations = {}
        total_inflation = 0.0
        
        # Calculate total inflation rate
        for container in self.containers:
            if container.is_alive:
                total_inflation += container.inflation_rate()
        
        # Allocate proportionally
        if total_inflation > 0:
            for container in self.containers:
                if container.is_alive:
                    fraction = container.inflation_rate() / total_inflation
                    allocations[container.id] = fraction
                else:
                    allocations[container.id] = 0.0
        
        return allocations
    
    def step(self) -> Dict[str, Any]:
        """
        Execute one simulation step.
        
        Returns statistics about this step.
        """
        self.generation += 1
        
        # Evaluate all containers
        alive_count = 0
        dead_count = 0
        best_fitness = 0.0
        best_container = None
        
        for container in self.containers:
            if container.is_alive:
                # Check timeout
                start_time = time.time()
                try:
                    container.evaluate()
                    elapsed_ms = (time.time() - start_time) * 1000
                    
                    if elapsed_ms > self.config.timeout_ms:
                        container.mark_dead("Timeout")
                        dead_count += 1
                    else:
                        alive_count += 1
                        if container.fitness > best_fitness:
                            best_fitness = container.fitness
                            best_container = container
                except Exception as e:
                    container.mark_dead(f"Evaluation error: {e}")
                    dead_count += 1
            else:
                dead_count += 1
        
        # Record history
        stats = {
            "generation": self.generation,
            "alive": alive_count,
            "dead": dead_count,
            "best_fitness": best_fitness,
            "best_container_id": best_container.id if best_container else None,
        }
        self.history.append(stats)
        
        return stats
    
    def get_survivors(self, top_k: int = 10) -> List['Container']:
        """Get the top-k containers by fitness."""
        alive = [c for c in self.containers if c.is_alive]
        return sorted(alive, key=lambda c: c.fitness, reverse=True)[:top_k]
    
    def cull(self, survival_rate: float = 0.2) -> int:
        """
        Remove low-fitness containers.
        
        Returns the number of containers culled.
        """
        alive = [c for c in self.containers if c.is_alive]
        if not alive:
            return 0
        
        # Sort by fitness
        alive.sort(key=lambda c: c.fitness, reverse=True)
        
        # Keep top survival_rate
        n_survive = max(1, int(len(alive) * survival_rate))
        
        culled = 0
        for container in alive[n_survive:]:
            container.mark_dead("Culled (low fitness)")
            culled += 1
        
        return culled
    
    def __str__(self) -> str:
        alive = sum(1 for c in self.containers if c.is_alive)
        return f"Localhost(gen={self.generation}, containers={len(self.containers)}, alive={alive})"

