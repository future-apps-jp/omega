"""
Population Management

Manages the population of containers through generations.
"""

from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
import random

from genesis.core.container import Container
from genesis.core.fitness import FitnessEvaluator
from genesis.evolution.mutation import Mutator, SpeciesMutator
from genesis.evolution.selection import Selection


@dataclass
class PopulationConfig:
    """Configuration for population management."""
    size: int = 100
    elite_size: int = 5
    survival_rate: float = 0.2
    mutation_rate: float = 0.1
    crossover_rate: float = 0.3
    species_mutation_rate: float = 0.05


@dataclass
class GenerationStats:
    """Statistics for a generation."""
    generation: int
    population_size: int
    alive_count: int
    dead_count: int
    best_fitness: float
    avg_fitness: float
    best_k: int
    avg_k: float
    best_species: str
    species_counts: Dict[str, int] = field(default_factory=dict)
    best_code: str = ""


class Population:
    """
    Manages the evolution of a population of containers.
    """
    
    def __init__(
        self,
        config: Optional[PopulationConfig] = None,
        fitness_evaluator: Optional[FitnessEvaluator] = None,
        mutator: Optional[Mutator] = None,
    ):
        self.config = config or PopulationConfig()
        self.fitness_evaluator = fitness_evaluator
        self.mutator = mutator or Mutator(
            mutation_rate=self.config.mutation_rate,
            crossover_rate=self.config.crossover_rate,
        )
        
        self.containers: List[Container] = []
        self.generation = 0
        self.history: List[GenerationStats] = []
    
    def initialize(
        self,
        generators: Dict[str, Callable[[], Container]],
        counts: Optional[Dict[str, int]] = None,
    ):
        """
        Initialize the population with containers from different species.
        
        Args:
            generators: Dict mapping species names to generator functions
            counts: Optional dict specifying how many of each species
        """
        self.containers = []
        
        if counts is None:
            # Equal distribution
            per_species = self.config.size // len(generators)
            counts = {species: per_species for species in generators}
        
        for species, generator in generators.items():
            n = counts.get(species, 0)
            for _ in range(n):
                container = generator()
                container.species = species
                self.containers.append(container)
        
        # Fill remaining slots if needed
        while len(self.containers) < self.config.size:
            species = random.choice(list(generators.keys()))
            container = generators[species]()
            container.species = species
            self.containers.append(container)
    
    def evaluate_all(self):
        """Evaluate fitness of all containers."""
        if self.fitness_evaluator is None:
            return
        
        for container in self.containers:
            if container.is_alive:
                self.fitness_evaluator.evaluate(container)
    
    def step(self) -> GenerationStats:
        """
        Execute one generation of evolution.
        
        Returns:
            Statistics for this generation
        """
        self.generation += 1
        
        # Evaluate fitness
        self.evaluate_all()
        
        # Record statistics
        stats = self._compute_stats()
        self.history.append(stats)
        
        # Selection
        survivors = Selection.truncation(
            self.containers,
            self.config.survival_rate
        )
        
        elite = Selection.elite(survivors, self.config.elite_size)
        
        # Generate next generation
        next_gen = [c.copy() for c in elite]  # Keep elite unchanged
        
        while len(next_gen) < self.config.size:
            # Select parents
            parent1 = random.choice(survivors)
            
            # Crossover only within same species
            same_species = [s for s in survivors if s.species == parent1.species]
            if len(same_species) > 1 and random.random() < self.config.crossover_rate:
                parent2 = random.choice(same_species)
                child = self.mutator.crossover_containers(parent1, parent2)
            else:
                child = parent1.copy()
            
            # Mutation
            try:
                child = self.mutator.mutate_container(child)
                child.invalidate_cache()
            except Exception:
                # If mutation fails, just use a copy
                child = parent1.copy()
            
            next_gen.append(child)
        
        self.containers = next_gen
        
        return stats
    
    def _compute_stats(self) -> GenerationStats:
        """Compute statistics for the current generation."""
        alive = [c for c in self.containers if c.is_alive]
        
        if not alive:
            return GenerationStats(
                generation=self.generation,
                population_size=len(self.containers),
                alive_count=0,
                dead_count=len(self.containers),
                best_fitness=0,
                avg_fitness=0,
                best_k=0,
                avg_k=0,
                best_species="None",
            )
        
        fitnesses = [c.fitness for c in alive]
        ks = [c.description_length() for c in alive]
        
        best = max(alive, key=lambda c: c.fitness)
        
        # Count species
        species_counts = {}
        for c in alive:
            species_counts[c.species] = species_counts.get(c.species, 0) + 1
        
        return GenerationStats(
            generation=self.generation,
            population_size=len(self.containers),
            alive_count=len(alive),
            dead_count=len(self.containers) - len(alive),
            best_fitness=best.fitness,
            avg_fitness=sum(fitnesses) / len(fitnesses),
            best_k=best.description_length(),
            avg_k=sum(ks) / len(ks),
            best_species=best.species,
            species_counts=species_counts,
            best_code=best.code(),
        )
    
    def get_best(self) -> Optional[Container]:
        """Get the best container by fitness."""
        alive = [c for c in self.containers if c.is_alive]
        if not alive:
            return None
        return max(alive, key=lambda c: c.fitness)
    
    def get_species_distribution(self) -> Dict[str, float]:
        """Get the distribution of species in the population."""
        alive = [c for c in self.containers if c.is_alive]
        if not alive:
            return {}
        
        counts = {}
        for c in alive:
            counts[c.species] = counts.get(c.species, 0) + 1
        
        total = len(alive)
        return {species: count / total for species, count in counts.items()}
    
    def inject_species(
        self,
        species: str,
        generator: Callable[[], Container],
        count: int,
    ):
        """
        Inject new individuals of a species into the population.
        
        This simulates the emergence of a new DSL (mutation event).
        """
        for _ in range(count):
            container = generator()
            container.species = species
            
            # Replace a random low-fitness individual
            alive = [c for c in self.containers if c.is_alive]
            if alive:
                worst = min(alive, key=lambda c: c.fitness)
                idx = self.containers.index(worst)
                self.containers[idx] = container
            else:
                self.containers.append(container)

