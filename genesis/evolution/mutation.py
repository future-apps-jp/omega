"""
Mutation Engine

Implements AST-based structural mutation for DSL evolution.
Key principle: No intelligence (LLM) involved - pure structural recombination.
"""

from typing import List, Optional, Callable
import random
import copy

from genesis.core.dsl import Gene, ValueGene
from genesis.core.container import Container


class Mutator:
    """
    AST-based mutation engine.
    
    Mutations are purely structural - no semantic understanding.
    This ensures that any emergent structure is truly discovered,
    not designed.
    """
    
    def __init__(
        self,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.3,
        gene_generators: Optional[List[Callable[[], Gene]]] = None,
    ):
        """
        Args:
            mutation_rate: Probability of point mutation
            crossover_rate: Probability of crossover
            gene_generators: Functions to generate new random genes
        """
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.gene_generators = gene_generators or []
    
    def mutate(self, program: Gene) -> Gene:
        """
        Apply mutation to a program.
        
        Mutations:
        1. Point mutation: Replace a random subtree
        2. Value mutation: Change a constant value
        """
        program = program.copy()
        
        if random.random() < self.mutation_rate:
            program = self._point_mutate(program)
        
        return program
    
    def _point_mutate(self, program: Gene) -> Gene:
        """Replace a random subtree with a new random subtree."""
        nodes = program.get_all_nodes()
        
        if not nodes:
            return program
        
        # Select a random node
        target = random.choice(nodes)
        
        # Generate a new subtree
        if self.gene_generators:
            generator = random.choice(self.gene_generators)
            new_subtree = generator()
            
            # Replace the target's content with the new subtree
            target.__dict__.update(new_subtree.__dict__)
        else:
            # Simple value mutation if no generators
            if isinstance(target, ValueGene):
                target.value = random.randint(0, 10)
        
        return program
    
    def crossover(self, parent1: Gene, parent2: Gene) -> Gene:
        """
        Perform crossover between two parents.
        
        Selects a random subtree from parent2 and inserts it
        into a random position in parent1.
        """
        if random.random() > self.crossover_rate:
            return parent1.copy()
        
        child = parent1.copy()
        
        # Get all nodes from both parents
        child_nodes = child.get_all_nodes()
        parent2_nodes = parent2.get_all_nodes()
        
        if not child_nodes or not parent2_nodes:
            return child
        
        # Select random nodes
        target = random.choice(child_nodes)
        donor = random.choice(parent2_nodes)
        
        # Replace target with copy of donor
        donor_copy = donor.copy()
        target.__dict__.update(donor_copy.__dict__)
        
        return child
    
    def mutate_container(self, container: Container) -> Container:
        """Mutate a container's program."""
        new_program = self.mutate(container.program)
        
        new_container = Container(
            program=new_program,
            species=container.species,
            task=container.task,
        )
        
        return new_container
    
    def crossover_containers(
        self,
        parent1: Container,
        parent2: Container
    ) -> Container:
        """Perform crossover between two containers."""
        new_program = self.crossover(parent1.program, parent2.program)
        
        # Inherit species from the more fit parent
        species = parent1.species if parent1.fitness >= parent2.fitness else parent2.species
        
        return Container(
            program=new_program,
            species=species,
            task=parent1.task,
        )


class SpeciesMutator(Mutator):
    """
    A mutator that can introduce new species.
    
    This allows "quantum-like" DSLs to emerge from classical ones.
    """
    
    def __init__(
        self,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.3,
        species_mutation_rate: float = 0.05,
        gene_generators: Optional[List[Callable[[], Gene]]] = None,
        species_generators: Optional[dict] = None,
    ):
        super().__init__(mutation_rate, crossover_rate, gene_generators)
        self.species_mutation_rate = species_mutation_rate
        self.species_generators = species_generators or {}
    
    def mutate_container(self, container: Container) -> Container:
        """Mutate a container, possibly changing species."""
        # Check for species mutation
        if random.random() < self.species_mutation_rate:
            return self._species_mutate(container)
        
        return super().mutate_container(container)
    
    def _species_mutate(self, container: Container) -> Container:
        """Mutate a container to a different species."""
        if not self.species_generators:
            return super().mutate_container(container)
        
        # Select a random species
        new_species = random.choice(list(self.species_generators.keys()))
        generator = self.species_generators[new_species]
        
        new_program = generator()
        
        return Container(
            program=new_program,
            species=new_species,
            task=container.task,
        )

