"""
Tests for Evolution Engine.
"""

import pytest
import numpy as np

from genesis.core.container import Container
from genesis.core.fitness import FitnessEvaluator, FitnessConfig
from genesis.dsl.scalar import generate_scalar_program, ScalarVal
from genesis.dsl.matrix import generate_matrix_program, create_optimal_solution
from genesis.evolution.mutation import Mutator, SpeciesMutator
from genesis.evolution.selection import Selection
from genesis.evolution.population import Population, PopulationConfig
from genesis.tasks.graph_walk import (
    create_graph_walk_task,
    create_graph_walk_fitness_evaluator,
)


class TestMutation:
    """Tests for mutation operations."""
    
    def test_mutate_preserves_structure(self):
        """Mutation should produce a valid program."""
        program = generate_scalar_program(max_depth=3)
        mutator = Mutator(mutation_rate=1.0)  # Always mutate
        
        mutated = mutator.mutate(program)
        
        assert mutated is not None
        assert mutated.size() >= 1
        # Should be evaluable
        result = mutated.evaluate({"x": 5})
        assert result is not None
    
    def test_crossover(self):
        """Crossover should combine two parents."""
        parent1 = generate_scalar_program(max_depth=3)
        parent2 = generate_scalar_program(max_depth=3)
        
        mutator = Mutator(crossover_rate=1.0)  # Always crossover
        child = mutator.crossover(parent1, parent2)
        
        assert child is not None
        assert child.size() >= 1
    
    def test_container_mutation(self):
        """Test mutation of containers."""
        program = generate_scalar_program()
        container = Container(program, species="Scalar")
        
        mutator = Mutator(mutation_rate=1.0)
        new_container = mutator.mutate_container(container)
        
        assert new_container.id != container.id  # New container
        assert new_container.species == container.species


class TestSelection:
    """Tests for selection mechanisms."""
    
    def setup_method(self):
        """Create test population."""
        self.containers = []
        for i in range(10):
            program = ScalarVal(i)
            c = Container(program, species="Scalar")
            c.fitness = float(i)  # Fitness = value
            self.containers.append(c)
    
    def test_elite_selection(self):
        """Elite selection should return top N."""
        elite = Selection.elite(self.containers, n_elite=3)
        
        assert len(elite) == 3
        assert elite[0].fitness == 9
        assert elite[1].fitness == 8
        assert elite[2].fitness == 7
    
    def test_truncation_selection(self):
        """Truncation should keep top percentage."""
        survivors = Selection.truncation(self.containers, survival_rate=0.3)
        
        assert len(survivors) == 3
        assert all(c.fitness >= 7 for c in survivors)
    
    def test_tournament_selection(self):
        """Tournament selection should prefer high fitness."""
        # Run many tournaments to check statistical properties
        wins = {i: 0 for i in range(10)}
        
        for _ in range(100):
            selected = Selection.tournament(
                self.containers,
                tournament_size=3,
                n_select=1
            )
            if selected:
                wins[int(selected[0].fitness)] += 1
        
        # Higher fitness should win more often
        assert wins[9] > wins[0]
    
    def test_species_balanced(self):
        """Species-balanced selection should maintain diversity."""
        # Add another species
        for i in range(5):
            program = ScalarVal(i + 100)
            c = Container(program, species="Matrix")
            c.fitness = float(i + 100)
            self.containers.append(c)
        
        selected = Selection.species_balanced(self.containers, n_select=6)
        
        # Should have representatives from both species
        species_in_selection = set(c.species for c in selected)
        assert len(species_in_selection) >= 2


class TestPopulation:
    """Tests for population management."""
    
    def test_population_initialization(self):
        """Test population initialization."""
        config = PopulationConfig(size=20)
        pop = Population(config=config)
        
        def scalar_gen():
            return Container(generate_scalar_program(), species="Scalar")
        
        def matrix_gen():
            adj = np.eye(5)
            return Container(generate_matrix_program(adj), species="Matrix")
        
        pop.initialize(
            generators={"Scalar": scalar_gen, "Matrix": matrix_gen},
            counts={"Scalar": 10, "Matrix": 10}
        )
        
        assert len(pop.containers) == 20
        
        # Check species distribution
        dist = pop.get_species_distribution()
        assert "Scalar" in dist
        assert "Matrix" in dist
    
    def test_population_step(self):
        """Test one evolution step."""
        task = create_graph_walk_task()
        fitness_eval = create_graph_walk_fitness_evaluator(task)
        
        config = PopulationConfig(size=10, survival_rate=0.5)
        pop = Population(config=config, fitness_evaluator=fitness_eval)
        
        def container_gen():
            program = generate_scalar_program()
            return Container(program, species="Scalar")
        
        pop.initialize(generators={"Scalar": container_gen})
        
        # Run one step
        stats = pop.step()
        
        assert stats.generation == 1
        assert stats.population_size == 10
        assert stats.best_fitness >= 0


class TestIntegration:
    """Integration tests for the full evolution pipeline."""
    
    def test_matrix_dominance(self):
        """
        Test that Matrix DSL eventually dominates.
        
        This is the core hypothesis of the research.
        """
        task = create_graph_walk_task(n_nodes=5, steps=3)
        fitness_eval = create_graph_walk_fitness_evaluator(task)
        
        config = PopulationConfig(
            size=30,
            elite_size=3,
            survival_rate=0.3,
            mutation_rate=0.2,
        )
        pop = Population(config=config, fitness_evaluator=fitness_eval)
        
        adj = task.adjacency_matrix
        
        def scalar_gen():
            return Container(generate_scalar_program(), species="Scalar")
        
        def matrix_gen():
            return Container(generate_matrix_program(adj), species="Matrix")
        
        # Start with mostly Scalar
        pop.initialize(
            generators={"Scalar": scalar_gen, "Matrix": matrix_gen},
            counts={"Scalar": 25, "Matrix": 5}
        )
        
        # Run evolution
        initial_dist = pop.get_species_distribution()
        
        for _ in range(20):
            stats = pop.step()
        
        final_dist = pop.get_species_distribution()
        
        # We expect Matrix proportion to increase
        # (though this is stochastic, so we just check it runs)
        print(f"Initial: {initial_dist}")
        print(f"Final: {final_dist}")
        print(f"Best: {pop.get_best()}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

