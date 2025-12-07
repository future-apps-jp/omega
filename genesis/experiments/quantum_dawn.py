#!/usr/bin/env python3
"""
Phase 25: "The Quantum Dawn" Experiment

This experiment demonstrates that Matrix DSLs (quantum-like structures)
naturally dominate over Scalar DSLs under the selection pressure of
minimal description length.

The task is graph walk: count paths from node A to node B in k steps.
- Scalar DSL: Must enumerate paths, O(k) or O(N) description
- Matrix DSL: A^k gives all paths at once, O(1) description

Expected result: Matrix DSL dominates after ~50-100 generations.
"""

import sys
import os
import argparse
import time
from datetime import datetime
from typing import Dict, List, Tuple
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from genesis.core.container import Container
from genesis.core.fitness import FitnessConfig
from genesis.dsl.scalar import generate_scalar_program
from genesis.dsl.matrix import generate_matrix_program, create_optimal_solution
from genesis.evolution.mutation import Mutator
from genesis.evolution.population import Population, PopulationConfig
from genesis.tasks.graph_walk import (
    create_graph_walk_task,
    create_graph_walk_fitness_evaluator,
    GraphWalkTask,
)


def create_generators(task: GraphWalkTask) -> Dict:
    """Create generator functions for each species."""
    adj = task.adjacency_matrix
    
    def scalar_gen():
        program = generate_scalar_program(max_depth=4)
        return Container(program, species="Scalar")
    
    def matrix_gen():
        program = generate_matrix_program(adj, max_depth=3)
        return Container(program, species="Matrix")
    
    return {"Scalar": scalar_gen, "Matrix": matrix_gen}


def run_experiment(
    n_nodes: int = 5,
    steps: int = 3,
    population_size: int = 50,
    generations: int = 100,
    initial_matrix_ratio: float = 0.2,
    mutation_rate: float = 0.15,
    verbose: bool = True,
) -> Dict:
    """
    Run the Quantum Dawn experiment.
    
    Args:
        n_nodes: Number of nodes in the graph
        steps: Number of steps for the walk
        population_size: Size of the population
        generations: Number of generations to evolve
        initial_matrix_ratio: Initial fraction of Matrix DSL
        mutation_rate: Mutation rate
        verbose: Print progress
    
    Returns:
        Dict with experiment results
    """
    print("=" * 60)
    print("Genesis-Matrix: The Quantum Dawn")
    print("=" * 60)
    
    # Create task
    task = create_graph_walk_task(
        n_nodes=n_nodes,
        steps=steps,
        start_node=0,
        end_node=n_nodes - 1,
    )
    print(f"\nTask: {task}")
    print(f"Target Value: {task.target_value}")
    
    # Show optimal solution
    optimal = create_optimal_solution(steps=steps)
    optimal_k = optimal.size()
    print(f"\nOptimal Solution: {optimal}")
    print(f"Optimal K: {optimal_k}")
    
    # Verify optimal solution
    result = optimal.evaluate({"A": task.adjacency_matrix})
    print(f"Optimal Result: {result}")
    assert result == task.target_value, "Optimal solution is incorrect!"
    
    # Create fitness evaluator
    fitness_config = FitnessConfig(
        k_penalty=1.0,  # Strong K penalty
        time_penalty=0.01,
        accuracy_weight=1000.0,
    )
    fitness_eval = create_graph_walk_fitness_evaluator(task, fitness_config)
    
    # Create population
    pop_config = PopulationConfig(
        size=population_size,
        elite_size=5,
        survival_rate=0.2,
        mutation_rate=mutation_rate,
        crossover_rate=0.3,
    )
    
    mutator = Mutator(
        mutation_rate=mutation_rate,
        crossover_rate=0.3,
    )
    
    population = Population(
        config=pop_config,
        fitness_evaluator=fitness_eval,
        mutator=mutator,
    )
    
    # Initialize with mostly Scalar
    generators = create_generators(task)
    n_matrix = int(population_size * initial_matrix_ratio)
    n_scalar = population_size - n_matrix
    
    population.initialize(
        generators=generators,
        counts={"Scalar": n_scalar, "Matrix": n_matrix},
    )
    
    print(f"\nInitial Population: {n_scalar} Scalar, {n_matrix} Matrix")
    
    # Evolution loop
    print("\n" + "-" * 60)
    print("Evolution Progress")
    print("-" * 60)
    
    history = []
    start_time = time.time()
    
    for gen in range(1, generations + 1):
        stats = population.step()
        history.append(stats)
        
        # Inject some matrix individuals periodically to simulate mutation
        if gen == 10:
            print("\n>>> [MUTATION EVENT] Matrix operations introduced!")
            population.inject_species(
                "Matrix",
                generators["Matrix"],
                count=int(population_size * 0.1)
            )
        
        # Print progress
        if verbose and (gen % 10 == 0 or gen <= 5 or stats.best_fitness > 500):
            dist = population.get_species_distribution()
            scalar_pct = dist.get("Scalar", 0) * 100
            matrix_pct = dist.get("Matrix", 0) * 100
            
            print(f"Gen {gen:3d} | "
                  f"Fitness: {stats.best_fitness:7.2f} | "
                  f"K: {stats.best_k:2d} | "
                  f"Scalar: {scalar_pct:5.1f}% | "
                  f"Matrix: {matrix_pct:5.1f}% | "
                  f"[{stats.best_species}]")
            
            if stats.best_k <= optimal_k + 2:
                print(f"  Best: {stats.best_code}")
        
        # Check for convergence
        if stats.best_fitness > 900 and "Matrix" in stats.best_species:
            print(f"\n!!! CONVERGENCE at Gen {gen} !!!")
            print(f"Matrix DSL has discovered the optimal structure!")
            break
    
    elapsed = time.time() - start_time
    
    # Final results
    print("\n" + "=" * 60)
    print("Final Results")
    print("=" * 60)
    
    final_dist = population.get_species_distribution()
    best = population.get_best()
    
    print(f"\nFinal Species Distribution:")
    for species, pct in final_dist.items():
        print(f"  {species}: {pct * 100:.1f}%")
    
    print(f"\nBest Individual:")
    print(f"  Species: {best.species}")
    print(f"  Fitness: {best.fitness:.2f}")
    print(f"  K: {best.description_length()}")
    print(f"  Code: {best.code()}")
    
    # Verify best result
    best_result = best.evaluate({"A": task.adjacency_matrix})
    print(f"  Result: {best_result}")
    print(f"  Target: {task.target_value}")
    print(f"  Error: {abs(task.target_value - float(best_result) if not isinstance(best_result, np.ndarray) else 'N/A')}")
    
    print(f"\nExecution Time: {elapsed:.2f}s")
    
    # Conclusion
    print("\n" + "-" * 60)
    dominant_species = max(final_dist, key=final_dist.get)
    if dominant_species == "Matrix" and final_dist["Matrix"] > 0.8:
        print("[CONCLUSION] Matrix DSL (A1) dominated the universe!")
        print("Quantum-like structures emerged through pure selection pressure.")
    elif final_dist.get("Matrix", 0) > final_dist.get("Scalar", 0):
        print("[CONCLUSION] Matrix DSL is winning but not yet dominant.")
        print("More generations may be needed for full convergence.")
    else:
        print("[CONCLUSION] Scalar DSL still dominates.")
        print("Consider adjusting parameters or running longer.")
    print("-" * 60)
    
    return {
        "task": str(task),
        "target": task.target_value,
        "generations": gen,
        "elapsed_time": elapsed,
        "final_distribution": final_dist,
        "best_species": best.species,
        "best_fitness": best.fitness,
        "best_k": best.description_length(),
        "best_code": best.code(),
        "history": [(h.generation, h.best_fitness, h.best_k, h.best_species) 
                   for h in history],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Genesis-Matrix: The Quantum Dawn Experiment"
    )
    parser.add_argument("--nodes", type=int, default=5,
                       help="Number of nodes in the graph")
    parser.add_argument("--steps", type=int, default=3,
                       help="Number of steps for the walk")
    parser.add_argument("--population", type=int, default=50,
                       help="Population size")
    parser.add_argument("--generations", type=int, default=100,
                       help="Number of generations")
    parser.add_argument("--matrix-ratio", type=float, default=0.2,
                       help="Initial ratio of Matrix DSL")
    parser.add_argument("--mutation-rate", type=float, default=0.15,
                       help="Mutation rate")
    parser.add_argument("--quiet", action="store_true",
                       help="Reduce output")
    
    args = parser.parse_args()
    
    results = run_experiment(
        n_nodes=args.nodes,
        steps=args.steps,
        population_size=args.population,
        generations=args.generations,
        initial_matrix_ratio=args.matrix_ratio,
        mutation_rate=args.mutation_rate,
        verbose=not args.quiet,
    )
    
    return results


if __name__ == "__main__":
    main()

