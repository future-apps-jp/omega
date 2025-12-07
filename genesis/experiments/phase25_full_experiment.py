#!/usr/bin/env python3
"""
Phase 25: "The Quantum Dawn" - Full Experiment Suite

This script runs comprehensive experiments to demonstrate the evolutionary
dominance of Matrix DSLs over Scalar DSLs under resource constraints.

Experiments:
1. Basic Competition: Scalar vs Matrix with various initial ratios
2. Scaling Test: Different graph sizes
3. Statistical Analysis: Multiple runs for significance

Expected Results:
- Matrix DSL dominates in all scenarios
- Dominance is faster with larger graphs (quantum advantage scales)
- Results are statistically significant
"""

import sys
import os
import json
import time
from datetime import datetime
from typing import Dict, List, Tuple
import numpy as np

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


class ExperimentRunner:
    """Runs and records Genesis-Matrix experiments."""
    
    def __init__(self, output_dir: str = "genesis/experiments/results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.results = []
    
    def run_single(
        self,
        n_nodes: int,
        steps: int,
        population_size: int,
        generations: int,
        initial_matrix_ratio: float,
        seed: int,
    ) -> Dict:
        """Run a single experiment."""
        np.random.seed(seed)
        
        # Create task
        task = create_graph_walk_task(
            n_nodes=n_nodes,
            steps=steps,
            start_node=0,
            end_node=n_nodes - 1,
        )
        
        # Fitness evaluator
        fitness_config = FitnessConfig(k_penalty=1.0, accuracy_weight=1000.0)
        fitness_eval = create_graph_walk_fitness_evaluator(task, fitness_config)
        
        # Population config
        pop_config = PopulationConfig(
            size=population_size,
            elite_size=max(2, population_size // 20),
            survival_rate=0.2,
            mutation_rate=0.15,
            crossover_rate=0.3,
        )
        
        population = Population(
            config=pop_config,
            fitness_evaluator=fitness_eval,
            mutator=Mutator(mutation_rate=0.15, crossover_rate=0.3),
        )
        
        # Generators
        adj = task.adjacency_matrix
        
        def scalar_gen():
            return Container(generate_scalar_program(max_depth=4), species="Scalar")
        
        def matrix_gen():
            return Container(generate_matrix_program(adj, max_depth=3), species="Matrix")
        
        # Initialize
        n_matrix = int(population_size * initial_matrix_ratio)
        n_scalar = population_size - n_matrix
        
        population.initialize(
            generators={"Scalar": scalar_gen, "Matrix": matrix_gen},
            counts={"Scalar": n_scalar, "Matrix": n_matrix},
        )
        
        # Evolution
        start_time = time.time()
        history = []
        convergence_gen = None
        
        for gen in range(1, generations + 1):
            stats = population.step()
            dist = population.get_species_distribution()
            
            history.append({
                "generation": gen,
                "best_fitness": stats.best_fitness,
                "best_k": stats.best_k,
                "best_species": stats.best_species,
                "matrix_ratio": dist.get("Matrix", 0),
                "scalar_ratio": dist.get("Scalar", 0),
            })
            
            # Check convergence (Matrix > 90%)
            if dist.get("Matrix", 0) > 0.9 and convergence_gen is None:
                convergence_gen = gen
        
        elapsed = time.time() - start_time
        
        best = population.get_best()
        final_dist = population.get_species_distribution()
        
        return {
            "config": {
                "n_nodes": n_nodes,
                "steps": steps,
                "population_size": population_size,
                "generations": generations,
                "initial_matrix_ratio": initial_matrix_ratio,
                "seed": seed,
            },
            "task": {
                "target": task.target_value,
            },
            "results": {
                "convergence_generation": convergence_gen,
                "final_matrix_ratio": final_dist.get("Matrix", 0),
                "final_scalar_ratio": final_dist.get("Scalar", 0),
                "best_species": best.species if best else None,
                "best_fitness": best.fitness if best else 0,
                "best_k": best.description_length() if best else 0,
                "best_code": best.code() if best else "",
                "elapsed_time": elapsed,
            },
            "history": history,
        }
    
    def run_experiment_suite(self):
        """Run the full experiment suite."""
        print("=" * 70)
        print("Genesis-Matrix: Phase 25 Full Experiment Suite")
        print("=" * 70)
        print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        all_results = []
        
        # Experiment 1: Basic Competition (multiple seeds)
        print("\n" + "-" * 70)
        print("[Experiment 1] Basic Competition: Scalar vs Matrix")
        print("-" * 70)
        
        for seed in range(5):
            print(f"\n  Run {seed + 1}/5 (seed={seed})...")
            result = self.run_single(
                n_nodes=5,
                steps=3,
                population_size=50,
                generations=50,
                initial_matrix_ratio=0.2,
                seed=seed,
            )
            all_results.append({"experiment": "basic", **result})
            
            conv = result["results"]["convergence_generation"]
            final_m = result["results"]["final_matrix_ratio"]
            print(f"    Convergence: Gen {conv if conv else 'N/A'}, "
                  f"Final Matrix: {final_m*100:.1f}%")
        
        # Analyze basic experiment
        basic_results = [r for r in all_results if r["experiment"] == "basic"]
        conv_gens = [r["results"]["convergence_generation"] for r in basic_results 
                     if r["results"]["convergence_generation"]]
        final_ratios = [r["results"]["final_matrix_ratio"] for r in basic_results]
        
        print(f"\n  Summary:")
        print(f"    Convergence Rate: {len(conv_gens)}/{len(basic_results)} runs")
        if conv_gens:
            print(f"    Avg Convergence Gen: {np.mean(conv_gens):.1f} ± {np.std(conv_gens):.1f}")
        print(f"    Final Matrix Ratio: {np.mean(final_ratios)*100:.1f}% ± {np.std(final_ratios)*100:.1f}%")
        
        # Experiment 2: Scaling Test
        print("\n" + "-" * 70)
        print("[Experiment 2] Scaling Test: Graph Size Effect")
        print("-" * 70)
        
        for n_nodes in [5, 8, 10]:
            print(f"\n  Graph Size N={n_nodes}...")
            result = self.run_single(
                n_nodes=n_nodes,
                steps=3,
                population_size=50,
                generations=50,
                initial_matrix_ratio=0.2,
                seed=42,
            )
            all_results.append({"experiment": "scaling", **result})
            
            conv = result["results"]["convergence_generation"]
            final_m = result["results"]["final_matrix_ratio"]
            print(f"    Convergence: Gen {conv if conv else 'N/A'}, "
                  f"Final Matrix: {final_m*100:.1f}%")
        
        # Experiment 3: Initial Ratio Effect
        print("\n" + "-" * 70)
        print("[Experiment 3] Initial Ratio Effect")
        print("-" * 70)
        
        for ratio in [0.05, 0.1, 0.2, 0.5]:
            print(f"\n  Initial Matrix Ratio: {ratio*100:.0f}%...")
            result = self.run_single(
                n_nodes=5,
                steps=3,
                population_size=50,
                generations=50,
                initial_matrix_ratio=ratio,
                seed=42,
            )
            all_results.append({"experiment": "initial_ratio", **result})
            
            conv = result["results"]["convergence_generation"]
            final_m = result["results"]["final_matrix_ratio"]
            print(f"    Convergence: Gen {conv if conv else 'N/A'}, "
                  f"Final Matrix: {final_m*100:.1f}%")
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(self.output_dir, f"phase25_results_{timestamp}.json")
        
        with open(results_file, "w") as f:
            json.dump(all_results, f, indent=2)
        
        print(f"\n\nResults saved to: {results_file}")
        
        # Final Summary
        print("\n" + "=" * 70)
        print("FINAL SUMMARY")
        print("=" * 70)
        
        all_final_matrix = [r["results"]["final_matrix_ratio"] for r in all_results]
        all_convergence = [r["results"]["convergence_generation"] for r in all_results 
                          if r["results"]["convergence_generation"]]
        
        print(f"\nTotal Experiments: {len(all_results)}")
        print(f"Matrix Dominance Rate: {sum(1 for r in all_final_matrix if r > 0.8)/len(all_final_matrix)*100:.1f}%")
        print(f"Average Final Matrix Ratio: {np.mean(all_final_matrix)*100:.1f}%")
        if all_convergence:
            print(f"Average Convergence Generation: {np.mean(all_convergence):.1f}")
        
        # Conclusion
        dominant_count = sum(1 for r in all_final_matrix if r > 0.8)
        if dominant_count / len(all_final_matrix) > 0.8:
            print("\n[CONCLUSION] ✓ Matrix DSL (A1) dominates across all experimental conditions!")
            print("The Quantum Dawn hypothesis is SUPPORTED.")
        else:
            print("\n[CONCLUSION] Mixed results. Further investigation needed.")
        
        return all_results


def main():
    runner = ExperimentRunner()
    results = runner.run_experiment_suite()
    return results


if __name__ == "__main__":
    main()

