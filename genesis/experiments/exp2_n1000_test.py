#!/usr/bin/env python3
"""
Experiment 2 with N=1000 population - Quick Test
"""

import sys
import os
import json
from datetime import datetime
import numpy as np
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from genesis.evolution.dsl_evolution import (
    EvolvableDSL,
    DSLEvolutionEngine,
    generate_evolvable_program,
)


def generate_random_graph(n_nodes: int, edge_prob: float = 0.4) -> np.ndarray:
    """Generate a random adjacency matrix."""
    A = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j and random.random() < edge_prob:
                A[i, j] = 1
    return A


def compute_target(A: np.ndarray, k: int, start: int, end: int) -> float:
    """Compute (A^k)[start, end] - the number of paths."""
    result = np.linalg.matrix_power(A, k)
    return float(result[start, end])


def evaluate_program_dynamic(program, dsl, n_tests=10):
    """Evaluate program on multiple random task instances."""
    total_fitness = 0.0
    successes = 0
    
    for _ in range(n_tests):
        n_nodes = random.randint(3, 10)
        k = random.randint(2, 6)
        A = generate_random_graph(n_nodes)
        start_node = 0
        end_node = n_nodes - 1
        target = compute_target(A, k, start_node, end_node)
        
        context = {"A": A, "x": A, "k": k, "N": n_nodes}
        try:
            result = program.evaluate(context, dsl)
            if isinstance(result, np.ndarray):
                if result.shape[0] > start_node and result.shape[1] > end_node:
                    result = result[start_node, end_node]
                else:
                    result = np.sum(result)
            
            error = abs(target - float(result))
            prog_k = program.size()
            
            if error < 0.5:
                fitness = 1000.0 / (1 + prog_k * 0.5)
                successes += 1
            else:
                fitness = 100.0 / (1 + error + prog_k * 0.5)
        except:
            fitness = 0.0
        
        total_fitness += fitness
    
    return total_fitness / n_tests, successes / n_tests


def run_experiment2a_n1000(generations=500, n_seeds=3):
    """Experiment 2A with N=1000 population."""
    population_size = 1000
    n_tests = 10
    
    print("\n" + "=" * 70)
    print("EXPERIMENT 2A (N=1000): Spontaneous Emergence Test")
    print("=" * 70)
    print(f"Population: {population_size}, Generations: {generations}")
    print(f"Task: N ~ U(3,10), k ~ U(2,6), {n_tests} tests/individual")
    
    all_results = []
    
    for seed in range(n_seeds):
        print(f"\n  Run {seed + 1}/{n_seeds} (seed={seed})...")
        np.random.seed(seed)
        random.seed(seed)
        
        dsl = EvolvableDSL("Genesis-2A-N1000")
        evolution_engine = DSLEvolutionEngine(dsl, invention_threshold=5, compression_benefit=0.3)
        
        population = [generate_evolvable_program(dsl, max_depth=4) for _ in range(population_size)]
        print(f"    Initial population: {len(population)} individuals")
        
        inventions = []
        best_success_ever = 0
        
        for gen in range(1, generations + 1):
            dsl.generation = gen
            
            # Evaluate all individuals
            fitness_scores = []
            success_rates = []
            for program in population:
                fitness, success = evaluate_program_dynamic(program, dsl, n_tests)
                fitness_scores.append(fitness)
                success_rates.append(success)
            
            best_fitness = max(fitness_scores)
            best_idx = fitness_scores.index(best_fitness)
            best_success = success_rates[best_idx]
            avg_success = np.mean(success_rates)
            
            if best_success > best_success_ever:
                best_success_ever = best_success
            
            # Invention check every 50 generations
            if gen % 50 == 0:
                new_op = evolution_engine.maybe_invent_operator(population, fitness_scores)
                if new_op:
                    inventions.append({"generation": gen, "operator": new_op})
            
            # Selection
            paired = list(zip(population, fitness_scores))
            paired.sort(key=lambda x: x[1], reverse=True)
            elite_size = max(10, population_size // 10)  # 10% elite
            survivors = [p.copy() for p, f in paired[:elite_size]]
            
            next_gen = survivors[:]
            while len(next_gen) < population_size:
                parent = random.choice(survivors)
                child = parent.copy()
                if random.random() < 0.2:
                    nodes = child.get_all_nodes()
                    if nodes:
                        target = random.choice(nodes)
                        new_sub = generate_evolvable_program(dsl, max_depth=2)
                        target.op = new_sub.op
                        target.children = new_sub.children
                        target.value = new_sub.value
                next_gen.append(child)
            
            population = next_gen
            
            if gen % 100 == 0 or gen == 1:
                print(f"    Gen {gen:4d}: Success={best_success*100:.0f}%, Avg={avg_success*100:.1f}%, Ops={len(dsl.operators)}, Matrix={dsl.has_matrix_ops()}")
        
        # Final statistics
        result = {
            "seed": seed,
            "final_success": best_success_ever,
            "final_operators": len(dsl.operators),
            "has_matrix": dsl.has_matrix_ops(),
            "inventions": len(inventions),
        }
        all_results.append(result)
        print(f"    Final: Success={best_success_ever*100:.0f}%, Ops={len(dsl.operators)}, Matrix={dsl.has_matrix_ops()}")
    
    # Summary
    print("\n" + "-" * 50)
    print("EXPERIMENT 2A (N=1000) SUMMARY")
    print("-" * 50)
    final_successes = [r["final_success"] for r in all_results]
    has_matrix = sum(1 for r in all_results if r["has_matrix"])
    print(f"Runs with Matrix emergence: {has_matrix}/{len(all_results)}")
    print(f"Final success rate: {np.mean(final_successes)*100:.1f}% ± {np.std(final_successes)*100:.1f}%")
    
    return all_results


if __name__ == "__main__":
    print("=" * 70)
    print("EXPERIMENT 2 WITH N=1000 POPULATION")
    print("=" * 70)
    
    # Run only Experiment 2A for quick test (3 seeds, 500 generations)
    results = run_experiment2a_n1000(generations=500, n_seeds=3)
    
    # Save results
    output_file = f"exp2a_n1000_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_file}")

