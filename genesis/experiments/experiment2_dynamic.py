#!/usr/bin/env python3
"""
Experiment 2 with Dynamic Task Evaluation

Key Changes from Fixed Task:
1. Graph size N ~ Uniform(3, 10)
2. Steps k ~ Uniform(2, 6)
3. Adjacency matrix A regenerated per evaluation
4. Meta-fitness: average over 10 random instances

This prevents scalar DSLs from "memorizing" a single solution.
"""

import sys
import os
import json
from datetime import datetime
from typing import Dict, List, Tuple
import numpy as np
from scipy import stats as scipy_stats
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from genesis.evolution.dsl_evolution import (
    EvolvableDSL,
    EvolvableGene,
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


def evaluate_program_dynamic(
    program: EvolvableGene,
    dsl: EvolvableDSL,
    n_tests: int = 10,
    n_range: Tuple[int, int] = (3, 10),
    k_range: Tuple[int, int] = (2, 6),
) -> Tuple[float, float]:
    """
    Evaluate program on multiple random task instances.
    
    Returns:
        (mean_fitness, success_rate)
    """
    total_fitness = 0.0
    successes = 0
    
    for _ in range(n_tests):
        # Generate random task
        n_nodes = random.randint(n_range[0], n_range[1])
        k = random.randint(k_range[0], k_range[1])
        A = generate_random_graph(n_nodes)
        start_node = 0
        end_node = n_nodes - 1
        target = compute_target(A, k, start_node, end_node)
        
        # Evaluate program
        context = {"A": A, "x": A, "k": k, "N": n_nodes}
        try:
            result = program.evaluate(context, dsl)
            if isinstance(result, np.ndarray):
                if result.shape[0] > start_node and result.shape[1] > end_node:
                    result = result[start_node, end_node]
                else:
                    result = np.sum(result)  # Fallback
            
            error = abs(target - float(result))
            prog_k = program.size()
            
            # Fitness: reward correct answers, penalize complexity
            if error < 0.5:  # Consider correct if within 0.5
                fitness = 1000.0 / (1 + prog_k * 0.5)
                successes += 1
            else:
                fitness = 100.0 / (1 + error + prog_k * 0.5)
        except Exception as e:
            fitness = 0.0
        
        total_fitness += fitness
    
    mean_fitness = total_fitness / n_tests
    success_rate = successes / n_tests
    
    return mean_fitness, success_rate


def run_experiment_2a_dynamic(
    population_size: int = 100,
    generations: int = 500,
    n_seeds: int = 5,
    n_tests: int = 10,
) -> Dict:
    """
    Experiment 2A with Dynamic Task Evaluation (NO injection).
    """
    print("=" * 70)
    print("EXPERIMENT 2A (DYNAMIC): Spontaneous Emergence Test")
    print("=" * 70)
    print(f"Population: {population_size}, Generations: {generations}")
    print(f"Task: N ~ U(3,10), k ~ U(2,6), {n_tests} tests/individual")
    
    all_results = []
    all_histories = []
    
    for seed in range(n_seeds):
        print(f"\n  Run {seed + 1}/{n_seeds} (seed={seed})...")
        np.random.seed(seed)
        random.seed(seed)
        
        dsl = EvolvableDSL("Genesis-2A-Dynamic")
        evolution_engine = DSLEvolutionEngine(dsl, invention_threshold=5, compression_benefit=0.3)
        
        population = [generate_evolvable_program(dsl, max_depth=4) for _ in range(population_size)]
        
        history = []
        inventions = []
        best_fitness_ever = 0
        best_success_rate_ever = 0
        
        for gen in range(1, generations + 1):
            dsl.generation = gen
            
            # Evaluate fitness with dynamic tasks
            fitness_scores = []
            success_rates = []
            for program in population:
                fitness, success_rate = evaluate_program_dynamic(program, dsl, n_tests)
                fitness_scores.append(fitness)
                success_rates.append(success_rate)
            
            best_fitness = max(fitness_scores)
            best_idx = fitness_scores.index(best_fitness)
            best_success = success_rates[best_idx]
            avg_fitness = np.mean(fitness_scores)
            avg_success = np.mean(success_rates)
            best_k = population[best_idx].size()
            
            if best_fitness > best_fitness_ever:
                best_fitness_ever = best_fitness
            if best_success > best_success_rate_ever:
                best_success_rate_ever = best_success
            
            # Try to invent new operator
            if gen % 50 == 0:
                new_op = evolution_engine.maybe_invent_operator(population, fitness_scores)
                if new_op:
                    inventions.append({"generation": gen, "operator": new_op})
            
            history.append({
                "generation": gen,
                "best_fitness": best_fitness,
                "avg_fitness": avg_fitness,
                "best_success_rate": best_success,
                "avg_success_rate": avg_success,
                "best_k": best_k,
                "n_operators": len(dsl.operators),
                "has_matrix": dsl.has_matrix_ops(),
            })
            
            # Selection and reproduction
            paired = list(zip(population, fitness_scores))
            paired.sort(key=lambda x: x[1], reverse=True)
            
            elite_size = max(5, population_size // 10)
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
                print(f"    Gen {gen:4d}: Fitness={best_fitness:.1f}, Success={best_success*100:.0f}%, K={best_k}, Matrix={dsl.has_matrix_ops()}")
        
        result = {
            "seed": seed,
            "n_operators": len(dsl.operators),
            "n_inventions": len(inventions),
            "has_matrix_ops": dsl.has_matrix_ops(),
            "best_fitness": best_fitness_ever,
            "best_success_rate": best_success_rate_ever,
            "final_avg_success": avg_success,
        }
        all_results.append(result)
        all_histories.append(history)
        
        print(f"    Final: Success={best_success_rate_ever*100:.0f}%, Matrix emerged: {dsl.has_matrix_ops()}")
    
    # Summary
    success_rates = [r["best_success_rate"] for r in all_results]
    mean_success = np.mean(success_rates)
    std_success = np.std(success_rates, ddof=1) if len(success_rates) > 1 else 0
    
    print("\n" + "-" * 50)
    print("EXPERIMENT 2A (DYNAMIC) SUMMARY")
    print("-" * 50)
    print(f"Best Success Rate: {mean_success*100:.1f}% ± {std_success*100:.1f}%")
    print(f"Matrix emerged: {sum(1 for r in all_results if r['has_matrix_ops'])}/{len(all_results)}")
    
    return {
        "config": {
            "experiment": "2A_dynamic",
            "description": "Spontaneous Emergence with Dynamic Tasks (NO INJECTION)",
            "population_size": population_size,
            "generations": generations,
            "n_tests": n_tests,
            "n_range": (3, 10),
            "k_range": (2, 6),
            "n_seeds": n_seeds,
        },
        "results": all_results,
        "histories": all_histories,
        "statistics": {
            "mean_success_rate": mean_success,
            "std_success_rate": std_success,
        },
    }


def run_experiment_2b_dynamic(
    population_size: int = 100,
    generations: int = 500,
    injection_gen: int = 200,
    n_seeds: int = 5,
    n_tests: int = 10,
) -> Dict:
    """
    Experiment 2B with Dynamic Task Evaluation (injection at specified gen).
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 2B (DYNAMIC): Simulated Discovery Test")
    print("=" * 70)
    print(f"Population: {population_size}, Generations: {generations}, Injection: Gen {injection_gen}")
    print(f"Task: N ~ U(3,10), k ~ U(2,6), {n_tests} tests/individual")
    
    all_results = []
    all_histories = []
    
    for seed in range(n_seeds):
        print(f"\n  Run {seed + 1}/{n_seeds} (seed={seed})...")
        np.random.seed(seed)
        random.seed(seed)
        
        dsl = EvolvableDSL("Genesis-2B-Dynamic")
        evolution_engine = DSLEvolutionEngine(dsl, invention_threshold=5, compression_benefit=0.3)
        
        population = [generate_evolvable_program(dsl, max_depth=4) for _ in range(population_size)]
        
        history = []
        inventions = []
        pre_injection_fitness = None
        pre_injection_success = None
        post_injection_fitness = None
        post_injection_success = None
        
        for gen in range(1, generations + 1):
            dsl.generation = gen
            
            # Evaluate fitness
            fitness_scores = []
            success_rates = []
            for program in population:
                fitness, success_rate = evaluate_program_dynamic(program, dsl, n_tests)
                fitness_scores.append(fitness)
                success_rates.append(success_rate)
            
            best_fitness = max(fitness_scores)
            best_idx = fitness_scores.index(best_fitness)
            best_success = success_rates[best_idx]
            avg_fitness = np.mean(fitness_scores)
            avg_success = np.mean(success_rates)
            best_k = population[best_idx].size()
            
            # Record pre-injection state
            if gen == injection_gen - 1:
                pre_injection_fitness = best_fitness
                pre_injection_success = best_success
            
            # INJECT matrix operations
            if gen == injection_gen:
                evolution_engine.inject_matrix_hint()
                inventions.append({"generation": gen, "operator": "MATMUL (injected)"})
                inventions.append({"generation": gen, "operator": "MATPOW (injected)"})
                
                # Create some matrix programs
                for i in range(population_size // 5):  # 20% of population
                    # Create MATPOW(x, k) program
                    matrix_program = EvolvableGene(
                        'MATPOW',
                        [EvolvableGene('VAR', value='x'), EvolvableGene('VAR', value='k')]
                    )
                    worst_idx = fitness_scores.index(min(fitness_scores))
                    population[worst_idx] = matrix_program
                    fitness_scores[worst_idx] = 500
            
            # Record post-injection state
            if gen == injection_gen + 5:  # Give 5 gens to stabilize
                post_injection_fitness = best_fitness
                post_injection_success = best_success
            
            # Try to invent
            if gen % 50 == 0 and gen != injection_gen:
                new_op = evolution_engine.maybe_invent_operator(population, fitness_scores)
                if new_op:
                    inventions.append({"generation": gen, "operator": new_op})
            
            history.append({
                "generation": gen,
                "best_fitness": best_fitness,
                "avg_fitness": avg_fitness,
                "best_success_rate": best_success,
                "avg_success_rate": avg_success,
                "best_k": best_k,
                "n_operators": len(dsl.operators),
                "has_matrix": dsl.has_matrix_ops(),
            })
            
            # Selection
            paired = list(zip(population, fitness_scores))
            paired.sort(key=lambda x: x[1], reverse=True)
            
            elite_size = max(5, population_size // 10)
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
            
            if gen % 100 == 0 or gen == 1 or gen == injection_gen or gen == injection_gen + 5:
                marker = " *** INJECTION ***" if gen == injection_gen else ""
                print(f"    Gen {gen:4d}: Fitness={best_fitness:.1f}, Success={best_success*100:.0f}%, K={best_k}{marker}")
        
        # Calculate jumps
        fitness_jump = (post_injection_fitness - pre_injection_fitness) if pre_injection_fitness and post_injection_fitness else None
        success_jump = (post_injection_success - pre_injection_success) if pre_injection_success is not None and post_injection_success is not None else None
        
        result = {
            "seed": seed,
            "n_operators": len(dsl.operators),
            "has_matrix_ops": dsl.has_matrix_ops(),
            "pre_injection_fitness": pre_injection_fitness,
            "post_injection_fitness": post_injection_fitness,
            "fitness_jump": fitness_jump,
            "pre_injection_success": pre_injection_success,
            "post_injection_success": post_injection_success,
            "success_jump": success_jump,
            "final_success": best_success,
        }
        all_results.append(result)
        all_histories.append(history)
        
        print(f"    Fitness Jump: {pre_injection_fitness:.1f} → {post_injection_fitness:.1f} (+{fitness_jump:.1f})")
        print(f"    Success Jump: {pre_injection_success*100:.0f}% → {post_injection_success*100:.0f}% (+{success_jump*100:.0f}%)")
    
    # Statistics
    fitness_jumps = [r["fitness_jump"] for r in all_results if r["fitness_jump"] is not None]
    success_jumps = [r["success_jump"] for r in all_results if r["success_jump"] is not None]
    
    mean_fitness_jump = np.mean(fitness_jumps) if fitness_jumps else 0
    std_fitness_jump = np.std(fitness_jumps, ddof=1) if len(fitness_jumps) > 1 else 0
    mean_success_jump = np.mean(success_jumps) if success_jumps else 0
    std_success_jump = np.std(success_jumps, ddof=1) if len(success_jumps) > 1 else 0
    
    print("\n" + "-" * 50)
    print("EXPERIMENT 2B (DYNAMIC) SUMMARY")
    print("-" * 50)
    print(f"Fitness Jump: +{mean_fitness_jump:.1f} ± {std_fitness_jump:.1f}")
    print(f"Success Jump: +{mean_success_jump*100:.1f}% ± {std_success_jump*100:.1f}%")
    
    return {
        "config": {
            "experiment": "2B_dynamic",
            "description": "Simulated Discovery with Dynamic Tasks (INJECTION)",
            "population_size": population_size,
            "generations": generations,
            "injection_gen": injection_gen,
            "n_tests": n_tests,
            "n_range": (3, 10),
            "k_range": (2, 6),
            "n_seeds": n_seeds,
        },
        "results": all_results,
        "histories": all_histories,
        "statistics": {
            "mean_fitness_jump": mean_fitness_jump,
            "std_fitness_jump": std_fitness_jump,
            "mean_success_jump": mean_success_jump,
            "std_success_jump": std_success_jump,
        },
    }


def main():
    print("=" * 70)
    print("EXPERIMENT 2 WITH DYNAMIC TASK EVALUATION")
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    print("\nKey Changes:")
    print("  - N ~ Uniform(3, 10)")
    print("  - k ~ Uniform(2, 6)")
    print("  - 10 random tests per individual")
    print("  - This prevents 'memorization' by scalar DSLs")
    
    # Run experiments
    exp2a = run_experiment_2a_dynamic(
        population_size=100,
        generations=500,
        n_seeds=5,
        n_tests=10,
    )
    
    exp2b = run_experiment_2b_dynamic(
        population_size=100,
        generations=500,
        injection_gen=200,
        n_seeds=5,
        n_tests=10,
    )
    
    # Save results
    output_dir = "genesis/experiments/results"
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {
        "experiment2a_dynamic": exp2a,
        "experiment2b_dynamic": exp2b,
        "timestamp": timestamp,
    }
    
    results_file = os.path.join(output_dir, f"experiment2_dynamic_{timestamp}.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=lambda x: None if x != x else x)
    
    print(f"\n\nResults saved to: {results_file}")
    
    # LaTeX output
    print("\n" + "=" * 70)
    print("LATEX-READY OUTPUT")
    print("=" * 70)
    
    print("\n% Experiment 2A (Dynamic) - Spontaneous Emergence")
    print("\\begin{table}[ht]")
    print("\\centering")
    print("\\begin{tabular}{cccc}")
    print("\\toprule")
    print("Run & Best Success (\\%) & Operators & Matrix Emerged \\\\")
    print("\\midrule")
    for r in exp2a["results"]:
        emerged = "Yes" if r["has_matrix_ops"] else "No"
        print(f"{r['seed']+1} & {r['best_success_rate']*100:.0f} & {r['n_operators']} & {emerged} \\\\")
    s = exp2a["statistics"]
    print("\\midrule")
    print(f"Mean & {s['mean_success_rate']*100:.1f} $\\pm$ {s['std_success_rate']*100:.1f} & -- & -- \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\caption{Experiment 2A (Dynamic): Spontaneous emergence test with dynamic tasks.}")
    print("\\label{tab:exp2a_dyn}")
    print("\\end{table}")
    
    print("\n% Experiment 2B (Dynamic) - Simulated Discovery")
    print("\\begin{table}[ht]")
    print("\\centering")
    print("\\begin{tabular}{cccccc}")
    print("\\toprule")
    print("Run & Pre-Inj (\\%) & Post-Inj (\\%) & Jump (\\%) & Fitness Jump \\\\")
    print("\\midrule")
    for r in exp2b["results"]:
        pre = r['pre_injection_success']*100 if r['pre_injection_success'] else 0
        post = r['post_injection_success']*100 if r['post_injection_success'] else 0
        jump = r['success_jump']*100 if r['success_jump'] else 0
        fjump = r['fitness_jump'] if r['fitness_jump'] else 0
        print(f"{r['seed']+1} & {pre:.0f} & {post:.0f} & +{jump:.0f} & +{fjump:.0f} \\\\")
    s2 = exp2b["statistics"]
    print("\\midrule")
    print(f"Mean & -- & -- & +{s2['mean_success_jump']*100:.1f} $\\pm$ {s2['std_success_jump']*100:.1f} & +{s2['mean_fitness_jump']:.1f} \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\caption{Experiment 2B (Dynamic): Simulated discovery with injection at Gen 200.}")
    print("\\label{tab:exp2b_dyn}")
    print("\\end{table}")
    
    return results


if __name__ == "__main__":
    main()

