#!/usr/bin/env python3
"""
All Experiments with Dynamic Task Evaluation

Experiment 1: Matrix DSL vs Scalar DSL Competition
Experiment 2A: Spontaneous Emergence (NO injection)
Experiment 2B: Simulated Discovery (with injection)

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

from genesis.core.container import Container
from genesis.dsl.scalar import generate_scalar_program
from genesis.dsl.matrix import generate_matrix_program
from genesis.evolution.mutation import Mutator
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


# =============================================================================
# EXPERIMENT 1: Matrix DSL vs Scalar DSL Competition (Dynamic)
# =============================================================================

def evaluate_container_dynamic(
    container: Container,
    n_tests: int = 10,
    n_range: Tuple[int, int] = (3, 10),
    k_range: Tuple[int, int] = (2, 6),
) -> Tuple[float, float]:
    """Evaluate a container on multiple random task instances."""
    total_fitness = 0.0
    successes = 0
    
    for _ in range(n_tests):
        n_nodes = random.randint(n_range[0], n_range[1])
        k = random.randint(k_range[0], k_range[1])
        A = generate_random_graph(n_nodes)
        start_node = 0
        end_node = n_nodes - 1
        target = compute_target(A, k, start_node, end_node)
        
        context = {"A": A, "x": A, "k": k, "N": n_nodes, "start": start_node, "end": end_node}
        
        try:
            result = container.program.evaluate(context)
            if isinstance(result, np.ndarray):
                if result.shape[0] > start_node and result.shape[1] > end_node:
                    result = result[start_node, end_node]
                else:
                    result = np.sum(result) % 1000  # Fallback
            
            error = abs(target - float(result))
            prog_k = container.program.size()
            
            if error < 0.5:
                fitness = 1000.0 / (1 + prog_k * 0.5)
                successes += 1
            else:
                fitness = 100.0 / (1 + error + prog_k * 0.5)
        except:
            fitness = 0.0
        
        total_fitness += fitness
    
    return total_fitness / n_tests, successes / n_tests


def run_experiment1_dynamic(
    population_size: int = 100,
    generations: int = 50,
    initial_matrix_ratio: float = 0.2,
    n_seeds: int = 10,
    n_tests: int = 10,
) -> Dict:
    """
    Experiment 1: Matrix DSL vs Scalar DSL Competition with Dynamic Tasks.
    """
    print("=" * 70)
    print("EXPERIMENT 1 (DYNAMIC): Matrix DSL Dominance")
    print("=" * 70)
    print(f"Population: {population_size}, Generations: {generations}, Seeds: {n_seeds}")
    print(f"Task: N ~ U(3,10), k ~ U(2,6), {n_tests} tests/individual")
    print(f"Initial Matrix ratio: {initial_matrix_ratio*100:.0f}%")
    
    all_results = []
    all_histories = []
    
    for seed in range(n_seeds):
        print(f"\n  Run {seed + 1}/{n_seeds} (seed={seed})...", end=" ")
        np.random.seed(seed)
        random.seed(seed)
        
        # Generate a reference adjacency matrix for program generation
        ref_A = generate_random_graph(5)
        
        # Initialize population
        n_matrix = int(population_size * initial_matrix_ratio)
        n_scalar = population_size - n_matrix
        
        population = []
        for _ in range(n_scalar):
            prog = generate_scalar_program(max_depth=4)
            population.append(Container(prog, species="Scalar"))
        for _ in range(n_matrix):
            prog = generate_matrix_program(ref_A, max_depth=3)
            population.append(Container(prog, species="Matrix"))
        
        random.shuffle(population)
        
        history = []
        convergence_gen = None
        
        for gen in range(1, generations + 1):
            # Evaluate fitness
            fitness_scores = []
            success_rates = []
            for container in population:
                fitness, success = evaluate_container_dynamic(container, n_tests)
                fitness_scores.append(fitness)
                success_rates.append(success)
                container.fitness = fitness
            
            # Count species
            n_matrix_curr = sum(1 for c in population if c.species == "Matrix")
            n_scalar_curr = len(population) - n_matrix_curr
            matrix_ratio = n_matrix_curr / len(population)
            
            best_fitness = max(fitness_scores)
            best_idx = fitness_scores.index(best_fitness)
            best_success = success_rates[best_idx]
            
            history.append({
                "generation": gen,
                "matrix_ratio": matrix_ratio,
                "scalar_ratio": 1 - matrix_ratio,
                "best_fitness": best_fitness,
                "best_success": best_success,
            })
            
            if matrix_ratio > 0.9 and convergence_gen is None:
                convergence_gen = gen
            
            # Selection
            paired = list(zip(population, fitness_scores))
            paired.sort(key=lambda x: x[1], reverse=True)
            
            elite_size = max(5, population_size // 20)
            survivors = [c for c, f in paired[:elite_size]]
            
            # Reproduction with mutation
            mutator = Mutator(mutation_rate=0.15, crossover_rate=0.3)
            next_gen = [c.copy() for c in survivors]
            
            while len(next_gen) < population_size:
                parent = random.choice(survivors)
                child = parent.copy()
                # Simple mutation: regenerate program of same species
                if random.random() < 0.15:
                    if child.species == "Scalar":
                        child.program = generate_scalar_program(max_depth=4)
                    else:
                        child.program = generate_matrix_program(ref_A, max_depth=3)
                next_gen.append(child)
            
            population = next_gen
        
        final_matrix_ratio = sum(1 for c in population if c.species == "Matrix") / len(population)
        
        result = {
            "seed": seed,
            "convergence_gen": convergence_gen,
            "final_matrix_ratio": final_matrix_ratio,
            "final_scalar_ratio": 1 - final_matrix_ratio,
            "best_success": max(success_rates),
        }
        all_results.append(result)
        all_histories.append(history)
        
        conv_str = str(convergence_gen) if convergence_gen else "N/A"
        print(f"Conv={conv_str}, Matrix={final_matrix_ratio*100:.0f}%")
    
    # Statistics
    conv_gens = [r["convergence_gen"] for r in all_results if r["convergence_gen"]]
    final_ratios = [r["final_matrix_ratio"] for r in all_results]
    
    if conv_gens:
        conv_mean = np.mean(conv_gens)
        conv_std = np.std(conv_gens, ddof=1) if len(conv_gens) > 1 else 0
        conv_ci = scipy_stats.t.interval(0.95, len(conv_gens)-1, loc=conv_mean, scale=scipy_stats.sem(conv_gens)) if len(conv_gens) > 1 else (conv_mean, conv_mean)
    else:
        conv_mean, conv_std, conv_ci = None, None, (None, None)
    
    ratio_mean = np.mean(final_ratios)
    ratio_std = np.std(final_ratios, ddof=1) if len(final_ratios) > 1 else 0
    
    print("\n" + "-" * 50)
    print("EXPERIMENT 1 (DYNAMIC) SUMMARY")
    print("-" * 50)
    print(f"Convergence Rate: {len(conv_gens)}/{len(all_results)}")
    if conv_mean:
        print(f"Convergence Gen: {conv_mean:.1f} ± {conv_std:.1f} (95% CI: [{conv_ci[0]:.1f}, {conv_ci[1]:.1f}])")
    print(f"Final Matrix %: {ratio_mean*100:.1f}% ± {ratio_std*100:.1f}%")
    
    return {
        "config": {
            "experiment": "1_dynamic",
            "description": "Matrix vs Scalar Competition with Dynamic Tasks",
            "population_size": population_size,
            "generations": generations,
            "initial_matrix_ratio": initial_matrix_ratio,
            "n_tests": n_tests,
            "n_range": (3, 10),
            "k_range": (2, 6),
            "n_seeds": n_seeds,
        },
        "results": all_results,
        "histories": all_histories,
        "statistics": {
            "convergence_rate": len(conv_gens) / len(all_results),
            "convergence_mean": conv_mean,
            "convergence_std": conv_std,
            "convergence_ci_95": conv_ci,
            "final_ratio_mean": ratio_mean,
            "final_ratio_std": ratio_std,
        },
    }


# =============================================================================
# EXPERIMENT 2A: Spontaneous Emergence (Dynamic, NO injection)
# =============================================================================

def evaluate_program_dynamic(
    program: EvolvableGene,
    dsl: EvolvableDSL,
    n_tests: int = 10,
    n_range: Tuple[int, int] = (3, 10),
    k_range: Tuple[int, int] = (2, 6),
) -> Tuple[float, float]:
    """Evaluate program on multiple random task instances."""
    total_fitness = 0.0
    successes = 0
    
    for _ in range(n_tests):
        n_nodes = random.randint(n_range[0], n_range[1])
        k = random.randint(k_range[0], k_range[1])
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


def run_experiment2a_dynamic(
    population_size: int = 100,
    generations: int = 500,
    n_seeds: int = 5,
    n_tests: int = 10,
) -> Dict:
    """Experiment 2A with Dynamic Task Evaluation (NO injection)."""
    print("\n" + "=" * 70)
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
        
        dsl = EvolvableDSL("Genesis-2A")
        evolution_engine = DSLEvolutionEngine(dsl, invention_threshold=5, compression_benefit=0.3)
        
        population = [generate_evolvable_program(dsl, max_depth=4) for _ in range(population_size)]
        
        history = []
        inventions = []
        best_success_ever = 0
        
        for gen in range(1, generations + 1):
            dsl.generation = gen
            
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
            
            if gen % 50 == 0:
                new_op = evolution_engine.maybe_invent_operator(population, fitness_scores)
                if new_op:
                    inventions.append({"generation": gen, "operator": new_op})
            
            history.append({
                "generation": gen,
                "best_fitness": best_fitness,
                "best_success": best_success,
                "avg_success": avg_success,
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
            
            if gen % 100 == 0 or gen == 1:
                print(f"    Gen {gen:4d}: Success={best_success*100:.0f}%, Ops={len(dsl.operators)}, Matrix={dsl.has_matrix_ops()}")
        
        result = {
            "seed": seed,
            "n_operators": len(dsl.operators),
            "has_matrix_ops": dsl.has_matrix_ops(),
            "best_success_rate": best_success_ever,
            "final_avg_success": avg_success,
        }
        all_results.append(result)
        all_histories.append(history)
        
        print(f"    Final: Best Success={best_success_ever*100:.0f}%, Matrix={dsl.has_matrix_ops()}")
    
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
            "n_seeds": n_seeds,
        },
        "results": all_results,
        "histories": all_histories,
        "statistics": {
            "mean_success_rate": mean_success,
            "std_success_rate": std_success,
        },
    }


# =============================================================================
# EXPERIMENT 2B: Simulated Discovery (Dynamic, with injection)
# =============================================================================

def run_experiment2b_dynamic(
    population_size: int = 100,
    generations: int = 500,
    injection_gen: int = 200,
    n_seeds: int = 5,
    n_tests: int = 10,
) -> Dict:
    """Experiment 2B with Dynamic Task Evaluation (with injection)."""
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
        
        dsl = EvolvableDSL("Genesis-2B")
        evolution_engine = DSLEvolutionEngine(dsl, invention_threshold=5, compression_benefit=0.3)
        
        population = [generate_evolvable_program(dsl, max_depth=4) for _ in range(population_size)]
        
        history = []
        pre_injection_success = None
        post_injection_success = None
        
        for gen in range(1, generations + 1):
            dsl.generation = gen
            
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
            
            if gen == injection_gen - 1:
                pre_injection_success = best_success
            
            # INJECT at specified generation
            if gen == injection_gen:
                evolution_engine.inject_matrix_hint()
                for i in range(population_size // 5):
                    matrix_program = EvolvableGene(
                        'MATPOW',
                        [EvolvableGene('VAR', value='x'), EvolvableGene('VAR', value='k')]
                    )
                    worst_idx = fitness_scores.index(min(fitness_scores))
                    population[worst_idx] = matrix_program
                    fitness_scores[worst_idx] = 500
            
            if gen == injection_gen + 10:
                post_injection_success = best_success
            
            if gen % 50 == 0 and gen != injection_gen:
                evolution_engine.maybe_invent_operator(population, fitness_scores)
            
            history.append({
                "generation": gen,
                "best_fitness": best_fitness,
                "best_success": best_success,
                "avg_success": avg_success,
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
            
            if gen % 100 == 0 or gen == 1 or gen == injection_gen or gen == injection_gen + 10:
                marker = " *** INJECTION ***" if gen == injection_gen else ""
                print(f"    Gen {gen:4d}: Success={best_success*100:.0f}%, Ops={len(dsl.operators)}{marker}")
        
        success_jump = (post_injection_success - pre_injection_success) if pre_injection_success is not None and post_injection_success is not None else None
        
        result = {
            "seed": seed,
            "n_operators": len(dsl.operators),
            "has_matrix_ops": dsl.has_matrix_ops(),
            "pre_injection_success": pre_injection_success,
            "post_injection_success": post_injection_success,
            "success_jump": success_jump,
            "final_success": best_success,
        }
        all_results.append(result)
        all_histories.append(history)
        
        if success_jump is not None:
            print(f"    Success Jump: {pre_injection_success*100:.0f}% → {post_injection_success*100:.0f}% (+{success_jump*100:.0f}%)")
    
    success_jumps = [r["success_jump"] for r in all_results if r["success_jump"] is not None]
    mean_jump = np.mean(success_jumps) if success_jumps else 0
    std_jump = np.std(success_jumps, ddof=1) if len(success_jumps) > 1 else 0
    
    print("\n" + "-" * 50)
    print("EXPERIMENT 2B (DYNAMIC) SUMMARY")
    print("-" * 50)
    print(f"Success Jump: +{mean_jump*100:.1f}% ± {std_jump*100:.1f}%")
    
    return {
        "config": {
            "experiment": "2B_dynamic",
            "description": "Simulated Discovery with Dynamic Tasks (INJECTION)",
            "population_size": population_size,
            "generations": generations,
            "injection_gen": injection_gen,
            "n_tests": n_tests,
            "n_seeds": n_seeds,
        },
        "results": all_results,
        "histories": all_histories,
        "statistics": {
            "mean_success_jump": mean_jump,
            "std_success_jump": std_jump,
        },
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("ALL EXPERIMENTS WITH DYNAMIC TASK EVALUATION")
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    print("\nDynamic Task Settings:")
    print("  - N ~ Uniform(3, 10)")
    print("  - k ~ Uniform(2, 6)")
    print("  - 10 random tests per individual per generation")
    print("  - This prevents 'memorization' by scalar DSLs")
    
    # Run all experiments
    exp1 = run_experiment1_dynamic(
        population_size=100,
        generations=50,
        initial_matrix_ratio=0.2,
        n_seeds=10,
        n_tests=10,
    )
    
    exp2a = run_experiment2a_dynamic(
        population_size=100,
        generations=500,
        n_seeds=5,
        n_tests=10,
    )
    
    exp2b = run_experiment2b_dynamic(
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
        "experiment1_dynamic": exp1,
        "experiment2a_dynamic": exp2a,
        "experiment2b_dynamic": exp2b,
        "timestamp": timestamp,
    }
    
    results_file = os.path.join(output_dir, f"all_experiments_dynamic_{timestamp}.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=lambda x: None if x != x else x)
    
    print(f"\n\nResults saved to: {results_file}")
    
    # LaTeX output
    print("\n" + "=" * 70)
    print("LATEX-READY OUTPUT")
    print("=" * 70)
    
    # Experiment 1 Table
    print("\n% Experiment 1 (Dynamic) - Matrix DSL Dominance")
    print("\\begin{table}[ht]")
    print("\\centering")
    print("\\begin{tabular}{ccc}")
    print("\\toprule")
    print("Run & Convergence (Gen) & Final Matrix (\\%) \\\\")
    print("\\midrule")
    for r in exp1["results"]:
        conv = str(r["convergence_gen"]) if r["convergence_gen"] else "N/A"
        print(f"{r['seed']+1} & {conv} & {r['final_matrix_ratio']*100:.0f} \\\\")
    s1 = exp1["statistics"]
    if s1["convergence_mean"]:
        print("\\midrule")
        print(f"Mean & {s1['convergence_mean']:.1f} $\\pm$ {s1['convergence_std']:.1f} & {s1['final_ratio_mean']*100:.0f} \\\\")
        ci = s1['convergence_ci_95']
        if ci[0]:
            print(f"95\\% CI & [{ci[0]:.1f}, {ci[1]:.1f}] & -- \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\caption{Experiment 1 (Dynamic): Matrix DSL dominance with dynamic task evaluation.}")
    print("\\label{tab:exp1_dyn}")
    print("\\end{table}")
    
    # Experiment 2A Table
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
    s2a = exp2a["statistics"]
    print("\\midrule")
    print(f"Mean & {s2a['mean_success_rate']*100:.1f} $\\pm$ {s2a['std_success_rate']*100:.1f} & -- & -- \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\caption{Experiment 2A (Dynamic): Spontaneous emergence test (NO injection).}")
    print("\\label{tab:exp2a_dyn}")
    print("\\end{table}")
    
    # Experiment 2B Table
    print("\n% Experiment 2B (Dynamic) - Simulated Discovery")
    print("\\begin{table}[ht]")
    print("\\centering")
    print("\\begin{tabular}{cccc}")
    print("\\toprule")
    print("Run & Pre-Inj (\\%) & Post-Inj (\\%) & Jump (\\%) \\\\")
    print("\\midrule")
    for r in exp2b["results"]:
        pre = r['pre_injection_success']*100 if r['pre_injection_success'] else 0
        post = r['post_injection_success']*100 if r['post_injection_success'] else 0
        jump = r['success_jump']*100 if r['success_jump'] else 0
        print(f"{r['seed']+1} & {pre:.0f} & {post:.0f} & +{jump:.0f} \\\\")
    s2b = exp2b["statistics"]
    print("\\midrule")
    print(f"Mean & -- & -- & +{s2b['mean_success_jump']*100:.1f} $\\pm$ {s2b['std_success_jump']*100:.1f} \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\caption{Experiment 2B (Dynamic): Simulated discovery (injection at Gen 200).}")
    print("\\label{tab:exp2b_dyn}")
    print("\\end{table}")
    
    return results


if __name__ == "__main__":
    main()

