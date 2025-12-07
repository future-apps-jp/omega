#!/usr/bin/env python3
"""
Experiment 2: Separated into 2A and 2B

Experiment 2A: Spontaneous Emergence Test
- 1000 generations, NO injection
- Tests whether matrix operations emerge de novo

Experiment 2B: Simulated Discovery Test  
- 1000 generations, injection at generation 300
- Tests post-discovery dynamics
"""

import sys
import os
import json
import time
from datetime import datetime
from typing import Dict, List
import numpy as np
from scipy import stats as scipy_stats
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from genesis.tasks.graph_walk import create_graph_walk_task
from genesis.evolution.dsl_evolution import (
    EvolvableDSL,
    EvolvableGene,
    DSLEvolutionEngine,
    generate_evolvable_program,
)


def run_experiment_2a(
    n_nodes: int = 5,
    steps: int = 3,
    population_size: int = 100,
    generations: int = 1000,
    n_seeds: int = 5,
) -> Dict:
    """
    Experiment 2A: Spontaneous Emergence Test
    
    NO injection - pure evolutionary search for 1000 generations.
    Tests whether matrix operations can emerge de novo.
    """
    print("=" * 70)
    print("EXPERIMENT 2A: Spontaneous Emergence Test (NO INJECTION)")
    print("=" * 70)
    print(f"Population: {population_size}, Generations: {generations}, Seeds: {n_seeds}")
    
    task = create_graph_walk_task(n_nodes=n_nodes, steps=steps, start_node=0, end_node=n_nodes-1)
    
    all_results = []
    all_histories = []
    
    for seed in range(n_seeds):
        print(f"\n  Run {seed + 1}/{n_seeds} (seed={seed})...")
        np.random.seed(seed)
        random.seed(seed)
        
        # Create evolvable DSL (scalar only - NO matrix ops ever)
        dsl = EvolvableDSL("Genesis-2A")
        evolution_engine = DSLEvolutionEngine(dsl, invention_threshold=5, compression_benefit=0.3)
        
        # Initialize population
        population = [generate_evolvable_program(dsl, max_depth=4) for _ in range(population_size)]
        
        history = []
        inventions = []
        best_fitness_ever = 0
        matrix_emerged = False  # Track if matrix-like ops ever emerge
        
        for gen in range(1, generations + 1):
            dsl.generation = gen
            
            # Evaluate fitness
            fitness_scores = []
            for program in population:
                context = {"A": task.adjacency_matrix, "x": task.adjacency_matrix, "k": task.steps}
                try:
                    result = program.evaluate(context, dsl)
                    if isinstance(result, np.ndarray):
                        try:
                            result = result[task.start_node, task.end_node]
                        except:
                            result = np.sum(result)
                    error = abs(task.target_value - float(result))
                    k = program.size()
                    fitness = 1000.0 / (1 + error * 100 + k * 0.5)
                except:
                    fitness = 0.0
                fitness_scores.append(fitness)
            
            best_fitness = max(fitness_scores)
            avg_fitness = np.mean(fitness_scores)
            best_idx = fitness_scores.index(best_fitness)
            best_k = population[best_idx].size()
            
            if best_fitness > best_fitness_ever:
                best_fitness_ever = best_fitness
            
            # Try to invent new operator (but NO matrix injection)
            new_op = None
            if gen % 50 == 0:
                new_op = evolution_engine.maybe_invent_operator(population, fitness_scores)
                if new_op:
                    inventions.append({"generation": gen, "operator": new_op})
                    # Check if any matrix-like operation emerged
                    if "MAT" in new_op.upper() or "MATRIX" in new_op.upper():
                        matrix_emerged = True
            
            history.append({
                "generation": gen,
                "best_fitness": best_fitness,
                "avg_fitness": avg_fitness,
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
            
            # Progress output
            if gen % 200 == 0 or gen == 1:
                print(f"    Gen {gen:4d}: Fitness={best_fitness:.2f}, K={best_k}, Ops={len(dsl.operators)}, Matrix={dsl.has_matrix_ops()}")
        
        result = {
            "seed": seed,
            "final_operators": list(dsl.operators.keys()),
            "n_operators": len(dsl.operators),
            "n_inventions": len(inventions),
            "has_matrix_ops": dsl.has_matrix_ops(),
            "matrix_emerged": matrix_emerged,
            "best_fitness": best_fitness_ever,
            "inventions": inventions,
        }
        all_results.append(result)
        all_histories.append(history)
        
        print(f"    Final: {len(dsl.operators)} operators, {len(inventions)} inventions, Matrix emerged: {matrix_emerged}")
    
    # Summary
    n_matrix_emerged = sum(1 for r in all_results if r["matrix_emerged"])
    
    print("\n" + "-" * 50)
    print("EXPERIMENT 2A SUMMARY")
    print("-" * 50)
    print(f"Matrix operations emerged: {n_matrix_emerged}/{len(all_results)} runs")
    print(f"Conclusion: {'Matrix CAN emerge spontaneously' if n_matrix_emerged > 0 else 'Matrix CANNOT emerge spontaneously (0/5 runs)'}")
    
    return {
        "config": {
            "experiment": "2A",
            "description": "Spontaneous Emergence Test (NO INJECTION)",
            "n_nodes": n_nodes,
            "steps": steps,
            "population_size": population_size,
            "generations": generations,
            "n_seeds": n_seeds,
            "injection": False,
        },
        "results": all_results,
        "histories": all_histories,
        "statistics": {
            "n_matrix_emerged": n_matrix_emerged,
            "emergence_rate": n_matrix_emerged / len(all_results),
        },
    }


def run_experiment_2b(
    n_nodes: int = 5,
    steps: int = 3,
    population_size: int = 100,
    generations: int = 1000,
    injection_gen: int = 300,
    n_seeds: int = 5,
) -> Dict:
    """
    Experiment 2B: Simulated Discovery Test
    
    Matrix operations injected at generation 300.
    Tests post-discovery dynamics.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 2B: Simulated Discovery Test (INJECTION at Gen 300)")
    print("=" * 70)
    print(f"Population: {population_size}, Generations: {generations}, Injection: Gen {injection_gen}, Seeds: {n_seeds}")
    
    task = create_graph_walk_task(n_nodes=n_nodes, steps=steps, start_node=0, end_node=n_nodes-1)
    
    all_results = []
    all_histories = []
    
    for seed in range(n_seeds):
        print(f"\n  Run {seed + 1}/{n_seeds} (seed={seed})...")
        np.random.seed(seed)
        random.seed(seed)
        
        # Create evolvable DSL
        dsl = EvolvableDSL("Genesis-2B")
        evolution_engine = DSLEvolutionEngine(dsl, invention_threshold=5, compression_benefit=0.3)
        
        # Initialize population
        population = [generate_evolvable_program(dsl, max_depth=4) for _ in range(population_size)]
        
        history = []
        inventions = []
        best_fitness_ever = 0
        pre_injection_fitness = None
        post_injection_fitness = None
        
        for gen in range(1, generations + 1):
            dsl.generation = gen
            
            # Evaluate fitness
            fitness_scores = []
            for program in population:
                context = {"A": task.adjacency_matrix, "x": task.adjacency_matrix, "k": task.steps}
                try:
                    result = program.evaluate(context, dsl)
                    if isinstance(result, np.ndarray):
                        try:
                            result = result[task.start_node, task.end_node]
                        except:
                            result = np.sum(result)
                    error = abs(task.target_value - float(result))
                    k = program.size()
                    fitness = 1000.0 / (1 + error * 100 + k * 0.5)
                except:
                    fitness = 0.0
                fitness_scores.append(fitness)
            
            best_fitness = max(fitness_scores)
            avg_fitness = np.mean(fitness_scores)
            best_idx = fitness_scores.index(best_fitness)
            best_k = population[best_idx].size()
            
            if best_fitness > best_fitness_ever:
                best_fitness_ever = best_fitness
            
            # Record pre-injection fitness
            if gen == injection_gen - 1:
                pre_injection_fitness = best_fitness
            
            # INJECT matrix operations at specified generation
            if gen == injection_gen:
                evolution_engine.inject_matrix_hint()
                inventions.append({"generation": gen, "operator": "MATMUL (injected)"})
                inventions.append({"generation": gen, "operator": "MATPOW (injected)"})
                
                # Inject some matrix programs
                for i in range(population_size // 10):
                    matrix_program = EvolvableGene(
                        'MATPOW',
                        [EvolvableGene('VAR', value='x'), EvolvableGene('CONST', value=task.steps)]
                    )
                    worst_idx = fitness_scores.index(min(fitness_scores))
                    population[worst_idx] = matrix_program
                    fitness_scores[worst_idx] = 500
            
            # Record post-injection fitness
            if gen == injection_gen + 1:
                post_injection_fitness = best_fitness
            
            # Try to invent new operator
            new_op = None
            if gen % 50 == 0 and gen != injection_gen:
                new_op = evolution_engine.maybe_invent_operator(population, fitness_scores)
                if new_op:
                    inventions.append({"generation": gen, "operator": new_op})
            
            history.append({
                "generation": gen,
                "best_fitness": best_fitness,
                "avg_fitness": avg_fitness,
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
            
            # Progress output
            if gen % 200 == 0 or gen == 1 or gen == injection_gen or gen == injection_gen + 1:
                marker = " *** INJECTION ***" if gen == injection_gen else ""
                print(f"    Gen {gen:4d}: Fitness={best_fitness:.2f}, K={best_k}, Ops={len(dsl.operators)}{marker}")
        
        # Calculate fitness jump
        fitness_jump = post_injection_fitness - pre_injection_fitness if pre_injection_fitness and post_injection_fitness else None
        
        result = {
            "seed": seed,
            "final_operators": list(dsl.operators.keys()),
            "n_operators": len(dsl.operators),
            "n_inventions": len(inventions),
            "has_matrix_ops": dsl.has_matrix_ops(),
            "best_fitness": best_fitness_ever,
            "pre_injection_fitness": pre_injection_fitness,
            "post_injection_fitness": post_injection_fitness,
            "fitness_jump": fitness_jump,
            "inventions": inventions,
        }
        all_results.append(result)
        all_histories.append(history)
        
        print(f"    Final: Fitness jump at injection: {pre_injection_fitness:.1f} → {post_injection_fitness:.1f} (+{fitness_jump:.1f})")
    
    # Statistics
    fitness_jumps = [r["fitness_jump"] for r in all_results if r["fitness_jump"] is not None]
    jump_mean = np.mean(fitness_jumps) if fitness_jumps else None
    jump_std = np.std(fitness_jumps, ddof=1) if len(fitness_jumps) > 1 else 0
    
    print("\n" + "-" * 50)
    print("EXPERIMENT 2B SUMMARY")
    print("-" * 50)
    if jump_mean:
        print(f"Fitness jump at injection: {jump_mean:.1f} ± {jump_std:.1f}")
    print(f"All runs have matrix ops after injection: {all(r['has_matrix_ops'] for r in all_results)}")
    
    return {
        "config": {
            "experiment": "2B",
            "description": "Simulated Discovery Test (INJECTION)",
            "n_nodes": n_nodes,
            "steps": steps,
            "population_size": population_size,
            "generations": generations,
            "injection_gen": injection_gen,
            "n_seeds": n_seeds,
            "injection": True,
        },
        "results": all_results,
        "histories": all_histories,
        "statistics": {
            "fitness_jump_mean": jump_mean,
            "fitness_jump_std": jump_std,
        },
    }


def main():
    print("=" * 70)
    print("EXPERIMENT 2: SEPARATED INTO 2A AND 2B")
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # Experiment 2A: NO injection
    exp2a_results = run_experiment_2a(
        n_nodes=5,
        steps=3,
        population_size=100,
        generations=1000,
        n_seeds=5,
    )
    
    # Experiment 2B: Injection at gen 300
    exp2b_results = run_experiment_2b(
        n_nodes=5,
        steps=3,
        population_size=100,
        generations=1000,
        injection_gen=300,
        n_seeds=5,
    )
    
    # Save results
    output_dir = "genesis/experiments/results"
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    combined_results = {
        "experiment2a": exp2a_results,
        "experiment2b": exp2b_results,
        "timestamp": timestamp,
    }
    
    results_file = os.path.join(output_dir, f"experiment2_separated_{timestamp}.json")
    with open(results_file, "w") as f:
        json.dump(combined_results, f, indent=2, default=lambda x: None if x != x else x)
    
    print(f"\n\nResults saved to: {results_file}")
    
    # Print LaTeX-ready output
    print("\n" + "=" * 70)
    print("LATEX-READY OUTPUT")
    print("=" * 70)
    
    print("\n% Experiment 2A Table (Spontaneous Emergence)")
    print("\\begin{table}[ht]")
    print("\\centering")
    print("\\begin{tabular}{cccc}")
    print("\\toprule")
    print("Run & Final Operators & Inventions & Matrix Emerged \\\\")
    print("\\midrule")
    for r in exp2a_results["results"]:
        emerged = "Yes" if r["matrix_emerged"] else "No"
        print(f"{r['seed']+1} & {r['n_operators']} & {r['n_inventions']} & {emerged} \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\caption{Experiment 2A: Spontaneous emergence test (NO injection, 1000 generations).}")
    print("\\label{tab:exp2a}")
    print("\\end{table}")
    
    print("\n% Experiment 2B Table (Simulated Discovery)")
    print("\\begin{table}[ht]")
    print("\\centering")
    print("\\begin{tabular}{ccccc}")
    print("\\toprule")
    print("Run & Pre-Injection & Post-Injection & Jump & Final Ops \\\\")
    print("\\midrule")
    for r in exp2b_results["results"]:
        print(f"{r['seed']+1} & {r['pre_injection_fitness']:.1f} & {r['post_injection_fitness']:.1f} & +{r['fitness_jump']:.1f} & {r['n_operators']} \\\\")
    s = exp2b_results["statistics"]
    print("\\midrule")
    print(f"Mean & -- & -- & +{s['fitness_jump_mean']:.1f} $\\pm$ {s['fitness_jump_std']:.1f} & -- \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\caption{Experiment 2B: Simulated discovery test (injection at Gen 300).}")
    print("\\label{tab:exp2b}")
    print("\\end{table}")
    
    return combined_results


if __name__ == "__main__":
    main()

