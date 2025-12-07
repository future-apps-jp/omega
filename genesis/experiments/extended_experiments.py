#!/usr/bin/env python3
"""
Extended Experiments for Paper Revision

Experiment 1: N=100, 50 generations, 10 seeds
Experiment 2: N=100, 1000 generations, 5 seeds

Outputs:
- Statistical results with confidence intervals
- Evolution curves for visualization
"""

import sys
import os
import json
import time
from datetime import datetime
from typing import Dict, List, Tuple
import numpy as np
from scipy import stats as scipy_stats

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
from genesis.evolution.dsl_evolution import (
    EvolvableDSL,
    EvolvableGene,
    DSLEvolutionEngine,
    generate_evolvable_program,
)


def run_experiment1(
    n_nodes: int = 5,
    steps: int = 3,
    population_size: int = 100,
    generations: int = 50,
    initial_matrix_ratio: float = 0.2,
    n_seeds: int = 10,
) -> Dict:
    """
    Experiment 1: Matrix DSL Dominance
    
    Extended version with N=100 population, multiple seeds.
    """
    print("=" * 70)
    print("EXPERIMENT 1: The Quantum Dawn (Extended)")
    print("=" * 70)
    print(f"Population: {population_size}, Generations: {generations}, Seeds: {n_seeds}")
    
    task = create_graph_walk_task(n_nodes=n_nodes, steps=steps, start_node=0, end_node=n_nodes-1)
    print(f"Task: {task}")
    
    all_results = []
    all_histories = []
    
    for seed in range(n_seeds):
        print(f"\n  Run {seed + 1}/{n_seeds} (seed={seed})...", end=" ")
        np.random.seed(seed)
        
        # Setup
        fitness_config = FitnessConfig(k_penalty=1.0, accuracy_weight=1000.0)
        fitness_eval = create_graph_walk_fitness_evaluator(task, fitness_config)
        
        pop_config = PopulationConfig(
            size=population_size,
            elite_size=max(5, population_size // 20),
            survival_rate=0.2,
            mutation_rate=0.15,
            crossover_rate=0.3,
        )
        
        population = Population(
            config=pop_config,
            fitness_evaluator=fitness_eval,
            mutator=Mutator(mutation_rate=0.15, crossover_rate=0.3),
        )
        
        adj = task.adjacency_matrix
        
        def scalar_gen():
            return Container(generate_scalar_program(max_depth=4), species="Scalar")
        
        def matrix_gen():
            return Container(generate_matrix_program(adj, max_depth=3), species="Matrix")
        
        n_matrix = int(population_size * initial_matrix_ratio)
        n_scalar = population_size - n_matrix
        
        population.initialize(
            generators={"Scalar": scalar_gen, "Matrix": matrix_gen},
            counts={"Scalar": n_scalar, "Matrix": n_matrix},
        )
        
        # Evolution
        history = []
        convergence_gen = None
        
        for gen in range(1, generations + 1):
            stats = population.step()
            dist = population.get_species_distribution()
            
            history.append({
                "generation": gen,
                "matrix_ratio": dist.get("Matrix", 0),
                "scalar_ratio": dist.get("Scalar", 0),
                "best_fitness": stats.best_fitness,
                "best_k": stats.best_k,
            })
            
            if dist.get("Matrix", 0) > 0.9 and convergence_gen is None:
                convergence_gen = gen
        
        final_dist = population.get_species_distribution()
        
        result = {
            "seed": seed,
            "convergence_gen": convergence_gen,
            "final_matrix_ratio": final_dist.get("Matrix", 0),
            "final_scalar_ratio": final_dist.get("Scalar", 0),
        }
        all_results.append(result)
        all_histories.append(history)
        
        print(f"Conv={convergence_gen if convergence_gen else 'N/A'}, "
              f"Final Matrix={final_dist.get('Matrix', 0)*100:.1f}%")
    
    # Statistics
    conv_gens = [r["convergence_gen"] for r in all_results if r["convergence_gen"]]
    final_ratios = [r["final_matrix_ratio"] for r in all_results]
    
    if conv_gens:
        conv_mean = np.mean(conv_gens)
        conv_std = np.std(conv_gens, ddof=1)
        conv_ci = scipy_stats.t.interval(0.95, len(conv_gens)-1, loc=conv_mean, scale=scipy_stats.sem(conv_gens))
    else:
        conv_mean, conv_std, conv_ci = None, None, (None, None)
    
    ratio_mean = np.mean(final_ratios)
    ratio_std = np.std(final_ratios, ddof=1)
    ratio_ci = scipy_stats.t.interval(0.95, len(final_ratios)-1, loc=ratio_mean, scale=scipy_stats.sem(final_ratios))
    
    print("\n" + "-" * 50)
    print("EXPERIMENT 1 SUMMARY")
    print("-" * 50)
    print(f"Convergence Rate: {len(conv_gens)}/{len(all_results)}")
    if conv_gens:
        print(f"Convergence Gen: {conv_mean:.2f} ± {conv_std:.2f} (95% CI: [{conv_ci[0]:.2f}, {conv_ci[1]:.2f}])")
    print(f"Final Matrix %: {ratio_mean*100:.1f}% ± {ratio_std*100:.1f}% (95% CI: [{ratio_ci[0]*100:.1f}%, {ratio_ci[1]*100:.1f}%])")
    
    return {
        "config": {
            "n_nodes": n_nodes,
            "steps": steps,
            "population_size": population_size,
            "generations": generations,
            "initial_matrix_ratio": initial_matrix_ratio,
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
            "final_ratio_ci_95": ratio_ci,
        },
    }


def run_experiment2(
    n_nodes: int = 5,
    steps: int = 3,
    population_size: int = 100,
    generations: int = 1000,
    n_seeds: int = 5,
) -> Dict:
    """
    Experiment 2: Evolution of Operators
    
    Extended version with N=100 population, 1000 generations.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Evolution of Operators (Extended)")
    print("=" * 70)
    print(f"Population: {population_size}, Generations: {generations}, Seeds: {n_seeds}")
    
    task = create_graph_walk_task(n_nodes=n_nodes, steps=steps, start_node=0, end_node=n_nodes-1)
    
    all_results = []
    all_histories = []
    
    for seed in range(n_seeds):
        print(f"\n  Run {seed + 1}/{n_seeds} (seed={seed})...")
        np.random.seed(seed)
        import random
        random.seed(seed)
        
        # Create evolvable DSL (scalar only initially)
        dsl = EvolvableDSL("Genesis")
        evolution_engine = DSLEvolutionEngine(dsl, invention_threshold=5, compression_benefit=0.3)
        
        # Initialize population
        population = [generate_evolvable_program(dsl, max_depth=4) for _ in range(population_size)]
        
        history = []
        inventions = []
        matrix_injection_gen = None
        best_fitness_ever = 0
        
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
            
            # Try to invent new operator
            new_op = None
            if gen % 50 == 0:
                new_op = evolution_engine.maybe_invent_operator(population, fitness_scores)
                if new_op:
                    inventions.append({"generation": gen, "operator": new_op})
            
            # Inject matrix operations at generation 300
            if gen == 300 and not matrix_injection_gen:
                evolution_engine.inject_matrix_hint()
                matrix_injection_gen = gen
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
                    fitness_scores[worst_idx] = 500  # Approximate
            
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
                # Mutate
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
            if gen % 100 == 0 or gen <= 10:
                print(f"    Gen {gen:4d}: Fitness={best_fitness:.2f}, K={best_k}, Ops={len(dsl.operators)}")
        
        result = {
            "seed": seed,
            "final_operators": list(dsl.operators.keys()),
            "n_operators": len(dsl.operators),
            "n_inventions": len(inventions),
            "has_matrix_ops": dsl.has_matrix_ops(),
            "matrix_injection_gen": matrix_injection_gen,
            "best_fitness": best_fitness_ever,
            "inventions": inventions,
        }
        all_results.append(result)
        all_histories.append(history)
    
    # Statistics
    n_ops_list = [r["n_operators"] for r in all_results]
    n_inv_list = [r["n_inventions"] for r in all_results]
    
    n_ops_mean = np.mean(n_ops_list)
    n_ops_std = np.std(n_ops_list, ddof=1)
    n_ops_ci = scipy_stats.t.interval(0.95, len(n_ops_list)-1, loc=n_ops_mean, scale=scipy_stats.sem(n_ops_list))
    
    n_inv_mean = np.mean(n_inv_list)
    n_inv_std = np.std(n_inv_list, ddof=1)
    n_inv_ci = scipy_stats.t.interval(0.95, len(n_inv_list)-1, loc=n_inv_mean, scale=scipy_stats.sem(n_inv_list))
    
    print("\n" + "-" * 50)
    print("EXPERIMENT 2 SUMMARY")
    print("-" * 50)
    print(f"Final Operators: {n_ops_mean:.1f} ± {n_ops_std:.1f} (95% CI: [{n_ops_ci[0]:.1f}, {n_ops_ci[1]:.1f}])")
    print(f"Inventions: {n_inv_mean:.1f} ± {n_inv_std:.1f} (95% CI: [{n_inv_ci[0]:.1f}, {n_inv_ci[1]:.1f}])")
    print(f"All runs have matrix ops: {all(r['has_matrix_ops'] for r in all_results)}")
    
    return {
        "config": {
            "n_nodes": n_nodes,
            "steps": steps,
            "population_size": population_size,
            "generations": generations,
            "n_seeds": n_seeds,
        },
        "results": all_results,
        "histories": all_histories,
        "statistics": {
            "n_operators_mean": n_ops_mean,
            "n_operators_std": n_ops_std,
            "n_operators_ci_95": n_ops_ci,
            "n_inventions_mean": n_inv_mean,
            "n_inventions_std": n_inv_std,
            "n_inventions_ci_95": n_inv_ci,
        },
    }


def create_evolution_plot_data(histories: List[List[Dict]], key: str = "matrix_ratio") -> Dict:
    """Create data for evolution plots with mean and CI."""
    n_gens = len(histories[0])
    n_seeds = len(histories)
    
    data_by_gen = []
    for gen_idx in range(n_gens):
        values = [histories[seed][gen_idx][key] for seed in range(n_seeds)]
        mean = np.mean(values)
        std = np.std(values, ddof=1) if n_seeds > 1 else 0
        sem = std / np.sqrt(n_seeds) if n_seeds > 1 else 0
        ci = 1.96 * sem  # 95% CI
        data_by_gen.append({
            "generation": gen_idx + 1,
            "mean": mean,
            "std": std,
            "ci_lower": mean - ci,
            "ci_upper": mean + ci,
        })
    
    return data_by_gen


def main():
    print("=" * 70)
    print("EXTENDED EXPERIMENTS FOR PAPER REVISION")
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # Experiment 1
    exp1_results = run_experiment1(
        n_nodes=5,
        steps=3,
        population_size=100,
        generations=50,
        initial_matrix_ratio=0.2,
        n_seeds=10,
    )
    
    # Experiment 2
    exp2_results = run_experiment2(
        n_nodes=5,
        steps=3,
        population_size=100,
        generations=1000,
        n_seeds=5,
    )
    
    # Create plot data
    exp1_plot_data = create_evolution_plot_data(exp1_results["histories"], "matrix_ratio")
    exp2_plot_data = create_evolution_plot_data(exp2_results["histories"], "best_fitness")
    
    # Save results
    output_dir = "genesis/experiments/results"
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    combined_results = {
        "experiment1": exp1_results,
        "experiment2": exp2_results,
        "plot_data": {
            "exp1_matrix_ratio": exp1_plot_data,
            "exp2_fitness": exp2_plot_data,
        },
        "timestamp": timestamp,
    }
    
    results_file = os.path.join(output_dir, f"extended_results_{timestamp}.json")
    with open(results_file, "w") as f:
        json.dump(combined_results, f, indent=2, default=lambda x: None if x != x else x)
    
    print(f"\n\nResults saved to: {results_file}")
    
    # Print LaTeX-ready tables
    print("\n" + "=" * 70)
    print("LATEX-READY OUTPUT")
    print("=" * 70)
    
    print("\n% Experiment 1 Table")
    print("\\begin{table}[ht]")
    print("\\centering")
    print("\\begin{tabular}{lcc}")
    print("\\toprule")
    print("Run (Seed) & Convergence Gen & Final Matrix \\% \\\\")
    print("\\midrule")
    for r in exp1_results["results"]:
        conv = r["convergence_gen"] if r["convergence_gen"] else "N/A"
        print(f"{r['seed']} & {conv} & {r['final_matrix_ratio']*100:.0f}\\% \\\\")
    s = exp1_results["statistics"]
    if s["convergence_mean"]:
        print("\\midrule")
        print(f"\\textbf{{Mean $\\pm$ SD}} & \\textbf{{{s['convergence_mean']:.1f} $\\pm$ {s['convergence_std']:.1f}}} & \\textbf{{{s['final_ratio_mean']*100:.0f}\\%}} \\\\")
        print(f"\\textbf{{95\\% CI}} & [{s['convergence_ci_95'][0]:.1f}, {s['convergence_ci_95'][1]:.1f}] & [{s['final_ratio_ci_95'][0]*100:.0f}\\%, {s['final_ratio_ci_95'][1]*100:.0f}\\%] \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\caption{Experiment 1: Matrix DSL dominance results (N=100, 10 runs).}")
    print("\\label{tab:exp1}")
    print("\\end{table}")
    
    print("\n% Experiment 2 Summary")
    s2 = exp2_results["statistics"]
    print(f"% Final operators: {s2['n_operators_mean']:.1f} ± {s2['n_operators_std']:.1f} (95% CI: [{s2['n_operators_ci_95'][0]:.1f}, {s2['n_operators_ci_95'][1]:.1f}])")
    print(f"% Inventions: {s2['n_inventions_mean']:.1f} ± {s2['n_inventions_std']:.1f} (95% CI: [{s2['n_inventions_ci_95'][0]:.1f}, {s2['n_inventions_ci_95'][1]:.1f}])")
    
    return combined_results


if __name__ == "__main__":
    main()

