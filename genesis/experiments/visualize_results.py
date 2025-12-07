#!/usr/bin/env python3
"""
Visualization of Extended Experiment Results

Generates publication-quality figures with:
- Evolution curves with confidence intervals
- Statistical summaries
"""

import sys
import os
import json
import glob
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Find the most recent results file
results_dir = "genesis/experiments/results"
results_files = glob.glob(os.path.join(results_dir, "extended_results_*.json"))
if not results_files:
    print("No results files found!")
    sys.exit(1)

latest_file = max(results_files, key=os.path.getmtime)
print(f"Loading results from: {latest_file}")

with open(latest_file, "r") as f:
    data = json.load(f)

exp1 = data["experiment1"]
exp2 = data["experiment2"]

# Create output directory for figures
fig_dir = "genesis/experiments/figures"
os.makedirs(fig_dir, exist_ok=True)

# Set matplotlib style
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'legend.fontsize': 10,
    'figure.figsize': (10, 6),
    'figure.dpi': 150,
})


def plot_experiment1():
    """Plot Experiment 1: Matrix ratio evolution with CI."""
    histories = exp1["histories"]
    n_seeds = len(histories)
    n_gens = len(histories[0])
    
    # Extract matrix ratios per generation
    generations = list(range(1, n_gens + 1))
    matrix_ratios = np.zeros((n_seeds, n_gens))
    
    for seed, hist in enumerate(histories):
        for gen_idx, gen_data in enumerate(hist):
            matrix_ratios[seed, gen_idx] = gen_data["matrix_ratio"]
    
    # Calculate statistics
    mean_ratio = np.mean(matrix_ratios, axis=0)
    std_ratio = np.std(matrix_ratios, axis=0, ddof=1)
    sem_ratio = std_ratio / np.sqrt(n_seeds)
    ci_lower = mean_ratio - 1.96 * sem_ratio
    ci_upper = mean_ratio + 1.96 * sem_ratio
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot individual runs in light color
    for seed in range(n_seeds):
        ax.plot(generations, matrix_ratios[seed, :] * 100, 
                alpha=0.2, color='steelblue', linewidth=0.5)
    
    # Plot mean with CI
    ax.plot(generations, mean_ratio * 100, 
            color='steelblue', linewidth=2, label='Mean')
    ax.fill_between(generations, ci_lower * 100, ci_upper * 100, 
                    alpha=0.3, color='steelblue', label='95% CI')
    
    # Add horizontal reference lines
    ax.axhline(y=90, color='red', linestyle='--', alpha=0.5, label='90% threshold')
    
    # Labels
    ax.set_xlabel('Generation')
    ax.set_ylabel('Matrix DSL Proportion (%)')
    ax.set_title('Experiment 1: Matrix DSL Dominance (N=100, 10 runs)')
    ax.legend(loc='lower right')
    ax.set_ylim(0, 105)
    ax.set_xlim(1, n_gens)
    ax.grid(True, alpha=0.3)
    
    # Add annotation
    conv_gen = exp1["statistics"]["convergence_mean"]
    conv_std = exp1["statistics"]["convergence_std"]
    ax.annotate(f'Convergence: Gen {conv_gen:.1f} ± {conv_std:.1f}',
                xy=(conv_gen, 92), xytext=(conv_gen + 10, 75),
                arrowprops=dict(arrowstyle='->', color='black'),
                fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Save
    fig.savefig(os.path.join(fig_dir, 'exp1_matrix_dominance.png'), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(fig_dir, 'exp1_matrix_dominance.pdf'), dpi=300, bbox_inches='tight')
    print(f"Saved: {fig_dir}/exp1_matrix_dominance.png")
    
    plt.close(fig)


def plot_experiment2():
    """Plot Experiment 2: Fitness and operator evolution."""
    histories = exp2["histories"]
    n_seeds = len(histories)
    n_gens = len(histories[0])
    
    # Extract data
    generations = list(range(1, n_gens + 1))
    best_fitness = np.zeros((n_seeds, n_gens))
    n_operators = np.zeros((n_seeds, n_gens))
    
    for seed, hist in enumerate(histories):
        for gen_idx, gen_data in enumerate(hist):
            best_fitness[seed, gen_idx] = gen_data["best_fitness"]
            n_operators[seed, gen_idx] = gen_data["n_operators"]
    
    # Calculate statistics
    mean_fitness = np.mean(best_fitness, axis=0)
    std_fitness = np.std(best_fitness, axis=0, ddof=1)
    sem_fitness = std_fitness / np.sqrt(n_seeds)
    ci_lower_f = mean_fitness - 1.96 * sem_fitness
    ci_upper_f = mean_fitness + 1.96 * sem_fitness
    
    mean_ops = np.mean(n_operators, axis=0)
    std_ops = np.std(n_operators, axis=0, ddof=1)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Subplot 1: Fitness evolution
    for seed in range(n_seeds):
        ax1.plot(generations, best_fitness[seed, :], 
                alpha=0.3, color='forestgreen', linewidth=0.5)
    
    ax1.plot(generations, mean_fitness, 
            color='forestgreen', linewidth=2, label='Mean Best Fitness')
    ax1.fill_between(generations, ci_lower_f, ci_upper_f, 
                    alpha=0.3, color='forestgreen', label='95% CI')
    
    # Mark matrix injection
    ax1.axvline(x=300, color='red', linestyle='--', alpha=0.7, label='Matrix Ops Injected (Gen 300)')
    
    ax1.set_ylabel('Best Fitness')
    ax1.set_title('Experiment 2: DSL Evolution (N=100, 1000 generations, 5 runs)')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    
    # Subplot 2: Operator count evolution
    for seed in range(n_seeds):
        ax2.plot(generations, n_operators[seed, :], 
                alpha=0.3, color='darkorange', linewidth=0.5)
    
    ax2.plot(generations, mean_ops, 
            color='darkorange', linewidth=2, label='Mean # Operators')
    ax2.fill_between(generations, mean_ops - std_ops, mean_ops + std_ops, 
                    alpha=0.3, color='darkorange', label='± 1 SD')
    
    ax2.axvline(x=300, color='red', linestyle='--', alpha=0.7)
    
    ax2.set_xlabel('Generation')
    ax2.set_ylabel('Number of Operators')
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    fig.savefig(os.path.join(fig_dir, 'exp2_dsl_evolution.png'), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(fig_dir, 'exp2_dsl_evolution.pdf'), dpi=300, bbox_inches='tight')
    print(f"Saved: {fig_dir}/exp2_dsl_evolution.png")
    
    plt.close(fig)


def plot_combined_summary():
    """Create a combined summary figure."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Exp1: Convergence times bar plot
    conv_gens = [r["convergence_gen"] for r in exp1["results"] if r["convergence_gen"]]
    ax1.bar(range(len(conv_gens)), conv_gens, color='steelblue', alpha=0.7)
    ax1.axhline(y=np.mean(conv_gens), color='red', linestyle='--', 
                label=f'Mean = {np.mean(conv_gens):.1f} ± {np.std(conv_gens, ddof=1):.1f}')
    ax1.set_xlabel('Run Index')
    ax1.set_ylabel('Convergence Generation')
    ax1.set_title('Experiment 1: Convergence Speed')
    ax1.legend()
    ax1.set_xticks(range(len(conv_gens)))
    ax1.grid(True, alpha=0.3)
    
    # Exp2: Final operator counts
    n_ops_list = [r["n_operators"] for r in exp2["results"]]
    n_inv_list = [r["n_inventions"] for r in exp2["results"]]
    
    x = np.arange(len(n_ops_list))
    width = 0.35
    
    bars1 = ax2.bar(x - width/2, n_ops_list, width, label='Final Operators', color='darkorange', alpha=0.7)
    bars2 = ax2.bar(x + width/2, n_inv_list, width, label='Inventions', color='forestgreen', alpha=0.7)
    
    ax2.set_xlabel('Run Index')
    ax2.set_ylabel('Count')
    ax2.set_title('Experiment 2: DSL Evolution Results')
    ax2.legend()
    ax2.set_xticks(x)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    fig.savefig(os.path.join(fig_dir, 'combined_summary.png'), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(fig_dir, 'combined_summary.pdf'), dpi=300, bbox_inches='tight')
    print(f"Saved: {fig_dir}/combined_summary.png")
    
    plt.close(fig)


def generate_latex_table():
    """Generate LaTeX tables for the paper."""
    print("\n" + "=" * 70)
    print("LATEX TABLES FOR PAPER")
    print("=" * 70)
    
    # Experiment 1 Table
    print("\n% Experiment 1: Matrix DSL Dominance")
    print("\\begin{table}[ht]")
    print("\\centering")
    print("\\begin{tabular}{ccc}")
    print("\\toprule")
    print("Run & Convergence (Gen) & Final Matrix (\\%) \\\\")
    print("\\midrule")
    
    for r in exp1["results"]:
        conv = str(r["convergence_gen"]) if r["convergence_gen"] else "N/A"
        print(f"{r['seed']+1} & {conv} & {r['final_matrix_ratio']*100:.0f} \\\\")
    
    s = exp1["statistics"]
    print("\\midrule")
    print(f"Mean & {s['convergence_mean']:.1f} & {s['final_ratio_mean']*100:.0f} \\\\")
    print(f"SD & {s['convergence_std']:.2f} & -- \\\\")
    ci = s['convergence_ci_95']
    print(f"95\\% CI & [{ci[0]:.1f}, {ci[1]:.1f}] & -- \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\caption{Experiment 1: Matrix DSL dominance results across 10 independent runs")
    print("with population size $N = 100$ and 50 generations.}")
    print("\\label{tab:exp1}")
    print("\\end{table}")
    
    # Experiment 2 Table
    print("\n% Experiment 2: DSL Evolution")
    print("\\begin{table}[ht]")
    print("\\centering")
    print("\\begin{tabular}{cccc}")
    print("\\toprule")
    print("Run & Final Operators & Inventions & Matrix Ops \\\\")
    print("\\midrule")
    
    for r in exp2["results"]:
        has_matrix = "Yes" if r["has_matrix_ops"] else "No"
        print(f"{r['seed']+1} & {r['n_operators']} & {r['n_inventions']} & {has_matrix} \\\\")
    
    s2 = exp2["statistics"]
    print("\\midrule")
    print(f"Mean & {s2['n_operators_mean']:.1f} & {s2['n_inventions_mean']:.1f} & -- \\\\")
    print(f"SD & {s2['n_operators_std']:.2f} & {s2['n_inventions_std']:.2f} & -- \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\caption{Experiment 2: DSL evolution results across 5 independent runs")
    print("with population size $N = 100$ and 1000 generations.}")
    print("\\label{tab:exp2}")
    print("\\end{table}")


def main():
    print("Generating visualizations...")
    plot_experiment1()
    plot_experiment2()
    plot_combined_summary()
    generate_latex_table()
    print("\nDone!")


if __name__ == "__main__":
    main()

