#!/usr/bin/env python3
"""
Phase 20: A1 vs Classical Complexity Comparison

This script performs a rigorous comparison of description lengths
between A1 and classical (NumPy) implementations of quantum protocols.

Key Features:
- Formal complexity metrics with explicit cost model
- Extended benchmark suite (Bell, GHZ, Teleportation + additional)
- Statistical analysis (mean, std, confidence intervals)
- Publication-ready output (tables, LaTeX)

Author: Hiroshi Kohashiguchi
Date: December 2025
"""

import sys
import os
import math
import json
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple
from datetime import datetime

# Add paths for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'a1'))

from core import A1, execute
from metrics import A1Metrics, ClassicalMetrics, compare, ComparisonResult
from gates import VOCABULARY_SIZE, QUANTUM_GATES


# =============================================================================
# Extended Benchmark Suite
# =============================================================================

EXTENDED_BENCHMARKS_A1 = {
    # Basic Protocols
    'bell': {
        'code': '(CNOT (H 0) 1)',
        'description': 'Bell state: (|00⟩ + |11⟩)/√2',
        'qubits': 2,
        'category': 'entanglement'
    },
    
    'ghz_3': {
        'code': '(CNOT (CNOT (H 0) 1) 2)',
        'description': 'GHZ-3 state: (|000⟩ + |111⟩)/√2',
        'qubits': 3,
        'category': 'entanglement'
    },
    
    'ghz_4': {
        'code': '(CNOT (CNOT (CNOT (H 0) 1) 2) 3)',
        'description': 'GHZ-4 state: 4-qubit GHZ',
        'qubits': 4,
        'category': 'entanglement'
    },
    
    # Superposition States
    'hadamard_1': {
        'code': '(H 0)',
        'description': 'Single-qubit superposition',
        'qubits': 1,
        'category': 'superposition'
    },
    
    'hadamard_4': {
        'code': '(BEGIN (H 0) (H 1) (H 2) (H 3))',
        'description': '4-qubit uniform superposition',
        'qubits': 4,
        'category': 'superposition'
    },
    
    # Teleportation
    'teleport': {
        'code': '''
        (DEFINE teleport
            (LAMBDA (psi alice bob)
                (BEGIN
                    (CNOT (H alice) bob)
                    (CNOT psi alice)
                    (H psi)
                    (MEASURE psi)
                    (MEASURE alice))))
        (teleport 0 1 2)
        ''',
        'description': 'Quantum teleportation protocol',
        'qubits': 3,
        'category': 'protocol'
    },
    
    # Superdense Coding (simplified - encode 10)
    'superdense': {
        'code': '''
        (BEGIN
            (CNOT (H 0) 1)
            (X 0)
            (CNOT 0 1)
            (H 0)
            (MEASURE 0)
            (MEASURE 1))
        ''',
        'description': 'Superdense coding (encode 10)',
        'qubits': 2,
        'category': 'protocol'
    },
    
    # Quantum Phase Estimation (simplified)
    'phase_kickback': {
        'code': '''
        (BEGIN
            (H 0)
            (CNOT 0 1)
            (H 0)
            (MEASURE 0))
        ''',
        'description': 'Phase kickback demonstration',
        'qubits': 2,
        'category': 'algorithm'
    },
    
    # Grover Iteration (single) - simplified
    'grover_oracle': {
        'code': '(H (Z (H 0)))',
        'description': 'Single Grover diffusion',
        'qubits': 1,
        'category': 'algorithm'
    },
}

EXTENDED_BENCHMARKS_NUMPY = {
    'bell': '''
import numpy as np

def bell_state():
    state = np.array([1, 0, 0, 0], dtype=complex)
    H = (1/np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex)
    CNOT = np.array([
        [1, 0, 0, 0], [0, 1, 0, 0],
        [0, 0, 0, 1], [0, 0, 1, 0]
    ], dtype=complex)
    I = np.eye(2, dtype=complex)
    H_full = np.kron(H, I)
    state = H_full @ state
    state = CNOT @ state
    return state

result = bell_state()
''',

    'ghz_3': '''
import numpy as np

def ghz_3_state():
    state = np.zeros(8, dtype=complex)
    state[0] = 1.0
    H = (1/np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex)
    I = np.eye(2, dtype=complex)
    CNOT = np.array([
        [1, 0, 0, 0], [0, 1, 0, 0],
        [0, 0, 0, 1], [0, 0, 1, 0]
    ], dtype=complex)
    H_full = np.kron(np.kron(H, I), I)
    state = H_full @ state
    CNOT_01 = np.kron(CNOT, I)
    state = CNOT_01 @ state
    CNOT_12 = np.kron(I, CNOT)
    state = CNOT_12 @ state
    return state

result = ghz_3_state()
''',

    'ghz_4': '''
import numpy as np

def ghz_4_state():
    state = np.zeros(16, dtype=complex)
    state[0] = 1.0
    H = (1/np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex)
    I = np.eye(2, dtype=complex)
    CNOT = np.array([
        [1, 0, 0, 0], [0, 1, 0, 0],
        [0, 0, 0, 1], [0, 0, 1, 0]
    ], dtype=complex)
    H_full = np.kron(np.kron(np.kron(H, I), I), I)
    state = H_full @ state
    CNOT_01 = np.kron(np.kron(CNOT, I), I)
    state = CNOT_01 @ state
    CNOT_12 = np.kron(np.kron(I, CNOT), I)
    state = CNOT_12 @ state
    CNOT_23 = np.kron(np.kron(I, I), CNOT)
    state = CNOT_23 @ state
    return state

result = ghz_4_state()
''',

    'hadamard_1': '''
import numpy as np

def hadamard_1():
    state = np.array([1, 0], dtype=complex)
    H = (1/np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex)
    state = H @ state
    return state

result = hadamard_1()
''',

    'hadamard_4': '''
import numpy as np

def hadamard_4():
    state = np.zeros(16, dtype=complex)
    state[0] = 1.0
    H = (1/np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex)
    I = np.eye(2, dtype=complex)
    H_full = np.kron(np.kron(np.kron(H, H), H), H)
    state = H_full @ state
    return state

result = hadamard_4()
''',

    'teleport': '''
import numpy as np
import random

def teleport(input_state):
    state = np.kron(input_state, np.array([1, 0, 0, 0], dtype=complex))
    H = (1/np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex)
    I = np.eye(2, dtype=complex)
    CNOT = np.array([
        [1, 0, 0, 0], [0, 1, 0, 0],
        [0, 0, 0, 1], [0, 0, 1, 0]
    ], dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    H_1 = np.kron(np.kron(I, H), I)
    state = H_1 @ state
    CNOT_12 = np.kron(I, CNOT)
    state = CNOT_12 @ state
    CNOT_01 = np.kron(CNOT, I)
    state = CNOT_01 @ state
    H_0 = np.kron(np.kron(H, I), I)
    state = H_0 @ state
    probs = np.abs(state)**2
    outcome = random.choices(range(8), weights=probs)[0]
    m0, m1 = (outcome >> 2) & 1, (outcome >> 1) & 1
    bob_state = np.zeros(2, dtype=complex)
    for i in range(2):
        idx = (m0 << 2) | (m1 << 1) | i
        bob_state[i] = state[idx]
    bob_state = bob_state / np.linalg.norm(bob_state)
    if m1: bob_state = X @ bob_state
    if m0: bob_state = Z @ bob_state
    return bob_state

input_state = np.array([1, 0], dtype=complex)
result = teleport(input_state)
''',

    'superdense': '''
import numpy as np
import random

def superdense(bit1, bit2):
    state = np.array([1, 0, 0, 0], dtype=complex)
    H = (1/np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex)
    I = np.eye(2, dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    CNOT = np.array([
        [1, 0, 0, 0], [0, 1, 0, 0],
        [0, 0, 0, 1], [0, 0, 1, 0]
    ], dtype=complex)
    H_full = np.kron(H, I)
    state = H_full @ state
    state = CNOT @ state
    if bit1:
        X_full = np.kron(X, I)
        state = X_full @ state
    if bit2:
        Z_full = np.kron(Z, I)
        state = Z_full @ state
    state = CNOT @ state
    H_full = np.kron(H, I)
    state = H_full @ state
    probs = np.abs(state)**2
    return probs

result = superdense(1, 0)
''',

    'phase_kickback': '''
import numpy as np

def phase_kickback():
    state = np.array([1, 0, 0, 0], dtype=complex)
    H = (1/np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex)
    I = np.eye(2, dtype=complex)
    CNOT = np.array([
        [1, 0, 0, 0], [0, 1, 0, 0],
        [0, 0, 0, 1], [0, 0, 1, 0]
    ], dtype=complex)
    H_full = np.kron(H, I)
    state = H_full @ state
    state = CNOT @ state
    state = H_full @ state
    probs = np.abs(state)**2
    return probs

result = phase_kickback()
''',

    'grover_oracle': '''
import numpy as np

def grover_step():
    state = np.array([1, 0], dtype=complex)
    H = (1/np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    state = H @ state
    state = Z @ state
    state = H @ state
    return state

result = grover_step()
''',
}


# =============================================================================
# Formal Complexity Analysis
# =============================================================================

@dataclass
class FormalComplexityResult:
    """Formal complexity analysis result with full details."""
    benchmark: str
    description: str
    category: str
    qubits: int
    
    # A1 metrics
    a1_tokens: int
    a1_bits: float
    a1_vocab_size: int
    
    # Classical metrics
    classical_tokens: int
    classical_bits: float
    classical_vocab_size: int
    classical_lines: int
    
    # Ratios
    token_ratio: float
    bits_ratio: float
    
    # Algorithmic probability (Chaitin-style)
    prob_a1: float       # 2^(-K_A1)
    prob_classical: float  # 2^(-K_Classical)
    prob_ratio: float     # P(A1) / P(Classical)


def analyze_benchmark(name: str) -> FormalComplexityResult:
    """Perform formal complexity analysis on a single benchmark."""
    a1_info = EXTENDED_BENCHMARKS_A1[name]
    a1_code = a1_info['code']
    classical_code = EXTENDED_BENCHMARKS_NUMPY[name]
    
    # A1 analysis
    a1_metrics = A1Metrics()
    a1_result = a1_metrics.analyze(a1_code)
    
    # Classical analysis
    classical_metrics = ClassicalMetrics()
    classical_result = classical_metrics.analyze(classical_code)
    
    # Calculate ratios
    token_ratio = classical_result.token_count / max(a1_result.token_count, 1)
    bits_ratio = classical_result.bits / max(a1_result.bits, 1)
    
    # Algorithmic probabilities (Chaitin: P = 2^(-K))
    prob_a1 = 2 ** (-a1_result.bits)
    prob_classical = 2 ** (-classical_result.bits)
    prob_ratio = prob_a1 / max(prob_classical, 1e-300)
    
    return FormalComplexityResult(
        benchmark=name,
        description=a1_info['description'],
        category=a1_info['category'],
        qubits=a1_info['qubits'],
        a1_tokens=a1_result.token_count,
        a1_bits=a1_result.bits,
        a1_vocab_size=a1_result.vocabulary_size,
        classical_tokens=classical_result.token_count,
        classical_bits=classical_result.bits,
        classical_vocab_size=classical_result.vocabulary_size,
        classical_lines=classical_result.source_lines,
        token_ratio=token_ratio,
        bits_ratio=bits_ratio,
        prob_a1=prob_a1,
        prob_classical=prob_classical,
        prob_ratio=prob_ratio
    )


def run_full_analysis() -> List[FormalComplexityResult]:
    """Run full complexity analysis on all benchmarks."""
    results = []
    for name in EXTENDED_BENCHMARKS_A1.keys():
        if name in EXTENDED_BENCHMARKS_NUMPY:
            result = analyze_benchmark(name)
            results.append(result)
    return results


# =============================================================================
# Statistical Summary
# =============================================================================

@dataclass
class StatisticalSummary:
    """Statistical summary of complexity comparisons."""
    n_benchmarks: int
    
    # Token statistics
    mean_token_ratio: float
    std_token_ratio: float
    min_token_ratio: float
    max_token_ratio: float
    
    # Bits statistics
    mean_bits_ratio: float
    std_bits_ratio: float
    min_bits_ratio: float
    max_bits_ratio: float
    
    # Probability statistics
    mean_log_prob_ratio: float  # log2(P_A1 / P_Classical)
    
    # Success criteria
    all_above_10x: bool


def compute_statistics(results: List[FormalComplexityResult]) -> StatisticalSummary:
    """Compute statistical summary from results."""
    token_ratios = [r.token_ratio for r in results]
    bits_ratios = [r.bits_ratio for r in results]
    log_prob_ratios = [math.log2(r.prob_ratio) if r.prob_ratio > 0 else 0 for r in results]
    
    n = len(results)
    
    mean_token = sum(token_ratios) / n
    std_token = math.sqrt(sum((x - mean_token)**2 for x in token_ratios) / n)
    
    mean_bits = sum(bits_ratios) / n
    std_bits = math.sqrt(sum((x - mean_bits)**2 for x in bits_ratios) / n)
    
    mean_log_prob = sum(log_prob_ratios) / n
    
    return StatisticalSummary(
        n_benchmarks=n,
        mean_token_ratio=mean_token,
        std_token_ratio=std_token,
        min_token_ratio=min(token_ratios),
        max_token_ratio=max(token_ratios),
        mean_bits_ratio=mean_bits,
        std_bits_ratio=std_bits,
        min_bits_ratio=min(bits_ratios),
        max_bits_ratio=max(bits_ratios),
        mean_log_prob_ratio=mean_log_prob,
        all_above_10x=all(r >= 10 for r in token_ratios)
    )


# =============================================================================
# Output Formatters
# =============================================================================

def print_detailed_report(results: List[FormalComplexityResult], stats: StatisticalSummary):
    """Print detailed comparison report."""
    print("=" * 80)
    print("A1 vs Classical Complexity Analysis — Phase 20")
    print("=" * 80)
    print(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Benchmarks: {stats.n_benchmarks}")
    print()
    
    # Detailed results table
    print("-" * 80)
    print(f"{'Benchmark':<15} {'Qubits':<7} {'A1 Tok':<8} {'NumPy Tok':<10} {'Ratio':<8} {'A1 Bits':<10} {'NumPy Bits':<12}")
    print("-" * 80)
    
    for r in results:
        print(f"{r.benchmark:<15} {r.qubits:<7} {r.a1_tokens:<8} {r.classical_tokens:<10} "
              f"{r.token_ratio:<8.1f} {r.a1_bits:<10.1f} {r.classical_bits:<12.1f}")
    
    print("-" * 80)
    
    # Statistical summary
    print("\n" + "=" * 80)
    print("Statistical Summary")
    print("=" * 80)
    print(f"\nToken Ratio:  {stats.mean_token_ratio:.1f}x ± {stats.std_token_ratio:.1f}x "
          f"(range: {stats.min_token_ratio:.1f}x - {stats.max_token_ratio:.1f}x)")
    print(f"Bits Ratio:   {stats.mean_bits_ratio:.1f}x ± {stats.std_bits_ratio:.1f}x "
          f"(range: {stats.min_bits_ratio:.1f}x - {stats.max_bits_ratio:.1f}x)")
    print(f"\nAlgorithmic Probability Gain: 2^{stats.mean_log_prob_ratio:.0f} (average)")
    
    # Success criteria
    print("\n" + "=" * 80)
    print("Success Criteria")
    print("=" * 80)
    print(f"All benchmarks > 10x token ratio: {'✅ YES' if stats.all_above_10x else '❌ NO'}")
    
    # Category breakdown
    print("\n" + "=" * 80)
    print("Category Breakdown")
    print("=" * 80)
    
    categories = {}
    for r in results:
        if r.category not in categories:
            categories[r.category] = []
        categories[r.category].append(r)
    
    for cat, cat_results in categories.items():
        avg_ratio = sum(r.token_ratio for r in cat_results) / len(cat_results)
        print(f"{cat.capitalize():<15}: {avg_ratio:.1f}x average ({len(cat_results)} benchmarks)")


def generate_latex_table(results: List[FormalComplexityResult]) -> str:
    """Generate LaTeX table for paper inclusion."""
    lines = [
        "\\begin{table}[ht]",
        "\\centering",
        "\\caption{A1 vs Classical Complexity Comparison}",
        "\\begin{tabular}{lcccccc}",
        "\\toprule",
        "Benchmark & Qubits & A1 Tokens & NumPy Tokens & Ratio & A1 Bits & NumPy Bits \\\\",
        "\\midrule"
    ]
    
    for r in results:
        lines.append(f"{r.benchmark.replace('_', '-')} & {r.qubits} & {r.a1_tokens} & "
                    f"{r.classical_tokens} & {r.token_ratio:.1f}x & "
                    f"{r.a1_bits:.0f} & {r.classical_bits:.0f} \\\\")
    
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}"
    ])
    
    return "\n".join(lines)


def generate_json_report(results: List[FormalComplexityResult], stats: StatisticalSummary) -> str:
    """Generate JSON report for programmatic use."""
    data = {
        'generated': datetime.now().isoformat(),
        'statistics': {
            'n_benchmarks': stats.n_benchmarks,
            'mean_token_ratio': stats.mean_token_ratio,
            'std_token_ratio': stats.std_token_ratio,
            'mean_bits_ratio': stats.mean_bits_ratio,
            'all_above_10x': stats.all_above_10x
        },
        'results': [
            {
                'benchmark': r.benchmark,
                'description': r.description,
                'category': r.category,
                'qubits': r.qubits,
                'a1_tokens': r.a1_tokens,
                'a1_bits': r.a1_bits,
                'classical_tokens': r.classical_tokens,
                'classical_bits': r.classical_bits,
                'token_ratio': r.token_ratio,
                'bits_ratio': r.bits_ratio
            }
            for r in results
        ]
    }
    return json.dumps(data, indent=2)


# =============================================================================
# Main
# =============================================================================

def main():
    """Run full Phase 20 analysis."""
    print("Running Phase 20: Complexity Metrics Analysis...\n")
    
    # Run analysis
    results = run_full_analysis()
    stats = compute_statistics(results)
    
    # Print detailed report
    print_detailed_report(results, stats)
    
    # Generate LaTeX table
    print("\n" + "=" * 80)
    print("LaTeX Table (for paper)")
    print("=" * 80)
    print(generate_latex_table(results))
    
    # Save JSON report
    json_path = os.path.join(os.path.dirname(__file__), 'complexity_results.json')
    with open(json_path, 'w') as f:
        f.write(generate_json_report(results, stats))
    print(f"\nJSON report saved to: {json_path}")
    
    # Return for testing
    return results, stats


if __name__ == "__main__":
    main()

