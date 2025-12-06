#!/usr/bin/env python3
"""
Phase 21: AWS Braket Experiments

Execute A1 programs on AWS Braket quantum hardware/simulators.

Features:
- A1 → Braket circuit transpilation
- SV1 simulator execution
- Statistical analysis (fidelity, TVD)
- Results export

Author: Hiroshi Kohashiguchi
Date: December 2025
"""

import sys
import os
import json
import time
import math
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import Counter

# Add paths for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'a1'))

# Load environment variables
from a1.config import config, load_dotenv
load_dotenv()

# AWS Braket imports
from braket.circuits import Circuit
from braket.devices import LocalSimulator
from braket.aws import AwsDevice

# Set AWS credentials from config
os.environ['AWS_ACCESS_KEY_ID'] = config.aws_access_key_id or ''
os.environ['AWS_SECRET_ACCESS_KEY'] = config.aws_secret_access_key or ''
os.environ['AWS_DEFAULT_REGION'] = config.aws_region


# =============================================================================
# A1 to Braket Transpiler
# =============================================================================

class A1BraketTranspiler:
    """
    Transpile A1 programs to Braket circuits.
    
    This is a simplified transpiler that handles basic A1 constructs.
    """
    
    def __init__(self):
        self.circuit = Circuit()
        self.env = {}
    
    def transpile(self, source: str) -> Circuit:
        """
        Transpile A1 source code to Braket circuit.
        
        Args:
            source: A1 source code
            
        Returns:
            Braket Circuit object
        """
        self.circuit = Circuit()
        tokens = self._tokenize(source)
        expressions = self._parse(tokens)
        
        for expr in expressions:
            self._eval(expr)
        
        return self.circuit
    
    def _tokenize(self, source: str) -> List[str]:
        """Tokenize A1 source."""
        # Remove comments
        lines = source.split('\n')
        cleaned = []
        for line in lines:
            comment_pos = line.find(';')
            if comment_pos != -1:
                line = line[:comment_pos]
            cleaned.append(line)
        source = ' '.join(cleaned)
        
        # Add spaces around parens
        source = source.replace('(', ' ( ').replace(')', ' ) ')
        return source.split()
    
    def _parse(self, tokens: List[str]) -> List:
        """Parse tokens into AST."""
        result = []
        while tokens:
            expr, tokens = self._read_expr(tokens)
            if expr is not None:
                result.append(expr)
        return result
    
    def _read_expr(self, tokens: List[str]) -> Tuple:
        """Read a single expression."""
        if not tokens:
            return None, tokens
        
        token = tokens[0]
        tokens = tokens[1:]
        
        if token == '(':
            lst = []
            while tokens and tokens[0] != ')':
                expr, tokens = self._read_expr(tokens)
                if expr is not None:
                    lst.append(expr)
            if tokens:
                tokens = tokens[1:]  # Skip ')'
            return lst, tokens
        elif token == ')':
            return None, tokens
        else:
            # Atom
            try:
                return int(token), tokens
            except ValueError:
                try:
                    return float(token), tokens
                except ValueError:
                    return token.upper(), tokens
    
    def _eval(self, expr) -> int:
        """Evaluate expression and return qubit index."""
        if isinstance(expr, (int, float)):
            return int(expr)
        
        if isinstance(expr, str):
            if expr in self.env:
                return self.env[expr]
            return expr
        
        if isinstance(expr, list):
            if not expr:
                return 0
            
            op = expr[0]
            
            # Special forms
            if op == 'DEFINE':
                name = expr[1]
                if isinstance(name, list):
                    # Function definition
                    fname = name[0]
                    params = name[1:]
                    body = expr[2]
                    self.env[fname] = ('LAMBDA', params, body)
                else:
                    value = self._eval(expr[2])
                    self.env[str(name)] = value
                return 0
            
            elif op == 'LAMBDA':
                return ('LAMBDA', expr[1], expr[2])
            
            elif op == 'BEGIN':
                result = 0
                for e in expr[1:]:
                    result = self._eval(e)
                return result
            
            elif op == 'IF':
                cond = self._eval(expr[1])
                if cond:
                    return self._eval(expr[2])
                else:
                    return self._eval(expr[3]) if len(expr) > 3 else 0
            
            # Quantum gates
            elif op == 'H':
                q = self._eval(expr[1])
                self.circuit.h(q)
                return q
            
            elif op == 'X':
                q = self._eval(expr[1])
                self.circuit.x(q)
                return q
            
            elif op == 'Y':
                q = self._eval(expr[1])
                self.circuit.y(q)
                return q
            
            elif op == 'Z':
                q = self._eval(expr[1])
                self.circuit.z(q)
                return q
            
            elif op == 'CNOT':
                c = self._eval(expr[1])
                t = self._eval(expr[2])
                self.circuit.cnot(c, t)
                return t
            
            elif op == 'CZ':
                c = self._eval(expr[1])
                t = self._eval(expr[2])
                self.circuit.cz(c, t)
                return t
            
            elif op == 'SWAP':
                q1 = self._eval(expr[1])
                q2 = self._eval(expr[2])
                self.circuit.swap(q1, q2)
                return q2
            
            elif op == 'RX':
                q = self._eval(expr[1])
                angle = self._eval(expr[2])
                self.circuit.rx(q, angle)
                return q
            
            elif op == 'RY':
                q = self._eval(expr[1])
                angle = self._eval(expr[2])
                self.circuit.ry(q, angle)
                return q
            
            elif op == 'RZ':
                q = self._eval(expr[1])
                angle = self._eval(expr[2])
                self.circuit.rz(q, angle)
                return q
            
            elif op == 'MEASURE':
                q = self._eval(expr[1])
                # Note: Braket handles measurement differently
                return q
            
            else:
                # Function call
                if op in self.env:
                    func = self.env[op]
                    if isinstance(func, tuple) and func[0] == 'LAMBDA':
                        _, params, body = func
                        args = [self._eval(a) for a in expr[1:]]
                        old_env = self.env.copy()
                        for p, a in zip(params, args):
                            self.env[str(p)] = a
                        result = self._eval(body)
                        self.env = old_env
                        return result
                
                return 0
        
        return 0


# =============================================================================
# Experiment Runner
# =============================================================================

@dataclass
class ExperimentResult:
    """Result of a single experiment."""
    benchmark: str
    backend: str
    shots: int
    counts: Dict[str, int]
    execution_time: float
    fidelity: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


def calculate_fidelity(counts: Dict[str, int], ideal: Dict[str, float]) -> float:
    """
    Calculate fidelity using Total Variation Distance.
    
    F = 1 - TVD(measured, ideal)
    TVD = 0.5 * sum(|p_measured - p_ideal|)
    """
    total = sum(counts.values())
    if total == 0:
        return 0.0
    
    measured = {k: v / total for k, v in counts.items()}
    
    all_keys = set(measured.keys()) | set(ideal.keys())
    tvd = 0.5 * sum(abs(measured.get(k, 0) - ideal.get(k, 0)) for k in all_keys)
    
    return 1 - tvd


# Benchmark definitions
BENCHMARKS = {
    'bell': {
        'a1_code': '(CNOT (H 0) 1)',
        'ideal': {'00': 0.5, '11': 0.5},
        'description': 'Bell state (|00⟩ + |11⟩)/√2',
        'qubits': 2
    },
    'ghz': {
        'a1_code': '(CNOT (CNOT (H 0) 1) 2)',
        'ideal': {'000': 0.5, '111': 0.5},
        'description': 'GHZ state (|000⟩ + |111⟩)/√2',
        'qubits': 3
    },
    'hadamard': {
        'a1_code': '(H 0)',
        'ideal': {'0': 0.5, '1': 0.5},
        'description': 'Single-qubit superposition',
        'qubits': 1
    },
    'superposition_2': {
        'a1_code': '(BEGIN (H 0) (H 1))',
        'ideal': {'00': 0.25, '01': 0.25, '10': 0.25, '11': 0.25},
        'description': '2-qubit uniform superposition',
        'qubits': 2
    },
}


# Global device cache to avoid re-creating devices
_device_cache = {}

def get_device(backend: str):
    """Get or create a device (cached)."""
    global _device_cache
    
    if backend in _device_cache:
        return _device_cache[backend]
    
    if backend == 'local':
        device = LocalSimulator()
    elif backend == 'sv1':
        device = AwsDevice(config.simulator_sv1)
    elif backend == 'ionq':
        if not config.enable_qpu_execution:
            raise ValueError("QPU execution is disabled. Set ENABLE_QPU_EXECUTION=true")
        device = AwsDevice(config.device_ionq)
    elif backend == 'rigetti':
        if not config.enable_qpu_execution:
            raise ValueError("QPU execution is disabled. Set ENABLE_QPU_EXECUTION=true")
        device = AwsDevice(config.device_rigetti)
    else:
        raise ValueError(f"Unknown backend: {backend}")
    
    _device_cache[backend] = device
    return device


def run_experiment(
    benchmark_name: str,
    backend: str = 'local',
    shots: int = 1000,
    verbose: bool = True
) -> ExperimentResult:
    """
    Run a single experiment.
    
    Args:
        benchmark_name: Name of benchmark from BENCHMARKS
        backend: 'local', 'sv1', 'ionq', 'rigetti'
        shots: Number of shots
        verbose: Print progress
        
    Returns:
        ExperimentResult
    """
    if benchmark_name not in BENCHMARKS:
        raise ValueError(f"Unknown benchmark: {benchmark_name}")
    
    benchmark = BENCHMARKS[benchmark_name]
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Benchmark: {benchmark_name}")
        print(f"Description: {benchmark['description']}")
        print(f"Backend: {backend}")
        print(f"Shots: {shots}")
        print(f"{'='*60}")
    
    # Transpile A1 to Braket circuit
    transpiler = A1BraketTranspiler()
    circuit = transpiler.transpile(benchmark['a1_code'])
    
    # Add measurement to all qubits
    n_qubits = benchmark['qubits']
    
    if verbose:
        print(f"\nA1 Code: {benchmark['a1_code']}")
        print(f"Circuit:\n{circuit}")
    
    # Get cached device
    device = get_device(backend)
    
    # Run
    start_time = time.time()
    
    if backend == 'local':
        task = device.run(circuit, shots=shots)
        result = task.result()
    else:
        # AWS execution
        s3_location = (config.s3_bucket, config.s3_prefix)
        task = device.run(circuit, s3_destination_folder=s3_location, shots=shots)
        
        if verbose:
            print(f"\nTask ARN: {task.id}")
            print("Waiting for completion...")
        
        result = task.result()
    
    execution_time = time.time() - start_time
    
    # Get measurement counts
    counts = dict(result.measurement_counts)
    
    # Calculate fidelity
    fidelity = calculate_fidelity(counts, benchmark['ideal'])
    
    if verbose:
        print(f"\nResults:")
        print(f"  Counts: {counts}")
        print(f"  Fidelity: {fidelity:.4f}")
        print(f"  Execution time: {execution_time:.2f}s")
    
    return ExperimentResult(
        benchmark=benchmark_name,
        backend=backend,
        shots=shots,
        counts=counts,
        execution_time=execution_time,
        fidelity=fidelity
    )


def run_all_experiments(
    backend: str = 'local',
    shots: int = 1000,
    n_trials: int = 1
) -> List[ExperimentResult]:
    """Run all benchmarks multiple times."""
    results = []
    
    for benchmark_name in BENCHMARKS:
        for trial in range(n_trials):
            print(f"\n[Trial {trial + 1}/{n_trials}] Running {benchmark_name}...")
            result = run_experiment(benchmark_name, backend, shots)
            results.append(result)
    
    return results


def print_summary(results: List[ExperimentResult]):
    """Print summary of all experiments."""
    print("\n" + "=" * 70)
    print("EXPERIMENT SUMMARY")
    print("=" * 70)
    
    # Group by benchmark
    by_benchmark = {}
    for r in results:
        if r.benchmark not in by_benchmark:
            by_benchmark[r.benchmark] = []
        by_benchmark[r.benchmark].append(r)
    
    print(f"\n{'Benchmark':<20} {'Backend':<10} {'Shots':<8} {'Fidelity':<12} {'Time (s)':<10}")
    print("-" * 70)
    
    for benchmark, rs in by_benchmark.items():
        fidelities = [r.fidelity for r in rs]
        times = [r.execution_time for r in rs]
        
        mean_fid = sum(fidelities) / len(fidelities)
        mean_time = sum(times) / len(times)
        
        if len(fidelities) > 1:
            std_fid = math.sqrt(sum((f - mean_fid)**2 for f in fidelities) / len(fidelities))
            fid_str = f"{mean_fid:.4f} ± {std_fid:.4f}"
        else:
            fid_str = f"{mean_fid:.4f}"
        
        print(f"{benchmark:<20} {rs[0].backend:<10} {rs[0].shots:<8} {fid_str:<12} {mean_time:.2f}")
    
    print("=" * 70)


def save_results(results: List[ExperimentResult], output_path: str):
    """Save results to JSON file."""
    data = {
        'generated': datetime.now().isoformat(),
        'config': {
            'region': config.aws_region,
            's3_bucket': config.s3_bucket,
        },
        'results': [
            {
                'benchmark': r.benchmark,
                'backend': r.backend,
                'shots': r.shots,
                'counts': r.counts,
                'fidelity': r.fidelity,
                'execution_time': r.execution_time,
                'timestamp': r.timestamp
            }
            for r in results
        ]
    }
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    """Run Phase 21 experiments."""
    import argparse
    
    parser = argparse.ArgumentParser(description='A1 AWS Braket Experiments')
    parser.add_argument('--backend', default='local', 
                       choices=['local', 'sv1', 'ionq', 'rigetti'],
                       help='Backend to use')
    parser.add_argument('--benchmark', default='all',
                       help='Benchmark to run (or "all")')
    parser.add_argument('--shots', type=int, default=1000,
                       help='Number of shots')
    parser.add_argument('--trials', type=int, default=1,
                       help='Number of trials per benchmark')
    parser.add_argument('--output', default=None,
                       help='Output JSON file')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Phase 21: AWS Braket Experiments")
    print("=" * 70)
    print(f"\nBackend: {args.backend}")
    print(f"Shots: {args.shots}")
    print(f"Trials: {args.trials}")
    
    if args.benchmark == 'all':
        results = run_all_experiments(args.backend, args.shots, args.trials)
    else:
        results = []
        for _ in range(args.trials):
            r = run_experiment(args.benchmark, args.backend, args.shots)
            results.append(r)
    
    print_summary(results)
    
    # Save results
    if args.output:
        save_results(results, args.output)
    else:
        output_dir = os.path.dirname(__file__)
        output_path = os.path.join(output_dir, f'results_{args.backend}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
        save_results(results, output_path)
    
    return results


if __name__ == "__main__":
    main()

