"""
A1 Metrics: Complexity Measurement Engine

This module implements Kolmogorov complexity approximation for A1 programs
and provides comparison with classical (NumPy) implementations.

Definition (A1-Complexity):
    K_A1(p) = |tokens(p)| × log₂(|V_A1|)
    
where:
    - tokens(p): program tokens (excluding parentheses)
    - |V_A1|: A1 vocabulary size (≈32)

Author: Hiroshi Kohashiguchi
Date: December 2025
"""

import math
import re
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass

# Handle both direct execution and package import
try:
    from .gates import VOCABULARY_SIZE, QUANTUM_GATES, A1_VOCABULARY
except ImportError:
    from gates import VOCABULARY_SIZE, QUANTUM_GATES, A1_VOCABULARY


# =============================================================================
# A1 Complexity Counter
# =============================================================================

@dataclass
class ComplexityResult:
    """
    Result of complexity measurement.
    
    Attributes:
        token_count: Number of tokens (excluding parentheses)
        vocabulary_size: Size of the vocabulary used
        bits: Estimated bits (tokens × log2(vocab_size))
        tokens: List of actual tokens
    """
    token_count: int
    vocabulary_size: int
    bits: float
    tokens: List[str]
    
    def __str__(self) -> str:
        return (f"ComplexityResult(tokens={self.token_count}, "
                f"bits={self.bits:.1f})")


class A1Metrics:
    """
    Complexity measurement for A1 programs.
    
    This class counts tokens in A1 source code and estimates
    Kolmogorov complexity using the formula:
    
        K_A1(p) ≈ |tokens(p)| × log₂(|V_A1|)
    
    Example:
        >>> metrics = A1Metrics()
        >>> result = metrics.analyze("(CNOT (H 0) 1)")
        >>> print(result.token_count)  # 4
        >>> print(result.bits)         # ~20
    """
    
    def __init__(self, vocabulary_size: int = VOCABULARY_SIZE):
        """
        Initialize the metrics calculator.
        
        Args:
            vocabulary_size: Size of A1 vocabulary (default: 32)
        """
        self.vocabulary_size = vocabulary_size
        self.log2_vocab = math.log2(vocabulary_size)
    
    def tokenize(self, source: str) -> List[str]:
        """
        Tokenize A1 source code, excluding parentheses.
        
        Args:
            source: A1 source code string
            
        Returns:
            List of tokens (symbols, numbers)
        """
        # Remove comments
        lines = source.split('\n')
        cleaned_lines = []
        for line in lines:
            comment_pos = line.find(';')
            if comment_pos != -1:
                line = line[:comment_pos]
            cleaned_lines.append(line)
        source = ' '.join(cleaned_lines)
        
        # Add spaces around parentheses
        source = source.replace('(', ' ').replace(')', ' ')
        
        # Split and filter
        tokens = [t for t in source.split() if t]
        
        return tokens
    
    def count_tokens(self, source: str) -> int:
        """
        Count tokens in A1 source code.
        
        Parentheses are not counted as they are structural.
        
        Args:
            source: A1 source code string
            
        Returns:
            Number of tokens
        """
        return len(self.tokenize(source))
    
    def complexity(self, source: str) -> float:
        """
        Calculate Kolmogorov complexity approximation in bits.
        
        Args:
            source: A1 source code string
            
        Returns:
            Estimated complexity in bits
        """
        token_count = self.count_tokens(source)
        return token_count * self.log2_vocab
    
    def analyze(self, source: str) -> ComplexityResult:
        """
        Perform full complexity analysis.
        
        Args:
            source: A1 source code string
            
        Returns:
            ComplexityResult with all metrics
        """
        tokens = self.tokenize(source)
        token_count = len(tokens)
        bits = token_count * self.log2_vocab
        
        return ComplexityResult(
            token_count=token_count,
            vocabulary_size=self.vocabulary_size,
            bits=bits,
            tokens=tokens
        )


# =============================================================================
# Classical (NumPy) Complexity Counter
# =============================================================================

@dataclass
class ClassicalComplexityResult:
    """
    Result of classical complexity measurement.
    """
    token_count: int
    vocabulary_size: int
    bits: float
    source_lines: int
    
    def __str__(self) -> str:
        return (f"ClassicalComplexityResult(tokens={self.token_count}, "
                f"bits={self.bits:.1f}, lines={self.source_lines})")


class ClassicalMetrics:
    """
    Complexity measurement for classical (NumPy) implementations.
    
    Uses the same token-based approach but with NumPy vocabulary.
    Classical vocabulary is larger (~256) due to:
    - NumPy functions (array, dot, kron, sqrt, etc.)
    - Operators (+, -, *, /, @, etc.)
    - Python keywords (import, def, return, etc.)
    - Literals (numbers, strings)
    
    Example:
        >>> metrics = ClassicalMetrics()
        >>> result = metrics.analyze(numpy_bell_code)
        >>> print(result.bits)  # ~800+
    """
    
    # Classical vocabulary is larger
    VOCABULARY_SIZE = 256
    
    def __init__(self):
        self.log2_vocab = math.log2(self.VOCABULARY_SIZE)
    
    def tokenize(self, source: str) -> List[str]:
        """
        Tokenize Python/NumPy source code.
        
        A simplified tokenizer that counts identifiers, operators, etc.
        """
        # Remove comments
        source = re.sub(r'#.*', '', source)
        
        # Remove string literals (replace with placeholder)
        source = re.sub(r'"[^"]*"', 'STR', source)
        source = re.sub(r"'[^']*'", 'STR', source)
        
        # Tokenize: split on whitespace and punctuation
        # Keep operators as separate tokens
        tokens = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*|[0-9]+\.?[0-9]*|[^\s\w]', source)
        
        # Filter out some noise
        tokens = [t for t in tokens if t.strip()]
        
        return tokens
    
    def count_tokens(self, source: str) -> int:
        """Count tokens in classical source code."""
        return len(self.tokenize(source))
    
    def complexity(self, source: str) -> float:
        """Calculate complexity in bits."""
        token_count = self.count_tokens(source)
        return token_count * self.log2_vocab
    
    def analyze(self, source: str) -> ClassicalComplexityResult:
        """Perform full complexity analysis."""
        tokens = self.tokenize(source)
        token_count = len(tokens)
        bits = token_count * self.log2_vocab
        source_lines = len([l for l in source.split('\n') if l.strip()])
        
        return ClassicalComplexityResult(
            token_count=token_count,
            vocabulary_size=self.VOCABULARY_SIZE,
            bits=bits,
            source_lines=source_lines
        )


# =============================================================================
# Comparison
# =============================================================================

@dataclass
class ComparisonResult:
    """
    Comparison between A1 and classical complexity.
    """
    task: str
    a1_result: ComplexityResult
    classical_result: ClassicalComplexityResult
    token_ratio: float
    bits_ratio: float
    
    def __str__(self) -> str:
        return (f"Comparison({self.task}):\n"
                f"  A1: {self.a1_result.token_count} tokens, {self.a1_result.bits:.1f} bits\n"
                f"  Classical: {self.classical_result.token_count} tokens, {self.classical_result.bits:.1f} bits\n"
                f"  Ratio: {self.token_ratio:.1f}x tokens, {self.bits_ratio:.1f}x bits")


def compare(task: str, a1_code: str, classical_code: str) -> ComparisonResult:
    """
    Compare A1 and classical implementations of the same task.
    
    Args:
        task: Name of the task (e.g., "Bell state")
        a1_code: A1 source code
        classical_code: Classical (NumPy) source code
        
    Returns:
        ComparisonResult with metrics and ratios
    """
    a1_metrics = A1Metrics()
    classical_metrics = ClassicalMetrics()
    
    a1_result = a1_metrics.analyze(a1_code)
    classical_result = classical_metrics.analyze(classical_code)
    
    token_ratio = classical_result.token_count / max(a1_result.token_count, 1)
    bits_ratio = classical_result.bits / max(a1_result.bits, 1)
    
    return ComparisonResult(
        task=task,
        a1_result=a1_result,
        classical_result=classical_result,
        token_ratio=token_ratio,
        bits_ratio=bits_ratio
    )


# =============================================================================
# Benchmark Definitions
# =============================================================================

# A1 code for benchmarks
BENCHMARK_A1 = {
    'bell': '(CNOT (H 0) 1)',
    
    'ghz': '(CNOT (CNOT (H 0) 1) 2)',
    
    'teleport': '''
        ; Quantum teleportation protocol
        (DEFINE teleport
            (LAMBDA (psi alice bob)
                (BEGIN
                    ; Create entangled pair
                    (CNOT (H alice) bob)
                    ; Bell measurement on psi and alice
                    (CNOT psi alice)
                    (H psi)
                    (MEASURE psi)
                    (MEASURE alice)
                )))
        (teleport 0 1 2)
    ''',
}

# NumPy code for benchmarks
BENCHMARK_NUMPY = {
    'bell': '''
import numpy as np

def bell_state():
    # Initial state |00>
    state = np.array([1, 0, 0, 0], dtype=complex)
    
    # Hadamard gate
    H = (1/np.sqrt(2)) * np.array([
        [1, 1],
        [1, -1]
    ], dtype=complex)
    
    # CNOT gate
    CNOT = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0]
    ], dtype=complex)
    
    # Identity for tensor product
    I = np.eye(2, dtype=complex)
    
    # Apply H to first qubit: H ⊗ I
    H_full = np.kron(H, I)
    state = H_full @ state
    
    # Apply CNOT
    state = CNOT @ state
    
    return state

result = bell_state()
''',
    
    'ghz': '''
import numpy as np

def ghz_state():
    # Initial state |000>
    state = np.zeros(8, dtype=complex)
    state[0] = 1.0
    
    # Gates
    H = (1/np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex)
    I = np.eye(2, dtype=complex)
    CNOT = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0]
    ], dtype=complex)
    
    # H on qubit 0
    H_full = np.kron(np.kron(H, I), I)
    state = H_full @ state
    
    # CNOT on qubits 0,1
    CNOT_01 = np.kron(CNOT, I)
    state = CNOT_01 @ state
    
    # CNOT on qubits 1,2
    CNOT_12 = np.kron(I, CNOT)
    state = CNOT_12 @ state
    
    return state

result = ghz_state()
''',
    
    'teleport': '''
import numpy as np
import random

def teleport(input_state):
    # Create 3-qubit state: input ⊗ |00>
    state = np.kron(input_state, np.array([1, 0, 0, 0], dtype=complex))
    
    # Gates
    H = (1/np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex)
    I = np.eye(2, dtype=complex)
    CNOT = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0]
    ], dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    
    # Create entangled pair between Alice (qubit 1) and Bob (qubit 2)
    # H on qubit 1
    H_1 = np.kron(np.kron(I, H), I)
    state = H_1 @ state
    
    # CNOT on qubits 1,2
    CNOT_12 = np.kron(I, CNOT)
    state = CNOT_12 @ state
    
    # Bell measurement: CNOT then H on input qubit
    # CNOT on qubits 0,1
    CNOT_01 = np.kron(CNOT, I)
    state = CNOT_01 @ state
    
    # H on qubit 0
    H_0 = np.kron(np.kron(H, I), I)
    state = H_0 @ state
    
    # Measure qubits 0 and 1 (simulated)
    probs = np.abs(state)**2
    outcome = random.choices(range(8), weights=probs)[0]
    m0, m1 = (outcome >> 2) & 1, (outcome >> 1) & 1
    
    # Apply corrections to Bob's qubit
    bob_state = np.zeros(2, dtype=complex)
    for i in range(2):
        idx = (m0 << 2) | (m1 << 1) | i
        bob_state[i] = state[idx]
    bob_state = bob_state / np.linalg.norm(bob_state)
    
    if m1:
        bob_state = X @ bob_state
    if m0:
        bob_state = Z @ bob_state
    
    return bob_state

# Test
input_state = np.array([1, 0], dtype=complex)  # |0>
result = teleport(input_state)
''',
}


def run_benchmarks() -> List[ComparisonResult]:
    """
    Run all benchmarks and return comparison results.
    
    Returns:
        List of ComparisonResult for each benchmark
    """
    results = []
    
    for name in ['bell', 'ghz', 'teleport']:
        result = compare(
            task=name.capitalize(),
            a1_code=BENCHMARK_A1[name],
            classical_code=BENCHMARK_NUMPY[name]
        )
        results.append(result)
    
    return results


def print_benchmark_report():
    """Print a formatted benchmark report."""
    results = run_benchmarks()
    
    print("=" * 60)
    print("A1 vs Classical Complexity Comparison")
    print("=" * 60)
    print()
    
    print(f"{'Task':<15} {'A1 Tokens':<12} {'NumPy Tokens':<15} {'Token Ratio':<12}")
    print("-" * 60)
    
    for r in results:
        print(f"{r.task:<15} {r.a1_result.token_count:<12} "
              f"{r.classical_result.token_count:<15} {r.token_ratio:.1f}x")
    
    print()
    print(f"{'Task':<15} {'A1 Bits':<12} {'NumPy Bits':<15} {'Bits Ratio':<12}")
    print("-" * 60)
    
    for r in results:
        print(f"{r.task:<15} {r.a1_result.bits:<12.1f} "
              f"{r.classical_result.bits:<15.1f} {r.bits_ratio:.1f}x")
    
    print()
    print("=" * 60)
    print(f"A1 Vocabulary Size: {VOCABULARY_SIZE}")
    print(f"Classical Vocabulary Size: {ClassicalMetrics.VOCABULARY_SIZE}")
    print("=" * 60)


if __name__ == "__main__":
    print_benchmark_report()

