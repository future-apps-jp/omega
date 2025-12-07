"""
DSL Self-Evolution Engine

Implements the mechanism for DSLs to evolve their own primitives.
This is the key mechanism for Phase 26: "Evolution of Operators"

The goal is to observe the spontaneous emergence of matrix-like operations
from a purely scalar DSL through evolutionary pressure.

Key Insight:
- When a pattern (e.g., repeated multiplication) becomes frequent and useful,
  it can be "compressed" into a new primitive operation
- This mimics how matrix operations might emerge from scalar operations
  when they provide compression advantage
"""

from typing import Dict, List, Set, Callable, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import Counter
import random
import copy
import numpy as np

from genesis.core.dsl import Gene, ValueGene, OperatorGene
from genesis.core.container import Container


@dataclass
class OperatorDefinition:
    """Definition of a DSL operator."""
    name: str
    arity: int  # Number of arguments
    implementation: Callable  # How to evaluate
    description: str = ""
    generation_born: int = 0  # When this operator was invented
    usage_count: int = 0


class EvolvableDSL:
    """
    A DSL that can evolve its own operators.
    
    Starts with basic scalar operations and can "invent" new operators
    when it discovers useful patterns.
    """
    
    # Initial primitive operations
    INITIAL_OPS = {
        'ADD': (2, lambda a, b: a + b),
        'SUB': (2, lambda a, b: a - b),
        'MUL': (2, lambda a, b: a * b),
        'CONST': (0, None),  # Special: constant value
        'VAR': (0, None),    # Special: variable reference
    }
    
    def __init__(self, name: str = "Evolvable"):
        self.name = name
        self.operators: Dict[str, OperatorDefinition] = {}
        self.generation = 0
        self.invention_history: List[Dict] = []
        
        # Initialize with basic ops
        for op_name, (arity, impl) in self.INITIAL_OPS.items():
            self.operators[op_name] = OperatorDefinition(
                name=op_name,
                arity=arity,
                implementation=impl,
                description="Primitive",
            )
    
    def get_operator(self, name: str) -> Optional[OperatorDefinition]:
        return self.operators.get(name)
    
    def add_operator(
        self,
        name: str,
        arity: int,
        implementation: Callable,
        description: str = "",
    ):
        """Add a new operator to the DSL."""
        self.operators[name] = OperatorDefinition(
            name=name,
            arity=arity,
            implementation=implementation,
            description=description,
            generation_born=self.generation,
        )
        self.invention_history.append({
            "generation": self.generation,
            "operator": name,
            "arity": arity,
            "description": description,
        })
    
    def has_matrix_ops(self) -> bool:
        """Check if DSL has matrix-like operations."""
        matrix_keywords = ['MATMUL', 'MATPOW', 'DOT', 'POWER', 'COMPOSE']
        return any(kw in op for op in self.operators for kw in matrix_keywords)


class EvolvableGene(Gene):
    """A gene that can use evolved operators."""
    
    def __init__(self, op: str, children: List['EvolvableGene'] = None, value: Any = None):
        super().__init__()
        self.op = op
        self.children = children or []
        self.value = value
    
    def evaluate(self, context: Dict[str, Any], dsl: EvolvableDSL = None) -> Any:
        """Evaluate using the DSL's current operators."""
        if self.op == 'CONST':
            return self.value
        
        if self.op == 'VAR':
            return context.get(self.value, 0)
        
        # Get operator from DSL
        if dsl:
            op_def = dsl.get_operator(self.op)
            if op_def and op_def.implementation:
                # Evaluate children
                child_values = [c.evaluate(context, dsl) for c in self.children]
                try:
                    result = op_def.implementation(*child_values)
                    # Overflow protection
                    if isinstance(result, (int, float)):
                        result = max(min(result, 1e12), -1e12)
                    return result
                except Exception:
                    return 0
        
        # Fallback to basic ops
        if self.op == 'ADD':
            return sum(c.evaluate(context, dsl) for c in self.children)
        if self.op == 'MUL':
            result = 1
            for c in self.children:
                result *= c.evaluate(context, dsl)
            return result
        
        return 0
    
    def size(self) -> int:
        if not self.children:
            return 1
        return 1 + sum(c.size() for c in self.children)
    
    def copy(self) -> 'EvolvableGene':
        return EvolvableGene(
            self.op,
            [c.copy() for c in self.children],
            self.value
        )
    
    def __str__(self) -> str:
        if self.op == 'CONST':
            return str(self.value)
        if self.op == 'VAR':
            return str(self.value)
        if not self.children:
            return self.op
        children_str = " ".join(str(c) for c in self.children)
        return f"({self.op} {children_str})"
    
    def get_all_nodes(self) -> List['EvolvableGene']:
        nodes = [self]
        for c in self.children:
            nodes.extend(c.get_all_nodes())
        return nodes
    
    def get_pattern_signature(self, depth: int = 2) -> str:
        """Get a structural signature for pattern detection."""
        if depth == 0 or not self.children:
            return self.op
        children_sigs = [c.get_pattern_signature(depth - 1) for c in self.children]
        return f"({self.op} {' '.join(children_sigs)})"


class PatternDetector:
    """
    Detects frequent patterns in programs that could be compressed
    into new operators.
    """
    
    def __init__(self, min_frequency: int = 5, min_size: int = 3):
        self.min_frequency = min_frequency
        self.min_size = min_size
        self.pattern_counts: Counter = Counter()
    
    def analyze_population(self, programs: List[EvolvableGene]):
        """Analyze a population for frequent patterns."""
        self.pattern_counts.clear()
        
        for program in programs:
            self._collect_patterns(program)
    
    def _collect_patterns(self, node: EvolvableGene, depth: int = 3):
        """Recursively collect pattern signatures."""
        if node.size() >= self.min_size:
            sig = node.get_pattern_signature(depth)
            self.pattern_counts[sig] += 1
        
        for child in node.children:
            self._collect_patterns(child, depth)
    
    def get_frequent_patterns(self) -> List[Tuple[str, int]]:
        """Get patterns that appear frequently enough to be worth compressing."""
        return [
            (pattern, count)
            for pattern, count in self.pattern_counts.most_common()
            if count >= self.min_frequency
        ]


class DSLEvolutionEngine:
    """
    Engine that evolves DSLs by inventing new operators.
    
    Key mechanism:
    1. Detect frequent patterns in successful programs
    2. If a pattern provides compression benefit, create new operator
    3. Replace pattern instances with new operator
    """
    
    def __init__(
        self,
        dsl: EvolvableDSL,
        invention_threshold: int = 5,
        compression_benefit: float = 0.3,
    ):
        self.dsl = dsl
        self.invention_threshold = invention_threshold
        self.compression_benefit = compression_benefit
        self.pattern_detector = PatternDetector(min_frequency=invention_threshold)
        self.invented_ops: List[str] = []
    
    def maybe_invent_operator(
        self,
        programs: List[EvolvableGene],
        fitness_scores: List[float],
    ) -> Optional[str]:
        """
        Analyze programs and potentially invent a new operator.
        
        Returns the name of the new operator if one was invented.
        """
        # Only analyze high-fitness programs
        threshold = np.percentile(fitness_scores, 70) if fitness_scores else 0
        good_programs = [
            p for p, f in zip(programs, fitness_scores)
            if f >= threshold
        ]
        
        if not good_programs:
            return None
        
        # Detect patterns
        self.pattern_detector.analyze_population(good_programs)
        frequent = self.pattern_detector.get_frequent_patterns()
        
        if not frequent:
            return None
        
        # Check each pattern for compression benefit
        for pattern, count in frequent:
            # Parse pattern to estimate size
            pattern_size = pattern.count('(') + pattern.count(')') // 2 + 1
            
            # Compression benefit: new op has size 1
            benefit = (pattern_size - 1) * count
            
            if benefit > self.compression_benefit * pattern_size * count:
                # Invent new operator!
                return self._create_operator_from_pattern(pattern, count)
        
        return None
    
    def _create_operator_from_pattern(self, pattern: str, frequency: int) -> str:
        """Create a new operator from a detected pattern."""
        # Generate unique name
        op_num = len(self.invented_ops)
        
        # Detect what kind of pattern this is
        if 'MUL' in pattern and pattern.count('MUL') >= 2:
            # Repeated multiplication -> POWER-like
            op_name = f"POWER_{op_num}"
            description = f"Discovered: repeated multiplication (freq={frequency})"
            
            def power_impl(base, exp=2):
                try:
                    return base ** int(exp)
                except:
                    return base * base
            
            self.dsl.add_operator(op_name, 2, power_impl, description)
            
        elif 'MUL' in pattern and 'ADD' in pattern:
            # Combined MUL and ADD -> COMPOSE-like
            op_name = f"COMPOSE_{op_num}"
            description = f"Discovered: multiply-add pattern (freq={frequency})"
            
            def compose_impl(a, b, c=1):
                return a * b + c
            
            self.dsl.add_operator(op_name, 3, compose_impl, description)
            
        else:
            # Generic new operator
            op_name = f"OP_{op_num}"
            description = f"Discovered pattern: {pattern[:30]}... (freq={frequency})"
            
            # Default implementation: identity or first child
            def generic_impl(*args):
                return args[0] if args else 0
            
            self.dsl.add_operator(op_name, 2, generic_impl, description)
        
        self.invented_ops.append(op_name)
        return op_name
    
    def inject_matrix_hint(self) -> str:
        """
        Inject a matrix operation as if it were "discovered".
        
        This simulates the scenario where evolution discovers that
        matrix operations provide massive compression.
        """
        # Add MATMUL-like operation
        op_name = "MATMUL"
        
        def matmul_impl(a, b):
            if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
                return np.dot(a, b)
            return a * b
        
        self.dsl.add_operator(
            op_name, 2, matmul_impl,
            "Matrix multiplication (emergent)"
        )
        
        # Add MATPOW-like operation
        op_name2 = "MATPOW"
        
        def matpow_impl(m, k):
            if isinstance(m, np.ndarray):
                return np.linalg.matrix_power(m, int(k))
            return m ** int(k)
        
        self.dsl.add_operator(
            op_name2, 2, matpow_impl,
            "Matrix power (emergent)"
        )
        
        self.invented_ops.extend([op_name, op_name2])
        return op_name


def generate_evolvable_program(
    dsl: EvolvableDSL,
    max_depth: int = 4,
    current_depth: int = 0,
) -> EvolvableGene:
    """Generate a random program using the current DSL vocabulary."""
    # Terminal condition
    if current_depth >= max_depth or (current_depth > 0 and random.random() < 0.3):
        if random.random() < 0.5:
            return EvolvableGene('CONST', value=random.randint(0, 5))
        else:
            return EvolvableGene('VAR', value='x')
    
    # Select operator (prefer newer operators slightly)
    ops = [op for op in dsl.operators if op not in ['CONST', 'VAR']]
    if not ops:
        return EvolvableGene('CONST', value=random.randint(0, 5))
    
    op = random.choice(ops)
    op_def = dsl.get_operator(op)
    arity = op_def.arity if op_def else 2
    
    children = [
        generate_evolvable_program(dsl, max_depth, current_depth + 1)
        for _ in range(arity)
    ]
    
    return EvolvableGene(op, children)

