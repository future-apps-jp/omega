"""
Species A: Scalar DSL (The Classical Beings)

A DSL limited to scalar operations: ADD, MUL, SUB, IF, LOOP.
These are the "classical beings" that can only compute one thing at a time.

For the graph walk task, they must enumerate paths one by one,
leading to O(k) or O(N) description length.
"""

from typing import Any, Dict, List, Optional
import random

from genesis.core.dsl import Gene, ValueGene, VariableGene, OperatorGene, DSL


class ScalarOp(OperatorGene):
    """
    Scalar operation gene.
    
    Supported operations:
    - ADD: Addition
    - SUB: Subtraction
    - MUL: Multiplication
    - IF: Conditional (returns left if right > 0, else 0)
    """
    
    OPERATIONS = ['ADD', 'SUB', 'MUL', 'IF']
    
    def __init__(self, op: str, children: List[Gene]):
        assert op in self.OPERATIONS, f"Unknown operation: {op}"
        super().__init__(op, children)
    
    def evaluate(self, context: Dict[str, Any]) -> Any:
        if len(self.children) < 2:
            return 0
        
        left = self.children[0].evaluate(context)
        right = self.children[1].evaluate(context)
        
        # Convert to scalar if needed
        left = self._to_scalar(left)
        right = self._to_scalar(right)
        
        if self.op == 'ADD':
            return left + right
        elif self.op == 'SUB':
            return left - right
        elif self.op == 'MUL':
            # Overflow protection
            result = left * right
            return max(min(result, 1e12), -1e12)
        elif self.op == 'IF':
            return left if right > 0 else 0
        
        return 0
    
    def _to_scalar(self, value: Any) -> float:
        """Convert value to scalar."""
        import numpy as np
        if isinstance(value, np.ndarray):
            return float(np.sum(value))
        try:
            return float(value)
        except (ValueError, TypeError):
            return 0.0
    
    def copy(self) -> 'ScalarOp':
        return ScalarOp(self.op, [c.copy() for c in self.children])


class ScalarVal(ValueGene):
    """A scalar constant value."""
    
    def copy(self) -> 'ScalarVal':
        return ScalarVal(self.value)


class ScalarVar(VariableGene):
    """A scalar variable reference."""
    
    def copy(self) -> 'ScalarVar':
        return ScalarVar(self.name)


# DSL Definition
ScalarDSL = DSL(
    name="Scalar",
    primitives=['ADD', 'SUB', 'MUL', 'IF', 'CONST', 'VAR']
)


def generate_scalar_program(max_depth: int = 4, current_depth: int = 0) -> Gene:
    """
    Generate a random scalar program.
    
    Args:
        max_depth: Maximum tree depth
        current_depth: Current depth (for recursion)
    
    Returns:
        A random Gene tree
    """
    # Terminal condition
    if current_depth >= max_depth or (current_depth > 0 and random.random() < 0.3):
        if random.random() < 0.5:
            return ScalarVal(random.randint(0, 5))
        else:
            return ScalarVar("x")
    
    # Generate operation
    op = random.choice(ScalarOp.OPERATIONS)
    left = generate_scalar_program(max_depth, current_depth + 1)
    right = generate_scalar_program(max_depth, current_depth + 1)
    
    return ScalarOp(op, [left, right])


def generate_scalar_individual(depth: int = 4) -> Gene:
    """Generate a scalar DSL individual for evolution."""
    return generate_scalar_program(max_depth=depth)

