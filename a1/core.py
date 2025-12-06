"""
A1 Core: Parser and Interpreter

This module implements the core of the A1 language:
- S-expression parser (tokenizer + reader)
- Environment-based interpreter
- Braket circuit generation

The key invariant is that `eval(expr)` is a homomorphism from 
A1-Expression to Braket-Circuit.

Author: Hiroshi Kohashiguchi
Date: December 2025
"""

from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass, field

# Try to import Braket, fall back to a mock for testing without AWS
try:
    from braket.circuits import Circuit
    BRAKET_AVAILABLE = True
except ImportError:
    BRAKET_AVAILABLE = False
    # Mock Circuit for local testing
    class Circuit:
        """Mock Circuit class for testing without Braket SDK."""
        def __init__(self):
            self.instructions = []
        
        def h(self, q: int) -> 'Circuit':
            self.instructions.append(('H', q))
            return self
        
        def x(self, q: int) -> 'Circuit':
            self.instructions.append(('X', q))
            return self
        
        def y(self, q: int) -> 'Circuit':
            self.instructions.append(('Y', q))
            return self
        
        def z(self, q: int) -> 'Circuit':
            self.instructions.append(('Z', q))
            return self
        
        def cnot(self, control: int, target: int) -> 'Circuit':
            self.instructions.append(('CNOT', control, target))
            return self
        
        def cz(self, control: int, target: int) -> 'Circuit':
            self.instructions.append(('CZ', control, target))
            return self
        
        def swap(self, q1: int, q2: int) -> 'Circuit':
            self.instructions.append(('SWAP', q1, q2))
            return self
        
        def rx(self, q: int, angle: float) -> 'Circuit':
            self.instructions.append(('RX', q, angle))
            return self
        
        def ry(self, q: int, angle: float) -> 'Circuit':
            self.instructions.append(('RY', q, angle))
            return self
        
        def rz(self, q: int, angle: float) -> 'Circuit':
            self.instructions.append(('RZ', q, angle))
            return self
        
        def measure(self, q: int) -> 'Circuit':
            # Braket uses different measurement API, but we'll handle it
            self.instructions.append(('MEASURE', q))
            return self
        
        def __str__(self) -> str:
            lines = ["Circuit:"]
            for inst in self.instructions:
                lines.append(f"  {inst}")
            return "\n".join(lines)
        
        def __repr__(self) -> str:
            return f"Circuit({self.instructions})"


# =============================================================================
# Tokenizer
# =============================================================================

def tokenize(source: str) -> List[str]:
    """
    Convert A1 source code into a list of tokens.
    
    Args:
        source: A1 source code string
        
    Returns:
        List of tokens (strings)
    
    Example:
        >>> tokenize("(CNOT (H 0) 1)")
        ['(', 'CNOT', '(', 'H', '0', ')', '1', ')']
    """
    # Add spaces around parentheses for easy splitting
    source = source.replace('(', ' ( ').replace(')', ' ) ')
    
    # Remove comments (lines starting with ;)
    lines = source.split('\n')
    cleaned_lines = []
    for line in lines:
        comment_pos = line.find(';')
        if comment_pos != -1:
            line = line[:comment_pos]
        cleaned_lines.append(line)
    source = ' '.join(cleaned_lines)
    
    # Split and filter empty strings
    tokens = source.split()
    return tokens


# =============================================================================
# Parser
# =============================================================================

class ParseError(Exception):
    """Raised when parsing fails."""
    pass


def parse(source: str) -> List[Any]:
    """
    Parse A1 source code into an AST (list of expressions).
    
    Args:
        source: A1 source code string
        
    Returns:
        List of parsed expressions
        
    Example:
        >>> parse("(H 0)")
        [['H', 0]]
        >>> parse("(CNOT (H 0) 1)")
        [['CNOT', ['H', 0], 1]]
    """
    tokens = tokenize(source)
    result = []
    
    while tokens:
        expr, tokens = read_from_tokens(tokens)
        if expr is not None:
            result.append(expr)
    
    return result


def read_from_tokens(tokens: List[str]) -> tuple:
    """
    Read an expression from a list of tokens.
    
    Args:
        tokens: List of tokens (modified in place)
        
    Returns:
        Tuple of (expression, remaining_tokens)
    """
    if not tokens:
        return None, tokens
    
    token = tokens[0]
    tokens = tokens[1:]
    
    if token == '(':
        # Read a list
        lst = []
        while tokens and tokens[0] != ')':
            expr, tokens = read_from_tokens(tokens)
            if expr is not None:
                lst.append(expr)
        
        if not tokens:
            raise ParseError("Unexpected EOF: missing ')'")
        
        # Pop the closing ')'
        tokens = tokens[1:]
        return lst, tokens
    
    elif token == ')':
        raise ParseError("Unexpected ')'")
    
    else:
        # Parse atom
        return parse_atom(token), tokens


def parse_atom(token: str) -> Union[int, float, str]:
    """
    Parse an atomic token into its appropriate type.
    
    Args:
        token: Token string
        
    Returns:
        int, float, or symbol (str)
    """
    # Try integer
    try:
        return int(token)
    except ValueError:
        pass
    
    # Try float
    try:
        return float(token)
    except ValueError:
        pass
    
    # Return as symbol (uppercase for consistency)
    return token.upper() if token.isalpha() else token


# =============================================================================
# Environment
# =============================================================================

@dataclass
class Environment:
    """
    Environment for variable bindings.
    
    Supports nested scopes via parent environment.
    """
    bindings: Dict[str, Any] = field(default_factory=dict)
    parent: Optional['Environment'] = None
    
    def get(self, name: str) -> Any:
        """Look up a variable in this environment or parents."""
        if name in self.bindings:
            return self.bindings[name]
        if self.parent is not None:
            return self.parent.get(name)
        raise NameError(f"Undefined variable: {name}")
    
    def set(self, name: str, value: Any) -> None:
        """Set a variable in this environment."""
        self.bindings[name] = value
    
    def extend(self, names: List[str], values: List[Any]) -> 'Environment':
        """Create a new environment with additional bindings."""
        new_env = Environment(parent=self)
        for name, value in zip(names, values):
            new_env.set(name, value)
        return new_env


# =============================================================================
# Lambda (Closure)
# =============================================================================

@dataclass
class Lambda:
    """
    A closure: function parameters, body, and captured environment.
    """
    params: List[str]
    body: Any
    env: Environment
    
    def __call__(self, *args) -> Any:
        """This will be handled by the interpreter."""
        raise NotImplementedError("Lambda must be called via interpreter.eval")


# =============================================================================
# A1 Interpreter
# =============================================================================

class A1:
    """
    The A1 Language Interpreter.
    
    A1 is a minimal Scheme dialect designed for quantum computation.
    Programs are S-expressions that evaluate to quantum circuits.
    
    Key invariant: eval(expr) produces a valid Braket circuit, and
    gate functions return the qubit index they acted on (for chaining).
    
    Example:
        >>> a1 = A1()
        >>> circuit = a1.run("(CNOT (H 0) 1)")
        >>> print(circuit)
    """
    
    def __init__(self):
        """Initialize the A1 interpreter with a fresh circuit and environment."""
        self.circuit = Circuit()
        self.env = self._make_global_env()
    
    def _make_global_env(self) -> Environment:
        """Create the global environment with built-in gates and functions."""
        env = Environment()
        
        # Register quantum gates
        # Each gate returns the qubit index it acted on (for chaining)
        
        # Single-qubit gates
        env.set('H', self._make_gate('h', 1))
        env.set('X', self._make_gate('x', 1))
        env.set('Y', self._make_gate('y', 1))
        env.set('Z', self._make_gate('z', 1))
        
        # Two-qubit gates
        env.set('CNOT', self._make_gate('cnot', 2))
        env.set('CZ', self._make_gate('cz', 2))
        env.set('SWAP', self._make_gate('swap', 2))
        
        # Rotation gates (qubit, angle)
        env.set('RX', self._make_rotation_gate('rx'))
        env.set('RY', self._make_rotation_gate('ry'))
        env.set('RZ', self._make_rotation_gate('rz'))
        
        # Measurement
        env.set('MEASURE', self._make_measure())
        
        # Math constants
        env.set('PI', 3.141592653589793)
        
        return env
    
    def _make_gate(self, gate_name: str, arity: int) -> Callable:
        """
        Create a gate function that applies a gate and returns qubit index.
        
        Args:
            gate_name: Name of the Braket gate method
            arity: Number of qubit arguments (1 or 2)
        """
        def gate_fn(*args):
            if len(args) != arity:
                raise TypeError(f"{gate_name.upper()} expects {arity} arguments, got {len(args)}")
            
            # Get the gate method from circuit
            method = getattr(self.circuit, gate_name)
            method(*[int(a) for a in args])
            
            # Return the last qubit index (for chaining)
            return int(args[-1])
        
        return gate_fn
    
    def _make_rotation_gate(self, gate_name: str) -> Callable:
        """Create a rotation gate function (qubit, angle)."""
        def rotation_fn(qubit, angle):
            method = getattr(self.circuit, gate_name)
            method(int(qubit), float(angle))
            return int(qubit)
        
        return rotation_fn
    
    def _make_measure(self) -> Callable:
        """Create the measurement function."""
        def measure_fn(qubit):
            self.circuit.measure(int(qubit))
            return int(qubit)
        
        return measure_fn
    
    def reset(self) -> None:
        """Reset the circuit to empty state."""
        self.circuit = Circuit()
    
    def eval(self, expr: Any, env: Optional[Environment] = None) -> Any:
        """
        Evaluate an A1 expression.
        
        Args:
            expr: Parsed A1 expression
            env: Environment (defaults to global)
            
        Returns:
            Result of evaluation (qubit index for gates, None for definitions)
        """
        if env is None:
            env = self.env
        
        # Number literal
        if isinstance(expr, (int, float)):
            return expr
        
        # Symbol lookup
        if isinstance(expr, str):
            return env.get(expr)
        
        # List (function application or special form)
        if isinstance(expr, list):
            if not expr:
                return None
            
            op = expr[0]
            
            # Special forms
            if op == 'DEFINE':
                # (DEFINE name value) or (DEFINE (name params...) body)
                if len(expr) == 3:
                    name = expr[1]
                    if isinstance(name, list):
                        # Function definition shorthand: (DEFINE (f x y) body)
                        fname = name[0]
                        params = name[1:]
                        body = expr[2]
                        env.set(fname, Lambda([str(p) for p in params], body, env))
                    else:
                        # Variable definition: (DEFINE x value)
                        value = self.eval(expr[2], env)
                        env.set(str(name), value)
                    return None
                else:
                    raise SyntaxError(f"DEFINE requires 2 arguments, got {len(expr) - 1}")
            
            elif op == 'LAMBDA':
                # (LAMBDA (params...) body)
                if len(expr) != 3:
                    raise SyntaxError(f"LAMBDA requires 2 arguments, got {len(expr) - 1}")
                params = expr[1]
                body = expr[2]
                return Lambda([str(p) for p in params], body, env)
            
            elif op == 'IF':
                # (IF condition then-expr else-expr)
                if len(expr) != 4:
                    raise SyntaxError(f"IF requires 3 arguments, got {len(expr) - 1}")
                condition = self.eval(expr[1], env)
                if condition:
                    return self.eval(expr[2], env)
                else:
                    return self.eval(expr[3], env)
            
            elif op == 'LET':
                # (LET ((name value) ...) body)
                if len(expr) != 3:
                    raise SyntaxError(f"LET requires 2 arguments, got {len(expr) - 1}")
                bindings = expr[1]
                body = expr[2]
                
                new_env = Environment(parent=env)
                for binding in bindings:
                    name = str(binding[0])
                    value = self.eval(binding[1], env)
                    new_env.set(name, value)
                
                return self.eval(body, new_env)
            
            elif op == 'QUOTE':
                # (QUOTE expr) - return unevaluated
                if len(expr) != 2:
                    raise SyntaxError(f"QUOTE requires 1 argument, got {len(expr) - 1}")
                return expr[1]
            
            elif op == 'BEGIN':
                # (BEGIN expr1 expr2 ...) - evaluate in sequence, return last
                result = None
                for e in expr[1:]:
                    result = self.eval(e, env)
                return result
            
            else:
                # Function application
                proc = self.eval(op, env)
                args = [self.eval(arg, env) for arg in expr[1:]]
                
                if isinstance(proc, Lambda):
                    # Create new environment with parameter bindings
                    new_env = proc.env.extend(proc.params, args)
                    return self.eval(proc.body, new_env)
                elif callable(proc):
                    return proc(*args)
                else:
                    raise TypeError(f"Cannot call {proc}")
        
        raise TypeError(f"Unknown expression type: {type(expr)}")
    
    def run(self, source: str) -> Circuit:
        """
        Parse and execute A1 source code.
        
        Args:
            source: A1 source code string
            
        Returns:
            The resulting Braket circuit
            
        Example:
            >>> a1 = A1()
            >>> circuit = a1.run("(CNOT (H 0) 1)")
        """
        expressions = parse(source)
        
        for expr in expressions:
            self.eval(expr)
        
        return self.circuit
    
    def run_fresh(self, source: str) -> Circuit:
        """
        Reset circuit and run source code.
        
        Useful for running independent programs.
        """
        self.reset()
        return self.run(source)


# =============================================================================
# Convenience Functions
# =============================================================================

def execute(source: str) -> Circuit:
    """
    Execute A1 source code and return the circuit.
    
    This is a convenience function that creates a fresh interpreter.
    
    Args:
        source: A1 source code string
        
    Returns:
        The resulting Braket circuit
    """
    a1 = A1()
    return a1.run(source)


# =============================================================================
# Main (for testing)
# =============================================================================

if __name__ == "__main__":
    # Test the parser
    print("=== Parser Tests ===")
    print(parse("(H 0)"))
    print(parse("(CNOT (H 0) 1)"))
    print(parse("(DEFINE make-bell (LAMBDA (q0 q1) (CNOT (H q0) q1)))"))
    
    # Test the interpreter
    print("\n=== Interpreter Tests ===")
    
    # Simple gate
    a1 = A1()
    circuit = a1.run("(H 0)")
    print(f"(H 0) => {circuit}")
    
    # Bell state
    a1 = A1()
    circuit = a1.run("(CNOT (H 0) 1)")
    print(f"(CNOT (H 0) 1) => {circuit}")
    
    # Function definition and call
    a1 = A1()
    circuit = a1.run("""
        (DEFINE make-bell
            (LAMBDA (q0 q1)
                (CNOT (H q0) q1)))
        (make-bell 0 1)
    """)
    print(f"make-bell => {circuit}")
    
    print("\n=== Success ===")

