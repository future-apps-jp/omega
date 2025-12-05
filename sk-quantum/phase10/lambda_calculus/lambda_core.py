"""
Lambda Calculus Core Implementation

This module provides a minimal implementation of untyped lambda calculus
for analyzing its algebraic structure and comparing with SK combinatory logic.

Key concepts:
- Lambda terms: variables, abstractions, applications
- β-reduction: (λx.M) N → M[x := N]
- Normal forms and Church-Rosser property

Goal: Verify that lambda calculus exhibits the same "classical" behavior
as SK combinatory logic, supporting the universality of our results.
"""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Set, List, Dict, Tuple
from enum import Enum
import hashlib


# =============================================================================
# Lambda Term AST
# =============================================================================

class Term(ABC):
    """Abstract base class for lambda terms."""
    
    @abstractmethod
    def free_vars(self) -> Set[str]:
        """Return set of free variables in this term."""
        pass
    
    @abstractmethod
    def substitute(self, var: str, replacement: 'Term') -> 'Term':
        """Substitute replacement for var in this term."""
        pass
    
    @abstractmethod
    def __eq__(self, other) -> bool:
        pass
    
    @abstractmethod
    def __hash__(self) -> int:
        pass
    
    @abstractmethod
    def to_string(self) -> str:
        """Convert to string representation."""
        pass
    
    def __repr__(self) -> str:
        return self.to_string()


@dataclass(frozen=True)
class Var(Term):
    """Variable term."""
    name: str
    
    def free_vars(self) -> Set[str]:
        return {self.name}
    
    def substitute(self, var: str, replacement: Term) -> Term:
        if self.name == var:
            return replacement
        return self
    
    def __eq__(self, other) -> bool:
        return isinstance(other, Var) and self.name == other.name
    
    def __hash__(self) -> int:
        return hash(('Var', self.name))
    
    def to_string(self) -> str:
        return self.name


@dataclass(frozen=True)
class Abs(Term):
    """Abstraction (λx.M)."""
    var: str
    body: Term
    
    def free_vars(self) -> Set[str]:
        return self.body.free_vars() - {self.var}
    
    def substitute(self, var: str, replacement: Term) -> Term:
        if var == self.var:
            # Bound variable, no substitution
            return self
        if self.var in replacement.free_vars():
            # Need alpha conversion to avoid capture
            new_var = fresh_var(self.var, self.body.free_vars() | replacement.free_vars())
            new_body = self.body.substitute(self.var, Var(new_var))
            return Abs(new_var, new_body.substitute(var, replacement))
        return Abs(self.var, self.body.substitute(var, replacement))
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, Abs):
            return False
        # Alpha equivalence: λx.M = λy.M[x:=y] if y not free in M
        if self.var == other.var:
            return self.body == other.body
        if other.var not in self.body.free_vars():
            return self.body.substitute(self.var, Var(other.var)) == other.body
        return False
    
    def __hash__(self) -> int:
        # Use canonical form for hashing
        return hash(('Abs', self.var, hash(self.body)))
    
    def to_string(self) -> str:
        return f"(λ{self.var}.{self.body.to_string()})"


@dataclass(frozen=True)
class App(Term):
    """Application (M N)."""
    func: Term
    arg: Term
    
    def free_vars(self) -> Set[str]:
        return self.func.free_vars() | self.arg.free_vars()
    
    def substitute(self, var: str, replacement: Term) -> Term:
        return App(
            self.func.substitute(var, replacement),
            self.arg.substitute(var, replacement)
        )
    
    def __eq__(self, other) -> bool:
        return (isinstance(other, App) and 
                self.func == other.func and 
                self.arg == other.arg)
    
    def __hash__(self) -> int:
        return hash(('App', hash(self.func), hash(self.arg)))
    
    def to_string(self) -> str:
        func_str = self.func.to_string()
        arg_str = self.arg.to_string()
        # Add parentheses for clarity
        if isinstance(self.arg, App):
            arg_str = f"({arg_str})"
        return f"({func_str} {arg_str})"


def fresh_var(base: str, avoid: Set[str]) -> str:
    """Generate a fresh variable name not in avoid set."""
    if base not in avoid:
        return base
    i = 0
    while f"{base}{i}" in avoid:
        i += 1
    return f"{base}{i}"


# =============================================================================
# Parser
# =============================================================================

def parse(s: str) -> Term:
    """
    Parse a lambda calculus expression.
    
    Syntax:
    - Variables: lowercase letters or names
    - Abstraction: λx.M or \\x.M
    - Application: M N (left associative)
    
    Examples:
    - "x"
    - "λx.x"
    - "(λx.x) y"
    - "λf.λx.f (f x)"
    """
    tokens = tokenize(s)
    term, rest = parse_term(tokens)
    if rest:
        raise SyntaxError(f"Unexpected tokens: {rest}")
    return term


def tokenize(s: str) -> List[str]:
    """Tokenize input string."""
    # Replace λ and \ with LAMBDA token
    s = s.replace('λ', ' LAMBDA ')
    s = s.replace('\\', ' LAMBDA ')
    s = s.replace('.', ' DOT ')
    s = s.replace('(', ' ( ')
    s = s.replace(')', ' ) ')
    return [t for t in s.split() if t]


def parse_term(tokens: List[str]) -> Tuple[Term, List[str]]:
    """Parse a term from tokens."""
    if not tokens:
        raise SyntaxError("Unexpected end of input")
    
    if tokens[0] == 'LAMBDA':
        # Abstraction
        return parse_abstraction(tokens[1:])
    else:
        # Application (left associative)
        return parse_application(tokens)


def parse_abstraction(tokens: List[str]) -> Tuple[Term, List[str]]:
    """Parse λx.M"""
    if not tokens:
        raise SyntaxError("Expected variable after λ")
    
    var = tokens[0]
    rest = tokens[1:]
    
    if not rest or rest[0] != 'DOT':
        raise SyntaxError("Expected '.' after variable in abstraction")
    
    body, rest = parse_term(rest[1:])
    return Abs(var, body), rest


def parse_application(tokens: List[str]) -> Tuple[Term, List[str]]:
    """Parse application (left associative)."""
    term, rest = parse_atom(tokens)
    
    while rest and rest[0] not in (')', 'DOT'):
        arg, rest = parse_atom(rest)
        term = App(term, arg)
    
    return term, rest


def parse_atom(tokens: List[str]) -> Tuple[Term, List[str]]:
    """Parse atomic term (variable or parenthesized expression)."""
    if not tokens:
        raise SyntaxError("Unexpected end of input")
    
    if tokens[0] == '(':
        term, rest = parse_term(tokens[1:])
        if not rest or rest[0] != ')':
            raise SyntaxError("Expected ')'")
        return term, rest[1:]
    elif tokens[0] == 'LAMBDA':
        return parse_abstraction(tokens[1:])
    else:
        # Variable
        return Var(tokens[0]), tokens[1:]


# =============================================================================
# β-Reduction
# =============================================================================

class ReductionStrategy(Enum):
    """Reduction strategies for λ-calculus."""
    NORMAL = "normal"           # Leftmost outermost first
    APPLICATIVE = "applicative" # Leftmost innermost first (call-by-value)
    LAZY = "lazy"               # Leftmost outermost, no reduction under λ


def find_redex(term: Term, strategy: ReductionStrategy = ReductionStrategy.NORMAL) -> Optional[Tuple[Term, 'Redex']]:
    """
    Find a β-redex in the term according to the given strategy.
    
    Returns (context, redex) or None if in normal form.
    """
    if isinstance(term, Var):
        return None
    
    if isinstance(term, Abs):
        if strategy == ReductionStrategy.LAZY:
            # Don't reduce under lambda in lazy strategy
            return None
        result = find_redex(term.body, strategy)
        if result:
            inner_term, redex = result
            return (Abs(term.var, inner_term), redex)
        return None
    
    if isinstance(term, App):
        # Check if this is a redex
        if isinstance(term.func, Abs):
            return (term, Redex(term.func.var, term.func.body, term.arg))
        
        # Try to find redex in subterms
        if strategy == ReductionStrategy.NORMAL:
            # Try function first
            result = find_redex(term.func, strategy)
            if result:
                inner_term, redex = result
                return (App(inner_term, term.arg), redex)
            result = find_redex(term.arg, strategy)
            if result:
                inner_term, redex = result
                return (App(term.func, inner_term), redex)
        else:
            # Try argument first (applicative)
            result = find_redex(term.arg, strategy)
            if result:
                inner_term, redex = result
                return (App(term.func, inner_term), redex)
            result = find_redex(term.func, strategy)
            if result:
                inner_term, redex = result
                return (App(inner_term, term.arg), redex)
        
        return None
    
    return None


@dataclass
class Redex:
    """A β-redex (λx.M) N."""
    var: str
    body: Term
    arg: Term
    
    def reduce(self) -> Term:
        """Perform β-reduction: (λx.M) N → M[x := N]"""
        return self.body.substitute(self.var, self.arg)


def beta_reduce_step(term: Term, strategy: ReductionStrategy = ReductionStrategy.NORMAL) -> Optional[Term]:
    """
    Perform one step of β-reduction.
    
    Returns the reduced term, or None if already in normal form.
    """
    result = find_redex(term, strategy)
    if result is None:
        return None
    
    context, redex = result
    reduced = redex.reduce()
    
    # Reconstruct term with reduced redex
    return substitute_in_context(context, redex, reduced)


def substitute_in_context(context: Term, redex: Redex, replacement: Term) -> Term:
    """Replace the redex in context with replacement."""
    if isinstance(context, App):
        if isinstance(context.func, Abs) and context.func.var == redex.var:
            return replacement
        # Recursively find and replace
        if isinstance(context.func, Abs):
            return App(context.func, substitute_in_context(context.arg, redex, replacement))
        return App(
            substitute_in_context(context.func, redex, replacement),
            context.arg
        )
    if isinstance(context, Abs):
        return Abs(context.var, substitute_in_context(context.body, redex, replacement))
    return context


def beta_reduce(term: Term, max_steps: int = 100, 
                strategy: ReductionStrategy = ReductionStrategy.NORMAL) -> Tuple[Term, List[Term]]:
    """
    Reduce term to normal form (if it exists within max_steps).
    
    Returns (final_term, reduction_history).
    """
    history = [term]
    current = term
    
    for _ in range(max_steps):
        next_term = beta_reduce_step(current, strategy)
        if next_term is None:
            # Normal form reached
            break
        history.append(next_term)
        current = next_term
    
    return current, history


def is_normal_form(term: Term) -> bool:
    """Check if term is in β-normal form."""
    return find_redex(term) is None


# =============================================================================
# Standard Combinators
# =============================================================================

# Identity: I = λx.x
I = Abs('x', Var('x'))

# K combinator: K = λx.λy.x
K = Abs('x', Abs('y', Var('x')))

# S combinator: S = λx.λy.λz.x z (y z)
S = Abs('x', Abs('y', Abs('z', 
    App(App(Var('x'), Var('z')), App(Var('y'), Var('z'))))))

# Omega: Ω = (λx.x x)(λx.x x) - diverges
omega_term = Abs('x', App(Var('x'), Var('x')))
Omega = App(omega_term, omega_term)

# Church numerals
def church_numeral(n: int) -> Term:
    """Create Church numeral for n."""
    # n = λf.λx.f^n x
    body = Var('x')
    for _ in range(n):
        body = App(Var('f'), body)
    return Abs('f', Abs('x', body))

# Boolean encodings
TRUE = Abs('t', Abs('f', Var('t')))   # λt.λf.t
FALSE = Abs('t', Abs('f', Var('f')))  # λt.λf.f


# =============================================================================
# Multiway Graph for Lambda Terms
# =============================================================================

@dataclass
class LambdaNode:
    """Node in the lambda reduction multiway graph."""
    term: Term
    term_id: str
    children: List['LambdaNode']
    
    def __init__(self, term: Term):
        self.term = term
        self.term_id = self._compute_id(term)
        self.children = []
    
    @staticmethod
    def _compute_id(term: Term) -> str:
        """Compute a unique ID for the term."""
        return hashlib.md5(term.to_string().encode()).hexdigest()[:8]


class LambdaMultiwayGraph:
    """
    Multiway graph of lambda term reductions.
    
    Similar to SK multiway graph, but for lambda calculus.
    """
    
    def __init__(self, root_term: Term, max_depth: int = 10, max_nodes: int = 100):
        self.root = LambdaNode(root_term)
        self.nodes: Dict[str, LambdaNode] = {self.root.term_id: self.root}
        self.max_depth = max_depth
        self.max_nodes = max_nodes
        self._build_graph(self.root, 0)
    
    def _build_graph(self, node: LambdaNode, depth: int):
        """Build the multiway graph by exploring all reduction paths."""
        if depth >= self.max_depth or len(self.nodes) >= self.max_nodes:
            return
        
        # Find all possible redexes (not just leftmost)
        redexes = self._find_all_redexes(node.term)
        
        for reduced_term in redexes:
            term_id = LambdaNode._compute_id(reduced_term)
            
            if term_id not in self.nodes:
                child = LambdaNode(reduced_term)
                self.nodes[term_id] = child
                node.children.append(child)
                self._build_graph(child, depth + 1)
            else:
                # Already visited, just add edge
                node.children.append(self.nodes[term_id])
    
    def _find_all_redexes(self, term: Term) -> List[Term]:
        """Find all possible one-step reductions of the term."""
        results = []
        self._collect_reductions(term, results)
        return results
    
    def _collect_reductions(self, term: Term, results: List[Term], 
                           make_term=lambda x: x):
        """Collect all possible reductions."""
        if isinstance(term, Var):
            return
        
        if isinstance(term, Abs):
            # Reduction inside abstraction
            def inner_make(t):
                return make_term(Abs(term.var, t))
            self._collect_reductions(term.body, results, inner_make)
        
        if isinstance(term, App):
            # If this is a redex, add its reduction
            if isinstance(term.func, Abs):
                reduced = term.func.body.substitute(term.func.var, term.arg)
                results.append(make_term(reduced))
            
            # Reductions inside function
            def func_make(t):
                return make_term(App(t, term.arg))
            self._collect_reductions(term.func, results, func_make)
            
            # Reductions inside argument
            def arg_make(t):
                return make_term(App(term.func, t))
            self._collect_reductions(term.arg, results, arg_make)
    
    @property
    def n_nodes(self) -> int:
        return len(self.nodes)
    
    @property
    def n_edges(self) -> int:
        return sum(len(n.children) for n in self.nodes.values())


if __name__ == "__main__":
    print("Lambda Calculus Core - Demo")
    print("=" * 50)
    
    # Test parsing
    print("\n--- Parsing ---")
    terms = [
        "x",
        "λx.x",
        "(λx.x) y",
        "λf.λx.f (f x)",
    ]
    for s in terms:
        t = parse(s)
        print(f"{s} → {t}")
    
    # Test reduction
    print("\n--- β-reduction ---")
    test_expr = "(λx.x) y"
    term = parse(test_expr)
    result, history = beta_reduce(term)
    print(f"{test_expr}")
    for i, t in enumerate(history):
        print(f"  Step {i}: {t}")
    
    # Test combinators
    print("\n--- Standard Combinators ---")
    print(f"I = {I}")
    print(f"K = {K}")
    print(f"S = {S}")
    
    # Test K combinator
    print("\n--- K a b → a ---")
    k_test = App(App(K, Var('a')), Var('b'))
    result, history = beta_reduce(k_test)
    for i, t in enumerate(history):
        print(f"  Step {i}: {t}")
    
    # Test multiway graph
    print("\n--- Multiway Graph ---")
    test_term = parse("(λx.x x) ((λy.y) z)")
    graph = LambdaMultiwayGraph(test_term, max_depth=5)
    print(f"Term: {test_term}")
    print(f"Nodes: {graph.n_nodes}, Edges: {graph.n_edges}")

