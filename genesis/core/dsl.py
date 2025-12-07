"""
DSL (Domain-Specific Language) Abstraction

Defines the base classes for DSL genes and programs.
Each DSL represents a "physical law" in the artificial universe.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import copy
import uuid


class Gene(ABC):
    """
    Abstract base class for DSL genes (AST nodes).
    
    Each gene represents a computational primitive or operation.
    """
    
    def __init__(self):
        self.id = str(uuid.uuid4())[:8]
    
    @abstractmethod
    def evaluate(self, context: Dict[str, Any]) -> Any:
        """Execute this gene and return the result."""
        pass
    
    @abstractmethod
    def size(self) -> int:
        """Return the description length (Kolmogorov complexity proxy)."""
        pass
    
    @abstractmethod
    def copy(self) -> 'Gene':
        """Create a deep copy of this gene."""
        pass
    
    @abstractmethod
    def __str__(self) -> str:
        """Return a string representation (code form)."""
        pass
    
    def get_all_nodes(self) -> List['Gene']:
        """Return all nodes in the subtree rooted at this gene."""
        return [self]


class ValueGene(Gene):
    """A gene representing a constant value."""
    
    def __init__(self, value: Any):
        super().__init__()
        self.value = value
    
    def evaluate(self, context: Dict[str, Any]) -> Any:
        return self.value
    
    def size(self) -> int:
        return 1
    
    def copy(self) -> 'ValueGene':
        return ValueGene(self.value)
    
    def __str__(self) -> str:
        return str(self.value)


class VariableGene(Gene):
    """A gene representing a variable reference."""
    
    def __init__(self, name: str):
        super().__init__()
        self.name = name
    
    def evaluate(self, context: Dict[str, Any]) -> Any:
        return context.get(self.name, 0)
    
    def size(self) -> int:
        return 1
    
    def copy(self) -> 'VariableGene':
        return VariableGene(self.name)
    
    def __str__(self) -> str:
        return self.name


class OperatorGene(Gene):
    """A gene representing an operation with children."""
    
    def __init__(self, op: str, children: List[Gene]):
        super().__init__()
        self.op = op
        self.children = children
    
    @abstractmethod
    def evaluate(self, context: Dict[str, Any]) -> Any:
        pass
    
    def size(self) -> int:
        return 1 + sum(child.size() for child in self.children)
    
    def copy(self) -> 'OperatorGene':
        return self.__class__(self.op, [c.copy() for c in self.children])
    
    def __str__(self) -> str:
        children_str = " ".join(str(c) for c in self.children)
        return f"({self.op} {children_str})"
    
    def get_all_nodes(self) -> List[Gene]:
        nodes = [self]
        for child in self.children:
            nodes.extend(child.get_all_nodes())
        return nodes


class DSL:
    """
    A Domain-Specific Language definition.
    
    Encapsulates the vocabulary (primitives) and semantics of a DSL.
    """
    
    def __init__(self, name: str, primitives: List[str]):
        self.name = name
        self.primitives = set(primitives)
        self.vocabulary_size = len(primitives)
    
    def bits_per_token(self) -> float:
        """Bits needed to encode one token."""
        import math
        return math.log2(max(self.vocabulary_size, 2))
    
    def __str__(self) -> str:
        return f"DSL({self.name}, {len(self.primitives)} primitives)"
    
    def __repr__(self) -> str:
        return self.__str__()

