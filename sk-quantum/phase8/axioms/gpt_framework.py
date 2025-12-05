"""
Generalized Probabilistic Theories (GPTs) Framework

This module provides a framework for analyzing computation models
using GPTs, enabling systematic comparison between classical and
quantum theories.

Key concepts:
- State Space: Convex set of allowed states
- Effects: Linear functionals giving measurement probabilities
- Transformations: Allowed operations on states

References:
- [Barrett2007] Information processing in generalized probabilistic theories
- [Janotta2014] Generalized probabilistic theories
"""

import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple, Optional, Callable
from enum import Enum
import itertools


class StateSpaceType(Enum):
    """Types of state spaces in GPTs."""
    SIMPLEX = "simplex"           # Classical (n-1 simplex for n outcomes)
    BALL = "ball"                  # Quantum-like (Bloch ball for qubit)
    HYPERCUBE = "hypercube"        # Box world
    POLYGON = "polygon"            # Regular polygon state spaces
    GENERAL_CONVEX = "general"     # General convex set


@dataclass
class StateSpace:
    """
    Represents a state space in GPTs.
    
    A state space is a convex subset of R^d that contains all
    physically allowed states.
    
    Attributes:
        dimension: Dimension of the embedding space
        extreme_points: Vertices of the convex set (pure states)
        space_type: Type of the state space
    """
    dimension: int
    extreme_points: np.ndarray  # Shape: (n_extreme, dimension)
    space_type: StateSpaceType
    
    def __post_init__(self):
        """Validate state space properties."""
        if len(self.extreme_points) == 0:
            raise ValueError("State space must have at least one extreme point")
        if self.extreme_points.shape[1] != self.dimension:
            raise ValueError(f"Extreme points dimension mismatch: "
                           f"expected {self.dimension}, got {self.extreme_points.shape[1]}")
    
    def is_valid_state(self, state: np.ndarray, tol: float = 1e-10) -> bool:
        """Check if a state is in the state space (convex hull of extreme points)."""
        # For simplicity, we check if state is a convex combination
        # This is a simplified check; full implementation would use LP
        return self._in_convex_hull(state, tol)
    
    def _in_convex_hull(self, point: np.ndarray, tol: float) -> bool:
        """Check if point is in convex hull of extreme points."""
        from scipy.optimize import linprog
        
        n_extreme = len(self.extreme_points)
        
        # We want to find coefficients c_i >= 0 with sum = 1
        # such that sum(c_i * v_i) = point
        # This is a linear feasibility problem
        
        # Minimize 0 (feasibility)
        c = np.zeros(n_extreme)
        
        # Equality constraints: A_eq @ x = b_eq
        # sum(c_i * v_i) = point AND sum(c_i) = 1
        A_eq = np.vstack([
            self.extreme_points.T,  # (dim, n_extreme)
            np.ones((1, n_extreme))  # sum constraint
        ])
        b_eq = np.concatenate([point, [1.0]])
        
        # Bounds: c_i >= 0
        bounds = [(0, None) for _ in range(n_extreme)]
        
        try:
            result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
            return result.success
        except:
            return False
    
    @property
    def n_extreme_points(self) -> int:
        """Number of extreme points (pure states)."""
        return len(self.extreme_points)
    
    def mixed_state(self, weights: np.ndarray) -> np.ndarray:
        """Create a mixed state from weights on extreme points."""
        if len(weights) != self.n_extreme_points:
            raise ValueError("Weights must match number of extreme points")
        if not np.isclose(np.sum(weights), 1.0):
            raise ValueError("Weights must sum to 1")
        if np.any(weights < 0):
            raise ValueError("Weights must be non-negative")
        return np.sum(weights[:, np.newaxis] * self.extreme_points, axis=0)


@dataclass
class Effect:
    """
    Represents an effect (measurement outcome) in GPTs.
    
    An effect is a linear functional e: StateSpace -> [0,1]
    giving the probability of a measurement outcome.
    
    Attributes:
        vector: The effect vector in the dual space
        name: Optional name for the effect
    """
    vector: np.ndarray
    name: str = ""
    
    def probability(self, state: np.ndarray) -> float:
        """Compute probability of this effect given a state."""
        prob = np.dot(self.vector, state)
        # Clip to [0, 1] for numerical stability
        return np.clip(prob, 0.0, 1.0)


@dataclass 
class Measurement:
    """
    A measurement is a collection of effects that sum to the unit effect.
    
    Attributes:
        effects: List of effects forming the measurement
        name: Optional name for the measurement
    """
    effects: List[Effect]
    name: str = ""
    
    def probabilities(self, state: np.ndarray) -> np.ndarray:
        """Compute probability distribution for this measurement."""
        probs = np.array([e.probability(state) for e in self.effects])
        # Normalize for numerical stability
        total = np.sum(probs)
        if total > 0:
            probs = probs / total
        return probs
    
    @property
    def n_outcomes(self) -> int:
        """Number of measurement outcomes."""
        return len(self.effects)


class Transformation(ABC):
    """
    Abstract base class for transformations in GPTs.
    
    A transformation maps states to states while preserving
    the convex structure.
    """
    
    @abstractmethod
    def apply(self, state: np.ndarray) -> np.ndarray:
        """Apply transformation to a state."""
        pass
    
    @abstractmethod
    def is_reversible(self) -> bool:
        """Check if transformation is reversible."""
        pass


class LinearTransformation(Transformation):
    """
    Linear transformation represented by a matrix.
    
    Attributes:
        matrix: The transformation matrix
        name: Optional name
    """
    
    def __init__(self, matrix: np.ndarray, name: str = ""):
        self.matrix = matrix
        self.name = name
    
    def apply(self, state: np.ndarray) -> np.ndarray:
        """Apply linear transformation."""
        return self.matrix @ state
    
    def is_reversible(self) -> bool:
        """Check if matrix is invertible."""
        try:
            np.linalg.inv(self.matrix)
            return True
        except np.linalg.LinAlgError:
            return False
    
    def compose(self, other: 'LinearTransformation') -> 'LinearTransformation':
        """Compose two linear transformations."""
        return LinearTransformation(
            self.matrix @ other.matrix,
            f"{self.name} ∘ {other.name}"
        )


class GPT:
    """
    A Generalized Probabilistic Theory.
    
    Combines state space, effects, and transformations into
    a complete theory.
    
    Attributes:
        state_space: The allowed states
        measurements: Available measurements
        transformations: Allowed transformations
        name: Name of the theory
    """
    
    def __init__(
        self,
        state_space: StateSpace,
        measurements: List[Measurement],
        transformations: List[Transformation],
        name: str = ""
    ):
        self.state_space = state_space
        self.measurements = measurements
        self.transformations = transformations
        self.name = name
    
    def __repr__(self) -> str:
        return (f"GPT(name='{self.name}', "
                f"dim={self.state_space.dimension}, "
                f"n_pure={self.state_space.n_extreme_points}, "
                f"n_measurements={len(self.measurements)}, "
                f"n_transformations={len(self.transformations)})")


# =============================================================================
# Classical Computation Model as GPT
# =============================================================================

def create_classical_bit_gpt(n_bits: int = 1) -> GPT:
    """
    Create a GPT representing classical n-bit computation.
    
    State space: (2^n - 1)-simplex (probability distributions over 2^n states)
    Transformations: Permutation matrices (reversible classical computation)
    
    Args:
        n_bits: Number of classical bits
        
    Returns:
        GPT representing classical computation
    """
    n_states = 2 ** n_bits
    
    # State space: vertices of simplex in R^{n_states}
    # Each vertex is a standard basis vector (pure state)
    extreme_points = np.eye(n_states)
    
    state_space = StateSpace(
        dimension=n_states,
        extreme_points=extreme_points,
        space_type=StateSpaceType.SIMPLEX
    )
    
    # Measurements: computational basis measurement
    effects = [Effect(np.eye(n_states)[i], name=f"|{i:0{n_bits}b}⟩") 
               for i in range(n_states)]
    comp_basis_measurement = Measurement(effects, name="computational_basis")
    
    # Transformations: all permutations (reversible classical)
    transformations = []
    for perm in itertools.permutations(range(n_states)):
        matrix = np.zeros((n_states, n_states))
        for i, j in enumerate(perm):
            matrix[j, i] = 1.0
        transformations.append(LinearTransformation(matrix, name=f"perm_{perm}"))
    
    return GPT(
        state_space=state_space,
        measurements=[comp_basis_measurement],
        transformations=transformations,
        name=f"Classical_{n_bits}bit"
    )


def create_qubit_gpt() -> GPT:
    """
    Create a GPT representing a single qubit.
    
    State space: Bloch ball (3D ball embedded in 4D)
    Transformations: Rotations (SU(2) represented as SO(3))
    
    Returns:
        GPT representing a qubit
    """
    # Bloch ball representation:
    # State ρ = (1/2)(I + r·σ) where |r| ≤ 1
    # We represent states as (1, r_x, r_y, r_z) in R^4
    
    # Extreme points: surface of Bloch sphere
    # For simplicity, we use vertices of an octahedron approximation
    bloch_vertices = np.array([
        [1, 1, 0, 0],   # |0⟩
        [1, -1, 0, 0],  # |1⟩
        [1, 0, 1, 0],   # |+⟩
        [1, 0, -1, 0],  # |-⟩
        [1, 0, 0, 1],   # |+i⟩
        [1, 0, 0, -1],  # |-i⟩
    ])
    
    state_space = StateSpace(
        dimension=4,
        extreme_points=bloch_vertices,
        space_type=StateSpaceType.BALL
    )
    
    # Measurements: Pauli measurements
    # Z measurement
    z_plus = Effect(np.array([0.5, 0.5, 0, 0]), name="|0⟩")
    z_minus = Effect(np.array([0.5, -0.5, 0, 0]), name="|1⟩")
    z_measurement = Measurement([z_plus, z_minus], name="Z")
    
    # X measurement  
    x_plus = Effect(np.array([0.5, 0, 0.5, 0]), name="|+⟩")
    x_minus = Effect(np.array([0.5, 0, -0.5, 0]), name="|-⟩")
    x_measurement = Measurement([x_plus, x_minus], name="X")
    
    # Y measurement
    y_plus = Effect(np.array([0.5, 0, 0, 0.5]), name="|+i⟩")
    y_minus = Effect(np.array([0.5, 0, 0, -0.5]), name="|-i⟩")
    y_measurement = Measurement([y_plus, y_minus], name="Y")
    
    # Transformations: Pauli rotations (simplified)
    # Identity
    I = LinearTransformation(np.eye(4), name="I")
    
    # X rotation (π): flips z component
    X = LinearTransformation(np.diag([1, 1, -1, -1]), name="X")
    
    # Z rotation (π): flips x, y components  
    Z = LinearTransformation(np.diag([1, -1, -1, 1]), name="Z")
    
    # Hadamard: swaps x and z
    H_matrix = np.array([
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
        [0, 1, 0, 0]
    ])
    H = LinearTransformation(H_matrix, name="H")
    
    return GPT(
        state_space=state_space,
        measurements=[z_measurement, x_measurement, y_measurement],
        transformations=[I, X, Z, H],
        name="Qubit"
    )


# =============================================================================
# State Space Comparison
# =============================================================================

@dataclass
class StateSpaceComparison:
    """
    Compare two state spaces and identify structural differences.
    """
    gpt1: GPT
    gpt2: GPT
    
    def dimension_ratio(self) -> float:
        """Ratio of dimensions."""
        return self.gpt2.state_space.dimension / self.gpt1.state_space.dimension
    
    def extreme_point_ratio(self) -> float:
        """Ratio of extreme points."""
        return (self.gpt2.state_space.n_extreme_points / 
                self.gpt1.state_space.n_extreme_points)
    
    def is_simplex(self, gpt: GPT) -> bool:
        """Check if state space is a simplex."""
        return gpt.state_space.space_type == StateSpaceType.SIMPLEX
    
    def is_ball(self, gpt: GPT) -> bool:
        """Check if state space is a ball."""
        return gpt.state_space.space_type == StateSpaceType.BALL
    
    def structural_difference(self) -> dict:
        """
        Analyze structural differences between the two GPTs.
        
        Returns:
            Dictionary with comparison results
        """
        return {
            'gpt1_name': self.gpt1.name,
            'gpt2_name': self.gpt2.name,
            'gpt1_type': self.gpt1.state_space.space_type.value,
            'gpt2_type': self.gpt2.state_space.space_type.value,
            'dimension_ratio': self.dimension_ratio(),
            'extreme_point_ratio': self.extreme_point_ratio(),
            'gpt1_is_simplex': self.is_simplex(self.gpt1),
            'gpt2_is_simplex': self.is_simplex(self.gpt2),
            'gpt1_is_ball': self.is_ball(self.gpt1),
            'gpt2_is_ball': self.is_ball(self.gpt2),
            'gpt1_n_transformations': len(self.gpt1.transformations),
            'gpt2_n_transformations': len(self.gpt2.transformations),
        }


def analyze_simplex_to_ball_gap(n_bits: int = 1) -> dict:
    """
    Analyze what additional structure is needed to go from
    simplex (classical) to ball (quantum).
    
    This addresses the key question:
    "What transforms the state space from simplex to Bloch sphere?"
    
    Args:
        n_bits: Number of bits to analyze
        
    Returns:
        Analysis of the gap between classical and quantum
    """
    classical = create_classical_bit_gpt(n_bits)
    quantum = create_qubit_gpt()  # 1 qubit for comparison
    
    comparison = StateSpaceComparison(classical, quantum)
    result = comparison.structural_difference()
    
    # Additional analysis
    result['gap_analysis'] = {
        'classical_pure_states': 2 ** n_bits,
        'quantum_pure_states': 'continuous (sphere surface)',
        'classical_mixed_states': 'simplex interior',
        'quantum_mixed_states': 'ball interior',
        'key_difference': 'continuous superposition vs discrete alternatives',
        'missing_structure': [
            'continuous amplitude',
            'complex phase',
            'non-orthogonal pure states',
            'non-commuting measurements'
        ]
    }
    
    return result


if __name__ == "__main__":
    # Demo: Compare classical and quantum GPTs
    print("=" * 60)
    print("GPT Framework Demo")
    print("=" * 60)
    
    # Create GPTs
    classical_1bit = create_classical_bit_gpt(1)
    classical_2bit = create_classical_bit_gpt(2)
    qubit = create_qubit_gpt()
    
    print(f"\n{classical_1bit}")
    print(f"{classical_2bit}")
    print(f"{qubit}")
    
    # Analyze gap
    print("\n" + "=" * 60)
    print("Simplex → Ball Gap Analysis")
    print("=" * 60)
    
    gap = analyze_simplex_to_ball_gap(1)
    for key, value in gap.items():
        if isinstance(value, dict):
            print(f"\n{key}:")
            for k, v in value.items():
                print(f"  {k}: {v}")
        else:
            print(f"{key}: {value}")

