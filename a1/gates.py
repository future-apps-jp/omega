"""
A1 Gates: Quantum Gate Definitions

This module defines the quantum gates available in the A1 language.
Each gate has a cost of 1 token for complexity measurement.

Gate Categories:
- Single-qubit gates: H, X, Y, Z
- Two-qubit gates: CNOT, CZ, SWAP
- Rotation gates: RX, RY, RZ (with angle parameter)
- Measurement: MEASURE

Author: Hiroshi Kohashiguchi
Date: December 2025
"""

from dataclasses import dataclass
from typing import List, Tuple
import math


@dataclass
class GateDefinition:
    """
    Definition of a quantum gate for A1.
    
    Attributes:
        name: Gate name (e.g., 'H', 'CNOT')
        arity: Number of qubit arguments
        has_angle: Whether the gate takes an angle parameter
        cost: Token cost (always 1 for primitives)
        description: Human-readable description
    """
    name: str
    arity: int
    has_angle: bool = False
    cost: int = 1
    description: str = ""


# =============================================================================
# Gate Definitions
# =============================================================================

QUANTUM_GATES = {
    # Single-qubit gates
    'H': GateDefinition(
        name='H',
        arity=1,
        description='Hadamard gate: creates superposition'
    ),
    'X': GateDefinition(
        name='X',
        arity=1,
        description='Pauli-X gate: bit flip'
    ),
    'Y': GateDefinition(
        name='Y',
        arity=1,
        description='Pauli-Y gate: combined bit and phase flip'
    ),
    'Z': GateDefinition(
        name='Z',
        arity=1,
        description='Pauli-Z gate: phase flip'
    ),
    'S': GateDefinition(
        name='S',
        arity=1,
        description='S gate: sqrt(Z), pi/2 phase'
    ),
    'T': GateDefinition(
        name='T',
        arity=1,
        description='T gate: sqrt(S), pi/4 phase'
    ),
    
    # Two-qubit gates
    'CNOT': GateDefinition(
        name='CNOT',
        arity=2,
        description='Controlled-NOT: flips target if control is |1>'
    ),
    'CZ': GateDefinition(
        name='CZ',
        arity=2,
        description='Controlled-Z: applies Z to target if control is |1>'
    ),
    'SWAP': GateDefinition(
        name='SWAP',
        arity=2,
        description='SWAP gate: exchanges two qubits'
    ),
    
    # Rotation gates
    'RX': GateDefinition(
        name='RX',
        arity=1,
        has_angle=True,
        description='Rotation around X-axis by angle theta'
    ),
    'RY': GateDefinition(
        name='RY',
        arity=1,
        has_angle=True,
        description='Rotation around Y-axis by angle theta'
    ),
    'RZ': GateDefinition(
        name='RZ',
        arity=1,
        has_angle=True,
        description='Rotation around Z-axis by angle theta'
    ),
    
    # Measurement
    'MEASURE': GateDefinition(
        name='MEASURE',
        arity=1,
        description='Measurement in computational basis'
    ),
}


# =============================================================================
# Gate Matrices (for simulation without Braket)
# =============================================================================

def get_gate_matrix(gate_name: str, angle: float = None):
    """
    Get the unitary matrix for a gate.
    
    This is useful for local simulation without Braket.
    
    Args:
        gate_name: Name of the gate
        angle: Rotation angle (for RX, RY, RZ)
        
    Returns:
        2D numpy array representing the gate matrix
    """
    import numpy as np
    
    sqrt2_inv = 1 / math.sqrt(2)
    
    matrices = {
        'H': np.array([
            [sqrt2_inv, sqrt2_inv],
            [sqrt2_inv, -sqrt2_inv]
        ], dtype=complex),
        
        'X': np.array([
            [0, 1],
            [1, 0]
        ], dtype=complex),
        
        'Y': np.array([
            [0, -1j],
            [1j, 0]
        ], dtype=complex),
        
        'Z': np.array([
            [1, 0],
            [0, -1]
        ], dtype=complex),
        
        'S': np.array([
            [1, 0],
            [0, 1j]
        ], dtype=complex),
        
        'T': np.array([
            [1, 0],
            [0, np.exp(1j * math.pi / 4)]
        ], dtype=complex),
        
        'CNOT': np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0]
        ], dtype=complex),
        
        'CZ': np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, -1]
        ], dtype=complex),
        
        'SWAP': np.array([
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1]
        ], dtype=complex),
    }
    
    # Rotation gates
    if gate_name == 'RX' and angle is not None:
        c, s = math.cos(angle / 2), math.sin(angle / 2)
        return np.array([
            [c, -1j * s],
            [-1j * s, c]
        ], dtype=complex)
    
    elif gate_name == 'RY' and angle is not None:
        c, s = math.cos(angle / 2), math.sin(angle / 2)
        return np.array([
            [c, -s],
            [s, c]
        ], dtype=complex)
    
    elif gate_name == 'RZ' and angle is not None:
        return np.array([
            [np.exp(-1j * angle / 2), 0],
            [0, np.exp(1j * angle / 2)]
        ], dtype=complex)
    
    return matrices.get(gate_name)


# =============================================================================
# Gate Set Information
# =============================================================================

def get_all_gates() -> List[str]:
    """Return list of all gate names."""
    return list(QUANTUM_GATES.keys())


def get_single_qubit_gates() -> List[str]:
    """Return list of single-qubit gate names."""
    return [name for name, gate in QUANTUM_GATES.items() 
            if gate.arity == 1 and not gate.has_angle]


def get_two_qubit_gates() -> List[str]:
    """Return list of two-qubit gate names."""
    return [name for name, gate in QUANTUM_GATES.items() 
            if gate.arity == 2]


def get_rotation_gates() -> List[str]:
    """Return list of rotation gate names."""
    return [name for name, gate in QUANTUM_GATES.items() 
            if gate.has_angle]


def total_gate_cost(gates: List[Tuple[str, ...]]) -> int:
    """
    Calculate total token cost for a list of gates.
    
    Args:
        gates: List of (gate_name, *args) tuples
        
    Returns:
        Total cost (each gate costs 1)
    """
    return len(gates)


# =============================================================================
# Vocabulary
# =============================================================================

# A1 vocabulary for complexity calculation
A1_VOCABULARY = {
    # Gates (cost 1 each)
    'gates': list(QUANTUM_GATES.keys()),
    
    # Special forms (cost 1 each)
    'special_forms': ['DEFINE', 'LAMBDA', 'IF', 'LET', 'QUOTE', 'BEGIN'],
    
    # Constants
    'constants': ['PI'],
}

def get_vocabulary_size() -> int:
    """
    Get the total vocabulary size of A1.
    
    Returns:
        Number of distinct tokens in A1 vocabulary
    """
    size = 0
    size += len(A1_VOCABULARY['gates'])          # 13 gates
    size += len(A1_VOCABULARY['special_forms'])  # 6 special forms
    size += len(A1_VOCABULARY['constants'])      # 1 constant
    size += 10  # digits 0-9 for qubit indices
    size += 2   # parentheses (not counted in complexity, but part of syntax)
    
    return size  # ≈ 32


VOCABULARY_SIZE = get_vocabulary_size()


if __name__ == "__main__":
    print("=== A1 Gate Definitions ===\n")
    
    for name, gate in QUANTUM_GATES.items():
        angle_str = " (+ angle)" if gate.has_angle else ""
        print(f"{name}: arity={gate.arity}{angle_str}, cost={gate.cost}")
        print(f"  {gate.description}\n")
    
    print(f"Total gates: {len(QUANTUM_GATES)}")
    print(f"Vocabulary size: {VOCABULARY_SIZE}")



