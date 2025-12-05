"""
Simple quantum circuit model for comparison.

Unlike classical reversible gates (permutation matrices),
quantum circuits use unitary matrices with complex entries.
"""

import numpy as np
from scipy import linalg
from typing import List, Tuple, Optional
from dataclasses import dataclass


# Pauli matrices
SIGMA_X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
SIGMA_Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
SIGMA_Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
IDENTITY = np.eye(2, dtype=np.complex128)

# Standard quantum gates
HADAMARD = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)
PHASE_S = np.array([[1, 0], [0, 1j]], dtype=np.complex128)
PHASE_T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=np.complex128)
CNOT = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 0]
], dtype=np.complex128)


@dataclass
class QuantumState:
    """Quantum state vector."""
    amplitudes: np.ndarray
    
    @property
    def probabilities(self) -> np.ndarray:
        return np.abs(self.amplitudes) ** 2
    
    @property
    def num_qubits(self) -> int:
        return int(np.log2(len(self.amplitudes)))
    
    def measure(self) -> int:
        """Simulate measurement, return outcome index."""
        probs = self.probabilities
        return np.random.choice(len(probs), p=probs)


class SimpleQuantumCircuit:
    """
    Simple quantum circuit with basic gates.
    
    Key difference from classical reversible gates:
    - Gates are UNITARY (complex entries), not just permutation
    - Hadamard creates superposition
    - Phase gates introduce complex phases
    """
    
    def __init__(self, num_qubits: int):
        """
        Initialize quantum circuit.
        
        Args:
            num_qubits: Number of qubits
        """
        self.num_qubits = num_qubits
        self.dim = 2 ** num_qubits
        self.gates: List[Tuple[str, np.ndarray]] = []
        
    def _embed_gate(self, gate: np.ndarray, target: int, 
                    control: Optional[int] = None) -> np.ndarray:
        """Embed single/two-qubit gate into full Hilbert space."""
        if control is None:
            # Single qubit gate
            ops = []
            for i in range(self.num_qubits):
                if i == target:
                    ops.append(gate)
                else:
                    ops.append(IDENTITY)
            
            result = ops[0]
            for op in ops[1:]:
                result = np.kron(result, op)
            return result
        else:
            # Controlled gate
            # |0><0| ⊗ I + |1><1| ⊗ gate
            P0 = np.array([[1, 0], [0, 0]], dtype=np.complex128)
            P1 = np.array([[0, 0], [0, 1]], dtype=np.complex128)
            
            # Build |0><0| ⊗ I ⊗ ... ⊗ I
            ops_0 = []
            ops_1 = []
            for i in range(self.num_qubits):
                if i == control:
                    ops_0.append(P0)
                    ops_1.append(P1)
                elif i == target:
                    ops_0.append(IDENTITY)
                    ops_1.append(gate)
                else:
                    ops_0.append(IDENTITY)
                    ops_1.append(IDENTITY)
            
            term_0 = ops_0[0]
            term_1 = ops_1[0]
            for i in range(1, len(ops_0)):
                term_0 = np.kron(term_0, ops_0[i])
                term_1 = np.kron(term_1, ops_1[i])
            
            return term_0 + term_1
    
    def h(self, target: int) -> 'SimpleQuantumCircuit':
        """Add Hadamard gate."""
        U = self._embed_gate(HADAMARD, target)
        self.gates.append(('H', U))
        return self
    
    def x(self, target: int) -> 'SimpleQuantumCircuit':
        """Add Pauli X gate (NOT)."""
        U = self._embed_gate(SIGMA_X, target)
        self.gates.append(('X', U))
        return self
    
    def z(self, target: int) -> 'SimpleQuantumCircuit':
        """Add Pauli Z gate."""
        U = self._embed_gate(SIGMA_Z, target)
        self.gates.append(('Z', U))
        return self
    
    def s(self, target: int) -> 'SimpleQuantumCircuit':
        """Add S (phase) gate."""
        U = self._embed_gate(PHASE_S, target)
        self.gates.append(('S', U))
        return self
    
    def t(self, target: int) -> 'SimpleQuantumCircuit':
        """Add T gate."""
        U = self._embed_gate(PHASE_T, target)
        self.gates.append(('T', U))
        return self
    
    def cx(self, control: int, target: int) -> 'SimpleQuantumCircuit':
        """Add CNOT gate."""
        U = self._embed_gate(SIGMA_X, target, control)
        self.gates.append(('CX', U))
        return self
    
    def cz(self, control: int, target: int) -> 'SimpleQuantumCircuit':
        """Add CZ gate."""
        U = self._embed_gate(SIGMA_Z, target, control)
        self.gates.append(('CZ', U))
        return self
    
    def get_unitary(self) -> np.ndarray:
        """Get the full circuit unitary."""
        U = np.eye(self.dim, dtype=np.complex128)
        for name, gate in self.gates:
            U = gate @ U
        return U
    
    def apply(self, state: QuantumState) -> QuantumState:
        """Apply circuit to a state."""
        U = self.get_unitary()
        new_amplitudes = U @ state.amplitudes
        return QuantumState(new_amplitudes)
    
    def analyze_structure(self) -> dict:
        """
        Analyze the algebraic structure of the circuit.
        
        Returns:
            Dictionary with structure properties
        """
        U = self.get_unitary()
        
        # Check unitarity
        is_unitary = np.allclose(U @ U.conj().T, np.eye(self.dim))
        
        # Check if it's a permutation matrix (classical)
        is_permutation = (
            is_unitary and
            np.allclose(np.abs(U), np.round(np.abs(U))) and
            np.allclose(np.sum(np.abs(U), axis=0), 1) and
            np.allclose(np.sum(np.abs(U), axis=1), 1)
        )
        
        # Check for genuine complex structure
        has_complex = np.any(np.abs(U.imag) > 1e-10)
        
        # Check for superposition creation (off-diagonal with equal magnitude)
        creates_superposition = False
        if has_complex or not is_permutation:
            # Check if applying to computational basis creates superposition
            for i in range(self.dim):
                basis = np.zeros(self.dim, dtype=np.complex128)
                basis[i] = 1
                output = U @ basis
                if np.sum(np.abs(output) > 1e-10) > 1:
                    creates_superposition = True
                    break
        
        # Eigenvalue analysis
        eigenvalues = np.linalg.eigvals(U)
        phases = np.angle(eigenvalues)
        
        # Check if phases include irrational multiples of π
        has_irrational_phase = False
        for phase in phases:
            if abs(phase) > 1e-10:
                ratio = phase / np.pi
                # Check if ratio is close to simple fraction
                is_simple = False
                for denom in range(1, 13):
                    for numer in range(-12, 13):
                        if abs(ratio - numer/denom) < 1e-6:
                            is_simple = True
                            break
                    if is_simple:
                        break
                if not is_simple:
                    has_irrational_phase = True
                    break
        
        return {
            'is_unitary': is_unitary,
            'is_permutation': is_permutation,
            'has_complex_entries': has_complex,
            'creates_superposition': creates_superposition,
            'has_irrational_phase': has_irrational_phase,
            'eigenvalues': eigenvalues,
            'determinant': np.linalg.det(U),
        }


def create_sample_circuits() -> List[Tuple[str, SimpleQuantumCircuit]]:
    """
    Create sample quantum circuits for comparison.
    
    Returns:
        List of (name, circuit) pairs
    """
    circuits = []
    
    # 1. Classical-like: just X gates (permutation)
    c1 = SimpleQuantumCircuit(2)
    c1.x(0).x(1)
    circuits.append(('Classical (X gates)', c1))
    
    # 2. With superposition: Hadamard
    c2 = SimpleQuantumCircuit(2)
    c2.h(0).h(1)
    circuits.append(('Superposition (H gates)', c2))
    
    # 3. With entanglement: Bell state creator
    c3 = SimpleQuantumCircuit(2)
    c3.h(0).cx(0, 1)
    circuits.append(('Entanglement (Bell)', c3))
    
    # 4. With phase: QFT-like
    c4 = SimpleQuantumCircuit(2)
    c4.h(0).s(0).h(1).cx(0, 1)
    circuits.append(('Phase (S gate)', c4))
    
    # 5. Complex circuit
    c5 = SimpleQuantumCircuit(3)
    c5.h(0).cx(0, 1).t(2).cx(1, 2).h(2)
    circuits.append(('Complex (3 qubit)', c5))
    
    return circuits


if __name__ == '__main__':
    print("=== Quantum Circuit Analysis ===\n")
    
    for name, circuit in create_sample_circuits():
        print(f"Circuit: {name}")
        props = circuit.analyze_structure()
        print(f"  Is unitary: {props['is_unitary']}")
        print(f"  Is permutation (classical): {props['is_permutation']}")
        print(f"  Has complex entries: {props['has_complex_entries']}")
        print(f"  Creates superposition: {props['creates_superposition']}")
        print(f"  Determinant: {props['determinant']:.4f}")
        print()

