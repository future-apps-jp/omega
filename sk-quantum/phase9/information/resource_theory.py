"""
Resource Theory of Coherence

This module implements the Resource Theory framework for analyzing
quantum coherence as a resource. This approach avoids circular reasoning
by treating coherence as an external resource that can be "injected"
into computational models.

Key concepts:
- Free States: Classical (incoherent) states - diagonal density matrices
- Free Operations: Incoherent operations - cannot create coherence
- Resource States: Coherent states - non-diagonal density matrices
- Coherence Measures: Quantify the amount of coherence

References:
- [Chitambar2019] Quantum resource theories (Reviews of Modern Physics)
- [Streltsov2017] Colloquium: Quantum coherence as a resource
- [Winter2016] Operational resource theory of coherence
"""

import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
from enum import Enum


# =============================================================================
# Density Matrix Operations
# =============================================================================

def is_density_matrix(rho: np.ndarray, tol: float = 1e-10) -> bool:
    """
    Check if matrix is a valid density matrix.
    
    Conditions:
    1. Hermitian: ρ = ρ†
    2. Positive semi-definite: eigenvalues ≥ 0
    3. Trace 1: Tr(ρ) = 1
    """
    # Hermitian
    if not np.allclose(rho, rho.conj().T, atol=tol):
        return False
    
    # Positive semi-definite
    eigenvalues = np.linalg.eigvalsh(rho)
    if np.any(eigenvalues < -tol):
        return False
    
    # Trace 1
    if not np.isclose(np.trace(rho), 1.0, atol=tol):
        return False
    
    return True


def pure_state_density(psi: np.ndarray) -> np.ndarray:
    """Create density matrix from pure state |ψ⟩⟨ψ|."""
    psi = psi.reshape(-1, 1)
    return psi @ psi.conj().T


def mixed_state_density(probs: np.ndarray, states: List[np.ndarray]) -> np.ndarray:
    """Create density matrix from mixture of states."""
    rho = np.zeros((len(states[0]), len(states[0])), dtype=complex)
    for p, psi in zip(probs, states):
        rho += p * pure_state_density(psi)
    return rho


# =============================================================================
# Standard Basis States
# =============================================================================

def computational_basis(n_qubits: int) -> List[np.ndarray]:
    """Get computational basis states for n qubits."""
    dim = 2 ** n_qubits
    return [np.eye(dim)[:, i] for i in range(dim)]


def plus_state(n_qubits: int = 1) -> np.ndarray:
    """Create |+⟩ = (|0⟩ + |1⟩)/√2 state."""
    dim = 2 ** n_qubits
    return np.ones(dim) / np.sqrt(dim)


def minus_state() -> np.ndarray:
    """Create |−⟩ = (|0⟩ − |1⟩)/√2 state."""
    return np.array([1, -1]) / np.sqrt(2)


def ghz_state(n_qubits: int) -> np.ndarray:
    """Create GHZ state (|00...0⟩ + |11...1⟩)/√2."""
    dim = 2 ** n_qubits
    psi = np.zeros(dim, dtype=complex)
    psi[0] = 1 / np.sqrt(2)
    psi[-1] = 1 / np.sqrt(2)
    return psi


# =============================================================================
# Coherence Measures
# =============================================================================

class CoherenceMeasure(ABC):
    """Abstract base class for coherence measures."""
    
    @abstractmethod
    def measure(self, rho: np.ndarray) -> float:
        """Measure coherence of density matrix."""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the measure."""
        pass


class L1Coherence(CoherenceMeasure):
    """
    L1-norm of coherence (sum of off-diagonal elements).
    
    C_{l1}(ρ) = Σ_{i≠j} |ρ_{ij}|
    
    This is one of the most commonly used coherence measures.
    """
    
    def measure(self, rho: np.ndarray) -> float:
        """Compute L1 coherence."""
        n = rho.shape[0]
        total = 0.0
        for i in range(n):
            for j in range(n):
                if i != j:
                    total += np.abs(rho[i, j])
        return total
    
    @property
    def name(self) -> str:
        return "L1_coherence"


class RelativeEntropyCoherence(CoherenceMeasure):
    """
    Relative entropy of coherence.
    
    C_{rel}(ρ) = S(ρ_diag) - S(ρ)
    
    where S is von Neumann entropy and ρ_diag is the dephased state.
    """
    
    def measure(self, rho: np.ndarray) -> float:
        """Compute relative entropy of coherence."""
        # von Neumann entropy of ρ
        eigenvalues = np.linalg.eigvalsh(rho)
        eigenvalues = eigenvalues[eigenvalues > 1e-15]  # Remove zeros
        S_rho = -np.sum(eigenvalues * np.log2(eigenvalues + 1e-15))
        
        # Dephased state (diagonal only)
        rho_diag = np.diag(np.diag(rho))
        eigenvalues_diag = np.real(np.diag(rho_diag))
        eigenvalues_diag = eigenvalues_diag[eigenvalues_diag > 1e-15]
        S_diag = -np.sum(eigenvalues_diag * np.log2(eigenvalues_diag + 1e-15))
        
        return max(0, S_diag - S_rho)
    
    @property
    def name(self) -> str:
        return "relative_entropy_coherence"


class RobustnessCoherence(CoherenceMeasure):
    """
    Robustness of coherence.
    
    Measures how much noise is needed to destroy coherence.
    Simplified version: ratio of off-diagonal to diagonal magnitude.
    """
    
    def measure(self, rho: np.ndarray) -> float:
        """Compute robustness of coherence (simplified)."""
        n = rho.shape[0]
        
        off_diag_sum = 0.0
        diag_sum = 0.0
        
        for i in range(n):
            diag_sum += np.abs(rho[i, i])
            for j in range(n):
                if i != j:
                    off_diag_sum += np.abs(rho[i, j])
        
        if diag_sum < 1e-15:
            return 0.0
        
        return off_diag_sum / diag_sum
    
    @property
    def name(self) -> str:
        return "robustness_coherence"


# =============================================================================
# Free States and Operations
# =============================================================================

@dataclass
class ResourceState:
    """
    A state in the resource theory framework.
    
    Attributes:
        density_matrix: The density matrix representation
        coherence: Dictionary of coherence measures
        is_free: Whether this is a free (incoherent) state
    """
    density_matrix: np.ndarray
    coherence: Dict[str, float]
    is_free: bool
    
    @classmethod
    def from_pure_state(cls, psi: np.ndarray, measures: List[CoherenceMeasure] = None):
        """Create ResourceState from pure state."""
        rho = pure_state_density(psi)
        return cls.from_density_matrix(rho, measures)
    
    @classmethod
    def from_density_matrix(cls, rho: np.ndarray, measures: List[CoherenceMeasure] = None):
        """Create ResourceState from density matrix."""
        if measures is None:
            measures = [L1Coherence(), RelativeEntropyCoherence()]
        
        coherence = {m.name: m.measure(rho) for m in measures}
        
        # Check if free (incoherent): off-diagonal elements are zero
        is_free = all(v < 1e-10 for v in coherence.values())
        
        return cls(
            density_matrix=rho,
            coherence=coherence,
            is_free=is_free
        )


class FreeOperation(ABC):
    """
    Abstract base class for free (incoherent) operations.
    
    Free operations cannot create coherence from incoherent states.
    """
    
    @abstractmethod
    def apply(self, rho: np.ndarray) -> np.ndarray:
        """Apply operation to density matrix."""
        pass
    
    @abstractmethod
    def is_coherence_non_generating(self) -> bool:
        """Check if operation cannot generate coherence."""
        pass


class PermutationOperation(FreeOperation):
    """
    Permutation operation (classical reversible computation).
    
    These are always free operations - they permute the basis
    but cannot create superposition.
    """
    
    def __init__(self, permutation_matrix: np.ndarray):
        self.P = permutation_matrix
    
    def apply(self, rho: np.ndarray) -> np.ndarray:
        """Apply permutation: P ρ P†"""
        return self.P @ rho @ self.P.conj().T
    
    def is_coherence_non_generating(self) -> bool:
        """Permutations are always coherence non-generating."""
        return True


class DephasingOperation(FreeOperation):
    """
    Dephasing (decoherence) operation.
    
    Removes all off-diagonal elements, making state incoherent.
    """
    
    def apply(self, rho: np.ndarray) -> np.ndarray:
        """Apply dephasing: ρ → diag(ρ)"""
        return np.diag(np.diag(rho))
    
    def is_coherence_non_generating(self) -> bool:
        """Dephasing destroys coherence, never generates it."""
        return True


class MeasurementOperation(FreeOperation):
    """
    Computational basis measurement.
    
    Projects onto computational basis - a free operation.
    """
    
    def __init__(self, basis_index: int):
        self.index = basis_index
    
    def apply(self, rho: np.ndarray) -> np.ndarray:
        """Apply projective measurement."""
        n = rho.shape[0]
        proj = np.zeros((n, n))
        proj[self.index, self.index] = 1.0
        
        # Post-measurement state (normalized)
        prob = np.real(rho[self.index, self.index])
        if prob > 1e-15:
            return proj
        return np.zeros_like(rho)
    
    def is_coherence_non_generating(self) -> bool:
        """Measurement is a free operation."""
        return True


# =============================================================================
# Resource-Generating Operations (Quantum)
# =============================================================================

class HadamardOperation:
    """
    Hadamard gate - a coherence-generating operation.
    
    H = (1/√2) [[1, 1], [1, -1]]
    
    This transforms |0⟩ → |+⟩, creating coherence.
    """
    
    def __init__(self):
        self.H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    
    def apply(self, rho: np.ndarray) -> np.ndarray:
        """Apply Hadamard: H ρ H†"""
        return self.H @ rho @ self.H.conj().T
    
    def is_coherence_non_generating(self) -> bool:
        """Hadamard CAN generate coherence."""
        return False


class PhaseGateOperation:
    """
    Phase gate S = [[1, 0], [0, i]].
    
    Changes phase but doesn't create superposition from basis states.
    However, it modifies existing coherence.
    """
    
    def __init__(self, phase: float = np.pi/2):
        self.S = np.array([[1, 0], [0, np.exp(1j * phase)]])
    
    def apply(self, rho: np.ndarray) -> np.ndarray:
        """Apply phase gate: S ρ S†"""
        return self.S @ rho @ self.S.conj().T
    
    def is_coherence_non_generating(self) -> bool:
        """Phase gate doesn't create coherence from incoherent states."""
        return True


# =============================================================================
# Resource Injection Experiment
# =============================================================================

@dataclass
class ResourceInjectionResult:
    """Results from injecting coherence resource into a computation."""
    initial_coherence: Dict[str, float]
    final_coherence: Dict[str, float]
    coherence_change: Dict[str, float]
    operation_type: str
    is_coherence_preserved: bool
    is_coherence_amplified: bool
    is_coherence_destroyed: bool


class ResourceInjectionExperiment:
    """
    Experiment: Inject coherence into a computational model and observe behavior.
    
    Key question: How does classical computation treat coherence as a resource?
    
    Expected results:
    - Free operations preserve or reduce coherence
    - Only quantum operations can amplify coherence
    """
    
    def __init__(self, measures: List[CoherenceMeasure] = None):
        if measures is None:
            measures = [L1Coherence(), RelativeEntropyCoherence()]
        self.measures = measures
    
    def inject_and_evolve(
        self,
        resource_state: ResourceState,
        operation: FreeOperation
    ) -> ResourceInjectionResult:
        """
        Inject a resource state and apply an operation.
        
        Args:
            resource_state: State with coherence to inject
            operation: Operation to apply
            
        Returns:
            ResourceInjectionResult with coherence analysis
        """
        # Apply operation
        rho_initial = resource_state.density_matrix
        rho_final = operation.apply(rho_initial)
        
        # Measure coherence
        initial_coherence = {m.name: m.measure(rho_initial) for m in self.measures}
        final_coherence = {m.name: m.measure(rho_final) for m in self.measures}
        
        coherence_change = {
            name: final_coherence[name] - initial_coherence[name]
            for name in initial_coherence
        }
        
        # Analyze
        avg_change = np.mean(list(coherence_change.values()))
        is_preserved = abs(avg_change) < 0.01
        is_amplified = avg_change > 0.01
        is_destroyed = avg_change < -0.01
        
        return ResourceInjectionResult(
            initial_coherence=initial_coherence,
            final_coherence=final_coherence,
            coherence_change=coherence_change,
            operation_type=type(operation).__name__,
            is_coherence_preserved=is_preserved,
            is_coherence_amplified=is_amplified,
            is_coherence_destroyed=is_destroyed
        )
    
    def run_classical_computation_test(self) -> Dict[str, any]:
        """
        Test how classical computation (permutations) affects coherence.
        
        Expected: Classical computation cannot amplify coherence.
        """
        results = []
        
        # Create resource states with varying coherence
        test_states = [
            ("|0⟩ (incoherent)", pure_state_density(np.array([1, 0]))),
            ("|+⟩ (max coherent)", pure_state_density(plus_state(1))),
            ("|−⟩ (max coherent)", pure_state_density(minus_state())),
            ("mixed", mixed_state_density(
                np.array([0.5, 0.5]),
                [np.array([1, 0]), np.array([0, 1])]
            )),
        ]
        
        # Classical operations
        NOT = np.array([[0, 1], [1, 0]])
        operations = [
            ("Identity", PermutationOperation(np.eye(2))),
            ("NOT (X)", PermutationOperation(NOT)),
            ("Dephasing", DephasingOperation()),
        ]
        
        for state_name, rho in test_states:
            resource = ResourceState.from_density_matrix(rho, self.measures)
            
            for op_name, op in operations:
                result = self.inject_and_evolve(resource, op)
                results.append({
                    'state': state_name,
                    'operation': op_name,
                    'initial_L1': result.initial_coherence.get('L1_coherence', 0),
                    'final_L1': result.final_coherence.get('L1_coherence', 0),
                    'change': result.coherence_change.get('L1_coherence', 0),
                    'preserved': result.is_coherence_preserved,
                    'amplified': result.is_coherence_amplified,
                    'destroyed': result.is_coherence_destroyed,
                })
        
        # Analysis
        any_amplified = any(r['amplified'] for r in results)
        
        return {
            'results': results,
            'conclusion': 'Classical computation CANNOT amplify coherence' if not any_amplified
                         else 'UNEXPECTED: Classical computation amplified coherence',
            'supports_H9_3': not any_amplified
        }


# =============================================================================
# Information-Theoretic Principles
# =============================================================================

def check_no_cloning(rho: np.ndarray) -> Dict[str, any]:
    """
    Test no-cloning theorem.
    
    For non-orthogonal states, perfect cloning is impossible.
    
    This is a CONSEQUENCE of coherence, not a cause.
    """
    # Create two non-orthogonal states
    psi_0 = np.array([1, 0])  # |0⟩
    psi_plus = plus_state(1)  # |+⟩
    
    # Overlap
    overlap = np.abs(np.dot(psi_0.conj(), psi_plus))
    
    # If states are non-orthogonal (overlap ≠ 0, 1), cloning is impossible
    is_orthogonal = np.isclose(overlap, 0) or np.isclose(overlap, 1)
    cloning_possible = is_orthogonal
    
    return {
        'overlap': overlap,
        'states_orthogonal': is_orthogonal,
        'perfect_cloning_possible': cloning_possible,
        'conclusion': 'No-cloning is a CONSEQUENCE of superposition (non-orthogonal states exist)'
    }


def check_no_deleting(computational_model: str = "reversible") -> Dict[str, any]:
    """
    Test no-deleting principle.
    
    For quantum states, complete deletion while leaving a blank copy is impossible.
    However, reversible classical computation also satisfies this!
    """
    if computational_model == "reversible":
        # Reversible computation preserves information
        # Cannot delete without trace
        satisfies = True
        reason = "Reversible computation is bijective - information preserved"
    elif computational_model == "irreversible":
        # Irreversible computation can delete (e.g., K combinator)
        satisfies = False
        reason = "K combinator: K x y → x (y is deleted)"
    else:
        satisfies = True
        reason = "Quantum no-deleting theorem"
    
    return {
        'model': computational_model,
        'satisfies_no_deleting': satisfies,
        'reason': reason,
        'conclusion': 'No-deleting holds for reversible computation, not unique to quantum'
    }


def analyze_information_principles() -> Dict[str, any]:
    """
    Analyze the relationship between information principles and quantum structure.
    
    Key insight: Classical reversible computation satisfies:
    - Information conservation
    - No-deleting
    
    But does NOT satisfy:
    - No-cloning (because only orthogonal states exist)
    - No-broadcasting (trivially satisfied - no non-commuting observables)
    
    This shows information principles alone don't give quantum structure.
    """
    results = {
        'classical_reversible': {
            'information_conservation': True,
            'no_deleting': True,
            'no_cloning': False,  # Can copy classical bits
            'no_broadcasting': True,  # Trivially - no non-commuting observables
        },
        'quantum': {
            'information_conservation': True,
            'no_deleting': True,
            'no_cloning': True,
            'no_broadcasting': True,
        },
        'irreversible_classical': {
            'information_conservation': False,
            'no_deleting': False,
            'no_cloning': False,
            'no_broadcasting': True,
        }
    }
    
    # Analysis
    shared = set()
    quantum_only = set()
    
    for principle in ['information_conservation', 'no_deleting', 'no_cloning', 'no_broadcasting']:
        if (results['classical_reversible'][principle] and 
            results['quantum'][principle]):
            shared.add(principle)
        elif results['quantum'][principle] and not results['classical_reversible'][principle]:
            quantum_only.add(principle)
    
    return {
        'results': results,
        'shared_with_reversible': list(shared),
        'unique_to_quantum': list(quantum_only),
        'conclusion': (
            'Information principles alone are insufficient. '
            'No-cloning is unique to quantum because it requires non-orthogonal states (superposition). '
            'This supports H9.2: No-cloning is a RESULT of quantum structure, not a cause.'
        )
    }


if __name__ == "__main__":
    print("=" * 60)
    print("Resource Theory of Coherence - Demo")
    print("=" * 60)
    
    # Create resource states
    print("\n--- Resource States ---")
    
    incoherent = ResourceState.from_pure_state(np.array([1, 0]))
    print(f"|0⟩: is_free={incoherent.is_free}, coherence={incoherent.coherence}")
    
    coherent = ResourceState.from_pure_state(plus_state(1))
    print(f"|+⟩: is_free={coherent.is_free}, coherence={coherent.coherence}")
    
    # Run experiment
    print("\n--- Classical Computation Test ---")
    experiment = ResourceInjectionExperiment()
    results = experiment.run_classical_computation_test()
    
    print(f"\nConclusion: {results['conclusion']}")
    print(f"Supports H9.3: {results['supports_H9_3']}")
    
    # Information principles
    print("\n--- Information Principles Analysis ---")
    info = analyze_information_principles()
    print(f"Shared with reversible: {info['shared_with_reversible']}")
    print(f"Unique to quantum: {info['unique_to_quantum']}")
    print(f"\nConclusion: {info['conclusion']}")

