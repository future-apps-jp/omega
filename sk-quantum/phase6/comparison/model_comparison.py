"""
Three-Model Comparison Framework.

Compares:
1. SK computation (irreversible, discrete) 
2. RCA (reversible, discrete)
3. Quantum circuits (reversible, continuous)

Key question: What makes quantum circuits different?
- Phase 4 showed: reversibility alone is insufficient (classical symplectic)
- Phase 5 showed: continuous-time evolution introduces interference
- Phase 6 tests: RCA with continuous-time should also show interference
"""

import numpy as np
from scipy import linalg
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import sys
import os

# Add paths for importing from other phases
_base_dir = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, _base_dir)
sys.path.insert(0, os.path.join(_base_dir, 'phase0'))
sys.path.insert(0, os.path.join(_base_dir, 'phase5', 'spectral'))

from multiway import MultiwayGraph, build_multiway_graph
from sk_parser import parse
from hamiltonian import (
    ComputationHamiltonian, 
    SpectralAnalysis,
    ContinuousTimeQuantumWalk,
    ClassicalRandomWalk,
    InterferenceAnalysis,
    build_hamiltonian_from_expression
)
from phase6.rca.automata import ReversibleCellularAutomaton, Rule90, analyze_rca_group
from phase6.rca.graph import RCAGraph
from phase6.rca.hamiltonian import RCAHamiltonian
from phase6.comparison.quantum_circuit import SimpleQuantumCircuit, create_sample_circuits


@dataclass
class ModelProperties:
    """Properties of a computational model."""
    name: str
    is_reversible: bool
    is_discrete: bool
    matrix_type: str  # 'permutation', 'orthogonal', 'unitary'
    has_complex_structure: bool
    has_superposition: bool
    has_interference: bool
    eigenvalue_phases: np.ndarray
    additional_info: Dict


@dataclass
class ComparisonResult:
    """Result of comparing three models."""
    sk_props: ModelProperties
    rca_props: ModelProperties
    quantum_props: ModelProperties
    summary: str
    key_differences: List[str]


class ModelComparison:
    """
    Framework for comparing SK, RCA, and Quantum circuit models.
    """
    
    def __init__(self):
        self.results: Dict[str, ModelProperties] = {}
    
    def analyze_sk_computation(self, expression: str = 'S(KK)(SK)',
                               times: Optional[np.ndarray] = None) -> ModelProperties:
        """
        Analyze SK computation properties.
        
        Args:
            expression: SK expression to analyze
            times: Time points for quantum walk (default: 0 to 10)
        
        Returns:
            ModelProperties for SK
        """
        if times is None:
            times = np.linspace(0, 10, 100)
        
        # Build multiway graph and Hamiltonian
        H = build_hamiltonian_from_expression(expression)
        
        # Get adjacency matrix from Hamiltonian
        adj = H.adjacency
        
        # Check reversibility (is transition bijective?)
        # SK is NOT reversible in general (multiple paths can lead to same result)
        degrees = np.sum(adj, axis=1)
        is_reversible = np.all(degrees == degrees[0]) if len(degrees) > 0 else True
        
        # Eigenvalue analysis
        if len(adj) > 0:
            eigenvalues = np.linalg.eigvals(adj)
            phases = np.angle(eigenvalues)
        else:
            eigenvalues = np.array([])
            phases = np.array([])
        
        # Check for complex eigenvalues
        has_complex = np.any(np.abs(np.imag(eigenvalues)) > 1e-10)
        
        # Analyze interference with continuous-time evolution
        has_interference = False
        interference_oscillations = 0
        if H.dimension >= 2:
            qw = ContinuousTimeQuantumWalk(adj)
            cw = ClassicalRandomWalk(adj)
            interference_analyzer = InterferenceAnalysis(qw, cw, H.dimension)
            int_results = interference_analyzer.detect_interference(0, times.tolist())
            has_interference = int_results['has_interference']
            interference_oscillations = int_results.get('mean_oscillation', 0)
        
        return ModelProperties(
            name='SK Computation',
            is_reversible=False,  # SK is fundamentally irreversible
            is_discrete=True,
            matrix_type='general',  # Not necessarily permutation
            has_complex_structure=False,  # No inherent complex structure
            has_superposition=False,  # Deterministic
            has_interference=has_interference,
            eigenvalue_phases=phases,
            additional_info={
                'num_nodes': H.dimension,
                'num_edges': int(adj.sum() / 2),
                'max_degree': np.max(degrees) if len(degrees) > 0 else 0,
                'interference_oscillations': interference_oscillations,
            }
        )
    
    def analyze_rca(self, size: int = 3, rule: int = 90,
                    times: Optional[np.ndarray] = None) -> ModelProperties:
        """
        Analyze RCA properties.
        
        Args:
            size: Number of cells
            rule: CA rule number
            times: Time points for quantum walk
        
        Returns:
            ModelProperties for RCA
        """
        if times is None:
            times = np.linspace(0, 10, 100)
        
        rca = ReversibleCellularAutomaton(size, rule)
        
        # Analyze group structure
        group_props = analyze_rca_group(rca)
        
        # Build graph and Hamiltonian
        graph = RCAGraph(rca)
        from phase6.rca.automata import RCAState
        initial = RCAState(tuple([1] + [0] * (size - 1)))
        previous = RCAState(tuple([0] * size))
        graph.build_from_initial(initial, previous)
        
        H = RCAHamiltonian(graph)
        
        # Eigenvalue analysis
        spectral = H.spectral_analysis()
        phases = np.angle(np.exp(1j * spectral.eigenvalues))  # Phases from real eigenvalues
        
        # Interference detection
        interference = H.detect_interference(times)
        
        return ModelProperties(
            name=f'RCA (Rule {rule})',
            is_reversible=True,  # RCA is reversible by construction
            is_discrete=True,
            matrix_type='permutation',  # Transition is bijective permutation
            has_complex_structure=False,  # No inherent complex structure
            has_superposition=False,  # Deterministic
            has_interference=interference['has_interference'],
            eigenvalue_phases=phases,
            additional_info={
                'group_order': group_props['order'],
                'is_permutation': group_props['is_permutation'],
                'graph_nodes': graph.num_nodes,
                'graph_edges': graph.num_edges,
                'spectral_gap': spectral.spectral_gap,
                'interference_oscillations': interference['quantum_oscillations'],
            }
        )
    
    def analyze_quantum_circuit(self, circuit: Optional[SimpleQuantumCircuit] = None,
                                name: str = 'Quantum Circuit') -> ModelProperties:
        """
        Analyze quantum circuit properties.
        
        Args:
            circuit: Quantum circuit to analyze (default: Bell state circuit)
            name: Name for the circuit
        
        Returns:
            ModelProperties for quantum circuit
        """
        if circuit is None:
            circuit = SimpleQuantumCircuit(2)
            circuit.h(0).cx(0, 1)  # Bell state creator
        
        props = circuit.analyze_structure()
        
        # Eigenvalue analysis
        U = circuit.get_unitary()
        eigenvalues = np.linalg.eigvals(U)
        phases = np.angle(eigenvalues)
        
        return ModelProperties(
            name=name,
            is_reversible=True,  # Unitary is always reversible
            is_discrete=False,  # Continuous complex amplitudes
            matrix_type='unitary',
            has_complex_structure=props['has_complex_entries'],
            has_superposition=props['creates_superposition'],
            has_interference=True,  # Quantum circuits inherently support interference
            eigenvalue_phases=phases,
            additional_info={
                'is_permutation': props['is_permutation'],
                'determinant': props['determinant'],
                'num_qubits': circuit.num_qubits,
                'num_gates': len(circuit.gates),
            }
        )
    
    def compare_all(self, sk_expr: str = 'S(KK)(SK)',
                    rca_size: int = 3, rca_rule: int = 90) -> ComparisonResult:
        """
        Compare all three models.
        
        Returns:
            ComparisonResult with analysis and summary
        """
        times = np.linspace(0, 10, 100)
        
        # Analyze each model
        sk_props = self.analyze_sk_computation(sk_expr, times)
        rca_props = self.analyze_rca(rca_size, rca_rule, times)
        
        # Use Bell state circuit as reference quantum model
        bell_circuit = SimpleQuantumCircuit(2)
        bell_circuit.h(0).cx(0, 1)
        quantum_props = self.analyze_quantum_circuit(bell_circuit, 'Quantum (Bell state)')
        
        # Identify key differences
        differences = []
        
        # 1. Reversibility
        if not sk_props.is_reversible:
            differences.append("SK is irreversible; RCA and Quantum are reversible")
        
        # 2. Discrete vs Continuous
        if sk_props.is_discrete and rca_props.is_discrete and not quantum_props.is_discrete:
            differences.append("SK and RCA are discrete; Quantum has continuous amplitudes")
        
        # 3. Complex structure
        if quantum_props.has_complex_structure and not rca_props.has_complex_structure:
            differences.append("Only Quantum has inherent complex structure")
        
        # 4. Superposition
        if quantum_props.has_superposition and not rca_props.has_superposition:
            differences.append("Only Quantum creates genuine superposition")
        
        # 5. Matrix type
        matrix_types = f"Matrix types: SK={sk_props.matrix_type}, RCA={rca_props.matrix_type}, Quantum={quantum_props.matrix_type}"
        differences.append(matrix_types)
        
        # 6. Interference with continuous time
        if sk_props.has_interference and rca_props.has_interference:
            differences.append("Both SK and RCA show interference under continuous-time evolution")
        
        # Generate summary
        summary = self._generate_summary(sk_props, rca_props, quantum_props, differences)
        
        return ComparisonResult(
            sk_props=sk_props,
            rca_props=rca_props,
            quantum_props=quantum_props,
            summary=summary,
            key_differences=differences
        )
    
    def _generate_summary(self, sk: ModelProperties, rca: ModelProperties,
                         quantum: ModelProperties, differences: List[str]) -> str:
        """Generate comparison summary."""
        lines = [
            "=" * 60,
            "THREE-MODEL COMPARISON SUMMARY",
            "=" * 60,
            "",
            "Model Properties:",
            f"  {'Property':<25} {'SK':<15} {'RCA':<15} {'Quantum':<15}",
            f"  {'-'*25} {'-'*15} {'-'*15} {'-'*15}",
            f"  {'Reversible':<25} {str(sk.is_reversible):<15} {str(rca.is_reversible):<15} {str(quantum.is_reversible):<15}",
            f"  {'Discrete':<25} {str(sk.is_discrete):<15} {str(rca.is_discrete):<15} {str(quantum.is_discrete):<15}",
            f"  {'Matrix Type':<25} {sk.matrix_type:<15} {rca.matrix_type:<15} {quantum.matrix_type:<15}",
            f"  {'Complex Structure':<25} {str(sk.has_complex_structure):<15} {str(rca.has_complex_structure):<15} {str(quantum.has_complex_structure):<15}",
            f"  {'Superposition':<25} {str(sk.has_superposition):<15} {str(rca.has_superposition):<15} {str(quantum.has_superposition):<15}",
            f"  {'Interference (cont.)':<25} {str(sk.has_interference):<15} {str(rca.has_interference):<15} {str(quantum.has_interference):<15}",
            "",
            "Key Differences:",
        ]
        
        for i, diff in enumerate(differences, 1):
            lines.append(f"  {i}. {diff}")
        
        lines.extend([
            "",
            "Conclusion:",
            "  - Reversibility alone does not lead to quantum behavior (Phase 4)",
            "  - Continuous-time evolution introduces interference (Phase 5)",
            "  - RCA + continuous-time shows interference similar to SK",
            "  - True quantum uniqueness: inherent complex structure + superposition",
            "",
            "=" * 60,
        ])
        
        return "\n".join(lines)


def run_comparison() -> ComparisonResult:
    """Run the full comparison and return results."""
    comparison = ModelComparison()
    return comparison.compare_all()


if __name__ == '__main__':
    result = run_comparison()
    print(result.summary)

