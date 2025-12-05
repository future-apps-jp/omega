"""
Hamiltonian construction for RCA graphs.

This enables continuous-time quantum walk on RCA,
allowing direct comparison with SK quantum walks.
"""

import numpy as np
from scipy import linalg
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from .automata import ReversibleCellularAutomaton, RCAState
from .graph import RCAGraph


@dataclass
class RCASpectralAnalysis:
    """Results of spectral analysis on RCA Hamiltonian."""
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    spectral_gap: float
    is_bipartite: bool
    bandwidth: float


class RCAHamiltonian:
    """
    Hamiltonian for continuous-time dynamics on RCA graph.
    
    H = A (adjacency matrix) or H = L (Laplacian)
    
    Time evolution: U(t) = exp(-i H t)
    """
    
    def __init__(self, graph: RCAGraph, use_laplacian: bool = False):
        """
        Initialize Hamiltonian from RCA graph.
        
        Args:
            graph: The RCA graph
            use_laplacian: If True, use Laplacian; else use adjacency
        """
        self.graph = graph
        self.use_laplacian = use_laplacian
        
        self.adjacency = graph.get_adjacency_matrix()
        self.laplacian = self._build_laplacian()
        
        self.H = self.laplacian if use_laplacian else self.adjacency
    
    def _build_laplacian(self) -> np.ndarray:
        """Build graph Laplacian: L = D - A."""
        degrees = np.sum(self.adjacency, axis=1)
        D = np.diag(degrees)
        return D - self.adjacency
    
    def spectral_analysis(self) -> RCASpectralAnalysis:
        """
        Perform spectral analysis of the Hamiltonian.
        
        Returns:
            RCASpectralAnalysis with eigenvalues, eigenvectors, etc.
        """
        if len(self.H) == 0:
            return RCASpectralAnalysis(
                eigenvalues=np.array([]),
                eigenvectors=np.array([[]]),
                spectral_gap=0.0,
                is_bipartite=False,
                bandwidth=0.0
            )
        
        eigenvalues, eigenvectors = np.linalg.eigh(self.H)
        
        # Sort by eigenvalue
        idx = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Spectral gap (second smallest for Laplacian, or gap from 0)
        if self.use_laplacian and len(eigenvalues) > 1:
            spectral_gap = eigenvalues[1]  # Algebraic connectivity
        elif len(eigenvalues) > 1:
            spectral_gap = eigenvalues[1] - eigenvalues[0]
        else:
            spectral_gap = 0.0
        
        # Check bipartiteness (adjacency has symmetric spectrum)
        is_bipartite = False
        if len(eigenvalues) > 0 and not self.use_laplacian:
            neg_evs = eigenvalues[eigenvalues < -1e-10]
            pos_evs = eigenvalues[eigenvalues > 1e-10]
            if len(neg_evs) == len(pos_evs):
                is_bipartite = np.allclose(np.sort(-neg_evs), np.sort(pos_evs))
        
        bandwidth = eigenvalues[-1] - eigenvalues[0] if len(eigenvalues) > 0 else 0.0
        
        return RCASpectralAnalysis(
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            spectral_gap=spectral_gap,
            is_bipartite=is_bipartite,
            bandwidth=bandwidth
        )
    
    def quantum_walk(self, initial_node: int, times: np.ndarray) -> np.ndarray:
        """
        Simulate continuous-time quantum walk.
        
        Args:
            initial_node: Starting node index
            times: Array of times to evaluate
        
        Returns:
            Array of shape (len(times), num_nodes) with probabilities
        """
        n = len(self.H)
        if n == 0:
            return np.array([[]])
        
        # Initial state: localized at initial_node
        psi_0 = np.zeros(n, dtype=np.complex128)
        psi_0[initial_node] = 1.0
        
        probabilities = np.zeros((len(times), n))
        
        for i, t in enumerate(times):
            # U(t) = exp(-i H t)
            U = linalg.expm(-1j * self.H * t)
            psi_t = U @ psi_0
            probabilities[i] = np.abs(psi_t) ** 2
        
        return probabilities
    
    def classical_walk(self, initial_node: int, times: np.ndarray,
                       dt: float = 0.01) -> np.ndarray:
        """
        Simulate classical random walk for comparison.
        
        Using continuous-time Markov chain: dp/dt = -L p
        
        Args:
            initial_node: Starting node index
            times: Array of times to evaluate
            dt: Time step for numerical integration
        
        Returns:
            Array of shape (len(times), num_nodes) with probabilities
        """
        n = len(self.H)
        if n == 0:
            return np.array([[]])
        
        # Transition rate matrix (generator)
        degrees = np.sum(self.adjacency, axis=1)
        # Avoid division by zero
        degrees = np.maximum(degrees, 1)
        Q = self.adjacency / degrees[:, np.newaxis]
        np.fill_diagonal(Q, 0)
        np.fill_diagonal(Q, -np.sum(Q, axis=1))
        
        # Initial distribution
        p_0 = np.zeros(n)
        p_0[initial_node] = 1.0
        
        probabilities = np.zeros((len(times), n))
        
        for i, t in enumerate(times):
            # P(t) = exp(Q t)
            P_t = linalg.expm(Q * t)
            p_t = P_t @ p_0
            # Ensure non-negative
            p_t = np.maximum(p_t, 0)
            # Normalize
            if np.sum(p_t) > 0:
                p_t = p_t / np.sum(p_t)
            probabilities[i] = p_t
        
        return probabilities
    
    def detect_interference(self, times: np.ndarray,
                           initial_node: int = 0) -> Dict:
        """
        Detect quantum interference by comparing quantum vs classical walks.
        
        Args:
            times: Time points to analyze
            initial_node: Starting node
        
        Returns:
            Dictionary with interference metrics
        """
        quantum_probs = self.quantum_walk(initial_node, times)
        classical_probs = self.classical_walk(initial_node, times)
        
        # Total variation distance at each time
        tvd = 0.5 * np.sum(np.abs(quantum_probs - classical_probs), axis=1)
        
        # Check for oscillation in return probability (quantum signature)
        return_probs_q = quantum_probs[:, initial_node]
        return_probs_c = classical_probs[:, initial_node]
        
        # Count sign changes in derivative (oscillations)
        diff_q = np.diff(return_probs_q)
        sign_changes_q = np.sum(np.diff(np.sign(diff_q)) != 0)
        
        diff_c = np.diff(return_probs_c)
        sign_changes_c = np.sum(np.diff(np.sign(diff_c)) != 0)
        
        # Interference detected if quantum has significantly more oscillations
        has_interference = sign_changes_q > sign_changes_c + 2
        
        # Also check TVD growth
        max_tvd = np.max(tvd)
        avg_tvd = np.mean(tvd)
        
        return {
            'has_interference': has_interference,
            'quantum_oscillations': sign_changes_q,
            'classical_oscillations': sign_changes_c,
            'max_tvd': max_tvd,
            'avg_tvd': avg_tvd,
            'return_prob_quantum': return_probs_q,
            'return_prob_classical': return_probs_c,
            'tvd_over_time': tvd,
        }


def build_rca_hamiltonian(size: int, rule: int = 90,
                          full_space: bool = False) -> RCAHamiltonian:
    """
    Convenience function to build RCA Hamiltonian.
    
    Args:
        size: Number of cells in RCA
        rule: CA rule number (default: 90)
        full_space: If True, build full state space graph
    
    Returns:
        RCAHamiltonian instance
    """
    rca = ReversibleCellularAutomaton(size, rule)
    graph = RCAGraph(rca)
    
    if full_space:
        graph.build_full_state_space()
    else:
        initial = RCAState(tuple([1] + [0] * (size - 1)))
        previous = RCAState(tuple([0] * size))
        graph.build_from_initial(initial, previous)
    
    return RCAHamiltonian(graph)


if __name__ == '__main__':
    # Test RCA Hamiltonian
    H = build_rca_hamiltonian(3, rule=90, full_space=False)
    
    print(f"RCA Hamiltonian (Rule 90, 3 cells)")
    print(f"  Graph nodes: {H.graph.num_nodes}")
    print(f"  Graph edges: {H.graph.num_edges}")
    
    # Spectral analysis
    spec = H.spectral_analysis()
    print(f"\nSpectral Analysis:")
    print(f"  Eigenvalues: {spec.eigenvalues[:5]}...")
    print(f"  Spectral gap: {spec.spectral_gap:.4f}")
    print(f"  Bandwidth: {spec.bandwidth:.4f}")
    
    # Interference detection
    times = np.linspace(0, 10, 100)
    interference = H.detect_interference(times)
    print(f"\nInterference Detection:")
    print(f"  Has interference: {interference['has_interference']}")
    print(f"  Quantum oscillations: {interference['quantum_oscillations']}")
    print(f"  Classical oscillations: {interference['classical_oscillations']}")
    print(f"  Max TVD: {interference['max_tvd']:.4f}")

