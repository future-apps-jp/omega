"""
Tests for Phase 5: Hamiltonian and Quantum Walk
"""

import pytest
import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'phase0'))

from .hamiltonian import (
    build_hamiltonian_from_expression,
    ComputationHamiltonian,
    SpectralAnalysis,
    ContinuousTimeQuantumWalk,
    ClassicalRandomWalk,
    InterferenceAnalysis,
)


class TestComputationHamiltonian:
    """計算ハミルトニアンのテスト"""
    
    def test_simple_expression(self):
        """単純な式でのハミルトニアン構築"""
        H = build_hamiltonian_from_expression("K a b")
        assert H.dimension >= 1
        assert H.adjacency is not None
        assert H.laplacian is not None
    
    def test_adjacency_symmetric(self):
        """隣接行列は対称"""
        H = build_hamiltonian_from_expression("S (K a) (K b) c")
        A = H.get_hamiltonian('adjacency')
        assert np.allclose(A, A.T)
    
    def test_laplacian_row_sum(self):
        """ラプラシアンの行和は0"""
        H = build_hamiltonian_from_expression("(K a b) (K c d)")
        L = H.get_hamiltonian('laplacian')
        row_sums = L.sum(axis=1)
        assert np.allclose(row_sums, 0)


class TestSpectralAnalysis:
    """スペクトル解析のテスト"""
    
    def test_eigenvalues_real(self):
        """実対称行列の固有値は実数"""
        H = build_hamiltonian_from_expression("S (K a) (K b) c")
        A = H.get_hamiltonian('adjacency')
        spectral = SpectralAnalysis(A)
        
        if np.iscomplexobj(spectral.eigenvalues):
            assert np.allclose(spectral.eigenvalues.imag, 0)
    
    def test_spectral_properties(self):
        """スペクトル解析の基本プロパティ"""
        H = build_hamiltonian_from_expression("(K a b) (K c d)")
        A = H.get_hamiltonian('adjacency')
        spectral = SpectralAnalysis(A)
        
        results = spectral.analyze()
        assert results['dimension'] == H.dimension
        assert results['bandwidth'] >= 0


class TestContinuousTimeQuantumWalk:
    """連続時間量子ウォークのテスト"""
    
    def test_unitary_evolution(self):
        """時間発展演算子はユニタリ"""
        H = build_hamiltonian_from_expression("S (K a) (K b) c")
        A = H.get_hamiltonian('adjacency')
        qw = ContinuousTimeQuantumWalk(A)
        
        assert qw.is_unitary(1.0)
        assert qw.is_unitary(2.5)
    
    def test_probability_conservation(self):
        """確率は保存される"""
        H = build_hamiltonian_from_expression("(K a b) (K c d)")
        A = H.get_hamiltonian('adjacency')
        qw = ContinuousTimeQuantumWalk(A)
        
        initial = np.zeros(H.dimension)
        initial[0] = 1.0
        
        for t in [0.5, 1.0, 2.0, 5.0]:
            prob = qw.probability_distribution(initial, t)
            assert np.isclose(prob.sum(), 1.0, atol=1e-10)
    
    def test_initial_state_preserved_at_t0(self):
        """t=0 では初期状態が保存"""
        H = build_hamiltonian_from_expression("K a b")
        if H.dimension < 2:
            pytest.skip("Graph too small")
        
        A = H.get_hamiltonian('adjacency')
        qw = ContinuousTimeQuantumWalk(A)
        
        initial = np.zeros(H.dimension)
        initial[0] = 1.0
        
        evolved = qw.evolve(initial, 0.0)
        assert np.allclose(evolved, initial)


class TestClassicalRandomWalk:
    """古典ランダムウォークのテスト"""
    
    def test_probability_conservation(self):
        """確率は保存される"""
        H = build_hamiltonian_from_expression("S (K a) (K b) c")
        A = H.get_hamiltonian('adjacency')
        cw = ClassicalRandomWalk(A)
        
        initial = np.zeros(H.dimension)
        initial[0] = 1.0
        
        for steps in [1, 5, 10]:
            prob = cw.evolve(initial, steps)
            assert np.isclose(prob.sum(), 1.0, atol=1e-10)
    
    def test_transition_row_stochastic(self):
        """遷移行列は行確率的"""
        H = build_hamiltonian_from_expression("(K a b) (K c d)")
        A = H.get_hamiltonian('adjacency')
        cw = ClassicalRandomWalk(A)
        
        row_sums = cw.transition.sum(axis=1)
        # 孤立ノードがなければ行和は1
        connected_rows = A.sum(axis=1) > 0
        assert np.allclose(row_sums[connected_rows], 1.0)


class TestInterferenceAnalysis:
    """干渉解析のテスト"""
    
    def test_compare_distributions(self):
        """量子と古典の分布比較"""
        H = build_hamiltonian_from_expression("S (K a) (K b) c")
        A = H.get_hamiltonian('adjacency')
        qw = ContinuousTimeQuantumWalk(A)
        cw = ClassicalRandomWalk(A)
        
        interference = InterferenceAnalysis(qw, cw, H.dimension)
        comparison = interference.compare_distributions(0, 1.0, 1)
        
        assert 'quantum_prob' in comparison
        assert 'classical_prob' in comparison
        assert 'total_variation' in comparison
    
    def test_detect_interference(self):
        """干渉検出"""
        H = build_hamiltonian_from_expression("(K a b) (K c d)")
        A = H.get_hamiltonian('adjacency')
        qw = ContinuousTimeQuantumWalk(A)
        cw = ClassicalRandomWalk(A)
        
        interference = InterferenceAnalysis(qw, cw, H.dimension)
        times = np.linspace(0.1, 5.0, 20)
        results = interference.detect_interference(0, times)
        
        assert 'has_interference' in results
        assert 'mean_oscillation' in results


class TestPhase4Consistency:
    """Phase 4 との整合性テスト"""
    
    def test_discrete_vs_continuous(self):
        """
        離散時間（整数t）vs 連続時間
        
        Phase 4 で確認したように、離散的な置換行列には複素構造がない。
        しかし、連続時間発展 exp(-iAt) は複素数を導入する。
        """
        H = build_hamiltonian_from_expression("S (K a) (K b) c")
        A = H.get_hamiltonian('adjacency')
        
        # 連続時間発展演算子
        qw = ContinuousTimeQuantumWalk(A)
        U = qw.evolution_operator(1.0)
        
        # U は複素行列
        assert np.iscomplexobj(U)
        
        # U はユニタリ（U†U = I）
        assert np.allclose(U @ U.conj().T, np.eye(len(U)))
        
        # しかし、隣接行列自体は実数
        assert not np.iscomplexobj(A)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

