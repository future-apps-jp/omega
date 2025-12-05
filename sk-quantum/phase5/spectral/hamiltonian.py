"""
Hamiltonian from Computation Graph
==================================

Phase 5: ハミルトニアンと干渉

目的:
    計算グラフの隣接行列をハミルトニアンとして定義し、
    連続時間発展による干渉の有無を検証する。

理論的背景:
    離散計算（置換行列）では複素構造が生じない（Phase 4で確認）。
    しかし、連続時間化 U(t) = exp(-iHt) では複素指数関数が現れ、
    干渉が生じる可能性がある。

    これは連続時間量子ウォーク（CTQW）と同じ構造：
    - H = A（隣接行列）または H = L（ラプラシアン）
    - 初期状態 |ψ(0)⟩ から時間発展 |ψ(t)⟩ = exp(-iHt)|ψ(0)⟩
    - 干渉パターンが観測されれば「量子的」
"""

from __future__ import annotations
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'phase0'))

from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional
import numpy as np
from numpy.linalg import eig, eigvals
from scipy.linalg import expm

from sk_parser import SKExpr, parse, to_string, to_canonical
from multiway import MultiwayGraph, build_multiway_graph, MultiwayNode


# =============================================================================
# Adjacency Matrix Construction
# =============================================================================

@dataclass
class ComputationHamiltonian:
    """
    計算グラフからハミルトニアン（隣接行列）を構築
    """
    graph: MultiwayGraph
    nodes: List[MultiwayNode] = field(default_factory=list, init=False)
    node_to_idx: Dict[str, int] = field(default_factory=dict, init=False)
    adjacency: np.ndarray = field(default=None, init=False)
    laplacian: np.ndarray = field(default=None, init=False)
    
    def __post_init__(self):
        self._build_matrices()
    
    def _build_matrices(self):
        """隣接行列とラプラシアンを構築"""
        # ノードのリスト化
        self.nodes = list(self.graph.nodes.values())
        self.node_to_idx = {node.node_id: i for i, node in enumerate(self.nodes)}
        
        n = len(self.nodes)
        self.adjacency = np.zeros((n, n), dtype=np.float64)
        
        # 辺を追加（各ノードの children から）
        for node in self.nodes:
            i = self.node_to_idx[node.node_id]
            for child in node.children.values():
                j = self.node_to_idx[child.node_id]
                self.adjacency[i, j] = 1.0
                self.adjacency[j, i] = 1.0  # 無向化
        
        # ラプラシアン L = D - A
        degree = np.diag(self.adjacency.sum(axis=1))
        self.laplacian = degree - self.adjacency
    
    @property
    def dimension(self) -> int:
        """ヒルベルト空間の次元"""
        return len(self.nodes)
    
    def get_hamiltonian(self, type: str = 'adjacency') -> np.ndarray:
        """
        ハミルトニアンを取得
        
        Args:
            type: 'adjacency' or 'laplacian'
        """
        if type == 'adjacency':
            return self.adjacency
        elif type == 'laplacian':
            return self.laplacian
        else:
            raise ValueError(f"Unknown type: {type}")
    
    def get_node_label(self, idx: int) -> str:
        """ノードのラベル（式の文字列表現）"""
        return to_string(self.nodes[idx].expr)


# =============================================================================
# Spectral Analysis
# =============================================================================

@dataclass
class SpectralAnalysis:
    """
    ハミルトニアンのスペクトル解析
    """
    hamiltonian: np.ndarray
    eigenvalues: np.ndarray = field(default=None, init=False)
    eigenvectors: np.ndarray = field(default=None, init=False)
    
    def __post_init__(self):
        self._compute_spectrum()
    
    def _compute_spectrum(self):
        """固有値・固有ベクトルを計算"""
        self.eigenvalues, self.eigenvectors = eig(self.hamiltonian)
        # 実対称行列なので固有値は実数のはず
        if np.allclose(self.eigenvalues.imag, 0):
            self.eigenvalues = self.eigenvalues.real
        # 固有値でソート
        idx = np.argsort(self.eigenvalues)
        self.eigenvalues = self.eigenvalues[idx]
        self.eigenvectors = self.eigenvectors[:, idx]
    
    @property
    def spectral_gap(self) -> float:
        """スペクトルギャップ（最小非ゼロ固有値）"""
        nonzero = self.eigenvalues[np.abs(self.eigenvalues) > 1e-10]
        if len(nonzero) == 0:
            return 0.0
        return np.min(np.abs(nonzero))
    
    @property
    def bandwidth(self) -> float:
        """帯域幅（最大固有値 - 最小固有値）"""
        return np.max(self.eigenvalues) - np.min(self.eigenvalues)
    
    def analyze(self) -> Dict:
        """スペクトルの詳細解析"""
        return {
            'dimension': len(self.eigenvalues),
            'eigenvalues': self.eigenvalues,
            'min_eigenvalue': np.min(self.eigenvalues),
            'max_eigenvalue': np.max(self.eigenvalues),
            'spectral_gap': self.spectral_gap,
            'bandwidth': self.bandwidth,
            'all_real': np.allclose(self.eigenvalues.imag, 0) if np.iscomplexobj(self.eigenvalues) else True,
            'degeneracy': self._count_degeneracy(),
        }
    
    def _count_degeneracy(self, tol: float = 1e-8) -> Dict[float, int]:
        """固有値の縮退度を計算"""
        unique, counts = np.unique(np.round(self.eigenvalues.real, 8), return_counts=True)
        return {float(v): int(c) for v, c in zip(unique, counts) if c > 1}


# =============================================================================
# Quantum Walk
# =============================================================================

@dataclass
class ContinuousTimeQuantumWalk:
    """
    連続時間量子ウォーク
    
    |ψ(t)⟩ = exp(-iHt)|ψ(0)⟩
    
    H は隣接行列またはラプラシアン
    """
    hamiltonian: np.ndarray
    
    def evolve(self, initial_state: np.ndarray, t: float) -> np.ndarray:
        """
        時間発展
        
        Args:
            initial_state: 初期状態ベクトル |ψ(0)⟩
            t: 時間
        
        Returns:
            |ψ(t)⟩ = exp(-iHt)|ψ(0)⟩
        """
        U = expm(-1j * self.hamiltonian * t)
        return U @ initial_state
    
    def probability_distribution(self, initial_state: np.ndarray, t: float) -> np.ndarray:
        """
        確率分布 |ψ(t)|²
        """
        psi_t = self.evolve(initial_state, t)
        return np.abs(psi_t) ** 2
    
    def evolution_operator(self, t: float) -> np.ndarray:
        """
        時間発展演算子 U(t) = exp(-iHt)
        """
        return expm(-1j * self.hamiltonian * t)
    
    def is_unitary(self, t: float, tol: float = 1e-10) -> bool:
        """U(t) がユニタリかどうか"""
        U = self.evolution_operator(t)
        I = np.eye(len(U))
        return np.allclose(U @ U.conj().T, I, atol=tol)


# =============================================================================
# Classical Random Walk (for comparison)
# =============================================================================

@dataclass
class ClassicalRandomWalk:
    """
    古典ランダムウォーク（比較用）
    
    遷移確率行列 P = D^{-1} A
    """
    adjacency: np.ndarray
    transition: np.ndarray = field(default=None, init=False)
    
    def __post_init__(self):
        degree = self.adjacency.sum(axis=1)
        # ゼロ除算を避ける
        degree[degree == 0] = 1
        self.transition = self.adjacency / degree[:, np.newaxis]
    
    def step(self, distribution: np.ndarray) -> np.ndarray:
        """
        1ステップの遷移
        
        p(t+1) = P^T p(t)
        """
        return self.transition.T @ distribution
    
    def evolve(self, initial: np.ndarray, steps: int) -> np.ndarray:
        """
        複数ステップの遷移
        """
        p = initial.copy()
        for _ in range(steps):
            p = self.step(p)
        return p
    
    def stationary_distribution(self) -> np.ndarray:
        """
        定常分布（主固有ベクトル）
        """
        eigenvalues, eigenvectors = eig(self.transition.T)
        # 固有値1に対応する固有ベクトル
        idx = np.argmin(np.abs(eigenvalues - 1))
        stationary = eigenvectors[:, idx].real
        return stationary / stationary.sum()


# =============================================================================
# Interference Detection
# =============================================================================

@dataclass
class InterferenceAnalysis:
    """
    干渉の検出と解析
    """
    quantum_walk: ContinuousTimeQuantumWalk
    classical_walk: ClassicalRandomWalk
    dimension: int
    
    def compare_distributions(self, initial_idx: int, t: float, 
                              classical_steps: int = None) -> Dict:
        """
        量子ウォークと古典ウォークの分布を比較
        
        Args:
            initial_idx: 初期状態のノードインデックス
            t: 量子ウォークの時間
            classical_steps: 古典ウォークのステップ数（None なら t を使用）
        """
        # 初期状態
        initial = np.zeros(self.dimension)
        initial[initial_idx] = 1.0
        
        # 量子ウォーク
        quantum_prob = self.quantum_walk.probability_distribution(initial, t)
        
        # 古典ウォーク
        if classical_steps is None:
            classical_steps = int(t)
        classical_prob = self.classical_walk.evolve(initial, max(1, classical_steps))
        
        # 比較指標
        return {
            'quantum_prob': quantum_prob,
            'classical_prob': classical_prob,
            'total_variation': 0.5 * np.sum(np.abs(quantum_prob - classical_prob)),
            'quantum_entropy': self._entropy(quantum_prob),
            'classical_entropy': self._entropy(classical_prob),
            'max_quantum_prob': np.max(quantum_prob),
            'max_classical_prob': np.max(classical_prob),
        }
    
    def _entropy(self, prob: np.ndarray) -> float:
        """Shannon エントロピー"""
        p = prob[prob > 1e-15]
        return -np.sum(p * np.log2(p))
    
    def detect_interference(self, initial_idx: int, times: List[float]) -> Dict:
        """
        干渉パターンを検出
        
        量子ウォークでは確率が時間に対して振動するが、
        古典ウォークでは単調に定常分布に近づく。
        """
        initial = np.zeros(self.dimension)
        initial[initial_idx] = 1.0
        
        quantum_probs = []
        for t in times:
            prob = self.quantum_walk.probability_distribution(initial, t)
            quantum_probs.append(prob)
        
        quantum_probs = np.array(quantum_probs)
        
        # 時間方向の振動を検出
        oscillation = np.std(quantum_probs, axis=0)
        
        return {
            'times': times,
            'quantum_probs': quantum_probs,
            'oscillation_per_node': oscillation,
            'mean_oscillation': np.mean(oscillation),
            'has_interference': np.mean(oscillation) > 0.01,
        }


# =============================================================================
# Main Analysis Functions
# =============================================================================

def build_hamiltonian_from_expression(expr_str: str, max_depth: int = 20) -> ComputationHamiltonian:
    """
    SK式から計算グラフを構築し、ハミルトニアンを生成
    """
    expr = parse(expr_str)
    graph = build_multiway_graph(expr, max_depth=max_depth)
    return ComputationHamiltonian(graph)


def analyze_expression(expr_str: str, max_depth: int = 20, verbose: bool = True) -> Dict:
    """
    SK式の完全な解析（スペクトル + 量子ウォーク）
    """
    results = {'expression': expr_str}
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"解析: {expr_str}")
        print(f"{'='*70}")
    
    # ハミルトニアン構築
    H = build_hamiltonian_from_expression(expr_str, max_depth)
    results['dimension'] = H.dimension
    results['n_nodes'] = len(H.nodes)
    
    # 辺数をカウント（隣接行列から）
    n_edges = int(H.adjacency.sum() / 2)  # 無向グラフなので2で割る
    results['n_edges'] = n_edges
    
    if verbose:
        print(f"\nグラフ構造:")
        print(f"  ノード数: {H.dimension}")
        print(f"  辺数: {n_edges}")
    
    if H.dimension < 2:
        if verbose:
            print("  ⚠️ グラフが小さすぎます（ノード数 < 2）")
        results['error'] = 'Graph too small'
        return results
    
    # スペクトル解析
    adj = H.get_hamiltonian('adjacency')
    spectral = SpectralAnalysis(adj)
    spec_results = spectral.analyze()
    results['spectral'] = spec_results
    
    if verbose:
        print(f"\nスペクトル解析:")
        print(f"  固有値範囲: [{spec_results['min_eigenvalue']:.4f}, {spec_results['max_eigenvalue']:.4f}]")
        print(f"  スペクトルギャップ: {spec_results['spectral_gap']:.4f}")
        print(f"  帯域幅: {spec_results['bandwidth']:.4f}")
        if spec_results['degeneracy']:
            print(f"  縮退: {spec_results['degeneracy']}")
    
    # 量子ウォーク
    qw = ContinuousTimeQuantumWalk(adj)
    cw = ClassicalRandomWalk(adj)
    
    # ユニタリ性の検証
    is_unitary = qw.is_unitary(1.0)
    results['is_unitary'] = is_unitary
    
    if verbose:
        print(f"\n量子ウォーク:")
        print(f"  U(t=1) はユニタリ: {is_unitary}")
    
    # 干渉解析
    interference = InterferenceAnalysis(qw, cw, H.dimension)
    times = np.linspace(0.1, 10, 50)
    int_results = interference.detect_interference(0, times)
    results['interference'] = {
        'has_interference': int_results['has_interference'],
        'mean_oscillation': int_results['mean_oscillation'],
    }
    
    if verbose:
        print(f"  平均振動: {int_results['mean_oscillation']:.4f}")
        print(f"  干渉あり: {int_results['has_interference']}")
    
    # 量子 vs 古典の比較
    comparison = interference.compare_distributions(0, 5.0, 5)
    results['quantum_vs_classical'] = {
        'total_variation': comparison['total_variation'],
        'quantum_entropy': comparison['quantum_entropy'],
        'classical_entropy': comparison['classical_entropy'],
    }
    
    if verbose:
        print(f"\n量子 vs 古典 (t=5):")
        print(f"  Total Variation: {comparison['total_variation']:.4f}")
        print(f"  量子エントロピー: {comparison['quantum_entropy']:.4f}")
        print(f"  古典エントロピー: {comparison['classical_entropy']:.4f}")
    
    return results


def run_phase5_analysis(verbose: bool = True) -> Dict:
    """
    Phase 5 の完全な解析を実行
    """
    results = {}
    
    if verbose:
        print("=" * 70)
        print("Phase 5: ハミルトニアンと干渉")
        print("=" * 70)
        print("\n目的: 連続時間量子ウォークで干渉が生じるかを検証")
        print("理論: U(t) = exp(-iAt) は複素数を導入し、干渉を生じさせる可能性")
    
    # テスト式
    test_expressions = [
        "S (K a) (K b) c",
        "(K a b) (K c d)",
        "(K a b) (K c d) (K e f)",
        "S (K a) (K b) (S c d e)",
    ]
    
    for expr_str in test_expressions:
        results[expr_str] = analyze_expression(expr_str, verbose=verbose)
    
    # 結論
    if verbose:
        print("\n" + "=" * 70)
        print("Phase 5: 結論")
        print("=" * 70)
        
        any_interference = any(
            r.get('interference', {}).get('has_interference', False)
            for r in results.values()
        )
        
        if any_interference:
            print("\n  🔔 干渉パターンが検出されました！")
            print("     連続時間量子ウォークでは、離散計算と異なる振る舞いが生じます。")
        else:
            print("\n  ✓ 干渉パターンは検出されませんでした。")
            print("     しかし、これは計算グラフの構造に依存する可能性があります。")
        
        print("\n  理論的考察:")
        print("    - 隣接行列 A は実対称行列 → 固有値は実数")
        print("    - U(t) = exp(-iAt) は複素行列 → ユニタリ")
        print("    - 複素構造は「連続時間化」によって導入される")
        print("    - これは「離散→連続」の極限で量子性が現れることを示唆")
    
    return results


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    results = run_phase5_analysis(verbose=True)

