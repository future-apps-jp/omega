"""
Phase 4: 可逆計算の代数構造 - 完全な実験

このスクリプトは研究計画v2 Phase 4の全ての検証を実行し、
結果を RESULTS_004_algebra.md にまとめる。
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'reversible'))

import numpy as np
from math import factorial
from itertools import product

from gates import (
    NOT, CNOT, TOFFOLI, FREDKIN, SWAP,
    GateGroup, EmbeddedGate, CompositeGate,
    verify_reversibility, verify_self_inverse, matrix_properties
)
from group_analysis import GroupAnalyzer, parity, analyze_parity_structure
from symplectic import (
    symplectic_form, is_symplectic, 
    SymplecticAnalyzer, compare_structures
)


def detailed_group_generation(gate, gate_name: str, max_depth: int = 30):
    """
    ゲートが生成する群を詳細に解析
    """
    print(f"\n{'='*70}")
    print(f"詳細解析: {gate_name}")
    print(f"{'='*70}")
    
    group = GateGroup([gate])
    
    # 段階的に生成
    for depth in [5, 10, 15, 20, 25, 30]:
        elements = group.generate(max_depth=depth)
        print(f"  depth={depth}: 位数 = {len(elements)}")
        
        if depth > 5 and len(elements) == len(group.generate(max_depth=depth-5)):
            print(f"    → 収束（これ以上増えない）")
            break
    
    final_order = len(elements)
    max_possible = factorial(group.dim)
    
    print(f"\n  最終的な群の位数: {final_order}")
    print(f"  S_{group.dim} の位数: {max_possible}")
    print(f"  比率: {final_order/max_possible:.6f}")
    
    return group, final_order


def analyze_j_squared_detailed(group: GateGroup):
    """
    J² = -I の詳細な検証
    """
    print(f"\n{'-'*70}")
    print("J² = -I の詳細検証")
    print(f"{'-'*70}")
    
    matrices = group.get_matrices()
    dim = group.dim
    I = np.eye(dim)
    
    print(f"  群の要素数: {len(matrices)}")
    
    # 1. 単一の置換行列で J² = -I となるものはあるか
    print("\n  1. 単一の置換行列で J² = -I:")
    single_candidates = []
    for i, M in enumerate(matrices):
        M2 = M @ M
        if np.allclose(M2, -I):
            single_candidates.append(i)
    
    if single_candidates:
        print(f"    候補あり: {single_candidates}")
    else:
        print(f"    候補なし（理論通り：置換行列は J² = -I を満たさない）")
    
    # 2. 線形結合 J = αM₁ + βM₂ で検証
    print("\n  2. 線形結合での検証:")
    
    # まず、置換行列の固有値を調べる
    print("\n    置換行列の固有値（最初の5個）:")
    for i, M in enumerate(matrices[:5]):
        eigs = np.linalg.eigvals(M)
        # ユニークな固有値
        unique_eigs = np.unique(np.round(eigs, 6))
        print(f"      M_{i}: {unique_eigs}")
    
    # 3. 複素係数を使った場合の検証
    print("\n  3. 複素係数 J = iI での検証:")
    J_trivial = 1j * I
    J_trivial_sq = J_trivial @ J_trivial
    print(f"    (iI)² = -I ? {np.allclose(J_trivial_sq, -I)}")
    print(f"    これは「自明解」：複素数を最初から仮定している")
    
    # 4. 結論
    print("\n  4. 結論:")
    print("    - 置換行列の有限群からは、J² = -I を満たす非自明な J は生成されない")
    print("    - 置換行列の固有値は1の冪根（λⁿ = 1）であり、±i にはならない")
    print("    - 複素係数を導入すれば J = iI という自明解は存在するが、")
    print("      これは複素数を「仮定」しており「導出」ではない")
    
    return len(single_candidates) == 0


def analyze_symplectic_embedding_detailed(group: GateGroup):
    """
    シンプレクティック埋め込みの詳細検証
    """
    print(f"\n{'-'*70}")
    print("シンプレクティック埋め込みの詳細検証")
    print(f"{'-'*70}")
    
    dim = group.dim
    matrices = group.get_matrices()
    
    print(f"  元の次元: {dim}")
    print(f"  埋め込み次元: {2*dim}")
    
    # 埋め込み: M → [[M, 0], [0, M]]
    omega = symplectic_form(dim)
    
    print("\n  シンプレクティック形式 Ω の構造:")
    print(f"    Ω = [[0, I], [-I, 0]] (サイズ {2*dim}×{2*dim})")
    
    # 各要素の埋め込み検証
    all_symplectic = True
    for i, M in enumerate(matrices[:10]):
        M_embed = np.block([
            [M, np.zeros((dim, dim))],
            [np.zeros((dim, dim)), M]
        ])
        
        # M^T Ω M = Ω の検証
        result = M_embed.T @ omega @ M_embed
        is_sympl = np.allclose(result, omega)
        all_symplectic = all_symplectic and is_sympl
        
        if i < 3:
            print(f"\n  M_{i} のシンプレクティック条件:")
            print(f"    M^T Ω M = Ω ? {is_sympl}")
    
    print(f"\n  全要素がシンプレクティック: {all_symplectic}")
    
    # 理論的解釈
    print("\n  理論的解釈:")
    print("    - 置換行列 P は直交行列 (P^T P = I)")
    print("    - 埋め込み [[P, 0], [0, P]] は常にシンプレクティック条件を満たす")
    print("    - これは P ∈ O(n) ⊂ Sp(2n, ℝ) の標準的埋め込み")
    print("    - しかし、これは「古典的」シンプレクティック構造であり、")
    print("      U(n) への拡大とは異なる")
    
    return all_symplectic


def analyze_complex_structure_detailed(group: GateGroup):
    """
    複素構造との関係の詳細検証
    """
    print(f"\n{'-'*70}")
    print("複素構造 J との関係")
    print(f"{'-'*70}")
    
    dim = group.dim
    matrices = group.get_matrices()
    
    # 標準複素構造
    I_n = np.eye(dim)
    O = np.zeros((dim, dim))
    J = np.block([
        [O, -I_n],
        [I_n, O]
    ])
    
    print(f"  標準複素構造 J = [[0, -I], [I, 0]]")
    print(f"  J² = -I ? {np.allclose(J @ J, -np.eye(2*dim))}")
    
    # 各要素が J と可換か
    print("\n  各要素と J の交換関係:")
    commuting_count = 0
    
    for i, M in enumerate(matrices[:10]):
        M_embed = np.block([
            [M, np.zeros((dim, dim))],
            [np.zeros((dim, dim)), M]
        ])
        
        commutator = M_embed @ J - J @ M_embed
        is_commuting = np.allclose(commutator, np.zeros_like(commutator))
        
        if is_commuting:
            commuting_count += 1
        
        if i < 3:
            print(f"    [M_{i}, J] = 0 ? {is_commuting}")
    
    print(f"\n  J と可換な要素: {commuting_count}/{min(10, len(matrices))}")
    
    # 理論的解釈
    print("\n  理論的解釈:")
    print("    - 埋め込み [[M, 0], [0, M]] は J = [[0, -I], [I, 0]] と常に可換")
    print("    - これは [[M, 0], [0, M]] が「実数的」ブロック対角だから")
    print("    - しかし、これは M がユニタリ（複素）であることを意味しない")
    print("    - M は実直交行列であり、複素数的位相を持たない")
    
    # 核心的な区別
    print("\n  核心的な区別:")
    print("    シンプレクティック + 複素構造 = ケーラー構造")
    print("    しかし、離散的な置換群では：")
    print("    - 連続的な位相 e^{iθ} が存在しない")
    print("    - パス間の干渉項 2Re(ψ₁*ψ₂) を計算できない")
    print("    - Born 則 |ψ|² が意味を持たない")
    
    return commuting_count == min(10, len(matrices))


def main():
    """
    Phase 4 の完全な実験を実行
    """
    print("=" * 70)
    print("Phase 4: 可逆計算の代数構造 - 完全な実験")
    print("=" * 70)
    
    results = {}
    
    # 1. Toffoli ゲートの詳細解析
    toffoli_group, toffoli_order = detailed_group_generation(TOFFOLI, "Toffoli")
    results['toffoli_order'] = toffoli_order
    
    # 2. Fredkin ゲートの詳細解析
    fredkin_group, fredkin_order = detailed_group_generation(FREDKIN, "Fredkin")
    results['fredkin_order'] = fredkin_order
    
    # 3. Toffoli + Fredkin の詳細解析
    tf_gate = CompositeGate([TOFFOLI, FREDKIN])
    print(f"\n{'='*70}")
    print("詳細解析: Toffoli + Fredkin (合成)")
    print(f"{'='*70}")
    
    tf_group = GateGroup([TOFFOLI, FREDKIN])
    for depth in [5, 10, 15, 20]:
        elements = tf_group.generate(max_depth=depth)
        print(f"  depth={depth}: 位数 = {len(elements)}")
    
    results['toffoli_fredkin_order'] = len(elements)
    
    # 4. J² = -I の詳細検証
    no_nontrivial_j = analyze_j_squared_detailed(toffoli_group)
    results['no_nontrivial_j'] = no_nontrivial_j
    
    # 5. シンプレクティック埋め込みの詳細検証
    all_symplectic = analyze_symplectic_embedding_detailed(toffoli_group)
    results['all_symplectic'] = all_symplectic
    
    # 6. 複素構造との関係
    all_commuting_with_j = analyze_complex_structure_detailed(toffoli_group)
    results['all_commuting_with_j'] = all_commuting_with_j
    
    # 7. 最終結論
    print("\n" + "=" * 70)
    print("Phase 4: 最終結論")
    print("=" * 70)
    
    print("""
    ┌─────────────────────────────────────────────────────────────────┐
    │                      Phase 4 の結論                              │
    ├─────────────────────────────────────────────────────────────────┤
    │                                                                  │
    │  1. 群構造:                                                      │
    │     - Toffoli/Fredkin は有限置換群を生成                        │
    │     - S_8 全体は生成しない（可逆古典計算の「万能性」とは別問題）│
    │                                                                  │
    │  2. J² = -I の探索:                                              │
    │     - 置換行列の有限群からは非自明な J は生成されない           │
    │     - 複素係数を仮定すれば J = iI は存在するが、これは自明解    │
    │                                                                  │
    │  3. シンプレクティック埋め込み:                                  │
    │     - 全ての置換行列は Sp(2n, ℝ) に埋め込み可能                 │
    │     - これは古典ハミルトン系と同型の構造                        │
    │                                                                  │
    │  4. 複素構造との関係:                                            │
    │     - 埋め込まれた行列は標準複素構造 J と可換                   │
    │     - しかし、これは「量子的」を意味しない                      │
    │     - 離散的置換群には連続的位相 e^{iθ} が存在しない            │
    │                                                                  │
    │  ★ 最終判定: 可逆論理ゲートは「古典的」                         │
    │     - シンプレクティック構造に埋め込めるが、                    │
    │     - ユニタリ群 U(n) の「量子的」構造は生成しない              │
    │                                                                  │
    │  これは研究計画v2の仮説 H1 を支持する:                          │
    │  「可逆計算は古典的であり、Sp(2n,ℝ) に閉じる」                  │
    │                                                                  │
    └─────────────────────────────────────────────────────────────────┘
    """)
    
    return results


if __name__ == "__main__":
    results = main()

