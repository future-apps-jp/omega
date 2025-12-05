"""
Group Analysis for Reversible Gates
====================================

Phase 4: 可逆計算の代数構造 - 群構造解析

目的:
    可逆論理ゲートが生成する群の構造を詳細に解析し、
    - 置換群 S_n の部分群としての特徴
    - 代数的閉包（closure）の性質
    - J² = -I を満たす要素の探索
    を行う。

理論的背景:
    Toffoli ゲートは計算万能（任意の可逆古典計算を実現可能）
    → 生成される群は S_{2^n} の「大きな」部分群
    しかし、この群は置換群であり、複素構造を含まない可能性が高い
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional
import numpy as np
from itertools import product, combinations
from functools import reduce
from math import factorial, gcd

from gates import (
    ReversibleGate, CompositeGate, GateGroup,
    NOT, CNOT, TOFFOLI, FREDKIN, SWAP,
    ToffoliGate, FredkinGate, EmbeddedGate, IdentityGate,
    matrix_properties
)


# =============================================================================
# Group Structure Analysis
# =============================================================================

@dataclass
class GroupAnalysisResult:
    """群解析の結果"""
    order: int
    generators: List[str]
    is_abelian: bool
    is_symmetric: bool
    center_size: int
    conjugacy_classes: int
    
    # 行列性質
    all_orthogonal: bool
    all_det_pm1: bool
    all_real_eigenvalues: bool
    
    # 複素構造
    has_j_squared_minus_i: bool
    j_candidates: List[np.ndarray]
    
    # 追加情報
    sample_eigenvalues: Dict[int, np.ndarray]


class GroupAnalyzer:
    """
    群の詳細な構造解析
    """
    
    def __init__(self, group: GateGroup):
        self.group = group
        self.matrices: List[np.ndarray] = []
        self.permutations: List[Tuple[int, ...]] = []
    
    def analyze(self, max_depth: int = 10) -> GroupAnalysisResult:
        """完全な群解析を実行"""
        
        # 群を生成
        perms = self.group.generate(max_depth)
        self.permutations = list(perms)
        self.matrices = self.group.get_matrices()
        
        order = len(self.permutations)
        dim = self.group.dim
        
        # 可換性の検証
        is_abelian = self._check_abelian()
        
        # 対称群か
        is_symmetric = (order == factorial(dim))
        
        # 中心の計算
        center_size = self._compute_center_size()
        
        # 共役類の数
        conjugacy_classes = self._count_conjugacy_classes()
        
        # 行列の性質
        all_orthogonal = True
        all_det_pm1 = True
        all_real_eigenvalues = True
        sample_eigenvalues = {}
        
        for i, M in enumerate(self.matrices[:min(100, len(self.matrices))]):
            props = matrix_properties(M)
            all_orthogonal = all_orthogonal and props['is_orthogonal']
            all_det_pm1 = all_det_pm1 and props['det_is_pm1']
            all_real_eigenvalues = all_real_eigenvalues and props['all_eigenvalues_real']
            
            if i < 10:
                sample_eigenvalues[i] = props['eigenvalues']
        
        # J² = -I の探索
        has_j, j_candidates = self._search_j_squared_minus_i()
        
        return GroupAnalysisResult(
            order=order,
            generators=[g.name for g in self.group.generators],
            is_abelian=is_abelian,
            is_symmetric=is_symmetric,
            center_size=center_size,
            conjugacy_classes=conjugacy_classes,
            all_orthogonal=all_orthogonal,
            all_det_pm1=all_det_pm1,
            all_real_eigenvalues=all_real_eigenvalues,
            has_j_squared_minus_i=has_j,
            j_candidates=j_candidates,
            sample_eigenvalues=sample_eigenvalues
        )
    
    def _check_abelian(self) -> bool:
        """群が可換かどうかを検証"""
        # サンプリングで検証
        n_samples = min(100, len(self.permutations))
        sample = self.permutations[:n_samples]
        
        for i in range(len(sample)):
            for j in range(i + 1, len(sample)):
                p1, p2 = sample[i], sample[j]
                # p1 ∘ p2
                comp1 = tuple(p1[p2[k]] for k in range(len(p1)))
                # p2 ∘ p1
                comp2 = tuple(p2[p1[k]] for k in range(len(p1)))
                
                if comp1 != comp2:
                    return False
        
        return True
    
    def _compute_center_size(self) -> int:
        """群の中心のサイズを計算"""
        center = []
        
        for p in self.permutations:
            is_central = True
            for q in self.permutations:
                # p ∘ q
                pq = tuple(p[q[k]] for k in range(len(p)))
                # q ∘ p
                qp = tuple(q[p[k]] for k in range(len(p)))
                
                if pq != qp:
                    is_central = False
                    break
            
            if is_central:
                center.append(p)
        
        return len(center)
    
    def _count_conjugacy_classes(self) -> int:
        """共役類の数を計算"""
        # 簡易実装：軌道を数える
        classified = set()
        n_classes = 0
        
        for p in self.permutations:
            if p in classified:
                continue
            
            # p の共役類
            for q in self.permutations:
                # q⁻¹ ∘ p ∘ q
                q_inv = self._inverse_perm(q)
                conj = self._compose_perm(q_inv, self._compose_perm(p, q))
                classified.add(conj)
            
            n_classes += 1
        
        return n_classes
    
    def _inverse_perm(self, p: Tuple[int, ...]) -> Tuple[int, ...]:
        """置換の逆を計算"""
        inv = [0] * len(p)
        for i, j in enumerate(p):
            inv[j] = i
        return tuple(inv)
    
    def _compose_perm(self, p1: Tuple[int, ...], p2: Tuple[int, ...]) -> Tuple[int, ...]:
        """置換の合成 p1 ∘ p2"""
        return tuple(p1[p2[i]] for i in range(len(p1)))
    
    def _search_j_squared_minus_i(self) -> Tuple[bool, List[np.ndarray]]:
        """
        J² = -I を満たす要素を探索
        
        置換行列は実数なので、J² = -I を満たすには
        J の固有値が ±i でなければならない。
        しかし置換行列の固有値は1の冪根（実数か複素共役ペア）なので、
        通常は見つからない。
        
        複素係数の線形結合を考える必要がある。
        """
        I = np.eye(self.group.dim)
        minus_I = -I
        candidates = []
        
        # 単一の置換行列では J² = -I は不可能
        # （置換行列の2乗は置換行列、-I は置換行列でない）
        
        # 複素係数の線形結合を試す
        # J = α₁M₁ + α₂M₂ + ... で J² = -I となるものを探す
        
        # まず、2つの行列の線形結合を試す
        n_matrices = min(20, len(self.matrices))
        
        for i in range(n_matrices):
            for j in range(i + 1, n_matrices):
                M1, M2 = self.matrices[i], self.matrices[j]
                
                # J = αM1 + βM2 で J² = -I を満たす (α, β) を探す
                # J² = α²M1² + αβ(M1M2 + M2M1) + β²M2² = -I
                
                # 簡略化：実数 α, β で試す（見つからないはず）
                for alpha in [1, -1, 0.5, -0.5]:
                    for beta in [1, -1, 0.5, -0.5, 1j, -1j]:
                        J = alpha * M1 + beta * M2
                        J_squared = J @ J
                        
                        if np.allclose(J_squared, minus_I):
                            candidates.append(J)
                
                # 純虚数係数も試す
                for alpha in [1j, -1j]:
                    J = alpha * M1
                    J_squared = J @ J
                    
                    if np.allclose(J_squared, minus_I):
                        candidates.append(J)
        
        # 単位行列に虚数をかけたもの（自明解）
        # J = iI → J² = -I （これは自明）
        trivial_J = 1j * I
        trivial_found = any(np.allclose(c, trivial_J) or np.allclose(c, -trivial_J) 
                           for c in candidates)
        
        # 非自明な解があるか
        nontrivial = [c for c in candidates 
                      if not (np.allclose(c, trivial_J) or np.allclose(c, -trivial_J))]
        
        return len(nontrivial) > 0, nontrivial


# =============================================================================
# Comparison with Known Groups
# =============================================================================

def analyze_toffoli_group(n_bits: int = 3, max_depth: int = 15) -> Dict:
    """
    Toffoli ゲートが生成する群を解析
    
    理論的背景：
    - Toffoli ゲートは計算万能（可逆古典計算）
    - 生成される群は A_{2^n}（交代群）を含む
    - 完全に S_{2^n}（対称群）を生成するには NOT と組み合わせが必要
    """
    results = {}
    
    # Toffoli のみ
    if n_bits == 3:
        group_t = GateGroup([TOFFOLI])
        results['toffoli_only'] = {
            'order': group_t.group_order(max_depth),
            'max_order': factorial(8),
            'is_symmetric': group_t.is_symmetric_group()
        }
    
    # Toffoli + CNOT（埋め込み）
    # 3-bit 空間での CNOT の埋め込み
    cnot_01 = EmbeddedGate(CNOT, [0, 1], 3)
    cnot_12 = EmbeddedGate(CNOT, [1, 2], 3)
    cnot_02 = EmbeddedGate(CNOT, [0, 2], 3)
    
    group_tc = GateGroup([TOFFOLI, cnot_01, cnot_12])
    results['toffoli_cnot'] = {
        'order': group_tc.group_order(max_depth),
        'max_order': factorial(8),
        'is_symmetric': group_tc.is_symmetric_group()
    }
    
    # Fredkin のみ
    group_f = GateGroup([FREDKIN])
    results['fredkin_only'] = {
        'order': group_f.group_order(max_depth),
        'max_order': factorial(8),
        'is_symmetric': group_f.is_symmetric_group()
    }
    
    return results


# =============================================================================
# Even/Odd Permutation Analysis
# =============================================================================

def parity(perm: Tuple[int, ...]) -> int:
    """
    置換のパリティ（偶置換なら 0、奇置換なら 1）
    
    置換を巡回置換に分解し、(n - サイクル数) mod 2 を計算
    """
    n = len(perm)
    visited = [False] * n
    n_cycles = 0
    
    for i in range(n):
        if visited[i]:
            continue
        
        # i から始まるサイクルをたどる
        j = i
        while not visited[j]:
            visited[j] = True
            j = perm[j]
        
        n_cycles += 1
    
    return (n - n_cycles) % 2


def analyze_parity_structure(group: GateGroup) -> Dict:
    """
    群のパリティ構造を解析
    
    - 全て偶置換 → 交代群 A_n の部分群
    - 奇置換を含む → 交代群より大きい
    """
    perms = list(group.generate())
    
    parities = [parity(p) for p in perms]
    n_even = sum(1 for p in parities if p == 0)
    n_odd = sum(1 for p in parities if p == 1)
    
    return {
        'total': len(perms),
        'n_even': n_even,
        'n_odd': n_odd,
        'all_even': n_odd == 0,
        'subgroup_of_alternating': n_odd == 0
    }


# =============================================================================
# Main Analysis
# =============================================================================

def run_group_analysis(verbose: bool = True) -> Dict:
    """
    完全な群構造解析を実行
    """
    results = {}
    
    if verbose:
        print("=" * 70)
        print("Phase 4: 群構造解析")
        print("=" * 70)
    
    # 1. Toffoli ゲートの群
    if verbose:
        print("\n1. Toffoli ゲートが生成する群")
        print("-" * 70)
    
    group_toffoli = GateGroup([TOFFOLI])
    analyzer = GroupAnalyzer(group_toffoli)
    toffoli_result = analyzer.analyze(max_depth=15)
    results['toffoli'] = toffoli_result
    
    if verbose:
        print(f"  Order: {toffoli_result.order}")
        print(f"  Is abelian: {toffoli_result.is_abelian}")
        print(f"  Is symmetric (S_8): {toffoli_result.is_symmetric}")
        print(f"  Center size: {toffoli_result.center_size}")
        print(f"  Conjugacy classes: {toffoli_result.conjugacy_classes}")
        print(f"  All orthogonal: {toffoli_result.all_orthogonal}")
        print(f"  All det = ±1: {toffoli_result.all_det_pm1}")
        print(f"  All real eigenvalues: {toffoli_result.all_real_eigenvalues}")
        print(f"  Has J² = -I (non-trivial): {toffoli_result.has_j_squared_minus_i}")
    
    # パリティ解析
    parity_result = analyze_parity_structure(group_toffoli)
    results['toffoli_parity'] = parity_result
    
    if verbose:
        print(f"\n  Parity structure:")
        print(f"    Even permutations: {parity_result['n_even']}")
        print(f"    Odd permutations: {parity_result['n_odd']}")
        print(f"    Subgroup of A_8: {parity_result['subgroup_of_alternating']}")
    
    # 2. Fredkin ゲートの群
    if verbose:
        print("\n2. Fredkin ゲートが生成する群")
        print("-" * 70)
    
    group_fredkin = GateGroup([FREDKIN])
    analyzer_f = GroupAnalyzer(group_fredkin)
    fredkin_result = analyzer_f.analyze(max_depth=15)
    results['fredkin'] = fredkin_result
    
    if verbose:
        print(f"  Order: {fredkin_result.order}")
        print(f"  Is abelian: {fredkin_result.is_abelian}")
        print(f"  All real eigenvalues: {fredkin_result.all_real_eigenvalues}")
        print(f"  Has J² = -I (non-trivial): {fredkin_result.has_j_squared_minus_i}")
    
    # 3. Toffoli + Fredkin の群
    if verbose:
        print("\n3. Toffoli + Fredkin が生成する群")
        print("-" * 70)
    
    group_tf = GateGroup([TOFFOLI, FREDKIN])
    analyzer_tf = GroupAnalyzer(group_tf)
    tf_result = analyzer_tf.analyze(max_depth=15)
    results['toffoli_fredkin'] = tf_result
    
    if verbose:
        print(f"  Order: {tf_result.order}")
        print(f"  Is symmetric (S_8): {tf_result.is_symmetric}")
        print(f"  Has J² = -I (non-trivial): {tf_result.has_j_squared_minus_i}")
    
    # 4. 結論
    if verbose:
        print("\n" + "=" * 70)
        print("結論")
        print("=" * 70)
        
        any_j = (toffoli_result.has_j_squared_minus_i or 
                 fredkin_result.has_j_squared_minus_i or
                 tf_result.has_j_squared_minus_i)
        
        if any_j:
            print("\n  🔔 非自明な J² = -I の候補が見つかりました！")
        else:
            print("\n  ✓ 可逆論理ゲートの生成する群には、")
            print("    非自明な J² = -I を満たす要素は見つかりませんでした。")
            print("\n  これは理論的に予想された結果です：")
            print("    - 置換行列は実数行列")
            print("    - 置換行列の固有値は1の冪根（|λ| = 1, λ^n = 1）")
            print("    - J² = -I は固有値 ±i を要求")
            print("    - しかし、有限巡回群の元として ±i は現れない")
        
        print("\n  群の性質：")
        all_real = (toffoli_result.all_real_eigenvalues and 
                    fredkin_result.all_real_eigenvalues)
        if all_real:
            print("    - 全ての行列の固有値は実数（または実数+複素共役ペア）")
            print("    - これは古典的シンプレクティック構造と整合")
        
        print("\n  次のステップ：")
        print("    → シンプレクティック群 Sp(2n,ℝ) への埋め込みを検証")
    
    return results


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    results = run_group_analysis(verbose=True)

