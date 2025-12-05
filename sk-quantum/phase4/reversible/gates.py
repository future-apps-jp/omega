"""
Reversible Logic Gates
======================

Phase 4: 可逆計算の代数構造

目的:
    可逆論理ゲート（Toffoli, Fredkin）の行列表現を実装し、
    それらが生成する群の構造を解析する。

理論的背景:
    - 可逆計算は古典ハミルトン力学に類似（シンプレクティック構造）
    - 量子計算はユニタリ群 U(n) に基づく（複素構造）
    - 問い: 可逆ゲートの生成する群は Sp(2n,ℝ) か U(n) か？

ゲートの定義:
    - NOT: x → ¬x
    - CNOT: (a, b) → (a, a ⊕ b)
    - Toffoli (CCNOT): (a, b, c) → (a, b, c ⊕ (a ∧ b))
    - Fredkin (CSWAP): (a, b, c) → (a, (¬a ∧ b) ∨ (a ∧ c), (¬a ∧ c) ∨ (a ∧ b))
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional, Callable
from abc import ABC, abstractmethod
from enum import Enum, auto
import numpy as np
from itertools import product, permutations
from functools import reduce


# =============================================================================
# Reversible Gate Abstract Base
# =============================================================================

class ReversibleGate(ABC):
    """
    可逆論理ゲートの抽象基底クラス
    
    n-bitの状態 (b₀, b₁, ..., b_{n-1}) を受け取り、
    同じサイズの状態を返す可逆な関数。
    """
    
    @property
    @abstractmethod
    def n_bits(self) -> int:
        """ゲートが作用するビット数"""
        pass
    
    @abstractmethod
    def apply(self, state: Tuple[int, ...]) -> Tuple[int, ...]:
        """
        ゲートを状態に適用
        
        Args:
            state: n-bit の状態 (各要素は 0 or 1)
        
        Returns:
            変換後の状態
        """
        pass
    
    @abstractmethod
    def inverse(self) -> 'ReversibleGate':
        """逆ゲートを返す"""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """ゲートの名前"""
        pass
    
    def __repr__(self) -> str:
        return self.name
    
    def to_matrix(self) -> np.ndarray:
        """
        ゲートの行列表現を生成
        
        2^n × 2^n の置換行列を返す
        """
        n = self.n_bits
        dim = 2 ** n
        matrix = np.zeros((dim, dim), dtype=np.float64)
        
        for i in range(dim):
            # i を n-bit 表現に変換
            state = tuple((i >> k) & 1 for k in range(n))
            # ゲートを適用
            output = self.apply(state)
            # 出力を整数に変換
            j = sum(b << k for k, b in enumerate(output))
            matrix[j, i] = 1.0
        
        return matrix
    
    def compose(self, other: 'ReversibleGate') -> 'CompositeGate':
        """
        ゲートを合成（self を後に適用）
        
        self ∘ other = self(other(x))
        """
        return CompositeGate([other, self])
    
    def __matmul__(self, other: 'ReversibleGate') -> 'CompositeGate':
        """@ 演算子で合成"""
        return self.compose(other)


# =============================================================================
# Basic Gates
# =============================================================================

class NOTGate(ReversibleGate):
    """NOT ゲート: x → ¬x"""
    
    @property
    def n_bits(self) -> int:
        return 1
    
    def apply(self, state: Tuple[int, ...]) -> Tuple[int, ...]:
        return (1 - state[0],)
    
    def inverse(self) -> 'NOTGate':
        return self  # NOT は自己逆
    
    @property
    def name(self) -> str:
        return "NOT"


class CNOTGate(ReversibleGate):
    """
    CNOT (Controlled-NOT) ゲート
    
    (control, target) → (control, control ⊕ target)
    """
    
    @property
    def n_bits(self) -> int:
        return 2
    
    def apply(self, state: Tuple[int, ...]) -> Tuple[int, ...]:
        a, b = state
        return (a, a ^ b)
    
    def inverse(self) -> 'CNOTGate':
        return self  # CNOT は自己逆
    
    @property
    def name(self) -> str:
        return "CNOT"


class ToffoliGate(ReversibleGate):
    """
    Toffoli (CCNOT) ゲート
    
    (a, b, c) → (a, b, c ⊕ (a ∧ b))
    
    2つの制御ビット (a, b) が共に 1 のとき、
    ターゲットビット c を反転
    """
    
    @property
    def n_bits(self) -> int:
        return 3
    
    def apply(self, state: Tuple[int, ...]) -> Tuple[int, ...]:
        a, b, c = state
        return (a, b, c ^ (a & b))
    
    def inverse(self) -> 'ToffoliGate':
        return self  # Toffoli は自己逆
    
    @property
    def name(self) -> str:
        return "Toffoli"


class FredkinGate(ReversibleGate):
    """
    Fredkin (CSWAP) ゲート
    
    (a, b, c) → (a, b', c')
    where:
        if a == 0: (b', c') = (b, c)
        if a == 1: (b', c') = (c, b)  (swap)
    
    制御ビット a が 1 のとき、b と c を交換
    """
    
    @property
    def n_bits(self) -> int:
        return 3
    
    def apply(self, state: Tuple[int, ...]) -> Tuple[int, ...]:
        a, b, c = state
        if a == 0:
            return (a, b, c)
        else:
            return (a, c, b)
    
    def inverse(self) -> 'FredkinGate':
        return self  # Fredkin は自己逆
    
    @property
    def name(self) -> str:
        return "Fredkin"


class SWAPGate(ReversibleGate):
    """
    SWAP ゲート
    
    (a, b) → (b, a)
    """
    
    @property
    def n_bits(self) -> int:
        return 2
    
    def apply(self, state: Tuple[int, ...]) -> Tuple[int, ...]:
        a, b = state
        return (b, a)
    
    def inverse(self) -> 'SWAPGate':
        return self  # SWAP は自己逆
    
    @property
    def name(self) -> str:
        return "SWAP"


class IdentityGate(ReversibleGate):
    """恒等ゲート"""
    
    def __init__(self, n_bits: int = 1):
        self._n_bits = n_bits
    
    @property
    def n_bits(self) -> int:
        return self._n_bits
    
    def apply(self, state: Tuple[int, ...]) -> Tuple[int, ...]:
        return state
    
    def inverse(self) -> 'IdentityGate':
        return self
    
    @property
    def name(self) -> str:
        return f"I_{self._n_bits}"


# =============================================================================
# Composite Gate
# =============================================================================

class CompositeGate(ReversibleGate):
    """
    合成ゲート
    
    複数のゲートを順に適用
    """
    
    def __init__(self, gates: List[ReversibleGate]):
        if not gates:
            raise ValueError("gates must not be empty")
        
        # 全ゲートが同じビット数であることを確認
        n = gates[0].n_bits
        for g in gates[1:]:
            if g.n_bits != n:
                raise ValueError(f"All gates must have same n_bits, got {n} and {g.n_bits}")
        
        self.gates = gates
        self._n_bits = n
    
    @property
    def n_bits(self) -> int:
        return self._n_bits
    
    def apply(self, state: Tuple[int, ...]) -> Tuple[int, ...]:
        result = state
        for gate in self.gates:
            result = gate.apply(result)
        return result
    
    def inverse(self) -> 'CompositeGate':
        """逆ゲート：各ゲートの逆を逆順に適用"""
        return CompositeGate([g.inverse() for g in reversed(self.gates)])
    
    @property
    def name(self) -> str:
        return " ∘ ".join(g.name for g in reversed(self.gates))
    
    def to_matrix(self) -> np.ndarray:
        """合成ゲートの行列：各ゲートの行列の積"""
        matrices = [g.to_matrix() for g in self.gates]
        return reduce(np.matmul, matrices)


# =============================================================================
# Embedded Gates (for larger state spaces)
# =============================================================================

class EmbeddedGate(ReversibleGate):
    """
    より大きな状態空間に埋め込まれたゲート
    
    n_total ビットの状態空間で、指定されたビットにのみ作用
    """
    
    def __init__(self, gate: ReversibleGate, target_bits: List[int], n_total: int):
        """
        Args:
            gate: 埋め込むゲート
            target_bits: ゲートが作用するビットのインデックス
            n_total: 全体のビット数
        """
        if len(target_bits) != gate.n_bits:
            raise ValueError(f"target_bits length must match gate.n_bits")
        if max(target_bits) >= n_total:
            raise ValueError(f"target_bits must be < n_total")
        if len(set(target_bits)) != len(target_bits):
            raise ValueError("target_bits must be unique")
        
        self.gate = gate
        self.target_bits = target_bits
        self._n_bits = n_total
    
    @property
    def n_bits(self) -> int:
        return self._n_bits
    
    def apply(self, state: Tuple[int, ...]) -> Tuple[int, ...]:
        # ターゲットビットの値を抽出
        sub_state = tuple(state[i] for i in self.target_bits)
        
        # ゲートを適用
        sub_result = self.gate.apply(sub_state)
        
        # 結果を元の状態に埋め込む
        result = list(state)
        for i, idx in enumerate(self.target_bits):
            result[idx] = sub_result[i]
        
        return tuple(result)
    
    def inverse(self) -> 'EmbeddedGate':
        return EmbeddedGate(self.gate.inverse(), self.target_bits, self._n_bits)
    
    @property
    def name(self) -> str:
        return f"{self.gate.name}@{self.target_bits}"


# =============================================================================
# Gate Group Generation
# =============================================================================

class GateGroup:
    """
    ゲートの集合が生成する群を解析
    """
    
    def __init__(self, generators: List[ReversibleGate]):
        """
        Args:
            generators: 生成元となるゲートのリスト
        """
        if not generators:
            raise ValueError("generators must not be empty")
        
        n = generators[0].n_bits
        for g in generators:
            if g.n_bits != n:
                raise ValueError("All generators must have same n_bits")
        
        self.generators = generators
        self.n_bits = n
        self.dim = 2 ** n
        
        # 生成される群の要素をキャッシュ
        self._elements: Optional[Set[Tuple[Tuple[int, ...], ...]]] = None
        self._matrices: Optional[List[np.ndarray]] = None
    
    def _state_to_permutation(self, gate: ReversibleGate) -> Tuple[Tuple[int, ...], ...]:
        """ゲートを状態の置換として表現"""
        perm = []
        for i in range(self.dim):
            state = tuple((i >> k) & 1 for k in range(self.n_bits))
            output = gate.apply(state)
            j = sum(b << k for k, b in enumerate(output))
            perm.append(j)
        return tuple(perm)
    
    def generate(self, max_depth: int = 10) -> Set[Tuple[int, ...]]:
        """
        生成元から群を生成
        
        Args:
            max_depth: 最大合成深さ
        
        Returns:
            生成された置換の集合
        """
        # 恒等変換
        identity = tuple(range(self.dim))
        
        # 生成元の置換表現
        gen_perms = []
        for g in self.generators:
            perm = self._state_to_permutation(g)
            gen_perms.append(perm)
            # 逆元も追加（自己逆でない場合）
            inv_perm = self._state_to_permutation(g.inverse())
            if inv_perm != perm:
                gen_perms.append(inv_perm)
        
        # BFS で群を生成
        elements = {identity}
        frontier = set(gen_perms)
        
        for depth in range(max_depth):
            if not frontier:
                break
            
            new_frontier = set()
            for perm in frontier:
                if perm not in elements:
                    elements.add(perm)
                    # 生成元との合成
                    for gen in gen_perms:
                        # perm ∘ gen
                        new_perm = tuple(perm[gen[i]] for i in range(self.dim))
                        if new_perm not in elements:
                            new_frontier.add(new_perm)
                        # gen ∘ perm
                        new_perm = tuple(gen[perm[i]] for i in range(self.dim))
                        if new_perm not in elements:
                            new_frontier.add(new_perm)
            
            frontier = new_frontier
        
        self._elements = elements
        return elements
    
    def group_order(self, max_depth: int = 10) -> int:
        """群の位数（要素数）"""
        if self._elements is None:
            self.generate(max_depth)
        return len(self._elements)
    
    def is_symmetric_group(self) -> bool:
        """生成される群が対称群 S_n と一致するか"""
        from math import factorial
        return self.group_order() == factorial(self.dim)
    
    def get_matrices(self) -> List[np.ndarray]:
        """群の全要素の行列表現"""
        if self._elements is None:
            self.generate()
        
        matrices = []
        for perm in self._elements:
            matrix = np.zeros((self.dim, self.dim), dtype=np.float64)
            for i, j in enumerate(perm):
                matrix[j, i] = 1.0
            matrices.append(matrix)
        
        self._matrices = matrices
        return matrices


# =============================================================================
# Analysis Functions
# =============================================================================

def verify_reversibility(gate: ReversibleGate) -> bool:
    """ゲートが可逆であることを検証"""
    n = gate.n_bits
    seen = set()
    
    for i in range(2 ** n):
        state = tuple((i >> k) & 1 for k in range(n))
        output = gate.apply(state)
        
        # 出力を整数に変換
        j = sum(b << k for k, b in enumerate(output))
        
        if j in seen:
            return False  # 衝突 → 非可逆
        seen.add(j)
    
    return len(seen) == 2 ** n


def verify_self_inverse(gate: ReversibleGate) -> bool:
    """ゲートが自己逆（g ∘ g = I）かを検証"""
    n = gate.n_bits
    
    for i in range(2 ** n):
        state = tuple((i >> k) & 1 for k in range(n))
        once = gate.apply(state)
        twice = gate.apply(once)
        
        if twice != state:
            return False
    
    return True


def matrix_properties(M: np.ndarray) -> Dict:
    """行列の基本的な性質を解析"""
    results = {}
    
    # 次元
    results['shape'] = M.shape
    
    # 直交性: M^T M = I ?
    MtM = M.T @ M
    I = np.eye(M.shape[0])
    results['is_orthogonal'] = np.allclose(MtM, I)
    
    # 行列式
    results['det'] = np.linalg.det(M)
    results['det_is_pm1'] = np.isclose(abs(results['det']), 1.0)
    
    # 固有値
    eigenvalues = np.linalg.eigvals(M)
    results['eigenvalues'] = eigenvalues
    
    # 全ての固有値が絶対値1か（ユニタリ的）
    results['all_eigenvalues_unit'] = np.allclose(np.abs(eigenvalues), 1.0)
    
    # 実固有値のみか
    results['all_eigenvalues_real'] = np.allclose(eigenvalues.imag, 0.0)
    
    # トレース
    results['trace'] = np.trace(M)
    
    return results


# =============================================================================
# Standard Gate Instances
# =============================================================================

# よく使うゲートのインスタンス
NOT = NOTGate()
CNOT = CNOTGate()
TOFFOLI = ToffoliGate()
FREDKIN = FredkinGate()
SWAP = SWAPGate()


# =============================================================================
# Test / Demo
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Phase 4: Reversible Logic Gates")
    print("=" * 70)
    
    # 基本ゲートの検証
    print("\n1. 基本ゲートの検証")
    print("-" * 70)
    
    gates = [NOT, CNOT, TOFFOLI, FREDKIN, SWAP]
    
    for gate in gates:
        rev = verify_reversibility(gate)
        self_inv = verify_self_inverse(gate)
        print(f"  {gate.name}: reversible={rev}, self_inverse={self_inv}")
    
    # 行列表現
    print("\n2. Toffoli ゲートの行列表現")
    print("-" * 70)
    
    M_toffoli = TOFFOLI.to_matrix()
    print(f"  Shape: {M_toffoli.shape}")
    print(f"  Matrix:\n{M_toffoli.astype(int)}")
    
    props = matrix_properties(M_toffoli)
    print(f"\n  Properties:")
    print(f"    is_orthogonal: {props['is_orthogonal']}")
    print(f"    det: {props['det']:.1f}")
    print(f"    all_eigenvalues_unit: {props['all_eigenvalues_unit']}")
    print(f"    all_eigenvalues_real: {props['all_eigenvalues_real']}")
    
    # 群の生成
    print("\n3. Toffoli ゲートが生成する群")
    print("-" * 70)
    
    group = GateGroup([TOFFOLI])
    order = group.group_order(max_depth=20)
    print(f"  Group order: {order}")
    print(f"  Is symmetric group S_8? {group.is_symmetric_group()}")

