"""
SK Operators as Linear Operators on Expression Space

SK計算の演算子を、式空間上の線形演算子として定式化する。

理論的背景:
    SK計算では、式 E に対して S や K を適用（アプリケーション）することで
    新しい式が得られる。これを線形演算子として捉える：
    
    Ŝ: |E⟩ → |S E⟩  (Sを左から適用)
    K̂: |E⟩ → |K E⟩  (Kを左から適用)
    
    重要な問い:
    - [Ŝ, K̂] = Ŝ K̂ - K̂ Ŝ ≠ 0 か？
    - この非可換性から重ね合わせは生じるか？
"""

import numpy as np
from typing import List, Dict, Set, Tuple, Optional, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'phase0'))
from sk_parser import SKExpr, S, K, Var, App, parse, to_string, to_canonical


@dataclass
class ExpressionBasis:
    """
    式空間の基底
    
    有限次元近似として、特定の深さまでの式を基底とする。
    """
    max_depth: int
    variables: List[str] = field(default_factory=lambda: ['a', 'b', 'c'])
    
    def __post_init__(self):
        self.expressions: List[SKExpr] = []
        self.expr_to_idx: Dict[str, int] = {}
        self._generate_basis()
    
    def _generate_basis(self):
        """基底となる式を生成"""
        # 原子的な式
        atoms = [S(), K()] + [Var(v) for v in self.variables]
        
        # 原子を登録
        for atom in atoms:
            canonical = to_canonical(atom)
            if canonical not in self.expr_to_idx:
                self.expr_to_idx[canonical] = len(self.expressions)
                self.expressions.append(atom)
        
        # 深さごとにアプリケーションを追加
        for depth in range(1, self.max_depth + 1):
            next_level = []
            for e1 in self.expressions.copy():
                for e2 in atoms:
                    app = App(e1, e2)
                    canonical = to_canonical(app)
                    if canonical not in self.expr_to_idx:
                        next_level.append(app)
            
            # 重複を避けながら追加
            for expr in next_level:
                canonical = to_canonical(expr)
                if canonical not in self.expr_to_idx:
                    self.expr_to_idx[canonical] = len(self.expressions)
                    self.expressions.append(expr)
            
            if len(self.expressions) > 1000:  # 計算量制限
                break
    
    @property
    def dimension(self) -> int:
        return len(self.expressions)
    
    def get_index(self, expr: SKExpr) -> Optional[int]:
        """式のインデックスを取得"""
        canonical = to_canonical(expr)
        return self.expr_to_idx.get(canonical)
    
    def get_expression(self, idx: int) -> SKExpr:
        """インデックスから式を取得"""
        return self.expressions[idx]


class SKOperator(ABC):
    """
    SK演算子の抽象基底クラス
    """
    
    def __init__(self, basis: ExpressionBasis):
        self.basis = basis
        self._matrix: Optional[np.ndarray] = None
    
    @abstractmethod
    def apply(self, expr: SKExpr) -> SKExpr:
        """演算子を式に適用"""
        pass
    
    def get_matrix(self) -> np.ndarray:
        """演算子の行列表現を取得"""
        if self._matrix is None:
            self._build_matrix()
        return self._matrix
    
    def _build_matrix(self):
        """行列表現を構築"""
        n = self.basis.dimension
        self._matrix = np.zeros((n, n), dtype=np.complex128)
        
        for j, expr in enumerate(self.basis.expressions):
            result = self.apply(expr)
            i = self.basis.get_index(result)
            if i is not None:
                self._matrix[i, j] = 1.0
    
    def __matmul__(self, other: 'SKOperator') -> np.ndarray:
        """行列積 self @ other"""
        return self.get_matrix() @ other.get_matrix()
    
    def commutator(self, other: 'SKOperator') -> np.ndarray:
        """交換子 [self, other] = self @ other - other @ self"""
        return self @ other - other @ self


class SOperator(SKOperator):
    """
    S演算子: |E⟩ → |S E⟩
    """
    
    def apply(self, expr: SKExpr) -> SKExpr:
        return App(S(), expr)


class KOperator(SKOperator):
    """
    K演算子: |E⟩ → |K E⟩
    """
    
    def apply(self, expr: SKExpr) -> SKExpr:
        return App(K(), expr)


class AppOperator(SKOperator):
    """
    特定の式を左から適用する演算子: |E⟩ → |F E⟩
    """
    
    def __init__(self, basis: ExpressionBasis, left_expr: SKExpr):
        super().__init__(basis)
        self.left_expr = left_expr
    
    def apply(self, expr: SKExpr) -> SKExpr:
        return App(self.left_expr, expr)


class ReductionOperator(SKOperator):
    """
    簡約演算子: |E⟩ → |reduce(E)⟩
    
    1ステップの簡約を行う（複数のRedexがある場合は最左最外を選択）
    """
    
    def __init__(self, basis: ExpressionBasis):
        super().__init__(basis)
        # 簡約関数をインポート
        from reduction import reduce_leftmost
        self._reduce = reduce_leftmost
    
    def apply(self, expr: SKExpr) -> SKExpr:
        result, _ = self._reduce(expr)
        return result if result else expr


@dataclass
class SKAlgebra:
    """
    SK演算子の代数
    
    S, K, および合成演算子の代数構造を解析する。
    """
    basis: ExpressionBasis
    S_op: SOperator = field(init=False)
    K_op: KOperator = field(init=False)
    
    def __post_init__(self):
        self.S_op = SOperator(self.basis)
        self.K_op = KOperator(self.basis)
    
    def commutator_SK(self) -> np.ndarray:
        """[Ŝ, K̂] を計算"""
        return self.S_op.commutator(self.K_op)
    
    def analyze_commutator(self) -> Dict:
        """交換子の解析"""
        C = self.commutator_SK()
        
        # フロベニウスノルム
        frobenius_norm = np.linalg.norm(C, 'fro')
        
        # スペクトルノルム（最大特異値）
        spectral_norm = np.linalg.norm(C, 2)
        
        # 非零要素の割合
        nonzero_ratio = np.count_nonzero(np.abs(C) > 1e-10) / C.size
        
        # 固有値
        eigenvalues = np.linalg.eigvals(C)
        
        # トレース（対角和）
        trace = np.trace(C)
        
        return {
            'commutator': C,
            'is_nonzero': frobenius_norm > 1e-10,
            'frobenius_norm': frobenius_norm,
            'spectral_norm': spectral_norm,
            'nonzero_ratio': nonzero_ratio,
            'eigenvalues': eigenvalues,
            'trace': trace,
            'dimension': self.basis.dimension,
        }
    
    def analyze_algebra_structure(self) -> Dict:
        """代数構造の解析"""
        S = self.S_op.get_matrix()
        K = self.K_op.get_matrix()
        
        # 各演算子の性質
        def analyze_operator(M, name):
            eigenvalues = np.linalg.eigvals(M)
            return {
                'name': name,
                'is_nilpotent': np.allclose(np.linalg.matrix_power(M, M.shape[0]), 0),
                'is_idempotent': np.allclose(M @ M, M),
                'rank': np.linalg.matrix_rank(M),
                'trace': np.trace(M),
                'eigenvalues': eigenvalues,
                'spectral_radius': np.max(np.abs(eigenvalues)),
            }
        
        S_props = analyze_operator(S, 'S')
        K_props = analyze_operator(K, 'K')
        
        # SK と KS
        SK = S @ K
        KS = K @ S
        
        return {
            'S': S_props,
            'K': K_props,
            'SK_equals_KS': np.allclose(SK, KS),
            'commutator_analysis': self.analyze_commutator(),
        }


def create_sk_algebra(max_depth: int = 2, 
                      variables: List[str] = None) -> SKAlgebra:
    """
    SK代数を構築
    
    Args:
        max_depth: 式の最大深さ
        variables: 使用する変数
    
    Returns:
        SKAlgebra インスタンス
    """
    if variables is None:
        variables = ['a', 'b', 'c']
    
    basis = ExpressionBasis(max_depth=max_depth, variables=variables)
    return SKAlgebra(basis=basis)


if __name__ == '__main__':
    print("=== SK Operator Algebra Analysis ===\n")
    
    # 代数を構築
    algebra = create_sk_algebra(max_depth=2)
    
    print(f"Basis dimension: {algebra.basis.dimension}")
    print(f"Sample expressions: {[to_string(e) for e in algebra.basis.expressions[:10]]}")
    
    # 交換子解析
    comm_analysis = algebra.analyze_commutator()
    print(f"\n[Ŝ, K̂] Analysis:")
    print(f"  Is non-zero: {comm_analysis['is_nonzero']}")
    print(f"  Frobenius norm: {comm_analysis['frobenius_norm']:.4f}")
    print(f"  Spectral norm: {comm_analysis['spectral_norm']:.4f}")
    print(f"  Non-zero ratio: {comm_analysis['nonzero_ratio']:.4f}")
    print(f"  Trace: {comm_analysis['trace']:.4f}")
    
    # 代数構造
    struct = algebra.analyze_algebra_structure()
    print(f"\nAlgebra structure:")
    print(f"  SK = KS: {struct['SK_equals_KS']}")

