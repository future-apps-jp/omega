"""
SK Reduction Operators Algebra
==============================

Phase 1A: 書き換え演算子の代数構造

目的:
    SK計算の書き換え操作を演算子として捉え、
    その代数的閉包が複素数体を含むかどうかを検証する。

演算子の定義:
    - Ŝ: S-reduction を適用する演算子
    - K̂: K-reduction を適用する演算子
    - Î: 恒等演算子
    - 合成: Ŝ∘K̂ など

検証項目:
    1. 演算子代数の生成元と関係式
    2. J² = -I を満たす J の探索
    3. Clifford代数への埋め込み可能性
"""

from __future__ import annotations
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'phase0'))

from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional, Callable, Union
from abc import ABC, abstractmethod
from enum import Enum, auto
import numpy as np
from itertools import product

from sk_parser import SKExpr, S, K, Var, App, parse, to_string, to_canonical
from reduction import (
    find_redexes, reduce_at_path, is_normal_form,
    reduce_s_redex, reduce_k_redex, is_s_redex, is_k_redex,
    RedexType, Redex
)
from multiway import MultiwayGraph, build_multiway_graph


# =============================================================================
# Abstract Operator
# =============================================================================

class Operator(ABC):
    """
    SK式上の演算子の抽象基底クラス
    """
    
    @abstractmethod
    def apply(self, expr: SKExpr) -> Optional[SKExpr]:
        """
        演算子を式に適用
        
        Returns:
            適用結果の式、または None（適用不可の場合）
        """
        pass
    
    @abstractmethod
    def __repr__(self) -> str:
        pass
    
    def __call__(self, expr: SKExpr) -> Optional[SKExpr]:
        return self.apply(expr)


# =============================================================================
# Basic Operators
# =============================================================================

class IdentityOp(Operator):
    """恒等演算子 Î"""
    
    def apply(self, expr: SKExpr) -> Optional[SKExpr]:
        return expr
    
    def __repr__(self) -> str:
        return "Î"


class SReductionOp(Operator):
    """
    S-reduction 演算子 Ŝ
    
    最外のS-redexを簡約（存在しない場合は None）
    """
    
    def apply(self, expr: SKExpr) -> Optional[SKExpr]:
        if is_s_redex(expr):
            return reduce_s_redex(expr)
        
        # 再帰的に S-redex を探す
        redexes = find_redexes(expr)
        s_redexes = [r for r in redexes if r.type == RedexType.S_REDEX]
        
        if s_redexes:
            # 最外（パスが最短）の S-redex を簡約
            outermost = min(s_redexes, key=lambda r: len(r.path))
            return reduce_at_path(expr, outermost.path)
        
        return None
    
    def __repr__(self) -> str:
        return "Ŝ"


class KReductionOp(Operator):
    """
    K-reduction 演算子 K̂
    
    最外のK-redexを簡約（存在しない場合は None）
    """
    
    def apply(self, expr: SKExpr) -> Optional[SKExpr]:
        if is_k_redex(expr):
            return reduce_k_redex(expr)
        
        # 再帰的に K-redex を探す
        redexes = find_redexes(expr)
        k_redexes = [r for r in redexes if r.type == RedexType.K_REDEX]
        
        if k_redexes:
            # 最外（パスが最短）の K-redex を簡約
            outermost = min(k_redexes, key=lambda r: len(r.path))
            return reduce_at_path(expr, outermost.path)
        
        return None
    
    def __repr__(self) -> str:
        return "K̂"


class PathReductionOp(Operator):
    """
    特定パスでの簡約演算子
    
    指定されたパスのRedexを簡約
    """
    
    def __init__(self, path: str, redex_type: RedexType = None):
        self.path = path
        self.redex_type = redex_type
    
    def apply(self, expr: SKExpr) -> Optional[SKExpr]:
        try:
            return reduce_at_path(expr, self.path)
        except:
            return None
    
    def __repr__(self) -> str:
        type_str = "S" if self.redex_type == RedexType.S_REDEX else "K" if self.redex_type == RedexType.K_REDEX else "?"
        return f"R̂({type_str}@{self.path or 'root'})"


# =============================================================================
# Composite Operators
# =============================================================================

class CompositeOp(Operator):
    """
    合成演算子: op1 ∘ op2 （op2 を先に適用）
    """
    
    def __init__(self, *operators: Operator):
        self.operators = list(operators)
    
    def apply(self, expr: SKExpr) -> Optional[SKExpr]:
        result = expr
        # 右から左へ適用（数学的な合成の慣習）
        for op in reversed(self.operators):
            if result is None:
                return None
            result = op.apply(result)
        return result
    
    def __repr__(self) -> str:
        return " ∘ ".join(repr(op) for op in self.operators)


class SumOp(Operator):
    """
    和演算子: op1 + op2
    
    両方の結果を持つ（量子的重ね合わせのモデル）
    ここでは単純に「いずれかが適用可能なら適用」とする
    """
    
    def __init__(self, *operators: Operator, coefficients: List[complex] = None):
        self.operators = list(operators)
        self.coefficients = coefficients or [1.0] * len(operators)
    
    def apply(self, expr: SKExpr) -> Optional[SKExpr]:
        # 最初に適用可能な演算子を適用（古典的近似）
        for op in self.operators:
            result = op.apply(expr)
            if result is not None:
                return result
        return None
    
    def apply_all(self, expr: SKExpr) -> List[Tuple[complex, SKExpr]]:
        """全ての演算子を適用し、係数付きで結果を返す"""
        results = []
        for coef, op in zip(self.coefficients, self.operators):
            result = op.apply(expr)
            if result is not None:
                results.append((coef, result))
        return results
    
    def __repr__(self) -> str:
        terms = []
        for coef, op in zip(self.coefficients, self.operators):
            if coef == 1:
                terms.append(repr(op))
            else:
                terms.append(f"{coef}·{repr(op)}")
        return " + ".join(terms)


# =============================================================================
# Operator Algebra
# =============================================================================

class OperatorAlgebra:
    """
    SK書き換え演算子の代数
    
    生成元: {Î, Ŝ, K̂}
    演算: 合成 (∘), 和 (+), スカラー倍
    
    検証項目:
    1. 関係式の導出
    2. J² = -I を満たす J の探索
    """
    
    def __init__(self):
        self.I = IdentityOp()
        self.S = SReductionOp()
        self.K = KReductionOp()
        
        # 生成元
        self.generators = {'I': self.I, 'S': self.S, 'K': self.K}
        
        # 合成表のキャッシュ
        self._composition_table: Dict[Tuple[str, str], str] = {}
    
    def compose(self, *ops: Operator) -> CompositeOp:
        """演算子を合成"""
        return CompositeOp(*ops)
    
    def sum(self, *ops: Operator, coefficients: List[complex] = None) -> SumOp:
        """演算子の和"""
        return SumOp(*ops, coefficients=coefficients)
    
    def test_relation(self, expr: SKExpr, op1: Operator, op2: Operator) -> bool:
        """
        2つの演算子が同じ結果を与えるか検証
        """
        result1 = op1.apply(expr)
        result2 = op2.apply(expr)
        
        if result1 is None and result2 is None:
            return True
        if result1 is None or result2 is None:
            return False
        
        return to_canonical(result1) == to_canonical(result2)
    
    def find_relations(self, test_exprs: List[SKExpr], max_depth: int = 2) -> List[str]:
        """
        演算子間の関係式を探索
        
        Args:
            test_exprs: テスト用のSK式
            max_depth: 合成の最大深さ
        
        Returns:
            発見された関係式のリスト
        """
        relations = []
        
        # 深さ1の演算子
        ops_d1 = [('I', self.I), ('S', self.S), ('K', self.K)]
        
        # 深さ2の演算子を生成
        ops_d2 = []
        for (n1, o1), (n2, o2) in product(ops_d1, ops_d1):
            name = f"{n1}∘{n2}"
            op = self.compose(o1, o2)
            ops_d2.append((name, op))
        
        all_ops = ops_d1 + ops_d2
        
        # 各ペアで関係を検証
        for i, (name1, op1) in enumerate(all_ops):
            for name2, op2 in all_ops[i+1:]:
                # 全テスト式で同じ結果を与えるか
                all_equal = True
                for expr in test_exprs:
                    if not self.test_relation(expr, op1, op2):
                        all_equal = False
                        break
                
                if all_equal:
                    relations.append(f"{name1} = {name2}")
        
        return relations
    
    def search_imaginary_unit(self, test_exprs: List[SKExpr], 
                               max_terms: int = 4) -> Optional[Operator]:
        """
        J² = -I を満たす演算子 J を探索
        
        これが見つかれば、虚数単位 i の代数的構造が存在する。
        
        Args:
            test_exprs: テスト用のSK式
            max_terms: 和演算子の最大項数
        
        Returns:
            J² = -I を満たす J（見つからなければ None）
        """
        # 基本演算子の組み合わせを試す
        base_ops = [self.I, self.S, self.K]
        
        # 合成演算子も含める
        composite_ops = [
            self.compose(self.S, self.K),
            self.compose(self.K, self.S),
            self.compose(self.S, self.S),
            self.compose(self.K, self.K),
        ]
        
        all_ops = base_ops + composite_ops
        
        # J² = -I の検証
        # 注: SK計算では -I は直接定義できない
        # 代わりに、J² の結果が特定のパターンを持つか検証
        
        for op in all_ops:
            J_squared = self.compose(op, op)
            
            # テスト式で J² の振る舞いを調べる
            patterns = []
            for expr in test_exprs:
                result = J_squared.apply(expr)
                if result is not None:
                    patterns.append((to_canonical(expr), to_canonical(result)))
            
            # パターンを分析
            # J² = I なら周期2の演算子
            # J² = -I は実数演算子では実現できないが、
            # 複素係数の和演算子で実現できる可能性がある
        
        return None  # 現段階では見つからない


# =============================================================================
# Matrix Representation
# =============================================================================

class MatrixRepresentation:
    """
    演算子の行列表現
    
    SK式の有限集合上での演算子の作用を行列で表現し、
    代数的性質を数値的に解析する。
    """
    
    def __init__(self, basis_exprs: List[SKExpr]):
        """
        Args:
            basis_exprs: 基底となるSK式のリスト
        """
        self.basis = basis_exprs
        self.dim = len(basis_exprs)
        
        # 基底のインデックス
        self.expr_to_idx = {to_canonical(e): i for i, e in enumerate(basis_exprs)}
    
    def operator_matrix(self, op: Operator) -> np.ndarray:
        """
        演算子を行列で表現
        
        M[i,j] = 1 if op(basis[j]) = basis[i], else 0
        """
        matrix = np.zeros((self.dim, self.dim), dtype=complex)
        
        for j, expr in enumerate(self.basis):
            result = op.apply(expr)
            if result is not None:
                canonical = to_canonical(result)
                if canonical in self.expr_to_idx:
                    i = self.expr_to_idx[canonical]
                    matrix[i, j] = 1.0
        
        return matrix
    
    def find_imaginary_structure(self) -> Dict:
        """
        虚数構造を探索
        
        行列表現で J² = -I を満たす J を探す
        """
        algebra = OperatorAlgebra()
        
        # 基本演算子の行列
        I_mat = self.operator_matrix(algebra.I)
        S_mat = self.operator_matrix(algebra.S)
        K_mat = self.operator_matrix(algebra.K)
        
        results = {
            'I': I_mat,
            'S': S_mat,
            'K': K_mat,
            'S²': S_mat @ S_mat,
            'K²': K_mat @ K_mat,
            'SK': S_mat @ K_mat,
            'KS': K_mat @ S_mat,
            '(SK)²': (S_mat @ K_mat) @ (S_mat @ K_mat),
            '(KS)²': (K_mat @ S_mat) @ (K_mat @ S_mat),
        }
        
        # 各行列の固有値を計算
        eigenvalues = {}
        for name, mat in results.items():
            try:
                eigs = np.linalg.eigvals(mat)
                eigenvalues[name] = eigs
            except:
                eigenvalues[name] = None
        
        results['eigenvalues'] = eigenvalues
        
        # J² = -I となる J を探索
        # J² の固有値が全て -1 なら、J は虚数単位的
        candidates = []
        for name, eigs in eigenvalues.items():
            if eigs is not None and len(eigs) > 0:
                # 固有値が ±i に近いか検証
                for eig in eigs:
                    if abs(abs(eig) - 1) < 0.01 and abs(eig.real) < 0.01:
                        candidates.append((name, eig))
        
        results['imaginary_candidates'] = candidates
        
        return results


# =============================================================================
# Clifford Algebra Analysis
# =============================================================================

class CliffordAnalysis:
    """
    Clifford代数との関連を分析
    
    Clifford代数 Cl(p,q) の生成元 γᵢ は：
    - γᵢ² = +1 (p個) または γᵢ² = -1 (q個)
    - γᵢγⱼ + γⱼγᵢ = 0 (i ≠ j)
    
    SK演算子がこのような関係を満たすか検証
    """
    
    def __init__(self, matrix_rep: MatrixRepresentation):
        self.matrix_rep = matrix_rep
    
    def anticommutator(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """反交換子 {A, B} = AB + BA"""
        return A @ B + B @ A
    
    def commutator(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """交換子 [A, B] = AB - BA"""
        return A @ B - B @ A
    
    def check_clifford_relations(self) -> Dict:
        """
        Clifford代数的関係を検証
        """
        algebra = OperatorAlgebra()
        
        S_mat = self.matrix_rep.operator_matrix(algebra.S)
        K_mat = self.matrix_rep.operator_matrix(algebra.K)
        I_mat = self.matrix_rep.operator_matrix(algebra.I)
        
        results = {}
        
        # S² の検証
        S_squared = S_mat @ S_mat
        results['S²'] = S_squared
        results['S² = I?'] = np.allclose(S_squared, I_mat)
        results['S² = -I?'] = np.allclose(S_squared, -I_mat)
        
        # K² の検証
        K_squared = K_mat @ K_mat
        results['K²'] = K_squared
        results['K² = I?'] = np.allclose(K_squared, I_mat)
        results['K² = -I?'] = np.allclose(K_squared, -I_mat)
        
        # 反交換子 {S, K}
        anticomm = self.anticommutator(S_mat, K_mat)
        results['{S, K}'] = anticomm
        results['{S, K} = 0?'] = np.allclose(anticomm, np.zeros_like(anticomm))
        
        # 交換子 [S, K]
        comm = self.commutator(S_mat, K_mat)
        results['[S, K]'] = comm
        results['[S, K] = 0?'] = np.allclose(comm, np.zeros_like(comm))
        
        # Clifford代数的構造のまとめ
        is_clifford_like = (
            (results['S² = I?'] or results['S² = -I?']) and
            (results['K² = I?'] or results['K² = -I?']) and
            results['{S, K} = 0?']
        )
        results['is_clifford_like'] = is_clifford_like
        
        return results


# =============================================================================
# Main Analysis
# =============================================================================

def run_algebraic_analysis(verbose: bool = True) -> Dict:
    """
    代数的構造の完全な解析を実行
    """
    results = {}
    
    if verbose:
        print("=" * 70)
        print("Phase 1A: 代数的構造の解析")
        print("=" * 70)
    
    # テスト用のSK式を生成
    test_expressions = [
        parse("S a b c"),
        parse("K a b"),
        parse("S K K a"),
        parse("S (K a) (K b) c"),
        parse("(K a b) (K c d)"),
        parse("S S K a b c"),
    ]
    
    if verbose:
        print(f"\nテスト式: {len(test_expressions)} 個")
        for expr in test_expressions:
            print(f"  {to_string(expr)}")
    
    # 演算子代数の構築
    algebra = OperatorAlgebra()
    
    # 関係式の探索
    if verbose:
        print("\n" + "-" * 70)
        print("1. 演算子間の関係式")
        print("-" * 70)
    
    relations = algebra.find_relations(test_expressions)
    results['relations'] = relations
    
    if verbose:
        if relations:
            for rel in relations:
                print(f"  {rel}")
        else:
            print("  自明でない関係式は見つかりませんでした")
    
    # 基底の構築（正規形まで簡約した式の集合）
    if verbose:
        print("\n" + "-" * 70)
        print("2. 行列表現の構築")
        print("-" * 70)
    
    # 小さな基底を使用
    basis_exprs = [
        parse("S"),
        parse("K"),
        parse("a"),
        parse("S K"),
        parse("K S"),
        parse("S a"),
        parse("K a"),
    ]
    
    matrix_rep = MatrixRepresentation(basis_exprs)
    
    if verbose:
        print(f"  基底の次元: {matrix_rep.dim}")
    
    # 虚数構造の探索
    if verbose:
        print("\n" + "-" * 70)
        print("3. 虚数構造 (J² = -I) の探索")
        print("-" * 70)
    
    imaginary_results = matrix_rep.find_imaginary_structure()
    results['imaginary_analysis'] = imaginary_results
    
    if verbose:
        print("\n  演算子行列の固有値:")
        for name, eigs in imaginary_results['eigenvalues'].items():
            if eigs is not None and len(eigs) > 0:
                eig_str = ", ".join(f"{e:.3f}" for e in eigs[:5])
                if len(eigs) > 5:
                    eig_str += "..."
                print(f"    {name}: [{eig_str}]")
        
        if imaginary_results['imaginary_candidates']:
            print("\n  虚数単位の候補 (固有値 ≈ ±i):")
            for name, eig in imaginary_results['imaginary_candidates']:
                print(f"    {name}: eigenvalue = {eig}")
        else:
            print("\n  虚数単位の候補: なし")
    
    # Clifford代数との関連
    if verbose:
        print("\n" + "-" * 70)
        print("4. Clifford代数的構造の検証")
        print("-" * 70)
    
    clifford = CliffordAnalysis(matrix_rep)
    clifford_results = clifford.check_clifford_relations()
    results['clifford_analysis'] = clifford_results
    
    if verbose:
        print(f"  S² = I? : {clifford_results['S² = I?']}")
        print(f"  S² = -I? : {clifford_results['S² = -I?']}")
        print(f"  K² = I? : {clifford_results['K² = I?']}")
        print(f"  K² = -I? : {clifford_results['K² = -I?']}")
        print(f"  {{S, K}} = 0? : {clifford_results['{S, K} = 0?']}")
        print(f"  [S, K] = 0? : {clifford_results['[S, K] = 0?']}")
        print(f"\n  Clifford代数的構造: {clifford_results['is_clifford_like']}")
    
    # 結論
    if verbose:
        print("\n" + "=" * 70)
        print("結論")
        print("=" * 70)
        
        has_imaginary = len(imaginary_results['imaginary_candidates']) > 0
        is_clifford = clifford_results['is_clifford_like']
        
        if has_imaginary or is_clifford:
            print("\n  🔔 複素構造の候補が見つかりました！")
            print("     さらなる検証が必要です。")
        else:
            print("\n  ✓ 現在の解析では複素構造は見つかりませんでした。")
            print("     より大きな基底での検証、または")
            print("     アプローチB（幾何学的構造）への移行を検討してください。")
    
    results['has_imaginary_structure'] = len(imaginary_results['imaginary_candidates']) > 0
    results['is_clifford_like'] = clifford_results['is_clifford_like']
    
    return results


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    results = run_algebraic_analysis(verbose=True)




