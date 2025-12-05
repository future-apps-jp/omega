"""
Commutator Analysis for SK Operators

交換子 [Ŝ, K̂] の詳細解析

理論的背景:
    量子力学では、非可換な観測量 [X̂, Ŷ] ≠ 0 が不確定性関係と
    干渉の源泉となる。例えば [x̂, p̂] = iℏ。
    
    Phase 6では、連続時間化で干渉が生じることを確認した。
    しかし、真の量子性（重ね合わせ）は量子回路のみが持つ。
    
    本解析では、[Ŝ, K̂] の構造を調べ、
    非可換性と重ね合わせ生成の関係を明らかにする。
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'phase0'))
from sk_parser import SKExpr, S, K, Var, App, parse, to_string

from .operators import SKAlgebra, ExpressionBasis, create_sk_algebra


@dataclass
class CommutatorAnalysis:
    """
    交換子の詳細解析
    """
    algebra: SKAlgebra
    
    def compute_commutator(self) -> np.ndarray:
        """[Ŝ, K̂] を計算"""
        return self.algebra.commutator_SK()
    
    def analyze_structure(self) -> Dict:
        """交換子の構造解析"""
        C = self.compute_commutator()
        n = C.shape[0]
        
        # 基本統計
        frobenius_norm = np.linalg.norm(C, 'fro')
        spectral_norm = np.linalg.norm(C, 2)
        nuclear_norm = np.linalg.norm(C, 'nuc')  # trace norm
        
        # スペクトル解析
        eigenvalues = np.linalg.eigvals(C)
        real_eigenvalues = eigenvalues[np.abs(eigenvalues.imag) < 1e-10].real
        complex_eigenvalues = eigenvalues[np.abs(eigenvalues.imag) >= 1e-10]
        
        # ランクと次元解析
        rank = np.linalg.matrix_rank(C)
        
        # トレース構造
        trace = np.trace(C)
        
        # 非零要素の解析
        nonzero_mask = np.abs(C) > 1e-10
        nonzero_count = np.count_nonzero(nonzero_mask)
        nonzero_positions = np.argwhere(nonzero_mask)
        
        return {
            'dimension': n,
            'is_nonzero': frobenius_norm > 1e-10,
            'frobenius_norm': frobenius_norm,
            'spectral_norm': spectral_norm,
            'nuclear_norm': nuclear_norm,
            'rank': rank,
            'trace': trace,
            'eigenvalues': eigenvalues,
            'num_real_eigenvalues': len(real_eigenvalues),
            'num_complex_eigenvalues': len(complex_eigenvalues),
            'nonzero_count': nonzero_count,
            'nonzero_ratio': nonzero_count / (n * n),
            'nonzero_positions': nonzero_positions,
        }
    
    def analyze_lie_structure(self) -> Dict:
        """
        リー代数的構造の解析
        
        [A, B] がリー括弧として振る舞うか検証
        """
        S = self.algebra.S_op.get_matrix()
        K = self.algebra.K_op.get_matrix()
        C = self.compute_commutator()  # [S, K]
        
        # 反対称性: [A, B] = -[B, A]
        C_KS = K @ S - S @ K  # [K, S]
        antisymmetric = np.allclose(C, -C_KS)
        
        # ヤコビ恒等式: [A, [B, C]] + [B, [C, A]] + [C, [A, B]] = 0
        # [S, [K, C]] + [K, [C, S]] + [C, [S, K]] = 0
        # C = [S, K] なので
        # [S, [K, C]] + [K, [C, S]] + [C, C] = 0
        KC = K @ C - C @ K  # [K, C]
        CS = C @ S - S @ C  # [C, S]
        SKC = S @ KC - KC @ S  # [S, [K, C]]
        KCS = K @ CS - CS @ K  # [K, [C, S]]
        CC = C @ C - C @ C  # [C, C] = 0
        
        jacobi_sum = SKC + KCS + CC
        jacobi_identity = np.allclose(jacobi_sum, 0)
        jacobi_error = np.linalg.norm(jacobi_sum, 'fro')
        
        return {
            'antisymmetric': antisymmetric,
            'jacobi_identity': jacobi_identity,
            'jacobi_error': jacobi_error,
            'is_lie_bracket': antisymmetric and jacobi_identity,
        }
    
    def analyze_uncertainty_relation(self) -> Dict:
        """
        不確定性関係的構造の解析
        
        量子力学では、[X̂, Ŷ] = iℏ Ẑ のような関係が不確定性を生む。
        SK代数で類似の構造があるか検証。
        """
        S = self.algebra.S_op.get_matrix()
        K = self.algebra.K_op.get_matrix()
        C = self.compute_commutator()
        
        # C が純虚数スカラー倍の単位行列に近いか
        # [X, P] = iℏI の形
        n = C.shape[0]
        I = np.eye(n)
        
        # C = αI + R (αは複素数、Rは残差)
        alpha = np.trace(C) / n
        residual = C - alpha * I
        residual_norm = np.linalg.norm(residual, 'fro')
        
        # Cの固有値が純虚数か
        eigenvalues = np.linalg.eigvals(C)
        pure_imaginary = np.allclose(eigenvalues.real, 0)
        
        # Robertson-Schrödinger 不確定性
        # Δ(S)Δ(K) ≥ |⟨[S,K]⟩|/2
        # 状態ベクトルを仮定して計算
        
        return {
            'trace_per_dim': alpha,
            'residual_norm': residual_norm,
            'is_scalar_multiple': residual_norm < 1e-5,
            'eigenvalues_pure_imaginary': pure_imaginary,
            'has_uncertainty_structure': abs(alpha.imag) > 1e-10,
        }
    
    def interpret_noncommutativity(self) -> Dict:
        """
        非可換性の物理的・計算論的解釈
        """
        analysis = self.analyze_structure()
        lie = self.analyze_lie_structure()
        uncertainty = self.analyze_uncertainty_relation()
        
        # 解釈
        interpretations = []
        
        if analysis['is_nonzero']:
            interpretations.append(
                "SK演算子は非可換: [Ŝ, K̂] ≠ 0"
            )
            
            if lie['is_lie_bracket']:
                interpretations.append(
                    "交換子はリー括弧の条件を満たす（リー代数構造あり）"
                )
            
            if uncertainty['has_uncertainty_structure']:
                interpretations.append(
                    "不確定性関係的構造が存在する可能性"
                )
            
            if analysis['num_complex_eigenvalues'] > 0:
                interpretations.append(
                    f"交換子は{analysis['num_complex_eigenvalues']}個の複素固有値を持つ"
                )
        else:
            interpretations.append(
                "SK演算子は可換: [Ŝ, K̂] = 0（この基底での近似において）"
            )
        
        # 重ね合わせとの関係
        superposition_relation = self._analyze_superposition_connection(analysis)
        
        return {
            'analysis': analysis,
            'lie_structure': lie,
            'uncertainty': uncertainty,
            'interpretations': interpretations,
            'superposition_relation': superposition_relation,
        }
    
    def _analyze_superposition_connection(self, analysis: Dict) -> Dict:
        """
        非可換性と重ね合わせの関係を分析
        """
        # 量子力学では、非可換性は:
        # 1. 同時固有状態の非存在 → 重ね合わせが必要
        # 2. 不確定性関係 → 確率的振る舞い
        
        C = self.compute_commutator()
        
        # 交換子の固有空間解析
        eigenvalues, eigenvectors = np.linalg.eig(C)
        
        # 零固有値の重複度（同時固有状態の数に関連）
        zero_eigenvalue_count = np.sum(np.abs(eigenvalues) < 1e-10)
        
        # 非零固有値の存在は、同時対角化不可能を示唆
        nonzero_eigenvalue_count = np.sum(np.abs(eigenvalues) >= 1e-10)
        
        conclusion = []
        
        if nonzero_eigenvalue_count > 0:
            conclusion.append(
                "S と K は同時対角化不可能（共通固有状態が制限される）"
            )
            conclusion.append(
                "これは量子力学の非可換観測量と類似の構造"
            )
        else:
            conclusion.append(
                "S と K は同時対角化可能（この近似において）"
            )
        
        # しかし、重要な違い
        conclusion.append(
            "注意: SK計算は離散的であり、連続的重ね合わせを自然には生成しない"
        )
        
        return {
            'zero_eigenvalue_count': zero_eigenvalue_count,
            'nonzero_eigenvalue_count': nonzero_eigenvalue_count,
            'simultaneously_diagonalizable': nonzero_eigenvalue_count == 0,
            'conclusions': conclusion,
        }


def run_commutator_analysis(max_depth: int = 3) -> Dict:
    """
    交換子解析を実行
    
    Args:
        max_depth: 式空間の最大深さ
    
    Returns:
        解析結果の辞書
    """
    algebra = create_sk_algebra(max_depth=max_depth)
    analyzer = CommutatorAnalysis(algebra)
    
    return analyzer.interpret_noncommutativity()


if __name__ == '__main__':
    print("=== Commutator [Ŝ, K̂] Analysis ===\n")
    
    results = run_commutator_analysis(max_depth=2)
    
    print("Structure Analysis:")
    analysis = results['analysis']
    print(f"  Dimension: {analysis['dimension']}")
    print(f"  Is non-zero: {analysis['is_nonzero']}")
    print(f"  Frobenius norm: {analysis['frobenius_norm']:.6f}")
    print(f"  Rank: {analysis['rank']}")
    print(f"  Trace: {analysis['trace']}")
    
    print(f"\nLie Structure:")
    lie = results['lie_structure']
    print(f"  Antisymmetric: {lie['antisymmetric']}")
    print(f"  Jacobi identity: {lie['jacobi_identity']}")
    print(f"  Is Lie bracket: {lie['is_lie_bracket']}")
    
    print(f"\nInterpretations:")
    for interp in results['interpretations']:
        print(f"  - {interp}")
    
    print(f"\nSuperposition Relation:")
    sup = results['superposition_relation']
    for conc in sup['conclusions']:
        print(f"  - {conc}")

