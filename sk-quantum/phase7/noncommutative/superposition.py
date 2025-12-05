"""
Superposition Analysis: Can Non-commutativity Generate Superposition?

重ね合わせ生成の条件を解析

Phase 6の発見:
    - 連続時間化で干渉は生じる
    - しかし、真の重ね合わせは量子回路のみ
    
Phase 7の問い:
    - 非可換性 [Ŝ, K̂] ≠ 0 は重ね合わせを生成するか？
    - SK計算から量子状態は導出可能か？
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
class SuperpositionAnalysis:
    """
    SK代数における重ね合わせの可能性を解析
    """
    algebra: SKAlgebra
    
    def analyze_basis_state_evolution(self) -> Dict:
        """
        基底状態の発展を解析
        
        古典的計算: |E⟩ → |E'⟩ (確定的)
        量子計算: |E⟩ → α|E₁⟩ + β|E₂⟩ (重ね合わせ)
        """
        S_matrix = self.algebra.S_op.get_matrix()
        K_matrix = self.algebra.K_op.get_matrix()
        
        n = self.algebra.basis.dimension
        results = []
        
        # 各基底状態について
        for i in range(min(n, 10)):  # 最初の10状態
            basis_state = np.zeros(n)
            basis_state[i] = 1.0
            
            # S演算の結果
            s_result = S_matrix @ basis_state
            s_nonzero = np.sum(np.abs(s_result) > 1e-10)
            
            # K演算の結果
            k_result = K_matrix @ basis_state
            k_nonzero = np.sum(np.abs(k_result) > 1e-10)
            
            results.append({
                'basis_index': i,
                'expression': to_string(self.algebra.basis.expressions[i]),
                'S_output_states': s_nonzero,
                'K_output_states': k_nonzero,
                'S_creates_superposition': s_nonzero > 1,
                'K_creates_superposition': k_nonzero > 1,
            })
        
        # 集計
        s_superposition_count = sum(1 for r in results if r['S_creates_superposition'])
        k_superposition_count = sum(1 for r in results if r['K_creates_superposition'])
        
        return {
            'individual_results': results,
            'S_superposition_ratio': s_superposition_count / len(results),
            'K_superposition_ratio': k_superposition_count / len(results),
            'overall_conclusion': self._interpret_superposition(results),
        }
    
    def _interpret_superposition(self, results: List[Dict]) -> str:
        """重ね合わせの解釈"""
        s_sup = any(r['S_creates_superposition'] for r in results)
        k_sup = any(r['K_creates_superposition'] for r in results)
        
        if s_sup or k_sup:
            return "SK演算子は形式的には複数の出力状態を生成する可能性があるが、これは多価関数的であり、量子的重ね合わせとは本質的に異なる"
        else:
            return "SK演算子は各基底状態を一意の状態に写像する（重ね合わせを生成しない）"
    
    def analyze_coherence_structure(self) -> Dict:
        """
        コヒーレンス構造の解析
        
        量子的重ね合わせには位相コヒーレンスが必要:
        |ψ⟩ = α|0⟩ + βe^{iφ}|1⟩
        
        SK計算では位相が自然に定義されないため、
        コヒーレンスが生じるかを検証
        """
        S_matrix = self.algebra.S_op.get_matrix()
        K_matrix = self.algebra.K_op.get_matrix()
        
        # 行列要素の位相
        S_phases = np.angle(S_matrix[S_matrix != 0])
        K_phases = np.angle(K_matrix[K_matrix != 0])
        
        # 位相の多様性
        S_phase_diversity = np.std(S_phases) if len(S_phases) > 0 else 0
        K_phase_diversity = np.std(K_phases) if len(K_phases) > 0 else 0
        
        # 非対角要素の存在（コヒーレンス項）
        n = S_matrix.shape[0]
        S_off_diag = np.sum(np.abs(S_matrix) > 1e-10) - np.sum(np.abs(np.diag(S_matrix)) > 1e-10)
        K_off_diag = np.sum(np.abs(K_matrix) > 1e-10) - np.sum(np.abs(np.diag(K_matrix)) > 1e-10)
        
        return {
            'S_phase_diversity': S_phase_diversity,
            'K_phase_diversity': K_phase_diversity,
            'S_off_diagonal_elements': S_off_diag,
            'K_off_diagonal_elements': K_off_diag,
            'has_coherence_structure': S_phase_diversity > 1e-10 or K_phase_diversity > 1e-10,
            'interpretation': self._interpret_coherence(S_phase_diversity, K_phase_diversity),
        }
    
    def _interpret_coherence(self, s_div: float, k_div: float) -> str:
        """コヒーレンスの解釈"""
        if s_div < 1e-10 and k_div < 1e-10:
            return "SK演算子は実数のみで構成され、位相コヒーレンスを持たない（古典的）"
        else:
            return "位相構造が検出されたが、これはSK計算の本質ではなく実装の詳細による可能性がある"
    
    def compare_with_quantum_gates(self) -> Dict:
        """
        量子ゲートとの比較
        
        Hadamardゲートなど、重ね合わせを生成するゲートとSK演算子を比較
        """
        S_matrix = self.algebra.S_op.get_matrix()
        K_matrix = self.algebra.K_op.get_matrix()
        
        # Hadamardゲート（2x2）
        H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
        
        # 比較メトリクス
        def gate_properties(M, name):
            # ユニタリ性
            is_unitary = np.allclose(M @ M.conj().T, np.eye(M.shape[0]))
            
            # エルミート性
            is_hermitian = np.allclose(M, M.conj().T)
            
            # 正規性
            is_normal = np.allclose(M @ M.conj().T, M.conj().T @ M)
            
            # 重ね合わせ生成（基底状態からの発散度）
            n = M.shape[0]
            superposition_score = 0
            for i in range(n):
                basis = np.zeros(n)
                basis[i] = 1
                output = M @ basis
                nonzero = np.sum(np.abs(output) > 1e-10)
                if nonzero > 1:
                    superposition_score += 1
            
            return {
                'name': name,
                'is_unitary': is_unitary,
                'is_hermitian': is_hermitian,
                'is_normal': is_normal,
                'superposition_score': superposition_score / n if n > 0 else 0,
            }
        
        s_props = gate_properties(S_matrix, 'S operator')
        k_props = gate_properties(K_matrix, 'K operator')
        h_props = gate_properties(H, 'Hadamard')
        
        return {
            'S': s_props,
            'K': k_props,
            'Hadamard': h_props,
            'key_difference': self._key_difference(s_props, k_props, h_props),
        }
    
    def _key_difference(self, s: Dict, k: Dict, h: Dict) -> List[str]:
        """主要な違いを特定"""
        differences = []
        
        if h['is_unitary'] and not (s['is_unitary'] and k['is_unitary']):
            differences.append(
                "Hadamardはユニタリだが、SK演算子は一般にユニタリではない"
            )
        
        if h['superposition_score'] > max(s['superposition_score'], k['superposition_score']):
            differences.append(
                "Hadamardは全ての基底状態で重ね合わせを生成するが、SK演算子はそうではない"
            )
        
        differences.append(
            "根本的違い: Hadamardは連続的複素振幅を持つが、SK演算子は離散的写像"
        )
        
        return differences
    
    def analyze_superposition_requirements(self) -> Dict:
        """
        重ね合わせ生成の要件分析
        
        なぜSK計算から重ね合わせが生じないか、その本質的理由を分析
        """
        analysis = self.analyze_basis_state_evolution()
        coherence = self.analyze_coherence_structure()
        comparison = self.compare_with_quantum_gates()
        
        requirements = [
            {
                'requirement': '連続的振幅',
                'quantum': '複素数 α, β ∈ ℂ',
                'sk': '離散的写像 (0 or 1)',
                'satisfied': False,
            },
            {
                'requirement': '位相コヒーレンス',
                'quantum': 'e^{iφ} による相対位相',
                'sk': '位相なし',
                'satisfied': coherence['has_coherence_structure'],
            },
            {
                'requirement': 'ユニタリ性',
                'quantum': 'UU† = I',
                'sk': '一般に非ユニタリ',
                'satisfied': comparison['S']['is_unitary'] or comparison['K']['is_unitary'],
            },
            {
                'requirement': '規格化',
                'quantum': '|α|² + |β|² = 1',
                'sk': '確率的解釈なし',
                'satisfied': False,
            },
        ]
        
        satisfied_count = sum(1 for r in requirements if r['satisfied'])
        
        conclusion = []
        conclusion.append(
            f"重ね合わせの要件: {satisfied_count}/{len(requirements)} 満たす"
        )
        
        if satisfied_count < len(requirements):
            conclusion.append(
                "SK計算から量子的重ね合わせは導出できない"
            )
            conclusion.append(
                "非可換性は必要条件かもしれないが、十分条件ではない"
            )
        
        return {
            'requirements': requirements,
            'satisfied_count': satisfied_count,
            'total_requirements': len(requirements),
            'conclusion': conclusion,
            'detailed_analysis': {
                'basis_evolution': analysis,
                'coherence': coherence,
                'gate_comparison': comparison,
            },
        }


def run_superposition_analysis(max_depth: int = 2) -> Dict:
    """
    重ね合わせ解析を実行
    """
    algebra = create_sk_algebra(max_depth=max_depth)
    analyzer = SuperpositionAnalysis(algebra)
    
    return analyzer.analyze_superposition_requirements()


if __name__ == '__main__':
    print("=== Superposition Analysis ===\n")
    
    results = run_superposition_analysis(max_depth=2)
    
    print("Requirements for Superposition:")
    for req in results['requirements']:
        status = "✓" if req['satisfied'] else "✗"
        print(f"  {status} {req['requirement']}")
        print(f"      Quantum: {req['quantum']}")
        print(f"      SK: {req['sk']}")
    
    print(f"\nConclusion:")
    for c in results['conclusion']:
        print(f"  - {c}")

