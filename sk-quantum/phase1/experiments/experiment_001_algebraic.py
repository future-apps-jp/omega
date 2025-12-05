"""
Phase 1A Experiment: 代数的構造からの複素数導出の検証
====================================================

目的:
    SK書き換え演算子の代数構造から、複素数体の要素（特に虚数単位 i）が
    自然に現れるかどうかを検証する。

検証項目:
    1. 閉じた基底（reduction closure）上での行列表現
    2. 演算子の固有値スペクトル
    3. J² = -I を満たす J の探索
    4. Clifford代数 Cl(p,q) への埋め込み可能性
    5. U(1) 群構造の検出

方法論:
    - SK式の有限状態空間上で演算子を行列表現
    - 行列の代数的性質を解析
    - 複素固有値の出現条件を調査
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'phase0'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'algebra'))

import numpy as np
from itertools import product, combinations
from typing import List, Dict, Set, Tuple

from sk_parser import SKExpr, S, K, Var, App, parse, to_string, to_canonical
from reduction import (
    find_redexes, reduce_at_path, is_normal_form, reduce_to_normal_form,
    reduce_leftmost, RedexType
)
from multiway import build_multiway_graph, enumerate_paths
from operators import (
    IdentityOp, SReductionOp, KReductionOp, OperatorAlgebra,
    MatrixRepresentation, CliffordAnalysis
)


# =============================================================================
# Experiment 1: Reduction Closure
# =============================================================================

def compute_reduction_closure(initial_exprs: List[SKExpr], 
                               max_size: int = 50,
                               max_steps: int = 5) -> Set[str]:
    """
    初期式から到達可能な全ての式（正規形まで）を計算
    
    これにより「閉じた」基底を構築できる
    """
    closure = set()
    frontier = set(to_canonical(e) for e in initial_exprs)
    
    for step in range(max_steps):
        new_frontier = set()
        
        for expr_str in frontier:
            if len(closure) >= max_size:
                break
            
            if expr_str in closure:
                continue
            
            closure.add(expr_str)
            
            try:
                expr = parse(expr_str)
                
                # S-reduction を試す
                s_op = SReductionOp()
                s_result = s_op(expr)
                if s_result is not None:
                    new_frontier.add(to_canonical(s_result))
                
                # K-reduction を試す
                k_op = KReductionOp()
                k_result = k_op(expr)
                if k_result is not None:
                    new_frontier.add(to_canonical(k_result))
            except:
                pass
        
        frontier = new_frontier - closure
        
        if not frontier:
            break
    
    return closure


# =============================================================================
# Experiment 2: Extended Matrix Analysis
# =============================================================================

def analyze_operator_spectrum(basis_exprs: List[SKExpr]) -> Dict:
    """
    演算子のスペクトル（固有値）を詳細に解析
    """
    matrix_rep = MatrixRepresentation(basis_exprs)
    algebra = OperatorAlgebra()
    
    results = {}
    
    # 基本演算子の行列
    I_mat = matrix_rep.operator_matrix(algebra.I)
    S_mat = matrix_rep.operator_matrix(algebra.S)
    K_mat = matrix_rep.operator_matrix(algebra.K)
    
    # 合成演算子
    operators = {
        'I': I_mat,
        'S': S_mat,
        'K': K_mat,
        'S+K': S_mat + K_mat,
        'S-K': S_mat - K_mat,
        'iS+K': 1j * S_mat + K_mat,  # 複素係数を許容
        'S+iK': S_mat + 1j * K_mat,
        'SK': S_mat @ K_mat,
        'KS': K_mat @ S_mat,
        '[S,K]': S_mat @ K_mat - K_mat @ S_mat,  # 交換子
        '{S,K}': S_mat @ K_mat + K_mat @ S_mat,  # 反交換子
    }
    
    for name, mat in operators.items():
        # 固有値計算
        try:
            eigenvalues = np.linalg.eigvals(mat)
            
            # 特徴的な固有値をチェック
            has_imaginary = any(abs(eig.imag) > 1e-10 for eig in eigenvalues)
            has_unit_circle = any(abs(abs(eig) - 1) < 1e-10 for eig in eigenvalues)
            has_minus_one = any(abs(eig + 1) < 1e-10 for eig in eigenvalues)
            
            results[name] = {
                'matrix': mat,
                'eigenvalues': eigenvalues,
                'has_imaginary': has_imaginary,
                'has_unit_circle': has_unit_circle,
                'has_minus_one': has_minus_one,
                'trace': np.trace(mat),
                'det': np.linalg.det(mat),
            }
        except:
            results[name] = {'error': 'Computation failed'}
    
    return results


# =============================================================================
# Experiment 3: Search for J² = -I
# =============================================================================

def search_J_squared_minus_I(basis_exprs: List[SKExpr], 
                              search_depth: int = 2) -> List[Dict]:
    """
    J² = -I を満たす演算子 J を系統的に探索
    
    探索空間:
        - 生成元 {I, S, K} の線形結合
        - 複素係数 a + bi を許容
    """
    matrix_rep = MatrixRepresentation(basis_exprs)
    algebra = OperatorAlgebra()
    
    I_mat = matrix_rep.operator_matrix(algebra.I)
    S_mat = matrix_rep.operator_matrix(algebra.S)
    K_mat = matrix_rep.operator_matrix(algebra.K)
    
    candidates = []
    
    # 線形結合 J = a*I + b*S + c*K (a,b,c は複素数) を探索
    # J² = -I となる条件を数値的に検索
    
    coefficient_range = [-1, -0.5, 0, 0.5, 1]
    complex_range = [1, 1j, -1, -1j]
    
    for a_real, a_imag in product(coefficient_range, coefficient_range):
        for b_real, b_imag in product(coefficient_range, coefficient_range):
            for c_real, c_imag in product(coefficient_range, coefficient_range):
                a = a_real + 1j * a_imag
                b = b_real + 1j * b_imag
                c = c_real + 1j * c_imag
                
                # ゼロの組み合わせはスキップ
                if a == 0 and b == 0 and c == 0:
                    continue
                
                J = a * I_mat + b * S_mat + c * K_mat
                J_squared = J @ J
                
                # -I に近いかチェック
                minus_I = -I_mat
                error = np.linalg.norm(J_squared - minus_I, 'fro')
                
                if error < 1e-6:
                    candidates.append({
                        'coefficients': (a, b, c),
                        'error': error,
                        'J': J,
                        'J²': J_squared,
                    })
    
    # より精密な探索（勾配法）
    # 最適化問題: min ||J² + I||² where J = aI + bS + cK
    
    return candidates


# =============================================================================
# Experiment 4: Pauli-like Matrices Search
# =============================================================================

def search_pauli_structure(basis_exprs: List[SKExpr]) -> Dict:
    """
    Pauli行列的構造を探索
    
    Pauli行列:
        σ₁ = [[0,1],[1,0]]
        σ₂ = [[0,-i],[i,0]]
        σ₃ = [[1,0],[0,-1]]
    
    満たす関係:
        σᵢ² = I
        σᵢσⱼ = iεᵢⱼₖσₖ (i≠j)
    """
    matrix_rep = MatrixRepresentation(basis_exprs)
    algebra = OperatorAlgebra()
    
    I_mat = matrix_rep.operator_matrix(algebra.I)
    S_mat = matrix_rep.operator_matrix(algebra.S)
    K_mat = matrix_rep.operator_matrix(algebra.K)
    
    # SK から 2次元部分空間を探す
    dim = S_mat.shape[0]
    
    results = {
        'dimension': dim,
        'S_eigenvalues': np.linalg.eigvals(S_mat),
        'K_eigenvalues': np.linalg.eigvals(K_mat),
        'pauli_candidates': [],
    }
    
    # S, K の非ゼロ行/列のみに注目して 2x2 部分行列を抽出
    non_zero_S = np.where(np.any(S_mat != 0, axis=1))[0]
    non_zero_K = np.where(np.any(K_mat != 0, axis=1))[0]
    
    common = list(set(non_zero_S) & set(non_zero_K))
    
    if len(common) >= 2:
        # 2x2 部分行列を検証
        for i, j in combinations(common, 2):
            idx = [i, j]
            S_sub = S_mat[np.ix_(idx, idx)]
            K_sub = K_mat[np.ix_(idx, idx)]
            
            # Pauli的関係をチェック
            S_squared = S_sub @ S_sub
            K_squared = K_sub @ K_sub
            SK = S_sub @ K_sub
            KS = K_sub @ S_sub
            
            is_s_squared_i = np.allclose(S_squared, np.eye(2))
            is_k_squared_i = np.allclose(K_squared, np.eye(2))
            
            if is_s_squared_i or is_k_squared_i:
                results['pauli_candidates'].append({
                    'indices': idx,
                    'S_sub': S_sub,
                    'K_sub': K_sub,
                    'S² = I': is_s_squared_i,
                    'K² = I': is_k_squared_i,
                })
    
    return results


# =============================================================================
# Experiment 5: Geometric Phase Analysis
# =============================================================================

def analyze_path_phases(initial_expr: SKExpr, max_depth: int = 5) -> Dict:
    """
    計算パスに沿った「位相」の解析
    
    仮説: パスの幾何学的性質が位相を定義する
    """
    graph = build_multiway_graph(initial_expr, max_depth=max_depth)
    paths = graph.get_all_paths()
    
    results = {
        'num_paths': len(paths),
        'paths': [],
    }
    
    for path in paths:
        path_info = {
            'length': len(path.nodes),
            'nodes': [to_canonical(node.expr) for node in path.nodes],
            'operations': [edge.redex_type for edge in path.edges],
        }
        
        # 位相の計算（仮説的）
        # S演算子 → +θ, K演算子 → -θ として累積
        theta = 0
        for edge in path.edges:
            if edge.redex_type == RedexType.S_REDEX:
                theta += np.pi / 4  # 仮の値
            elif edge.redex_type == RedexType.K_REDEX:
                theta -= np.pi / 4
        
        path_info['accumulated_phase'] = theta
        path_info['phase_factor'] = np.exp(1j * theta)
        
        results['paths'].append(path_info)
    
    # パス間の位相差
    if len(results['paths']) > 1:
        phases = [p['accumulated_phase'] for p in results['paths']]
        results['phase_differences'] = [phases[i] - phases[0] for i in range(1, len(phases))]
    
    return results


# =============================================================================
# Main Experiment
# =============================================================================

def run_experiment():
    print("=" * 80)
    print("Phase 1A 実験: 代数的構造からの複素数導出の検証")
    print("=" * 80)
    
    # 実験1: Reduction Closure
    print("\n" + "-" * 80)
    print("実験1: Reduction Closure の構築")
    print("-" * 80)
    
    initial = [
        parse("S K K"),
        parse("S (S K) K"),
        parse("S K S"),
    ]
    
    closure = compute_reduction_closure(initial, max_size=30, max_steps=5)
    print(f"  Closure サイズ: {len(closure)}")
    print(f"  式の例: {list(closure)[:5]}")
    
    # closure を基底として使用
    basis_exprs = [parse(e) for e in list(closure)[:20]]
    
    # 実験2: スペクトル解析
    print("\n" + "-" * 80)
    print("実験2: 演算子のスペクトル解析")
    print("-" * 80)
    
    spectrum_results = analyze_operator_spectrum(basis_exprs)
    
    for name, data in spectrum_results.items():
        if 'error' in data:
            continue
        
        eigs = data['eigenvalues']
        eig_str = ", ".join(f"{e:.3f}" for e in eigs[:5])
        
        flags = []
        if data['has_imaginary']:
            flags.append("虚数")
        if data['has_unit_circle']:
            flags.append("単位円")
        if data['has_minus_one']:
            flags.append("-1")
        
        flag_str = " [" + ", ".join(flags) + "]" if flags else ""
        print(f"  {name:12s}: trace={data['trace']:.3f}, det={data['det']:.3f}{flag_str}")
    
    # 実験3: J² = -I の探索
    print("\n" + "-" * 80)
    print("実験3: J² = -I を満たす J の探索")
    print("-" * 80)
    
    j_candidates = search_J_squared_minus_I(basis_exprs)
    
    if j_candidates:
        print(f"  候補数: {len(j_candidates)}")
        for i, cand in enumerate(j_candidates[:3]):
            a, b, c = cand['coefficients']
            print(f"    J_{i+1} = ({a})I + ({b})S + ({c})K, error={cand['error']:.6f}")
    else:
        print("  J² = -I を満たす J は見つかりませんでした")
    
    # 実験4: Pauli構造
    print("\n" + "-" * 80)
    print("実験4: Pauli行列的構造の探索")
    print("-" * 80)
    
    pauli_results = search_pauli_structure(basis_exprs)
    print(f"  基底次元: {pauli_results['dimension']}")
    print(f"  Pauli候補数: {len(pauli_results['pauli_candidates'])}")
    
    for cand in pauli_results['pauli_candidates'][:2]:
        print(f"    indices={cand['indices']}, S²=I: {cand['S² = I']}, K²=I: {cand['K² = I']}")
    
    # 実験5: パス位相
    print("\n" + "-" * 80)
    print("実験5: 計算パスに沿った位相解析")
    print("-" * 80)
    
    test_expr = parse("S (K a) (K b) c")
    phase_results = analyze_path_phases(test_expr, max_depth=4)
    
    print(f"  初期式: {to_string(test_expr)}")
    print(f"  パス数: {phase_results['num_paths']}")
    
    for i, path in enumerate(phase_results['paths'][:3]):
        ops = [str(o).split('.')[-1] for o in path['operations']]
        print(f"    パス{i+1}: len={path['length']}, ops={ops}, phase={path['accumulated_phase']:.4f}")
    
    if 'phase_differences' in phase_results:
        print(f"  位相差: {phase_results['phase_differences']}")
    
    # 結論
    print("\n" + "=" * 80)
    print("実験結論")
    print("=" * 80)
    
    found_complex = (
        len(j_candidates) > 0 or
        any(d.get('has_imaginary', False) for d in spectrum_results.values()) or
        len(pauli_results['pauli_candidates']) > 0
    )
    
    if found_complex:
        print("\n  🔔 複素数構造の兆候が見つかりました！")
        print("     詳細な解析が必要です。")
    else:
        print("\n  結果: 直接的な複素構造は見つかりませんでした。")
        print("\n  考察:")
        print("    1. SK演算子の行列表現は疎で縮退が多い")
        print("    2. 有限基底上の解析では限界がある")
        print("    3. 複素構造は演算子代数ではなく、")
        print("       パスの幾何学（アプローチB）に現れる可能性がある")
        print("\n  推奨: アプローチB（幾何学的構造）への移行を検討")
    
    return {
        'closure_size': len(closure),
        'spectrum': spectrum_results,
        'j_candidates': j_candidates,
        'pauli': pauli_results,
        'phase': phase_results,
        'found_complex': found_complex,
    }


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    results = run_experiment()

