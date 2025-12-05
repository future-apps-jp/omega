"""
Information-Theoretic Approach to Phase Derivation
===================================================

Phase 2: 情報理論的アプローチ

目的:
    SK計算のパスに沿った情報量の変化から、位相を「計算」する。
    これは Phase 1A/1B とは異なり、位相を「仮定」するのではなく
    情報理論的な量から「導出」することを目指す。

理論的背景:
    1. Kolmogorov複雑性: 最短記述長としての情報量
    2. Landauer原理: 情報消去 → エネルギー散逸 (kT ln 2)
    3. K-combinator: 情報を捨てる操作
    4. S-combinator: 情報を複製・再配置する操作

仮説:
    - 情報消去量 ΔK から位相を計算できる
    - Φ ∝ ΔK（情報消去量に比例した位相）
"""

from __future__ import annotations
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'phase0'))

from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional, Callable
import numpy as np
import zlib
import hashlib
import cmath

from sk_parser import (
    SKExpr, S, K, Var, App, parse, to_string, to_canonical,
    size, depth, variables
)
from reduction import find_redexes, reduce_at_path, is_normal_form, RedexType, Redex
from multiway import (
    MultiwayGraph, MultiwayNode, ReductionEdge, Path,
    build_multiway_graph
)


# =============================================================================
# Kolmogorov Complexity Approximation
# =============================================================================

def kolmogorov_size(expr: SKExpr) -> int:
    """
    最も単純な近似: 式のサイズ（ノード数）
    """
    return size(expr)


def kolmogorov_string_length(expr: SKExpr) -> int:
    """
    正準形文字列の長さ
    """
    return len(to_canonical(expr))


def kolmogorov_compressed(expr: SKExpr) -> int:
    """
    圧縮後のサイズ（zlib圧縮）
    
    Kolmogorov複雑性の上界近似
    """
    canonical = to_canonical(expr).encode('utf-8')
    compressed = zlib.compress(canonical, level=9)
    return len(compressed)


def kolmogorov_depth_weighted(expr: SKExpr) -> float:
    """
    深さで重み付けしたサイズ
    
    深い部分木はより「複雑」とみなす
    """
    return size(expr) + 0.5 * depth(expr)


def kolmogorov_variable_entropy(expr: SKExpr) -> float:
    """
    変数のエントロピー
    
    変数の多様性を考慮した情報量
    """
    vars_set = variables(expr)
    if not vars_set:
        return 0.0
    
    # 単純に変数の数をカウント
    return len(vars_set)


# =============================================================================
# Information Erasure Tracking
# =============================================================================

@dataclass
class InformationChange:
    """
    1ステップでの情報量変化
    """
    source_expr: SKExpr
    target_expr: SKExpr
    redex_type: RedexType
    redex_path: str
    
    # 各種複雑性指標の変化
    delta_size: int = 0
    delta_string_length: int = 0
    delta_compressed: int = 0
    delta_depth: float = 0.0
    delta_variables: float = 0.0
    
    # 消去された情報
    erased_subexpr: Optional[SKExpr] = None
    erased_size: int = 0
    
    def __post_init__(self):
        self.delta_size = kolmogorov_size(self.target_expr) - kolmogorov_size(self.source_expr)
        self.delta_string_length = kolmogorov_string_length(self.target_expr) - kolmogorov_string_length(self.source_expr)
        
        try:
            self.delta_compressed = kolmogorov_compressed(self.target_expr) - kolmogorov_compressed(self.source_expr)
        except:
            self.delta_compressed = 0
        
        self.delta_depth = kolmogorov_depth_weighted(self.target_expr) - kolmogorov_depth_weighted(self.source_expr)
        self.delta_variables = kolmogorov_variable_entropy(self.target_expr) - kolmogorov_variable_entropy(self.source_expr)
        
        # K-redex の場合、消去された部分式を特定
        if self.redex_type == RedexType.K_REDEX:
            self._find_erased_subexpr()
    
    def _find_erased_subexpr(self):
        """K x y → x で消去される y を特定"""
        # K x y の形を探す
        if isinstance(self.source_expr, App):
            if isinstance(self.source_expr.func, App):
                if isinstance(self.source_expr.func.func, K):
                    # (K x) y → x, y が消去される
                    self.erased_subexpr = self.source_expr.arg
                    self.erased_size = kolmogorov_size(self.erased_subexpr)
    
    @property
    def is_information_erasing(self) -> bool:
        """情報消去操作かどうか"""
        return self.redex_type == RedexType.K_REDEX
    
    @property
    def total_delta(self) -> float:
        """総合的な情報量変化"""
        return self.delta_size + 0.1 * self.delta_compressed


@dataclass
class PathInformation:
    """
    パス全体の情報量変化
    """
    changes: List[InformationChange]
    start_expr: SKExpr
    end_expr: SKExpr
    
    @property
    def total_erasure(self) -> int:
        """総消去量"""
        return sum(c.erased_size for c in self.changes if c.is_information_erasing)
    
    @property
    def num_k_reductions(self) -> int:
        """K簡約の回数"""
        return sum(1 for c in self.changes if c.is_information_erasing)
    
    @property
    def num_s_reductions(self) -> int:
        """S簡約の回数"""
        return sum(1 for c in self.changes if not c.is_information_erasing)
    
    @property
    def total_delta_size(self) -> int:
        """総サイズ変化"""
        return sum(c.delta_size for c in self.changes)
    
    @property
    def total_delta_compressed(self) -> int:
        """総圧縮サイズ変化"""
        return sum(c.delta_compressed for c in self.changes)
    
    def summary(self) -> Dict:
        return {
            'num_steps': len(self.changes),
            'num_k_reductions': self.num_k_reductions,
            'num_s_reductions': self.num_s_reductions,
            'total_erasure': self.total_erasure,
            'total_delta_size': self.total_delta_size,
            'total_delta_compressed': self.total_delta_compressed,
        }


# =============================================================================
# Information-Based Phase Calculation
# =============================================================================

class InformationPhaseCalculator:
    """
    情報量から位相を計算
    
    仮説: Φ = f(ΔK) where ΔK は情報消去量
    """
    
    def __init__(self, phase_formula: str = 'linear'):
        """
        Args:
            phase_formula: 位相計算式
                - 'linear': Φ = α * ΔK
                - 'logarithmic': Φ = α * log(1 + ΔK)
                - 'landauer': Φ = kT * ln(2) * ΔK (正規化)
        """
        self.phase_formula = phase_formula
    
    def compute_phase(self, path_info: PathInformation, alpha: float = 0.1) -> complex:
        """
        パスの情報量変化から位相因子を計算
        
        Returns:
            exp(iΦ) where Φ は計算された位相
        """
        if self.phase_formula == 'linear':
            # 線形: Φ = α * (消去量 - 生成量)
            # K は情報消去、S は情報複製
            phi = alpha * (path_info.total_erasure - path_info.num_s_reductions)
        
        elif self.phase_formula == 'logarithmic':
            # 対数: より緩やかな依存性
            erasure = path_info.total_erasure
            phi = alpha * np.log(1 + erasure)
        
        elif self.phase_formula == 'landauer':
            # Landauer原理に基づく
            # kT ln(2) ≈ 2.87 × 10^-21 J at 300K
            # 正規化して π/4 程度のスケールに
            phi = (np.pi / 4) * path_info.num_k_reductions
        
        elif self.phase_formula == 'size_change':
            # サイズ変化に基づく
            phi = alpha * path_info.total_delta_size
        
        elif self.phase_formula == 'compressed':
            # 圧縮サイズ変化に基づく
            phi = alpha * path_info.total_delta_compressed
        
        else:
            phi = 0.0
        
        return cmath.exp(1j * phi)
    
    def compute_interference(self, path1_info: PathInformation, 
                            path2_info: PathInformation,
                            alpha: float = 0.1) -> Dict:
        """
        2つのパスの干渉を計算
        """
        phase1 = self.compute_phase(path1_info, alpha)
        phase2 = self.compute_phase(path2_info, alpha)
        
        # 位相差
        phase_diff = cmath.phase(phase1) - cmath.phase(phase2)
        
        # 干渉項
        interference = 2 * (phase1 * phase2.conjugate()).real
        
        # 情報量差
        info_diff = {
            'delta_erasure': path1_info.total_erasure - path2_info.total_erasure,
            'delta_k_reductions': path1_info.num_k_reductions - path2_info.num_k_reductions,
            'delta_s_reductions': path1_info.num_s_reductions - path2_info.num_s_reductions,
            'delta_size': path1_info.total_delta_size - path2_info.total_delta_size,
        }
        
        return {
            'phase1': cmath.phase(phase1),
            'phase2': cmath.phase(phase2),
            'phase_diff': phase_diff,
            'interference': interference,
            'info_diff': info_diff,
            'is_constructive': interference > 0,
            'is_destructive': interference < 0,
        }


# =============================================================================
# Path Information Extraction
# =============================================================================

def extract_path_information(graph: MultiwayGraph) -> List[PathInformation]:
    """
    グラフから全パスの情報量変化を抽出
    """
    paths = graph.get_all_paths()
    path_infos = []
    
    for path in paths:
        if len(path.nodes) < 2:
            continue
        
        changes = []
        for edge in path.edges:
            change = InformationChange(
                source_expr=edge.source.expr,
                target_expr=edge.target.expr,
                redex_type=edge.redex_type,
                redex_path=edge.redex_path,
            )
            changes.append(change)
        
        path_info = PathInformation(
            changes=changes,
            start_expr=path.nodes[0].expr,
            end_expr=path.nodes[-1].expr,
        )
        path_infos.append(path_info)
    
    return path_infos


# =============================================================================
# Main Analysis
# =============================================================================

def run_information_analysis(expr: SKExpr, max_depth: int = 10, verbose: bool = True) -> Dict:
    """
    情報理論的解析を実行
    """
    results = {}
    
    if verbose:
        print("=" * 70)
        print("Phase 2: 情報理論的アプローチによる位相導出")
        print("=" * 70)
        print(f"\n対象式: {to_string(expr)}")
        print(f"初期サイズ: {kolmogorov_size(expr)}")
        print(f"初期圧縮サイズ: {kolmogorov_compressed(expr)}")
    
    # グラフ構築
    graph = build_multiway_graph(expr, max_depth=max_depth)
    
    if verbose:
        print(f"ノード数: {len(graph.nodes)}")
    
    # パス情報抽出
    path_infos = extract_path_information(graph)
    
    if verbose:
        print(f"パス数: {len(path_infos)}")
    
    results['num_paths'] = len(path_infos)
    results['path_summaries'] = [p.summary() for p in path_infos]
    
    if len(path_infos) < 2:
        if verbose:
            print("\n⚠️ 複数パスが必要です（干渉計算のため）")
        results['has_multiple_paths'] = False
        return results
    
    results['has_multiple_paths'] = True
    
    # 各位相計算式での解析
    formulas = ['linear', 'logarithmic', 'landauer', 'size_change', 'compressed']
    
    if verbose:
        print("\n" + "-" * 70)
        print("パスの情報量サマリー")
        print("-" * 70)
        
        for i, path_info in enumerate(path_infos[:5]):
            summary = path_info.summary()
            print(f"\n  パス {i+1}:")
            print(f"    ステップ数: {summary['num_steps']}")
            print(f"    K簡約: {summary['num_k_reductions']}, S簡約: {summary['num_s_reductions']}")
            print(f"    消去量: {summary['total_erasure']}")
            print(f"    サイズ変化: {summary['total_delta_size']}")
    
    if verbose:
        print("\n" + "-" * 70)
        print("位相計算式ごとの干渉解析")
        print("-" * 70)
    
    results['interference_analysis'] = {}
    
    for formula in formulas:
        calculator = InformationPhaseCalculator(formula)
        
        # 全パスペアでの干渉
        interferences = []
        for i, p1 in enumerate(path_infos):
            for j, p2 in enumerate(path_infos[i+1:], i+1):
                # 同じ終点を持つパスのみ
                if to_canonical(p1.end_expr) != to_canonical(p2.end_expr):
                    continue
                
                interference = calculator.compute_interference(p1, p2)
                interferences.append({
                    'path1_idx': i,
                    'path2_idx': j,
                    **interference,
                })
        
        results['interference_analysis'][formula] = interferences
        
        if verbose:
            print(f"\n📊 計算式: {formula}")
            
            if not interferences:
                print("   同じ終点を持つパスペアがありません")
                continue
            
            # 統計
            phase_diffs = [inf['phase_diff'] for inf in interferences]
            
            constructive = sum(1 for inf in interferences if inf['is_constructive'])
            destructive = sum(1 for inf in interferences if inf['is_destructive'])
            
            print(f"   パスペア数: {len(interferences)}")
            print(f"   位相差範囲: [{min(phase_diffs):.4f}, {max(phase_diffs):.4f}]")
            print(f"   建設的干渉: {constructive}, 破壊的干渉: {destructive}")
            
            # サンプル表示
            if interferences:
                inf = interferences[0]
                print(f"   サンプル: φ₁={inf['phase1']:.4f}, φ₂={inf['phase2']:.4f}, Δφ={inf['phase_diff']:.4f}")
    
    # 重要な発見の検出
    if verbose:
        print("\n" + "=" * 70)
        print("結論")
        print("=" * 70)
    
    # 非自明な位相差があるか
    has_nontrivial_phase = False
    for formula, interferences in results['interference_analysis'].items():
        for inf in interferences:
            if abs(inf['phase_diff']) > 1e-6:
                has_nontrivial_phase = True
                break
    
    results['has_nontrivial_phase'] = has_nontrivial_phase
    
    # 情報量差と位相差の相関
    correlations = {}
    for formula, interferences in results['interference_analysis'].items():
        if not interferences:
            continue
        
        phase_diffs = [inf['phase_diff'] for inf in interferences]
        info_diffs = [inf['info_diff']['delta_erasure'] for inf in interferences]
        
        if len(set(phase_diffs)) > 1 and len(set(info_diffs)) > 1:
            try:
                correlation = np.corrcoef(phase_diffs, info_diffs)[0, 1]
                correlations[formula] = correlation
            except:
                pass
    
    results['info_phase_correlations'] = correlations
    
    if verbose:
        if has_nontrivial_phase:
            print("\n✓ 非自明な位相差が計算されました。")
            print("  情報量差から位相が「計算」されています（仮定ではなく）。")
            
            if correlations:
                print("\n  情報量-位相差の相関:")
                for formula, corr in correlations.items():
                    if not np.isnan(corr):
                        print(f"    {formula}: r = {corr:.4f}")
        else:
            print("\n⚠️ 全てのパスで位相差がゼロでした。")
            print("   情報量変化がパス間で同一である可能性があります。")
    
    return results


def analyze_multiple_expressions(verbose: bool = True) -> Dict:
    """
    複数の式で情報理論的解析を実行
    """
    test_expressions = [
        "S (K a) (K b) c",
        "(K a b) (K c d)",
        "S (K a b) c d",
        "(K a b) (K c d) (K e f)",
        "S (K a) (K b) (S c d e)",
    ]
    
    all_results = {}
    
    for expr_str in test_expressions:
        try:
            expr = parse(expr_str)
            if verbose:
                print(f"\n{'='*70}")
                print(f"式: {expr_str}")
            
            results = run_information_analysis(expr, max_depth=8, verbose=verbose)
            all_results[expr_str] = results
        except Exception as e:
            all_results[expr_str] = {'error': str(e)}
            if verbose:
                print(f"  Error: {e}")
    
    return all_results


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    # 単一式のテスト
    expr = parse("S (K a) (K b) c")
    results = run_information_analysis(expr, verbose=True)
    
    print("\n\n" + "=" * 70)
    print("複数式での解析")
    print("=" * 70)
    
    all_results = analyze_multiple_expressions(verbose=True)
    
    # サマリー
    print("\n\n" + "=" * 70)
    print("全体サマリー")
    print("=" * 70)
    
    for expr_str, results in all_results.items():
        if 'error' in results:
            status = f"❌ Error"
        elif not results.get('has_multiple_paths', False):
            status = "⚪ 単一パス"
        elif results.get('has_nontrivial_phase', False):
            status = "🔔 非自明位相あり"
        else:
            status = "✓ 位相差ゼロ"
        
        print(f"  {expr_str:30s} : {status}")


