"""
Path Space Holonomy for SK Computation
======================================

Phase 1B: パス空間の幾何学的構造

目的:
    SK計算のパス空間に「接続」を定義し、そのホロノミー（曲率効果）から
    位相構造が現れるかどうかを検証する。

理論的背景:
    1. Berry位相: パラメータ空間のループに沿った位相シフト
    2. ホロノミー群: 全てのループに沿った平行移動の集合
    3. U(1) 構造: 量子力学の位相は U(1) 群で記述される

検証項目:
    1. パス空間への接続の定義
    2. ループ（同一終端への異なるパス）の列挙
    3. ホロノミーの計算と U(1) 構造の検出
"""

from __future__ import annotations
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'phase0'))

from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional, Callable
from enum import Enum, auto
import numpy as np
from itertools import combinations
import cmath

from sk_parser import SKExpr, S, K, Var, App, parse, to_string, to_canonical, size
from reduction import find_redexes, reduce_at_path, is_normal_form, RedexType, Redex
from multiway import (
    MultiwayGraph, MultiwayNode, ReductionEdge, Path,
    build_multiway_graph
)


# =============================================================================
# Path Operations
# =============================================================================

@dataclass
class PathOperation:
    """
    パス上の単一操作（辺）
    
    Attributes:
        redex_type: S または K
        redex_path: 式内の位置
        source_canonical: 元の式の正準形
        target_canonical: 結果の式の正準形
    """
    redex_type: RedexType
    redex_path: str
    source_canonical: str
    target_canonical: str
    
    def __repr__(self):
        type_str = "S" if self.redex_type == RedexType.S_REDEX else "K"
        return f"{type_str}@{self.redex_path or 'root'}"


@dataclass
class ComputationPath:
    """
    計算パス（操作の列）
    
    Attributes:
        operations: 操作の列
        start: 開始式の正準形
        end: 終了式の正準形
    """
    operations: List[PathOperation]
    start: str
    end: str
    
    @property
    def length(self) -> int:
        return len(self.operations)
    
    @property
    def signature(self) -> str:
        """パスのシグネチャ（操作タイプの列）"""
        return "-".join(str(op) for op in self.operations)
    
    def operation_types(self) -> List[RedexType]:
        """操作タイプの列"""
        return [op.redex_type for op in self.operations]


# =============================================================================
# Loop Detection
# =============================================================================

@dataclass
class ComputationLoop:
    """
    計算ループ（同じ始点・終点を持つパスペア）
    
    Attributes:
        path1: 第1パス
        path2: 第2パス
        start: 共通の開始点
        end: 共通の終了点
    """
    path1: ComputationPath
    path2: ComputationPath
    start: str
    end: str
    
    @property
    def area(self) -> int:
        """ループの「面積」（パス長の差の絶対値）"""
        return abs(self.path1.length - self.path2.length)
    
    def operation_difference(self) -> Dict[str, int]:
        """操作タイプの差（S数の差、K数の差）"""
        s1 = sum(1 for op in self.path1.operations if op.redex_type == RedexType.S_REDEX)
        k1 = sum(1 for op in self.path1.operations if op.redex_type == RedexType.K_REDEX)
        s2 = sum(1 for op in self.path2.operations if op.redex_type == RedexType.S_REDEX)
        k2 = sum(1 for op in self.path2.operations if op.redex_type == RedexType.K_REDEX)
        
        return {
            'delta_S': s1 - s2,
            'delta_K': k1 - k2,
        }


def extract_computation_paths(graph: MultiwayGraph) -> List[ComputationPath]:
    """
    MultiwayGraphから計算パスを抽出
    """
    paths = graph.get_all_paths()
    computation_paths = []
    
    for path in paths:
        if len(path.nodes) < 2:
            continue
        
        operations = []
        for edge in path.edges:
            op = PathOperation(
                redex_type=edge.redex_type,
                redex_path=edge.redex_path,
                source_canonical=to_canonical(edge.source.expr),
                target_canonical=to_canonical(edge.target.expr),
            )
            operations.append(op)
        
        comp_path = ComputationPath(
            operations=operations,
            start=to_canonical(path.nodes[0].expr),
            end=to_canonical(path.nodes[-1].expr),
        )
        computation_paths.append(comp_path)
    
    return computation_paths


def find_loops(paths: List[ComputationPath]) -> List[ComputationLoop]:
    """
    同じ始点・終点を持つパスペアからループを構成
    """
    loops = []
    
    # 始点・終点でグループ化
    groups: Dict[Tuple[str, str], List[ComputationPath]] = {}
    for path in paths:
        key = (path.start, path.end)
        if key not in groups:
            groups[key] = []
        groups[key].append(path)
    
    # 各グループ内でペアを作成
    for (start, end), group_paths in groups.items():
        if len(group_paths) < 2:
            continue
        
        for p1, p2 in combinations(group_paths, 2):
            # 同一パスはスキップ
            if p1.signature == p2.signature:
                continue
            
            loop = ComputationLoop(
                path1=p1,
                path2=p2,
                start=start,
                end=end,
            )
            loops.append(loop)
    
    return loops


# =============================================================================
# Connection and Holonomy
# =============================================================================

class Connection:
    """
    パス空間上の接続（位相の割り当て方）
    
    接続は各操作に位相を割り当てる関数
    """
    
    def __init__(self, phase_function: Callable[[PathOperation], complex]):
        """
        Args:
            phase_function: 操作 → 位相因子 の関数
        """
        self.phase_function = phase_function
    
    def parallel_transport(self, path: ComputationPath) -> complex:
        """
        パスに沿った平行移動（位相の累積）
        
        Returns:
            累積位相因子 exp(iΦ)
        """
        total_phase = complex(1.0, 0.0)
        
        for op in path.operations:
            phase = self.phase_function(op)
            total_phase *= phase
        
        return total_phase
    
    def holonomy(self, loop: ComputationLoop) -> complex:
        """
        ループのホロノミー
        
        ホロノミー = path1の位相 / path2の位相
        = path1の位相 × path2の逆位相
        
        U(1) の要素として返す
        """
        phase1 = self.parallel_transport(loop.path1)
        phase2 = self.parallel_transport(loop.path2)
        
        # 位相2の逆元と位相1の積
        if abs(phase2) < 1e-10:
            return complex(float('nan'), float('nan'))
        
        holonomy = phase1 / phase2
        
        # 正規化して U(1) に
        if abs(holonomy) > 1e-10:
            holonomy = holonomy / abs(holonomy)
        
        return holonomy


# =============================================================================
# Predefined Connections
# =============================================================================

def constant_phase_connection(s_phase: float, k_phase: float) -> Connection:
    """
    定数位相接続: S → exp(iθ_S), K → exp(iθ_K)
    """
    def phase_func(op: PathOperation) -> complex:
        if op.redex_type == RedexType.S_REDEX:
            return cmath.exp(1j * s_phase)
        else:
            return cmath.exp(1j * k_phase)
    
    return Connection(phase_func)


def depth_dependent_connection(base_s: float, base_k: float) -> Connection:
    """
    深さ依存接続: 位相が式内の位置（深さ）に依存
    """
    def phase_func(op: PathOperation) -> complex:
        depth = len(op.redex_path) if op.redex_path else 0
        
        if op.redex_type == RedexType.S_REDEX:
            return cmath.exp(1j * base_s / (depth + 1))
        else:
            return cmath.exp(1j * base_k / (depth + 1))
    
    return Connection(phase_func)


def complexity_dependent_connection(alpha: float = 0.1) -> Connection:
    """
    複雑性依存接続: 位相が式のサイズ変化に依存
    """
    def phase_func(op: PathOperation) -> complex:
        try:
            src_size = len(op.source_canonical)
            tgt_size = len(op.target_canonical)
            delta = tgt_size - src_size
            return cmath.exp(1j * alpha * delta)
        except:
            return complex(1.0, 0.0)
    
    return Connection(phase_func)


def information_erasure_connection(k_phase: float = np.pi) -> Connection:
    """
    情報消去接続: K演算子（情報消去）に π 位相を割り当て
    
    Landauer原理に基づく: 情報消去は不可逆操作
    """
    def phase_func(op: PathOperation) -> complex:
        if op.redex_type == RedexType.K_REDEX:
            return cmath.exp(1j * k_phase)
        else:
            return complex(1.0, 0.0)
    
    return Connection(phase_func)


# =============================================================================
# Holonomy Group Analysis
# =============================================================================

class HolonomyGroupAnalysis:
    """
    ホロノミー群の解析
    
    全てのループのホロノミーを計算し、
    それらが生成する群を推定する
    """
    
    def __init__(self, connection: Connection):
        self.connection = connection
    
    def analyze_loops(self, loops: List[ComputationLoop]) -> Dict:
        """
        全ループのホロノミーを計算
        """
        results = {
            'num_loops': len(loops),
            'holonomies': [],
            'phases': [],
            'is_trivial': True,
            'contains_u1': False,
        }
        
        for loop in loops:
            h = self.connection.holonomy(loop)
            
            if np.isnan(h.real) or np.isnan(h.imag):
                continue
            
            results['holonomies'].append({
                'loop': loop,
                'holonomy': h,
                'phase': cmath.phase(h),
                'magnitude': abs(h),
            })
            results['phases'].append(cmath.phase(h))
            
            # 非自明なホロノミーがあるか
            if abs(h - 1.0) > 1e-6:
                results['is_trivial'] = False
        
        # U(1) 構造の検出
        # 位相が連続的に分布しているか
        if len(results['phases']) > 1:
            phases = np.array(results['phases'])
            unique_phases = len(set(np.round(phases, 4)))
            
            # 複数の異なる位相があれば U(1) の可能性
            if unique_phases > 1:
                results['contains_u1'] = True
            
            results['phase_statistics'] = {
                'mean': float(np.mean(phases)),
                'std': float(np.std(phases)),
                'min': float(np.min(phases)),
                'max': float(np.max(phases)),
                'unique_count': unique_phases,
            }
        
        return results
    
    def check_group_closure(self, loops: List[ComputationLoop]) -> Dict:
        """
        ホロノミーが群を成すかチェック
        
        条件:
        1. 単位元の存在（自明ループ）
        2. 逆元の存在（逆向きループ）
        3. 結合律（合成の一貫性）
        """
        results = {
            'has_identity': False,
            'has_inverses': False,
            'is_abelian': True,  # U(1) はアーベル群
        }
        
        holonomies = []
        for loop in loops:
            h = self.connection.holonomy(loop)
            if not (np.isnan(h.real) or np.isnan(h.imag)):
                holonomies.append(h)
        
        if not holonomies:
            return results
        
        # 単位元のチェック
        for h in holonomies:
            if abs(h - 1.0) < 1e-6:
                results['has_identity'] = True
                break
        
        # 逆元のチェック
        for h in holonomies:
            h_inv = 1.0 / h if abs(h) > 1e-10 else None
            if h_inv:
                for h2 in holonomies:
                    if abs(h2 - h_inv) < 1e-6:
                        results['has_inverses'] = True
                        break
        
        return results


# =============================================================================
# Main Analysis
# =============================================================================

def run_holonomy_analysis(expr: SKExpr, max_depth: int = 10, verbose: bool = True) -> Dict:
    """
    完全なホロノミー解析を実行
    """
    results = {}
    
    if verbose:
        print("=" * 70)
        print("Phase 1B: パス空間のホロノミー解析")
        print("=" * 70)
        print(f"\n対象式: {to_string(expr)}")
    
    # グラフ構築
    graph = build_multiway_graph(expr, max_depth=max_depth)
    
    # 辺の総数を計算
    total_edges = sum(len(node.children) for node in graph.nodes.values())
    
    if verbose:
        print(f"ノード数: {len(graph.nodes)}")
        print(f"辺数: {total_edges}")
    
    # パス抽出
    paths = extract_computation_paths(graph)
    
    if verbose:
        print(f"パス数: {len(paths)}")
    
    results['num_paths'] = len(paths)
    results['paths'] = paths
    
    # ループ検出
    loops = find_loops(paths)
    
    if verbose:
        print(f"ループ数: {len(loops)}")
    
    results['num_loops'] = len(loops)
    results['loops'] = loops
    
    if not loops:
        if verbose:
            print("\n⚠️ ループが見つかりませんでした")
            print("   同じ終端に至る複数のパスが必要です")
        results['has_loops'] = False
        return results
    
    results['has_loops'] = True
    
    # 各接続でのホロノミー解析
    connections = {
        'constant_S_K': constant_phase_connection(np.pi/4, -np.pi/4),
        'constant_S_only': constant_phase_connection(np.pi/2, 0),
        'depth_dependent': depth_dependent_connection(np.pi/2, np.pi/4),
        'complexity': complexity_dependent_connection(0.1),
        'info_erasure': information_erasure_connection(np.pi),
    }
    
    if verbose:
        print("\n" + "-" * 70)
        print("接続ごとのホロノミー解析")
        print("-" * 70)
    
    results['connections'] = {}
    
    for name, conn in connections.items():
        analyzer = HolonomyGroupAnalysis(conn)
        analysis = analyzer.analyze_loops(loops)
        group_check = analyzer.check_group_closure(loops)
        
        results['connections'][name] = {
            'analysis': analysis,
            'group': group_check,
        }
        
        if verbose:
            print(f"\n📐 接続: {name}")
            print(f"   非自明なホロノミー: {'はい' if not analysis['is_trivial'] else 'いいえ'}")
            print(f"   U(1) 構造の候補: {'はい' if analysis['contains_u1'] else 'いいえ'}")
            
            if 'phase_statistics' in analysis:
                stats = analysis['phase_statistics']
                print(f"   位相統計: mean={stats['mean']:.4f}, std={stats['std']:.4f}")
                print(f"   位相範囲: [{stats['min']:.4f}, {stats['max']:.4f}]")
                print(f"   異なる位相数: {stats['unique_count']}")
            
            if analysis['holonomies']:
                print(f"   ホロノミーサンプル:")
                for h_data in analysis['holonomies'][:3]:
                    phase = h_data['phase']
                    print(f"     phase={phase:.4f} (≈ {phase/np.pi:.2f}π)")
    
    # 結論
    if verbose:
        print("\n" + "=" * 70)
        print("結論")
        print("=" * 70)
        
        any_nontrivial = any(
            not r['analysis']['is_trivial'] 
            for r in results['connections'].values()
        )
        any_u1 = any(
            r['analysis']['contains_u1'] 
            for r in results['connections'].values()
        )
        
        if any_u1:
            print("\n🔔 U(1) 構造の候補が見つかりました！")
            print("   ただし、これは接続の定義に依存しています。")
            print("   「導出」されたのではなく、「仮定」から生じた可能性があります。")
        elif any_nontrivial:
            print("\n⚠️ 非自明なホロノミーが見つかりました。")
            print("   しかし、U(1) 構造とは言えません。")
        else:
            print("\n✓ 全ての接続でホロノミーが自明でした。")
            print("   パス空間の幾何学から位相構造は現れませんでした。")
    
    results['has_nontrivial_holonomy'] = any(
        not r['analysis']['is_trivial'] 
        for r in results['connections'].values()
    )
    results['has_u1_structure'] = any(
        r['analysis']['contains_u1'] 
        for r in results['connections'].values()
    )
    
    return results


def analyze_multiple_expressions(verbose: bool = True) -> Dict:
    """
    複数の式でホロノミー解析を実行
    """
    test_expressions = [
        "S (K a) (K b) c",
        "(K a b) (K c d)",
        "S (K a b) c d",
        "(K a b) (K c d) (K e f)",
        "S S K a b c",
        "S (K a) (K b) (S c d e)",
    ]
    
    all_results = {}
    
    for expr_str in test_expressions:
        try:
            expr = parse(expr_str)
            if verbose:
                print(f"\n{'='*70}")
                print(f"式: {expr_str}")
            
            results = run_holonomy_analysis(expr, max_depth=8, verbose=verbose)
            all_results[expr_str] = results
        except Exception as e:
            all_results[expr_str] = {'error': str(e)}
    
    return all_results


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    # 単一式のテスト
    expr = parse("S (K a) (K b) c")
    results = run_holonomy_analysis(expr, verbose=True)
    
    print("\n\n" + "=" * 70)
    print("複数式での解析")
    print("=" * 70)
    
    # 複数式でのテスト
    all_results = analyze_multiple_expressions(verbose=True)
    
    # サマリー
    print("\n\n" + "=" * 70)
    print("全体サマリー")
    print("=" * 70)
    
    for expr_str, results in all_results.items():
        if 'error' in results:
            status = f"❌ Error: {results['error']}"
        elif not results.get('has_loops', False):
            status = "⚪ ループなし"
        elif results.get('has_u1_structure', False):
            status = "🔔 U(1) 候補あり"
        elif results.get('has_nontrivial_holonomy', False):
            status = "⚠️ 非自明ホロノミー"
        else:
            status = "✓ 自明"
        
        print(f"  {expr_str:30s} : {status}")

