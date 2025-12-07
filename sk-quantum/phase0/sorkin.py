"""
Sorkin's Quantum Measure Formula
================================

Day 6 実装: Sorkin公式 I₂, I₃ の計算

Sorkinの定理（1994）:
    量子干渉が生じるのは、確率測度が以下の条件を満たすとき：
    - I₂(A,B) = P(A∪B) - P(A) - P(B) ≠ 0  （2次干渉あり）
    - I₃(A,B,C) = 0  （3次干渉なし）
    
    この条件を満たす最小の数体系は複素数である。

本実装では:
    - A, B, C を「パス（計算経路）」と見なす
    - P(A) = パスAの確率
    - P(A∪B) = パスAまたはBを通る確率（同じ終端への確率の和）
    - I₂, I₃ を計算し、量子的非加法性を検証
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Tuple, Set, Optional
from itertools import combinations

from sk_parser import parse, to_canonical
from multiway import (
    MultiwayGraph, Path, MultiwayNode, 
    build_multiway_graph, enumerate_paths
)
from probability import (
    ProbabilityModel, ProbabilityCalculator,
    UniformModel, LengthWeightedModel, ComplexityWeightedModel,
    BranchWeightedModel
)


# =============================================================================
# Sorkin Interference Measures
# =============================================================================

@dataclass
class InterferenceMeasure:
    """
    干渉測度の結果
    
    Attributes:
        paths: 使用したパスの組
        I2: 2次干渉項
        I3: 3次干渉項（3パスの場合）
        is_quantum: 量子的非加法性があるか (I2≠0 and I3=0)
    """
    paths: Tuple[Path, ...]
    I2: float
    I3: Optional[float]
    
    @property
    def is_quantum_like(self) -> bool:
        """量子的非加法性の条件を満たすか"""
        if self.I3 is None:
            # 2パスの場合、I2≠0 のみで判定
            return abs(self.I2) > 1e-10
        else:
            # 3パスの場合、I2≠0 かつ I3≈0
            return abs(self.I2) > 1e-10 and abs(self.I3) < 1e-10
    
    @property
    def is_classical(self) -> bool:
        """古典的（加法的）か"""
        return abs(self.I2) < 1e-10


def compute_I2(
    path_a: Path, 
    path_b: Path,
    calc: ProbabilityCalculator
) -> float:
    """
    2次干渉項 I₂(A,B) を計算
    
    I₂(A,B) = P(A∪B) - P(A) - P(B)
    
    古典確率では I₂ = 0（加法性）
    量子確率では I₂ ≠ 0（干渉項）
    
    Args:
        path_a: パスA
        path_b: パスB
        calc: 確率計算機
    
    Returns:
        I₂ の値
    """
    P_A = calc.get_path_probability(path_a)
    P_B = calc.get_path_probability(path_b)
    
    # P(A∪B): AまたはBを通る確率
    # 同じ終端への異なるパスの場合、P(A∪B) = P(A) + P(B)（古典）
    # ただし、量子的には干渉項が入る
    
    # ここでの解釈：
    # - AとBが排他的なら P(A∪B) = P(A) + P(B)
    # - AとBが同じ終端に至るなら、終端への確率 P(terminal) を使う
    
    if path_a.end == path_b.end:
        # 同じ終端への異なるパス
        P_union = calc.get_terminal_probability(path_a.end)
    else:
        # 異なる終端へのパス（排他的）
        P_union = P_A + P_B
    
    return P_union - P_A - P_B


def compute_I3(
    path_a: Path,
    path_b: Path,
    path_c: Path,
    calc: ProbabilityCalculator
) -> float:
    """
    3次干渉項 I₃(A,B,C) を計算
    
    I₃(A,B,C) = P(A∪B∪C) - P(A∪B) - P(B∪C) - P(C∪A) + P(A) + P(B) + P(C)
    
    量子力学では I₃ = 0（3次以上の干渉なし）
    
    Args:
        path_a, path_b, path_c: 3つのパス
        calc: 確率計算機
    
    Returns:
        I₃ の値
    """
    P_A = calc.get_path_probability(path_a)
    P_B = calc.get_path_probability(path_b)
    P_C = calc.get_path_probability(path_c)
    
    # P(A∪B), P(B∪C), P(C∪A)
    def P_union_2(p1: Path, p2: Path) -> float:
        if p1.end == p2.end:
            return calc.get_terminal_probability(p1.end)
        else:
            return calc.get_path_probability(p1) + calc.get_path_probability(p2)
    
    P_AB = P_union_2(path_a, path_b)
    P_BC = P_union_2(path_b, path_c)
    P_CA = P_union_2(path_c, path_a)
    
    # P(A∪B∪C)
    terminals = {path_a.end, path_b.end, path_c.end}
    P_ABC = sum(calc.get_terminal_probability(t) for t in terminals)
    
    return P_ABC - P_AB - P_BC - P_CA + P_A + P_B + P_C


# =============================================================================
# Sorkin Analysis
# =============================================================================

class SorkinAnalyzer:
    """
    Sorkin公式による量子性の分析
    """
    
    def __init__(self, graph: MultiwayGraph, model: ProbabilityModel):
        self.graph = graph
        self.model = model
        self.calc = ProbabilityCalculator(graph, model)
        self.paths = graph.get_all_paths()
    
    def analyze_pair(self, path_a: Path, path_b: Path) -> InterferenceMeasure:
        """2パスの干渉を分析"""
        I2 = compute_I2(path_a, path_b, self.calc)
        return InterferenceMeasure(
            paths=(path_a, path_b),
            I2=I2,
            I3=None
        )
    
    def analyze_triple(self, path_a: Path, path_b: Path, 
                       path_c: Path) -> InterferenceMeasure:
        """3パスの干渉を分析"""
        # 代表的なI2を計算（A,Bのペア）
        I2 = compute_I2(path_a, path_b, self.calc)
        I3 = compute_I3(path_a, path_b, path_c, self.calc)
        return InterferenceMeasure(
            paths=(path_a, path_b, path_c),
            I2=I2,
            I3=I3
        )
    
    def analyze_all_pairs(self) -> List[InterferenceMeasure]:
        """全パスペアの干渉を分析"""
        results = []
        for path_a, path_b in combinations(self.paths, 2):
            results.append(self.analyze_pair(path_a, path_b))
        return results
    
    def analyze_all_triples(self) -> List[InterferenceMeasure]:
        """全パス3つ組の干渉を分析"""
        results = []
        for path_a, path_b, path_c in combinations(self.paths, 3):
            results.append(self.analyze_triple(path_a, path_b, path_c))
        return results
    
    def find_quantum_signatures(self) -> Dict:
        """
        量子的シグネチャを探索
        
        Returns:
            分析結果の辞書
        """
        pair_results = self.analyze_all_pairs()
        triple_results = self.analyze_all_triples()
        
        # 量子的ペアを探索
        quantum_pairs = [r for r in pair_results if r.is_quantum_like]
        classical_pairs = [r for r in pair_results if r.is_classical]
        
        # 量子的3つ組を探索
        quantum_triples = [r for r in triple_results if r.is_quantum_like]
        
        return {
            "model": self.model.name,
            "total_paths": len(self.paths),
            "total_pairs": len(pair_results),
            "quantum_pairs": len(quantum_pairs),
            "classical_pairs": len(classical_pairs),
            "total_triples": len(triple_results),
            "quantum_triples": len(quantum_triples),
            "I2_values": [r.I2 for r in pair_results],
            "I3_values": [r.I3 for r in triple_results if r.I3 is not None],
            "has_quantum_signature": len(quantum_pairs) > 0,
            "pair_results": pair_results,
            "triple_results": triple_results,
        }


# =============================================================================
# Main Analysis Function
# =============================================================================

def verify_sorkin(
    source: str,
    models: List[ProbabilityModel] = None,
    verbose: bool = True
) -> Dict[str, Dict]:
    """
    SK式に対してSorkin公式を検証
    
    Args:
        source: SK式の文字列
        models: 使用する確率モデルのリスト
        verbose: 詳細出力するか
    
    Returns:
        {モデル名: 分析結果} の辞書
    """
    expr = parse(source)
    graph = build_multiway_graph(expr)
    
    if models is None:
        models = [
            UniformModel(),
            LengthWeightedModel(alpha=0.5),
            ComplexityWeightedModel(beta=0.1),
            BranchWeightedModel(graph),
        ]
    
    results = {}
    
    for model in models:
        analyzer = SorkinAnalyzer(graph, model)
        result = analyzer.find_quantum_signatures()
        results[model.name] = result
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Model: {model.name}")
            print(f"{'='*60}")
            print(f"Total paths: {result['total_paths']}")
            print(f"Pairs analyzed: {result['total_pairs']}")
            print(f"  Quantum pairs (I₂≠0): {result['quantum_pairs']}")
            print(f"  Classical pairs (I₂=0): {result['classical_pairs']}")
            
            if result['total_triples'] > 0:
                print(f"Triples analyzed: {result['total_triples']}")
                print(f"  Quantum triples (I₂≠0, I₃=0): {result['quantum_triples']}")
            
            if result['I2_values']:
                I2_nonzero = [v for v in result['I2_values'] if abs(v) > 1e-10]
                if I2_nonzero:
                    print(f"\nNon-zero I₂ values:")
                    for v in I2_nonzero[:5]:
                        print(f"  I₂ = {v:.6f}")
                    if len(I2_nonzero) > 5:
                        print(f"  ... and {len(I2_nonzero)-5} more")
            
            if result['has_quantum_signature']:
                print(f"\n🔔 QUANTUM SIGNATURE DETECTED!")
            else:
                print(f"\n✓ Classical behavior (I₂ = 0 for all pairs)")
    
    return results


def quick_check(source: str) -> bool:
    """
    クイックチェック: 量子的シグネチャがあるか
    
    Args:
        source: SK式の文字列
    
    Returns:
        量子的シグネチャがあればTrue
    """
    results = verify_sorkin(source, verbose=False)
    return any(r['has_quantum_signature'] for r in results.values())


# =============================================================================
# Main (for testing)
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Sorkin Formula Verification - Day 6")
    print("=" * 70)
    
    test_cases = [
        # 単純なケース
        ("K a b", "Single path (no branching)"),
        
        # 分岐があるケース
        ("(K a b) (K c d)", "Two K-redexes (branching)"),
        
        # 研究例
        ("S (K a) (K b) c", "Research example"),
        
        # より複雑な例
        ("S (K a b) c d", "Nested redexes"),
        
        # さらに複雑な例
        ("(K a b) (K c d) (K e f)", "Three K-redexes"),
    ]
    
    all_results = {}
    
    for source, description in test_cases:
        print(f"\n{'#'*70}")
        print(f"# Test: {description}")
        print(f"# Expression: {source}")
        print(f"{'#'*70}")
        
        results = verify_sorkin(source)
        all_results[source] = results
    
    # サマリー
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    
    for source, results in all_results.items():
        has_quantum = any(r['has_quantum_signature'] for r in results.values())
        status = "🔔 Quantum" if has_quantum else "✓ Classical"
        print(f"{source}: {status}")
    
    print(f"\n{'='*70}")
    print("Analysis completed!")
    print(f"{'='*70}")



