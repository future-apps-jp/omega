"""
SK Path Probability Definitions
===============================

Day 5 実装: 計算パスへの確率の割り当て

確率定義の候補:
1. 等重率 (Uniform): 全パスに等しい確率
2. パス長重み (Length-weighted): 短いパスほど高確率
3. 複雑性重み (Complexity-weighted): 単純なパスほど高確率
4. 分岐重み (Branch-weighted): 各分岐で等確率に分配

Sorkin公式で使用するための確率定義を提供。
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Callable, Set
from math import exp, log

from sk_parser import SKExpr, to_canonical, size
from multiway import MultiwayGraph, Path, MultiwayNode, build_multiway_graph


# =============================================================================
# Abstract Probability Model
# =============================================================================

class ProbabilityModel(ABC):
    """
    確率モデルの抽象基底クラス
    
    パスに確率を割り当てる方法を定義
    """
    
    @abstractmethod
    def path_weight(self, path: Path) -> float:
        """
        パスの重み（非正規化）を計算
        
        Args:
            path: 計算パス
        
        Returns:
            パスの重み（正の実数）
        """
        pass
    
    def path_probability(self, path: Path, all_paths: List[Path]) -> float:
        """
        パスの確率（正規化済み）を計算
        
        Args:
            path: 計算パス
            all_paths: 全パスのリスト
        
        Returns:
            正規化された確率
        """
        total_weight = sum(self.path_weight(p) for p in all_paths)
        if total_weight == 0:
            return 0.0
        return self.path_weight(path) / total_weight
    
    def terminal_probability(self, terminal: MultiwayNode, 
                            paths: List[Path]) -> float:
        """
        終端への到達確率（その終端への全パスの確率の和）
        
        Args:
            terminal: 終端ノード
            paths: 全パスのリスト
        
        Returns:
            終端への到達確率
        """
        terminal_paths = [p for p in paths if p.end == terminal]
        return sum(self.path_probability(p, paths) for p in terminal_paths)
    
    @property
    @abstractmethod
    def name(self) -> str:
        """モデル名"""
        pass


# =============================================================================
# Concrete Probability Models
# =============================================================================

class UniformModel(ProbabilityModel):
    """
    等重率モデル: 全パスに等しい重み
    
    P(path) = 1 / (total number of paths)
    """
    
    def path_weight(self, path: Path) -> float:
        return 1.0
    
    @property
    def name(self) -> str:
        return "Uniform"


class LengthWeightedModel(ProbabilityModel):
    """
    パス長重みモデル: 短いパスほど高確率
    
    weight(path) = exp(-α * length)
    
    α > 0 で短いパスを優遇
    """
    
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
    
    def path_weight(self, path: Path) -> float:
        return exp(-self.alpha * path.length)
    
    @property
    def name(self) -> str:
        return f"LengthWeighted(α={self.alpha})"


class ComplexityWeightedModel(ProbabilityModel):
    """
    複雑性重みモデル: 単純なパスほど高確率
    
    weight(path) = 2^(-K(path))
    
    K(path) = パスの複雑性（簡約ステップの式サイズの和で近似）
    
    Kolmogorov複雑性の近似
    """
    
    def __init__(self, beta: float = 0.1):
        self.beta = beta
    
    def path_weight(self, path: Path) -> float:
        # パスの複雑性: 各ステップの式のサイズの和
        complexity = sum(size(node.expr) for node in path.nodes)
        return 2 ** (-self.beta * complexity)
    
    @property
    def name(self) -> str:
        return f"ComplexityWeighted(β={self.beta})"


class BranchWeightedModel(ProbabilityModel):
    """
    分岐重みモデル: 各分岐点で等確率に分配
    
    weight(path) = ∏_{分岐点 v} 1/(v での分岐数)
    
    これは「ランダムウォーク」確率に対応
    """
    
    def __init__(self, graph: MultiwayGraph):
        self.graph = graph
        # 各ノードの分岐数をキャッシュ
        self._branching_factors: Dict[str, int] = {}
        for node in graph.nodes.values():
            self._branching_factors[node.node_id] = max(len(node.children), 1)
    
    def path_weight(self, path: Path) -> float:
        weight = 1.0
        for node in path.nodes[:-1]:  # 終端以外
            branching = self._branching_factors.get(node.node_id, 1)
            weight /= branching
        return weight
    
    @property
    def name(self) -> str:
        return "BranchWeighted"


class ReductionTypeWeightedModel(ProbabilityModel):
    """
    簡約タイプ重みモデル: S簡約とK簡約に異なる重みを付与
    
    weight(path) = ∏_{edge} w(edge.redex_type)
    
    S簡約とK簡約の「作用」の違いを反映
    """
    
    def __init__(self, s_weight: float = 1.0, k_weight: float = 1.0):
        self.s_weight = s_weight
        self.k_weight = k_weight
    
    def path_weight(self, path: Path) -> float:
        from reduction import RedexType
        
        weight = 1.0
        for edge in path.edges:
            if edge.redex_type == RedexType.S_REDEX:
                weight *= self.s_weight
            else:
                weight *= self.k_weight
        return weight
    
    @property
    def name(self) -> str:
        return f"ReductionTypeWeighted(S={self.s_weight}, K={self.k_weight})"


# =============================================================================
# Probability Calculator
# =============================================================================

@dataclass
class PathProbabilities:
    """パスと確率の対応"""
    path: Path
    weight: float
    probability: float


class ProbabilityCalculator:
    """
    確率計算のためのユーティリティクラス
    """
    
    def __init__(self, graph: MultiwayGraph, model: ProbabilityModel):
        self.graph = graph
        self.model = model
        self.paths = graph.get_all_paths()
        
        # 確率を事前計算
        self._probabilities: Dict[str, PathProbabilities] = {}
        for path in self.paths:
            weight = model.path_weight(path)
            prob = model.path_probability(path, self.paths)
            self._probabilities[path.path_id] = PathProbabilities(path, weight, prob)
    
    def get_path_probability(self, path: Path) -> float:
        """パスの確率を取得"""
        return self._probabilities[path.path_id].probability
    
    def get_terminal_probability(self, terminal: MultiwayNode) -> float:
        """終端への到達確率を取得"""
        return self.model.terminal_probability(terminal, self.paths)
    
    def get_all_probabilities(self) -> List[PathProbabilities]:
        """全パスの確率情報を取得"""
        return list(self._probabilities.values())
    
    def get_paths_to_terminal(self, terminal: MultiwayNode) -> List[PathProbabilities]:
        """特定の終端へのパスの確率情報を取得"""
        terminal_paths = [p for p in self.paths if p.end == terminal]
        return [self._probabilities[p.path_id] for p in terminal_paths]
    
    def summary(self) -> Dict:
        """確率計算の要約"""
        terminal_probs = {}
        for terminal in self.graph.terminals:
            expr_str = to_canonical(terminal.expr)
            terminal_probs[expr_str] = self.get_terminal_probability(terminal)
        
        return {
            "model": self.model.name,
            "total_paths": len(self.paths),
            "terminal_probabilities": terminal_probs,
            "path_probabilities": {
                pp.path.path_id: pp.probability 
                for pp in self._probabilities.values()
            }
        }


# =============================================================================
# Utility Functions
# =============================================================================

def compute_probabilities(
    source: str,
    models: List[ProbabilityModel] = None
) -> Dict[str, ProbabilityCalculator]:
    """
    SK式に対して複数の確率モデルで確率を計算
    
    Args:
        source: SK式の文字列
        models: 使用する確率モデルのリスト
    
    Returns:
        {モデル名: ProbabilityCalculator} の辞書
    """
    from sk_parser import parse
    
    expr = parse(source)
    graph = build_multiway_graph(expr)
    
    if models is None:
        models = [
            UniformModel(),
            LengthWeightedModel(alpha=0.5),
            ComplexityWeightedModel(beta=0.1),
            BranchWeightedModel(graph),
        ]
    
    return {
        model.name: ProbabilityCalculator(graph, model)
        for model in models
    }


def print_probability_comparison(source: str) -> None:
    """
    複数の確率モデルの比較を表示
    """
    from sk_parser import parse
    
    print(f"Expression: {source}")
    print("=" * 60)
    
    calculators = compute_probabilities(source)
    
    for name, calc in calculators.items():
        print(f"\n{name}:")
        summary = calc.summary()
        
        print(f"  Terminal probabilities:")
        for terminal, prob in summary["terminal_probabilities"].items():
            print(f"    {terminal}: {prob:.4f}")
        
        print(f"  Path probabilities:")
        for path_id, prob in summary["path_probabilities"].items():
            print(f"    {path_id}: {prob:.4f}")


# =============================================================================
# Main (for testing)
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("SK Probability Models Test - Day 5")
    print("=" * 70)
    
    test_cases = [
        "K a b",
        "(K a b) (K c d)",
        "S (K a) (K b) c",
        "S (K a b) c d",
    ]
    
    for source in test_cases:
        print(f"\n{'='*70}")
        print_probability_comparison(source)
    
    print(f"\n{'='*70}")
    print("All tests completed!")
    print(f"{'='*70}")




