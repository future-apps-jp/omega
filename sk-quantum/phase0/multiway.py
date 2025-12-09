"""
SK Combinator Multiway Graph
============================

Day 3-4 実装: 全計算パスの列挙（Multiway Graph）

Multiway Graphとは:
- SK計算で複数のRedexがある場合、どのRedexを先に簡約するかで
  異なる計算パスが生まれる
- 全ての可能な簡約順序を木構造で表現したもの
- Sorkin公式で「経路A, B, C」として扱われる

用語:
- Path: 初期状態から終状態への簡約列
- Branch: 分岐点（複数のRedexが存在する状態）
- Terminal: 正規形（これ以上簡約できない状態）
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Tuple, Iterator
from collections import defaultdict
import hashlib

from sk_parser import SKExpr, S, K, Var, App, parse, to_string, to_canonical
from reduction import (
    find_redexes, reduce_at_path, is_normal_form,
    RedexType, Redex
)


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class MultiwayNode:
    """
    Multiway Graphのノード（計算状態）
    
    Attributes:
        expr: この状態のSK式
        node_id: ユニークID（式のハッシュ）
        is_terminal: 正規形かどうか
        children: 子ノードへの辺（redex_path -> child_node）
    """
    expr: SKExpr
    node_id: str = field(default="", init=False)
    is_terminal: bool = field(default=False, init=False)
    children: Dict[str, MultiwayNode] = field(default_factory=dict, init=False)
    redex_info: Dict[str, Redex] = field(default_factory=dict, init=False)
    
    def __post_init__(self):
        self.node_id = self._compute_id()
        self.is_terminal = is_normal_form(self.expr)
    
    def _compute_id(self) -> str:
        """式からユニークIDを計算"""
        canonical = to_canonical(self.expr)
        return hashlib.md5(canonical.encode()).hexdigest()[:8]
    
    def __hash__(self):
        return hash(self.node_id)
    
    def __eq__(self, other):
        if not isinstance(other, MultiwayNode):
            return False
        return self.node_id == other.node_id


@dataclass
class ReductionEdge:
    """
    簡約による遷移（辺）
    
    Attributes:
        source: 元のノード
        target: 簡約後のノード
        redex_path: 簡約したRedexのパス
        redex_type: Redexの種類（S or K）
    """
    source: MultiwayNode
    target: MultiwayNode
    redex_path: str
    redex_type: RedexType


@dataclass
class Path:
    """
    初期状態から終状態への計算パス
    
    Attributes:
        nodes: パス上のノード列
        edges: パス上の辺列
        path_id: パスのユニークID
    """
    nodes: List[MultiwayNode]
    edges: List[ReductionEdge]
    
    @property
    def path_id(self) -> str:
        """パスのID（辺のredex_pathを連結）"""
        return "-".join(e.redex_path or "root" for e in self.edges)
    
    @property
    def start(self) -> MultiwayNode:
        return self.nodes[0]
    
    @property
    def end(self) -> MultiwayNode:
        return self.nodes[-1]
    
    @property
    def length(self) -> int:
        return len(self.edges)
    
    def __repr__(self) -> str:
        steps = [to_string(self.nodes[0].expr)]
        for edge in self.edges:
            rtype = "S" if edge.redex_type == RedexType.S_REDEX else "K"
            steps.append(f"--[{rtype}@{edge.redex_path}]-->")
            steps.append(to_string(edge.target.expr))
        return " ".join(steps)


class MultiwayGraph:
    """
    SK計算のMultiway Graph
    
    全ての可能な簡約パスを保持する構造
    
    Attributes:
        root: 初期状態のノード
        nodes: 全ノードの集合（node_id -> node）
        terminals: 終端ノード（正規形）の集合
    """
    
    def __init__(self, root_expr: SKExpr):
        self.root = MultiwayNode(root_expr)
        self.nodes: Dict[str, MultiwayNode] = {self.root.node_id: self.root}
        self.terminals: Set[MultiwayNode] = set()
        
        if self.root.is_terminal:
            self.terminals.add(self.root)
    
    def add_node(self, expr: SKExpr) -> MultiwayNode:
        """ノードを追加（既存なら再利用）"""
        node = MultiwayNode(expr)
        if node.node_id in self.nodes:
            return self.nodes[node.node_id]
        
        self.nodes[node.node_id] = node
        if node.is_terminal:
            self.terminals.add(node)
        
        return node
    
    def add_edge(self, source: MultiwayNode, target: MultiwayNode, 
                 redex_path: str, redex: Redex) -> None:
        """辺を追加"""
        source.children[redex_path] = target
        source.redex_info[redex_path] = redex
    
    def get_all_paths(self) -> List[Path]:
        """
        ルートから全終端への全パスを列挙
        
        Returns:
            パスのリスト
        """
        all_paths: List[Path] = []
        
        def dfs(current: MultiwayNode, 
                nodes: List[MultiwayNode], 
                edges: List[ReductionEdge]) -> None:
            
            if current.is_terminal:
                all_paths.append(Path(nodes.copy(), edges.copy()))
                return
            
            for redex_path, child in current.children.items():
                redex = current.redex_info[redex_path]
                edge = ReductionEdge(
                    source=current,
                    target=child,
                    redex_path=redex_path,
                    redex_type=redex.type
                )
                nodes.append(child)
                edges.append(edge)
                dfs(child, nodes, edges)
                nodes.pop()
                edges.pop()
        
        dfs(self.root, [self.root], [])
        return all_paths
    
    def get_paths_to(self, terminal: MultiwayNode) -> List[Path]:
        """
        特定の終端への全パスを取得
        
        Args:
            terminal: 終端ノード
        
        Returns:
            その終端へのパスのリスト
        """
        return [p for p in self.get_all_paths() if p.end == terminal]
    
    def get_terminal_expressions(self) -> List[SKExpr]:
        """全終端の式を取得"""
        return [t.expr for t in self.terminals]
    
    def statistics(self) -> Dict[str, int]:
        """グラフの統計情報"""
        all_paths = self.get_all_paths()
        return {
            "total_nodes": len(self.nodes),
            "terminal_nodes": len(self.terminals),
            "total_paths": len(all_paths),
            "max_path_length": max((p.length for p in all_paths), default=0),
            "min_path_length": min((p.length for p in all_paths), default=0),
        }


# =============================================================================
# Graph Construction
# =============================================================================

def build_multiway_graph(
    expr: SKExpr,
    max_depth: int = 100,
    max_nodes: int = 10000
) -> MultiwayGraph:
    """
    SK式からMultiway Graphを構築
    
    Args:
        expr: 初期のSK式
        max_depth: 最大探索深さ
        max_nodes: 最大ノード数
    
    Returns:
        構築されたMultiwayGraph
    
    Raises:
        RuntimeError: 制限を超えた場合
    """
    graph = MultiwayGraph(expr)
    
    # BFS で探索
    queue: List[Tuple[MultiwayNode, int]] = [(graph.root, 0)]
    visited: Set[str] = {graph.root.node_id}
    
    while queue:
        current, depth = queue.pop(0)
        
        if depth >= max_depth:
            continue
        
        if len(graph.nodes) >= max_nodes:
            raise RuntimeError(f"Exceeded max nodes limit: {max_nodes}")
        
        if current.is_terminal:
            continue
        
        # 全Redexを探索
        redexes = find_redexes(current.expr)
        
        for redex in redexes:
            # このRedexで簡約
            try:
                new_expr = reduce_at_path(current.expr, redex.path)
            except Exception:
                continue
            
            # ノードを追加/取得
            child = graph.add_node(new_expr)
            graph.add_edge(current, child, redex.path, redex)
            
            # 未訪問なら探索キューに追加
            if child.node_id not in visited:
                visited.add(child.node_id)
                queue.append((child, depth + 1))
    
    return graph


# =============================================================================
# Utility Functions
# =============================================================================

def print_multiway_graph(graph: MultiwayGraph, max_paths: int = 20) -> None:
    """
    Multiway Graphを表示
    
    Args:
        graph: 表示するグラフ
        max_paths: 表示する最大パス数
    """
    stats = graph.statistics()
    
    print(f"Initial: {to_string(graph.root.expr)}")
    print(f"         {to_canonical(graph.root.expr)}")
    print()
    print(f"Statistics:")
    print(f"  Total nodes: {stats['total_nodes']}")
    print(f"  Terminal nodes: {stats['terminal_nodes']}")
    print(f"  Total paths: {stats['total_paths']}")
    print(f"  Path length: {stats['min_path_length']} - {stats['max_path_length']}")
    print()
    
    # 終端ごとにパスを表示
    print("Terminals:")
    for terminal in graph.terminals:
        paths = graph.get_paths_to(terminal)
        print(f"\n  → {to_string(terminal.expr)} ({len(paths)} paths)")
        
        for i, path in enumerate(paths[:max_paths]):
            if i >= max_paths:
                print(f"      ... and {len(paths) - max_paths} more paths")
                break
            
            # パスIDを表示
            print(f"      Path {i+1}: {path.path_id}")


def enumerate_paths(source: str, max_depth: int = 50) -> List[Path]:
    """
    SK式文字列から全パスを列挙
    
    Args:
        source: SK式の文字列
        max_depth: 最大探索深さ
    
    Returns:
        全パスのリスト
    """
    expr = parse(source)
    graph = build_multiway_graph(expr, max_depth=max_depth)
    return graph.get_all_paths()


def count_paths_to_terminals(source: str) -> Dict[str, int]:
    """
    各終端への経路数をカウント
    
    Args:
        source: SK式の文字列
    
    Returns:
        {終端の式: 経路数} の辞書
    """
    expr = parse(source)
    graph = build_multiway_graph(expr)
    
    result = {}
    for terminal in graph.terminals:
        paths = graph.get_paths_to(terminal)
        result[to_canonical(terminal.expr)] = len(paths)
    
    return result


# =============================================================================
# Visualization (Text-based)
# =============================================================================

def visualize_as_tree(graph: MultiwayGraph, max_depth: int = 5) -> str:
    """
    グラフをテキストツリーとして可視化
    
    Args:
        graph: 可視化するグラフ
        max_depth: 表示する最大深さ
    
    Returns:
        ツリー形式の文字列
    """
    lines = []
    
    def render(node: MultiwayNode, prefix: str, depth: int) -> None:
        if depth > max_depth:
            lines.append(f"{prefix}...")
            return
        
        expr_str = to_string(node.expr)
        terminal_mark = " ●" if node.is_terminal else ""
        lines.append(f"{prefix}{expr_str}{terminal_mark}")
        
        children = list(node.children.items())
        for i, (redex_path, child) in enumerate(children):
            is_last = (i == len(children) - 1)
            redex = node.redex_info[redex_path]
            rtype = "S" if redex.type == RedexType.S_REDEX else "K"
            
            connector = "└─" if is_last else "├─"
            new_prefix = prefix.replace("├─", "│ ").replace("└─", "  ")
            new_prefix += "  │ " if not is_last else "    "
            
            lines.append(f"{prefix}{'  │' if not is_last else '   '}")
            lines.append(f"{prefix}{connector}[{rtype}@{redex_path or 'root'}]")
            render(child, new_prefix.rstrip() + " ", depth + 1)
    
    render(graph.root, "", 0)
    return "\n".join(lines)


# =============================================================================
# Main (for testing)
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("SK Multiway Graph Test - Day 3-4")
    print("=" * 70)
    
    # テストケース
    test_cases = [
        # 単純なケース（分岐なし）
        ("K a b", "K-reduction only"),
        
        # S-reduction
        ("S K K a", "SKK identity"),
        
        # 複数のRedexによる分岐
        ("(K a b) (K c d)", "Two K-redexes"),
        
        # 研究例
        ("S (K a) (K b) c", "Research example"),
        
        # より複雑な例
        ("S (K a b) c d", "Nested redexes"),
    ]
    
    for source, description in test_cases:
        print(f"\n{'='*70}")
        print(f"Test: {description}")
        print(f"Expression: {source}")
        print(f"{'='*70}")
        
        expr = parse(source)
        graph = build_multiway_graph(expr)
        
        print_multiway_graph(graph)
        
        print("\nTree visualization:")
        print(visualize_as_tree(graph))
    
    # パス数のカウント
    print(f"\n{'='*70}")
    print("Path Count Summary")
    print(f"{'='*70}")
    
    for source, description in test_cases:
        counts = count_paths_to_terminals(source)
        total = sum(counts.values())
        print(f"\n{source}:")
        print(f"  Total paths: {total}")
        for terminal, count in counts.items():
            print(f"  → {terminal}: {count} paths")
    
    print(f"\n{'='*70}")
    print("All tests completed!")
    print(f"{'='*70}")




