"""
SK Multiway Graph Tests
=======================

Day 3-4: Multiway Graphのユニットテスト

実行方法:
    python3 test_multiway.py
"""

import unittest
from sk_parser import parse, to_canonical
from multiway import (
    MultiwayNode, MultiwayGraph, Path,
    build_multiway_graph, enumerate_paths, count_paths_to_terminals
)


class TestMultiwayNode(unittest.TestCase):
    """MultiwayNodeのテスト"""
    
    def test_node_id_unique(self):
        """同じ式は同じIDを持つ"""
        expr1 = parse("S K K")
        expr2 = parse("S K K")
        node1 = MultiwayNode(expr1)
        node2 = MultiwayNode(expr2)
        self.assertEqual(node1.node_id, node2.node_id)
    
    def test_different_expr_different_id(self):
        """異なる式は異なるIDを持つ"""
        node1 = MultiwayNode(parse("S K K"))
        node2 = MultiwayNode(parse("K S K"))
        self.assertNotEqual(node1.node_id, node2.node_id)
    
    def test_terminal_detection(self):
        """正規形の検出"""
        # 正規形
        self.assertTrue(MultiwayNode(parse("S")).is_terminal)
        self.assertTrue(MultiwayNode(parse("S K")).is_terminal)
        self.assertTrue(MultiwayNode(parse("S K K")).is_terminal)
        
        # 非正規形（Redexあり）
        self.assertFalse(MultiwayNode(parse("K a b")).is_terminal)
        self.assertFalse(MultiwayNode(parse("S a b c")).is_terminal)


class TestMultiwayGraphBasic(unittest.TestCase):
    """MultiwayGraphの基本テスト"""
    
    def test_single_step_graph(self):
        """1ステップで終了するグラフ"""
        graph = build_multiway_graph(parse("K a b"))
        
        self.assertEqual(len(graph.nodes), 2)  # 初期 + 終端
        self.assertEqual(len(graph.terminals), 1)
        
        paths = graph.get_all_paths()
        self.assertEqual(len(paths), 1)
        self.assertEqual(paths[0].length, 1)
    
    def test_already_normal_form(self):
        """既に正規形の場合"""
        graph = build_multiway_graph(parse("S K K"))
        
        self.assertEqual(len(graph.nodes), 1)
        self.assertEqual(len(graph.terminals), 1)
        self.assertTrue(graph.root.is_terminal)
        
        paths = graph.get_all_paths()
        self.assertEqual(len(paths), 1)
        self.assertEqual(paths[0].length, 0)
    
    def test_multi_step_graph(self):
        """複数ステップのグラフ"""
        graph = build_multiway_graph(parse("S K K a"))
        
        # S K K a → K a (K a) → a
        self.assertEqual(len(graph.terminals), 1)
        
        paths = graph.get_all_paths()
        self.assertEqual(len(paths), 1)
        self.assertEqual(paths[0].length, 2)


class TestMultiwayGraphBranching(unittest.TestCase):
    """分岐のテスト"""
    
    def test_two_redexes_branching(self):
        """2つのRedexによる分岐"""
        # (K a b) (K c d) has two K-redexes
        graph = build_multiway_graph(parse("(K a b) (K c d)"))
        
        paths = graph.get_all_paths()
        self.assertEqual(len(paths), 2)  # 2つのパス
        
        # 両方のパスが同じ終端に到達
        terminals = {p.end.node_id for p in paths}
        self.assertEqual(len(terminals), 1)  # 1つの終端
    
    def test_research_example_branching(self):
        """研究例: S (K a) (K b) c"""
        graph = build_multiway_graph(parse("S (K a) (K b) c"))
        
        paths = graph.get_all_paths()
        self.assertEqual(len(paths), 2)  # 2つのパス
        
        # 終端の式を確認
        terminal_exprs = graph.get_terminal_expressions()
        self.assertEqual(len(terminal_exprs), 1)
        self.assertEqual(to_canonical(terminal_exprs[0]), "(a b)")
    
    def test_nested_redexes(self):
        """ネストしたRedex"""
        # S (K a b) c d has S-redex and nested K-redex
        graph = build_multiway_graph(parse("S (K a b) c d"))
        
        paths = graph.get_all_paths()
        self.assertEqual(len(paths), 2)


class TestPathProperties(unittest.TestCase):
    """Pathプロパティのテスト"""
    
    def test_path_length(self):
        """パスの長さ"""
        paths = enumerate_paths("K a b")
        self.assertEqual(len(paths), 1)
        self.assertEqual(paths[0].length, 1)
    
    def test_path_start_end(self):
        """パスの始点と終点"""
        paths = enumerate_paths("K a b")
        path = paths[0]
        
        self.assertEqual(to_canonical(path.start.expr), "((K a) b)")
        self.assertEqual(to_canonical(path.end.expr), "a")
    
    def test_path_id_unique(self):
        """異なるパスは異なるIDを持つ"""
        paths = enumerate_paths("(K a b) (K c d)")
        self.assertEqual(len(paths), 2)
        
        path_ids = {p.path_id for p in paths}
        self.assertEqual(len(path_ids), 2)


class TestCountPathsToTerminals(unittest.TestCase):
    """終端へのパス数カウントのテスト"""
    
    def test_single_terminal(self):
        """単一終端へのパス数"""
        counts = count_paths_to_terminals("K a b")
        self.assertEqual(len(counts), 1)
        self.assertEqual(counts["a"], 1)
    
    def test_multiple_paths_to_same_terminal(self):
        """同じ終端への複数パス"""
        counts = count_paths_to_terminals("(K a b) (K c d)")
        self.assertEqual(len(counts), 1)  # 1つの終端
        self.assertEqual(list(counts.values())[0], 2)  # 2つのパス
    
    def test_research_example(self):
        """研究例のパス数"""
        counts = count_paths_to_terminals("S (K a) (K b) c")
        self.assertEqual(len(counts), 1)
        self.assertEqual(counts["(a b)"], 2)


class TestGraphStatistics(unittest.TestCase):
    """統計情報のテスト"""
    
    def test_statistics_basic(self):
        """基本的な統計"""
        graph = build_multiway_graph(parse("K a b"))
        stats = graph.statistics()
        
        self.assertIn("total_nodes", stats)
        self.assertIn("terminal_nodes", stats)
        self.assertIn("total_paths", stats)
        self.assertIn("max_path_length", stats)
        self.assertIn("min_path_length", stats)
    
    def test_statistics_values(self):
        """統計値の検証"""
        graph = build_multiway_graph(parse("S K K a"))
        stats = graph.statistics()
        
        self.assertEqual(stats["terminal_nodes"], 1)
        self.assertEqual(stats["total_paths"], 1)
        self.assertEqual(stats["max_path_length"], 2)


class TestEdgeCases(unittest.TestCase):
    """エッジケースのテスト"""
    
    def test_atom_graph(self):
        """アトムのみのグラフ"""
        graph = build_multiway_graph(parse("S"))
        
        self.assertEqual(len(graph.nodes), 1)
        self.assertEqual(len(graph.terminals), 1)
        self.assertTrue(graph.root.is_terminal)
    
    def test_complex_expression(self):
        """複雑な式"""
        # これが終了することを確認（無限ループしない）
        graph = build_multiway_graph(parse("S S S a b c"), max_depth=20)
        
        # 少なくとも1つのパスが存在
        paths = graph.get_all_paths()
        self.assertGreater(len(paths), 0)


class TestSorkinPreparation(unittest.TestCase):
    """Sorkin公式の前準備テスト"""
    
    def test_multiple_paths_exist(self):
        """複数パスの存在確認"""
        # Sorkin公式には同じ終端への複数パスが必要
        graph = build_multiway_graph(parse("(K a b) (K c d)"))
        
        for terminal in graph.terminals:
            paths = graph.get_paths_to(terminal)
            self.assertGreater(len(paths), 1, 
                "Need multiple paths for Sorkin formula")
    
    def test_path_differentiation(self):
        """パスの区別が可能"""
        paths = enumerate_paths("(K a b) (K c d)")
        
        # パスのIDで区別可能
        ids = [p.path_id for p in paths]
        self.assertEqual(len(ids), len(set(ids)), "Path IDs should be unique")
        
        # パスの詳細が異なる
        path_details = [
            tuple(e.redex_path for e in p.edges) for p in paths
        ]
        self.assertEqual(len(path_details), len(set(path_details)))


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("Running SK Multiway Graph Tests...")
    print("=" * 60)
    unittest.main(verbosity=2)



