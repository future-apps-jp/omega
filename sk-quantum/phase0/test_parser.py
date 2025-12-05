"""
SK Parser Tests
===============

Day 1: SK式パーサのユニットテスト

実行方法:
    python -m pytest test_parser.py -v
    または
    python test_parser.py
"""

import unittest
from sk_parser import (
    S, K, Var, App,
    parse, to_string, to_canonical,
    size, depth, variables, substitute,
    tokenize, TokenType
)


class TestTokenizer(unittest.TestCase):
    """トークナイザのテスト"""
    
    def test_tokenize_s(self):
        """Sコンビネータのトークン化"""
        tokens = tokenize("S")
        self.assertEqual(len(tokens), 2)  # S + EOF
        self.assertEqual(tokens[0].type, TokenType.S)
    
    def test_tokenize_k(self):
        """Kコンビネータのトークン化"""
        tokens = tokenize("K")
        self.assertEqual(len(tokens), 2)
        self.assertEqual(tokens[0].type, TokenType.K)
    
    def test_tokenize_var(self):
        """変数のトークン化"""
        tokens = tokenize("a")
        self.assertEqual(len(tokens), 2)
        self.assertEqual(tokens[0].type, TokenType.VAR)
        self.assertEqual(tokens[0].value, "a")
    
    def test_tokenize_parens(self):
        """括弧のトークン化"""
        tokens = tokenize("(S K)")
        self.assertEqual(len(tokens), 5)  # ( S K ) EOF
        self.assertEqual(tokens[0].type, TokenType.LPAREN)
        self.assertEqual(tokens[3].type, TokenType.RPAREN)
    
    def test_tokenize_whitespace(self):
        """空白の処理"""
        tokens = tokenize("  S   K  ")
        self.assertEqual(len(tokens), 3)  # S K EOF
    
    def test_tokenize_invalid_char(self):
        """不正な文字のエラー"""
        with self.assertRaises(ValueError):
            tokenize("S + K")


class TestParserBasic(unittest.TestCase):
    """パーサの基本テスト"""
    
    def test_parse_s(self):
        """Sのパース"""
        expr = parse("S")
        self.assertIsInstance(expr, S)
    
    def test_parse_k(self):
        """Kのパース"""
        expr = parse("K")
        self.assertIsInstance(expr, K)
    
    def test_parse_var(self):
        """変数のパース"""
        expr = parse("a")
        self.assertIsInstance(expr, Var)
        self.assertEqual(expr.name, "a")
    
    def test_parse_simple_app(self):
        """単純なApplicationのパース"""
        expr = parse("S K")
        self.assertIsInstance(expr, App)
        self.assertIsInstance(expr.func, S)
        self.assertIsInstance(expr.arg, K)


class TestParserLeftAssociativity(unittest.TestCase):
    """左結合のテスト"""
    
    def test_left_associativity_3(self):
        """3項の左結合: S K K = (S K) K"""
        expr = parse("S K K")
        # (S K) K
        self.assertIsInstance(expr, App)
        self.assertIsInstance(expr.func, App)  # (S K)
        self.assertIsInstance(expr.arg, K)
        self.assertIsInstance(expr.func.func, S)
        self.assertIsInstance(expr.func.arg, K)
    
    def test_left_associativity_4(self):
        """4項の左結合: S K K K = ((S K) K) K"""
        expr = parse("S K K K")
        # ((S K) K) K
        self.assertIsInstance(expr, App)
        self.assertIsInstance(expr.arg, K)
        self.assertIsInstance(expr.func, App)  # (S K) K
        self.assertIsInstance(expr.func.func, App)  # S K


class TestParserParentheses(unittest.TestCase):
    """括弧のテスト"""
    
    def test_redundant_parens(self):
        """冗長な括弧"""
        expr1 = parse("(S)")
        expr2 = parse("S")
        self.assertEqual(repr(expr1), repr(expr2))
    
    def test_parens_change_associativity(self):
        """括弧による結合順序の変更"""
        # S K K = (S K) K
        expr1 = parse("S K K")
        # S (K K) ≠ S K K
        expr2 = parse("S (K K)")
        
        # expr1: (S K) K
        self.assertIsInstance(expr1.func, App)
        self.assertIsInstance(expr1.arg, K)
        
        # expr2: S (K K)
        self.assertIsInstance(expr2.func, S)
        self.assertIsInstance(expr2.arg, App)
    
    def test_nested_parens(self):
        """ネストした括弧"""
        expr = parse("((S K) K)")
        self.assertEqual(to_canonical(expr), "((S K) K)")
    
    def test_complex_parens(self):
        """複雑な括弧"""
        expr = parse("S (K a) (K b)")
        # (S (K a)) (K b)
        self.assertIsInstance(expr, App)
        self.assertIsInstance(expr.arg, App)  # K b
        self.assertIsInstance(expr.func, App)  # S (K a)


class TestToString(unittest.TestCase):
    """文字列変換のテスト"""
    
    def test_to_string_simple(self):
        """単純な式の文字列化"""
        self.assertEqual(to_string(parse("S")), "S")
        self.assertEqual(to_string(parse("K")), "K")
        self.assertEqual(to_string(parse("a")), "a")
    
    def test_to_string_app(self):
        """Applicationの文字列化"""
        self.assertEqual(to_string(parse("S K")), "S K")
        self.assertEqual(to_string(parse("S K K")), "S K K")
    
    def test_to_string_nested(self):
        """ネストした式の文字列化（括弧付き）"""
        expr = parse("S (K K)")
        self.assertEqual(to_string(expr), "S (K K)")


class TestToCanonical(unittest.TestCase):
    """正規形文字列のテスト"""
    
    def test_canonical_simple(self):
        """単純な式の正規形"""
        self.assertEqual(to_canonical(parse("S")), "S")
        self.assertEqual(to_canonical(parse("K")), "K")
    
    def test_canonical_app(self):
        """Applicationの正規形（完全括弧付け）"""
        self.assertEqual(to_canonical(parse("S K")), "(S K)")
        self.assertEqual(to_canonical(parse("S K K")), "((S K) K)")
        self.assertEqual(to_canonical(parse("S (K K)")), "(S (K K))")


class TestSize(unittest.TestCase):
    """サイズ計算のテスト"""
    
    def test_size_atom(self):
        """アトムのサイズ"""
        self.assertEqual(size(parse("S")), 1)
        self.assertEqual(size(parse("K")), 1)
        self.assertEqual(size(parse("a")), 1)
    
    def test_size_app(self):
        """Applicationのサイズ"""
        self.assertEqual(size(parse("S K")), 3)  # App + S + K
        self.assertEqual(size(parse("S K K")), 5)  # App + (App + S + K) + K


class TestDepth(unittest.TestCase):
    """深さ計算のテスト"""
    
    def test_depth_atom(self):
        """アトムの深さ"""
        self.assertEqual(depth(parse("S")), 1)
    
    def test_depth_app(self):
        """Applicationの深さ"""
        self.assertEqual(depth(parse("S K")), 2)
        self.assertEqual(depth(parse("S K K")), 3)
        self.assertEqual(depth(parse("S (K K)")), 3)


class TestVariables(unittest.TestCase):
    """変数収集のテスト"""
    
    def test_no_variables(self):
        """変数なし"""
        self.assertEqual(variables(parse("S K K")), set())
    
    def test_single_variable(self):
        """単一変数"""
        self.assertEqual(variables(parse("S a")), {"a"})
    
    def test_multiple_variables(self):
        """複数変数"""
        self.assertEqual(variables(parse("S (K a) (K b)")), {"a", "b"})


class TestSubstitute(unittest.TestCase):
    """置換のテスト"""
    
    def test_substitute_var(self):
        """変数の置換"""
        expr = parse("a")
        result = substitute(expr, "a", parse("K"))
        self.assertIsInstance(result, K)
    
    def test_substitute_in_app(self):
        """Application内の変数置換"""
        expr = parse("S a")
        result = substitute(expr, "a", parse("K"))
        self.assertEqual(to_canonical(result), "(S K)")
    
    def test_substitute_complex(self):
        """複雑な置換"""
        expr = parse("S (K a) (K b)")
        result = substitute(expr, "a", parse("S"))
        self.assertIn("S", to_string(result))


class TestResearchExamples(unittest.TestCase):
    """研究計画書の例のテスト"""
    
    def test_research_example_1(self):
        """研究例: S (K a) (K b)"""
        expr = parse("S (K a) (K b)")
        self.assertEqual(to_canonical(expr), "((S (K a)) (K b))")
        self.assertEqual(size(expr), 9)
    
    def test_research_example_2(self):
        """研究例: S S S"""
        expr = parse("S S S")
        self.assertEqual(to_canonical(expr), "((S S) S)")


class TestEdgeCases(unittest.TestCase):
    """エッジケースのテスト"""
    
    def test_empty_string(self):
        """空文字列"""
        with self.assertRaises(ValueError):
            parse("")
    
    def test_unmatched_lparen(self):
        """閉じ括弧なし"""
        with self.assertRaises(ValueError):
            parse("(S K")
    
    def test_unmatched_rparen(self):
        """開き括弧なし"""
        with self.assertRaises(ValueError):
            parse("S K)")
    
    def test_empty_parens(self):
        """空の括弧"""
        with self.assertRaises(ValueError):
            parse("()")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    # 簡易テスト実行
    print("Running SK Parser Tests...")
    print("=" * 60)
    
    # unittest を実行
    unittest.main(verbosity=2)

