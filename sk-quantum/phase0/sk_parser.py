"""
SK Combinator Parser
====================

Day 1 実装: SK式のAST定義とパーサ

SK計算の基本規則:
- S x y z → x z (y z)
- K x y → x

AST構造:
- S: Sコンビネータ
- K: Kコンビネータ
- Var(name): 変数（小文字アルファベット）
- App(func, arg): 関数適用（左結合）

構文例:
- "S" → S
- "K" → K
- "a" → Var("a")
- "S K" → App(S, K)
- "S K K" → App(App(S, K), K)
- "(S K) K" → App(App(S, K), K)
- "S (K K)" → App(S, App(K, K))
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Union, List, Tuple
from enum import Enum, auto


# =============================================================================
# AST Definition
# =============================================================================

class SKExpr:
    """SK式の基底クラス"""
    pass


@dataclass(frozen=True)
class S(SKExpr):
    """Sコンビネータ: S x y z → x z (y z)"""
    def __repr__(self) -> str:
        return "S"
    
    def __str__(self) -> str:
        return "S"


@dataclass(frozen=True)
class K(SKExpr):
    """Kコンビネータ: K x y → x"""
    def __repr__(self) -> str:
        return "K"
    
    def __str__(self) -> str:
        return "K"


@dataclass(frozen=True)
class Var(SKExpr):
    """変数（テスト・実験用）"""
    name: str
    
    def __repr__(self) -> str:
        return f"Var({self.name!r})"
    
    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True)
class App(SKExpr):
    """関数適用: (func arg)"""
    func: SKExpr
    arg: SKExpr
    
    def __repr__(self) -> str:
        return f"App({self.func!r}, {self.arg!r})"
    
    def __str__(self) -> str:
        # 可読性のための文字列表現
        func_str = str(self.func)
        arg_str = str(self.arg)
        
        # 引数が App の場合は括弧で囲む
        if isinstance(self.arg, App):
            arg_str = f"({arg_str})"
        
        return f"{func_str} {arg_str}"


# 型エイリアス
Expr = Union[S, K, Var, App]


# =============================================================================
# Tokenizer
# =============================================================================

class TokenType(Enum):
    S = auto()       # S
    K = auto()       # K
    VAR = auto()     # 小文字変数
    LPAREN = auto()  # (
    RPAREN = auto()  # )
    EOF = auto()     # 終端


@dataclass
class Token:
    type: TokenType
    value: str
    position: int  # デバッグ用


def tokenize(source: str) -> List[Token]:
    """
    SK式をトークン列に変換
    
    Args:
        source: SK式の文字列
    
    Returns:
        トークンのリスト
    
    Raises:
        ValueError: 不正な文字が含まれる場合
    """
    tokens: List[Token] = []
    pos = 0
    
    while pos < len(source):
        char = source[pos]
        
        # 空白をスキップ
        if char.isspace():
            pos += 1
            continue
        
        # Sコンビネータ
        if char == 'S':
            tokens.append(Token(TokenType.S, 'S', pos))
            pos += 1
            continue
        
        # Kコンビネータ
        if char == 'K':
            tokens.append(Token(TokenType.K, 'K', pos))
            pos += 1
            continue
        
        # 左括弧
        if char == '(':
            tokens.append(Token(TokenType.LPAREN, '(', pos))
            pos += 1
            continue
        
        # 右括弧
        if char == ')':
            tokens.append(Token(TokenType.RPAREN, ')', pos))
            pos += 1
            continue
        
        # 小文字変数
        if char.islower():
            tokens.append(Token(TokenType.VAR, char, pos))
            pos += 1
            continue
        
        # 不正な文字
        raise ValueError(f"Unexpected character '{char}' at position {pos}")
    
    tokens.append(Token(TokenType.EOF, '', pos))
    return tokens


# =============================================================================
# Parser
# =============================================================================

class Parser:
    """
    SK式のパーサ（再帰下降構文解析）
    
    文法:
        expr    := atom+
        atom    := S | K | VAR | '(' expr ')'
    
    Application は左結合:
        S K K = (S K) K
    """
    
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.pos = 0
    
    def current(self) -> Token:
        """現在のトークンを取得"""
        return self.tokens[self.pos]
    
    def advance(self) -> Token:
        """次のトークンに進む"""
        token = self.current()
        if self.pos < len(self.tokens) - 1:
            self.pos += 1
        return token
    
    def parse(self) -> SKExpr:
        """
        SK式をパース
        
        Returns:
            パースされたAST
        
        Raises:
            ValueError: 構文エラーの場合
        """
        expr = self.parse_expr()
        
        if self.current().type != TokenType.EOF:
            raise ValueError(f"Unexpected token {self.current()} after expression")
        
        return expr
    
    def parse_expr(self) -> SKExpr:
        """
        式をパース（左結合のApplication）
        
        expr := atom+
        """
        atoms: List[SKExpr] = []
        
        while self.current().type in (TokenType.S, TokenType.K, TokenType.VAR, TokenType.LPAREN):
            atoms.append(self.parse_atom())
        
        if not atoms:
            raise ValueError(f"Expected expression, got {self.current()}")
        
        # 左結合でApplicationを構築
        result = atoms[0]
        for atom in atoms[1:]:
            result = App(result, atom)
        
        return result
    
    def parse_atom(self) -> SKExpr:
        """
        アトムをパース
        
        atom := S | K | VAR | '(' expr ')'
        """
        token = self.current()
        
        if token.type == TokenType.S:
            self.advance()
            return S()
        
        if token.type == TokenType.K:
            self.advance()
            return K()
        
        if token.type == TokenType.VAR:
            self.advance()
            return Var(token.value)
        
        if token.type == TokenType.LPAREN:
            self.advance()  # '(' を消費
            expr = self.parse_expr()
            
            if self.current().type != TokenType.RPAREN:
                raise ValueError(f"Expected ')', got {self.current()}")
            
            self.advance()  # ')' を消費
            return expr
        
        raise ValueError(f"Unexpected token {token}")


# =============================================================================
# Public API
# =============================================================================

def parse(source: str) -> SKExpr:
    """
    SK式文字列をASTにパース
    
    Args:
        source: SK式の文字列
    
    Returns:
        パースされたAST
    
    Examples:
        >>> parse("S")
        S
        >>> parse("K")
        K
        >>> parse("S K")
        App(S, K)
        >>> parse("S K K")
        App(App(S, K), K)
        >>> parse("S (K K)")
        App(S, App(K, K))
    """
    tokens = tokenize(source)
    parser = Parser(tokens)
    return parser.parse()


def to_string(expr: SKExpr) -> str:
    """
    ASTを文字列に変換（可読形式）
    
    Args:
        expr: SK式のAST
    
    Returns:
        SK式の文字列表現
    """
    return str(expr)


def to_canonical(expr: SKExpr) -> str:
    """
    ASTを正規形文字列に変換（括弧を明示）
    
    Args:
        expr: SK式のAST
    
    Returns:
        完全に括弧付けされた文字列
    """
    if isinstance(expr, S):
        return "S"
    elif isinstance(expr, K):
        return "K"
    elif isinstance(expr, Var):
        return expr.name
    elif isinstance(expr, App):
        return f"({to_canonical(expr.func)} {to_canonical(expr.arg)})"
    else:
        raise TypeError(f"Unknown expression type: {type(expr)}")


# =============================================================================
# Utility Functions
# =============================================================================

def size(expr: SKExpr) -> int:
    """
    式のサイズ（ノード数）を計算
    
    Args:
        expr: SK式のAST
    
    Returns:
        ASTのノード数
    """
    if isinstance(expr, (S, K, Var)):
        return 1
    elif isinstance(expr, App):
        return 1 + size(expr.func) + size(expr.arg)
    else:
        raise TypeError(f"Unknown expression type: {type(expr)}")


def depth(expr: SKExpr) -> int:
    """
    式の深さを計算
    
    Args:
        expr: SK式のAST
    
    Returns:
        ASTの深さ
    """
    if isinstance(expr, (S, K, Var)):
        return 1
    elif isinstance(expr, App):
        return 1 + max(depth(expr.func), depth(expr.arg))
    else:
        raise TypeError(f"Unknown expression type: {type(expr)}")


def variables(expr: SKExpr) -> set:
    """
    式に含まれる変数を収集
    
    Args:
        expr: SK式のAST
    
    Returns:
        変数名の集合
    """
    if isinstance(expr, S) or isinstance(expr, K):
        return set()
    elif isinstance(expr, Var):
        return {expr.name}
    elif isinstance(expr, App):
        return variables(expr.func) | variables(expr.arg)
    else:
        raise TypeError(f"Unknown expression type: {type(expr)}")


def substitute(expr: SKExpr, var_name: str, replacement: SKExpr) -> SKExpr:
    """
    変数を式で置換
    
    Args:
        expr: 元の式
        var_name: 置換する変数名
        replacement: 置換後の式
    
    Returns:
        置換後の式
    """
    if isinstance(expr, S) or isinstance(expr, K):
        return expr
    elif isinstance(expr, Var):
        if expr.name == var_name:
            return replacement
        return expr
    elif isinstance(expr, App):
        return App(
            substitute(expr.func, var_name, replacement),
            substitute(expr.arg, var_name, replacement)
        )
    else:
        raise TypeError(f"Unknown expression type: {type(expr)}")


# =============================================================================
# Main (for testing)
# =============================================================================

if __name__ == "__main__":
    # 基本テスト
    test_cases = [
        "S",
        "K",
        "a",
        "S K",
        "K S",
        "S K K",
        "S S S",
        "(S K)",
        "(S K) K",
        "S (K K)",
        "S (K a) (K b)",
        "((S K) K)",
        "S (S (S K))",
    ]
    
    print("=" * 60)
    print("SK Parser Test - Day 1")
    print("=" * 60)
    
    for source in test_cases:
        try:
            expr = parse(source)
            print(f"\nInput:     {source!r}")
            print(f"AST:       {expr!r}")
            print(f"String:    {to_string(expr)}")
            print(f"Canonical: {to_canonical(expr)}")
            print(f"Size:      {size(expr)}")
            print(f"Depth:     {depth(expr)}")
            if variables(expr):
                print(f"Variables: {variables(expr)}")
        except Exception as e:
            print(f"\nInput: {source!r}")
            print(f"Error: {e}")
    
    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)

