"""
SK Combinator Reduction
=======================

Day 2 実装: β簡約（1ステップ）

SK計算の簡約規則:
- S x y z → x z (y z)   （S-reduction）
- K x y → x             （K-reduction）

用語:
- Redex (Reducible Expression): 簡約可能な部分式
- Normal Form: これ以上簡約できない式
- Head Normal Form: 最外の演算子が簡約できない式
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List, Tuple
from enum import Enum, auto

from sk_parser import SKExpr, S, K, Var, App, parse, to_string, to_canonical


# =============================================================================
# Redex Types
# =============================================================================

class RedexType(Enum):
    """Redexの種類"""
    S_REDEX = auto()  # S x y z
    K_REDEX = auto()  # K x y


@dataclass
class Redex:
    """
    Redex（簡約可能な部分式）の情報
    
    Attributes:
        type: Redexの種類（S or K）
        path: ルートからRedexへのパス（'L'=左, 'R'=右）
        expr: Redex自体の式
    """
    type: RedexType
    path: str  # e.g., "LR" means go Left then Right
    expr: SKExpr


# =============================================================================
# Redex Detection
# =============================================================================

def is_s_redex(expr: SKExpr) -> bool:
    """
    式がS-redex（S x y z の形）かどうか判定
    
    S x y z = ((S x) y) z = App(App(App(S, x), y), z)
    """
    # App(App(App(S, x), y), z) の形をチェック
    if not isinstance(expr, App):
        return False
    
    # expr = App(something, z)
    if not isinstance(expr.func, App):
        return False
    
    # expr.func = App(something2, y)
    if not isinstance(expr.func.func, App):
        return False
    
    # expr.func.func = App(S, x)
    return isinstance(expr.func.func.func, S)


def is_k_redex(expr: SKExpr) -> bool:
    """
    式がK-redex（K x y の形）かどうか判定
    
    K x y = (K x) y = App(App(K, x), y)
    """
    # App(App(K, x), y) の形をチェック
    if not isinstance(expr, App):
        return False
    
    # expr = App(something, y)
    if not isinstance(expr.func, App):
        return False
    
    # expr.func = App(K, x)
    return isinstance(expr.func.func, K)


def find_redexes(expr: SKExpr, path: str = "") -> List[Redex]:
    """
    式中の全Redexを探索
    
    Args:
        expr: 探索対象の式
        path: 現在のパス（ルートからの経路）
    
    Returns:
        見つかったRedexのリスト
    """
    redexes: List[Redex] = []
    
    # S-redex チェック
    if is_s_redex(expr):
        redexes.append(Redex(RedexType.S_REDEX, path, expr))
    
    # K-redex チェック
    if is_k_redex(expr):
        redexes.append(Redex(RedexType.K_REDEX, path, expr))
    
    # 再帰的に子ノードを探索
    if isinstance(expr, App):
        redexes.extend(find_redexes(expr.func, path + "L"))
        redexes.extend(find_redexes(expr.arg, path + "R"))
    
    return redexes


def has_redex(expr: SKExpr) -> bool:
    """式にRedexが存在するかどうか"""
    return len(find_redexes(expr)) > 0


def is_normal_form(expr: SKExpr) -> bool:
    """式が正規形（Redexなし）かどうか"""
    return not has_redex(expr)


# =============================================================================
# Single-Step Reduction
# =============================================================================

def reduce_s_redex(expr: SKExpr) -> SKExpr:
    """
    S-redex を1ステップ簡約
    
    S x y z → x z (y z)
    
    S x y z = App(App(App(S, x), y), z)
    →  App(App(x, z), App(y, z))
    """
    if not is_s_redex(expr):
        raise ValueError("Expression is not an S-redex")
    
    # expr = App(App(App(S, x), y), z)
    z = expr.arg
    y = expr.func.arg
    x = expr.func.func.arg
    
    # x z (y z) = App(App(x, z), App(y, z))
    return App(App(x, z), App(y, z))


def reduce_k_redex(expr: SKExpr) -> SKExpr:
    """
    K-redex を1ステップ簡約
    
    K x y → x
    
    K x y = App(App(K, x), y)
    → x
    """
    if not is_k_redex(expr):
        raise ValueError("Expression is not a K-redex")
    
    # expr = App(App(K, x), y)
    x = expr.func.arg
    
    return x


def reduce_at_path(expr: SKExpr, path: str) -> SKExpr:
    """
    指定されたパスにあるRedexを簡約
    
    Args:
        expr: 元の式
        path: Redexへのパス
    
    Returns:
        簡約後の式
    """
    if path == "":
        # ルートのRedexを簡約
        if is_s_redex(expr):
            return reduce_s_redex(expr)
        elif is_k_redex(expr):
            return reduce_k_redex(expr)
        else:
            raise ValueError("No redex at root")
    
    if not isinstance(expr, App):
        raise ValueError(f"Cannot follow path '{path}' in non-App expression")
    
    direction = path[0]
    rest_path = path[1:]
    
    if direction == "L":
        new_func = reduce_at_path(expr.func, rest_path)
        return App(new_func, expr.arg)
    elif direction == "R":
        new_arg = reduce_at_path(expr.arg, rest_path)
        return App(expr.func, new_arg)
    else:
        raise ValueError(f"Invalid path direction: {direction}")


def reduce_leftmost(expr: SKExpr) -> Optional[SKExpr]:
    """
    最左のRedexを1ステップ簡約（Normal Order Reduction）
    
    Args:
        expr: 元の式
    
    Returns:
        簡約後の式、またはNone（正規形の場合）
    """
    redexes = find_redexes(expr)
    
    if not redexes:
        return None  # 正規形
    
    # パスが最も短い（=最外の）、同じ長さなら最左のRedexを選択
    redexes.sort(key=lambda r: (len(r.path), r.path))
    leftmost = redexes[0]
    
    return reduce_at_path(expr, leftmost.path)


def reduce_outermost(expr: SKExpr) -> Optional[SKExpr]:
    """
    最外のRedexを1ステップ簡約
    
    Args:
        expr: 元の式
    
    Returns:
        簡約後の式、またはNone（正規形の場合）
    """
    redexes = find_redexes(expr)
    
    if not redexes:
        return None
    
    # パスが最も短いRedexを選択
    outermost = min(redexes, key=lambda r: len(r.path))
    
    return reduce_at_path(expr, outermost.path)


def reduce_innermost(expr: SKExpr) -> Optional[SKExpr]:
    """
    最内のRedexを1ステップ簡約（Applicative Order Reduction）
    
    Args:
        expr: 元の式
    
    Returns:
        簡約後の式、またはNone（正規形の場合）
    """
    redexes = find_redexes(expr)
    
    if not redexes:
        return None
    
    # パスが最も長いRedexを選択
    innermost = max(redexes, key=lambda r: len(r.path))
    
    return reduce_at_path(expr, innermost.path)


# =============================================================================
# Multi-Step Reduction
# =============================================================================

@dataclass
class ReductionStep:
    """簡約の1ステップを記録"""
    before: SKExpr
    after: SKExpr
    redex_type: RedexType
    redex_path: str


def reduce_to_normal_form(
    expr: SKExpr,
    strategy: str = "leftmost",
    max_steps: int = 1000
) -> Tuple[SKExpr, List[ReductionStep]]:
    """
    正規形まで簡約（ステップ数制限付き）
    
    Args:
        expr: 元の式
        strategy: 簡約戦略 ("leftmost", "outermost", "innermost")
        max_steps: 最大ステップ数
    
    Returns:
        (正規形, 簡約ステップのリスト)
    
    Raises:
        RuntimeError: 最大ステップ数を超えた場合
    """
    reduce_func = {
        "leftmost": reduce_leftmost,
        "outermost": reduce_outermost,
        "innermost": reduce_innermost,
    }.get(strategy, reduce_leftmost)
    
    steps: List[ReductionStep] = []
    current = expr
    
    for _ in range(max_steps):
        redexes = find_redexes(current)
        if not redexes:
            return current, steps  # 正規形に到達
        
        # 次の式を計算
        next_expr = reduce_func(current)
        if next_expr is None:
            return current, steps
        
        # どのRedexが簡約されたか特定
        for redex in redexes:
            try:
                reduced = reduce_at_path(current, redex.path)
                if repr(reduced) == repr(next_expr):
                    steps.append(ReductionStep(
                        before=current,
                        after=next_expr,
                        redex_type=redex.type,
                        redex_path=redex.path
                    ))
                    break
            except:
                pass
        else:
            # 特定できなかった場合
            steps.append(ReductionStep(
                before=current,
                after=next_expr,
                redex_type=redexes[0].type,
                redex_path=redexes[0].path
            ))
        
        current = next_expr
    
    raise RuntimeError(f"Reduction did not terminate within {max_steps} steps")


# =============================================================================
# Utility Functions
# =============================================================================

def step_by_step(expr: SKExpr, max_steps: int = 20) -> None:
    """
    簡約過程を表示
    
    Args:
        expr: 元の式
        max_steps: 最大ステップ数
    """
    print(f"Initial: {to_string(expr)}")
    print(f"         {to_canonical(expr)}")
    print()
    
    try:
        normal_form, steps = reduce_to_normal_form(expr, max_steps=max_steps)
        
        for i, step in enumerate(steps, 1):
            redex_name = "S" if step.redex_type == RedexType.S_REDEX else "K"
            print(f"Step {i}: {redex_name}-reduction at path '{step.redex_path}'")
            print(f"      → {to_string(step.after)}")
            print(f"        {to_canonical(step.after)}")
            print()
        
        print(f"Normal form: {to_string(normal_form)}")
        print(f"Steps: {len(steps)}")
        
    except RuntimeError as e:
        print(f"Error: {e}")


def evaluate(source: str, max_steps: int = 1000) -> SKExpr:
    """
    SK式を評価（正規形まで簡約）
    
    Args:
        source: SK式の文字列
        max_steps: 最大ステップ数
    
    Returns:
        正規形
    """
    expr = parse(source)
    normal_form, _ = reduce_to_normal_form(expr, max_steps=max_steps)
    return normal_form


# =============================================================================
# Main (for testing)
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("SK Reduction Test - Day 2")
    print("=" * 60)
    
    # テストケース
    test_cases = [
        # K-reduction
        ("K a b", "K a b → a"),
        ("K S K", "K S K → S"),
        
        # S-reduction
        ("S K K a", "S K K a → K a (K a) → a"),
        
        # 複合
        ("S (K S) K a b", "S (K S) K a b → ..."),
        
        # 研究例
        ("S (K a) (K b) c", "S (K a) (K b) c → K a c (K b c) → a (K b c) → a b"),
    ]
    
    for source, description in test_cases:
        print(f"\n{'='*60}")
        print(f"Test: {description}")
        print(f"{'='*60}")
        expr = parse(source)
        step_by_step(expr)
    
    # Redex検出テスト
    print(f"\n{'='*60}")
    print("Redex Detection Test")
    print(f"{'='*60}")
    
    redex_tests = [
        "S",
        "K",
        "S K",
        "K a b",
        "S a b c",
        "S (K a) (K b) c",
    ]
    
    for source in redex_tests:
        expr = parse(source)
        redexes = find_redexes(expr)
        print(f"\n{source}:")
        print(f"  Canonical: {to_canonical(expr)}")
        print(f"  Redexes: {len(redexes)}")
        for r in redexes:
            rtype = "S" if r.type == RedexType.S_REDEX else "K"
            print(f"    - {rtype}-redex at path '{r.path}'")
    
    print(f"\n{'='*60}")
    print("All tests completed!")
    print(f"{'='*60}")



