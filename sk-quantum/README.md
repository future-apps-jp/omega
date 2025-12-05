# SK-Quantum: SK計算と量子構造の関係を検証するプロジェクト

## 概要

このプロジェクトは、SKコンビネータ計算（抽象度10）と量子力学の複素振幅構造（抽象度9）との間に関係があるかどうかを検証する研究実装です。

## プロジェクト構造

```
sk-quantum/
├── README.md
├── phase0/                    # 初期実験（Python）
│   ├── sk_parser.py          # ✅ Day 1: SK式パーサ
│   ├── test_parser.py        # ✅ Day 1: パーサテスト
│   ├── reduction.py          # 🔲 Day 2: β簡約
│   ├── redex.py              # 🔲 Day 3: Redex探索
│   ├── multiway.py           # 🔲 Day 4: Multiway graph
│   ├── probability.py        # 🔲 Day 5: 確率定義
│   ├── sorkin.py             # 🔲 Day 6: Sorkin公式
│   └── experiments/
│       └── experiment_001.ipynb  # 🔲 Day 7: 最初の実験
└── src/                       # 本格実装（Haskell）予定
```

## 進捗状況

### Phase 0: 初期実験

| Day | タスク | ステータス | 成果物 |
|-----|--------|-----------|--------|
| Day 1 | SK式AST定義 + パーサ | ✅ 完了 | `sk_parser.py` |
| Day 2 | β簡約の実装 | ✅ 完了 | `reduction.py` |
| Day 3 | Redex探索器 | 🔲 未着手 | `redex.py` |
| Day 4 | Multiway graph構築 | 🔲 未着手 | `multiway.py` |
| Day 5 | 確率定義の実装 | 🔲 未着手 | `probability.py` |
| Day 6 | Sorkin公式 I₂, I₃ | 🔲 未着手 | `sorkin.py` |
| Day 7 | 実験実行 + 分析 | 🔲 未着手 | `experiment_001.ipynb` |

## 使用方法

### Day 1: SK式パーサ

```python
from sk_parser import parse, to_string, to_canonical, S, K, App

# SK式をパース
expr = parse("S (K a) (K b)")
print(expr)           # S (K a) (K b)
print(repr(expr))     # App(App(S, App(K, Var('a'))), App(K, Var('b')))
print(to_canonical(expr))  # ((S (K a)) (K b))

# サイズと深さ
from sk_parser import size, depth
print(size(expr))     # 9
print(depth(expr))    # 4
```

### テスト実行

```bash
cd phase0
python3 test_parser.py
```

## SK計算の基本規則

```
S x y z → x z (y z)
K x y → x
```

## 研究目標

1. SK計算の分岐確率が Sorkin の2次非加法性条件 (I₂ ≠ 0, I₃ = 0) を満たすかどうかを検証
2. 満たす場合、複素振幅が数学的に要請されることを確認
3. 満たさない場合、SK計算が本質的に古典的であることを確認

## 参考文献

- Sorkin, R.D. (1994). Quantum mechanics as quantum measure theory. Modern Physics Letters A.
- Curry, H.B., & Feys, R. (1958). Combinatory Logic, Vol. I. North-Holland.

