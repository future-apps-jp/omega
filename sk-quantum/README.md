# SK-Quantum: SK計算と量子構造の関係を検証するプロジェクト

## 概要

このプロジェクトは、SKコンビネータ計算（抽象度10）と量子力学の複素振幅構造（抽象度9）との間に関係があるかどうかを検証する研究実装です。

## プロジェクト構造

```
sk-quantum/
├── README.md
├── phase0/                    # ✅ Phase 0 完了
│   ├── sk_parser.py          # ✅ SK式パーサ (37 tests)
│   ├── reduction.py          # ✅ β簡約 + Redex探索 (31 tests)
│   ├── multiway.py           # ✅ Multiway graph (21 tests)
│   ├── probability.py        # ✅ 確率定義 (4モデル)
│   ├── sorkin.py             # ✅ Sorkin公式 I₂, I₃
│   └── experiments/
│       └── experiment_001.py # ✅ Sorkin公式検証
├── phase1/                    # ✅ Phase 1 完了
│   ├── algebra/              # ✅ Phase 1A 代数的構造 (21 tests)
│   │   └── operators.py      # ✅ 書き換え演算子代数
│   ├── geometry/             # ✅ Phase 1B 幾何学的構造 (19 tests)
│   │   └── holonomy.py       # ✅ ホロノミー解析
│   └── experiments/
│       ├── RESULTS_001.md    # ✅ Phase 1A 結果
│       └── RESULTS_002_holonomy.md # ✅ Phase 1B 結果
├── phase2/                    # ✅ Phase 2 完了
│   ├── information/          # ✅ 情報理論的アプローチ (15 tests)
│   │   └── complexity.py     # ✅ Kolmogorov複雑性・位相計算
│   └── experiments/
│       └── RESULTS_003_information.md # ✅ Phase 2 結果
└── src/                       # 本格実装（Haskell）予定
```

## 進捗状況

### Phase 0: 初期実験

| Day | タスク | ステータス | 成果物 |
|-----|--------|-----------|--------|
| Day 1 | SK式AST定義 + パーサ | ✅ 完了 | `sk_parser.py` |
| Day 2 | β簡約の実装 | ✅ 完了 | `reduction.py` |
| Day 3 | Redex探索器 | ✅ 完了 | `reduction.py` (統合) |
| Day 4 | Multiway graph構築 | ✅ 完了 | `multiway.py` |
| Day 5 | 確率定義の実装 | ✅ 完了 | `probability.py` |
| Day 6 | Sorkin公式 I₂, I₃ | ✅ 完了 | `sorkin.py` |
| Day 7 | 実験実行 + 分析 | ✅ 完了 | `experiments/experiment_001.py` |

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

## 実験結果

### Phase 0: Sorkin公式による量子性検証 ✅

| Expression | Paths | I₂≠0 | Status |
|------------|-------|------|--------|
| `(K a b) (K c d)` | 2 | 0 | ✓ Classical |
| `S (K a) (K b) c` | 2 | 0 | ✓ Classical |
| `(K a b) (K c d) (K e f)` | 6 | 15 | 🔔 見かけの非加法性* |

**結論**: 現在の確率定義では古典的。I₂ ≠ 0 は P(A∪B) 定義に起因。

### Phase 1A: 代数的構造からの複素数導出 ✅

| 検証項目 | 結果 |
|----------|------|
| J² = -I の候補 | 1250個（全て自明解） |
| Clifford構造 | なし |
| Pauli構造 | なし |

**結論**: J² = -I の候補は全て `J = -iI + ...` の形で、複素係数を最初から導入した自明解。

### Phase 1B: パス空間のホロノミー解析 ✅

| Expression | ループ数 | U(1) 候補 |
|------------|----------|-----------|
| `S (K a) (K b) c` | 1 | いいえ |
| `(K a b) (K c d) (K e f)` | 15 | いいえ |
| `S (K a) (K b) (S c d e)` | **120** | **はい** |

**結論**: U(1) 構造は見つかったが、**接続の定義に依存**。位相を「仮定」しており「導出」ではない。

### Phase 2: 情報理論的アプローチ ✅

| Expression | パス数 | 位相差 | 計算式 |
|------------|--------|--------|--------|
| `S (K a) (K b) c` | 2 | 0 | 全て |
| `(K a b) (K c d) (K e f)` | 6 | 0 | 全て |
| `S (K a) (K b) (S c d e)` | 16 | **≠0** | **linear** |

**結論**: 情報量から位相を「計算」することは可能だが、計算式の選択に任意性が残る。

---

## 全体総括

| Phase | 目標 | 結果 | 位相の由来 |
|-------|------|------|------------|
| Phase 0 | Sorkin公式検証 | 古典的 | N/A |
| Phase 1A | 代数的導出 | 自明解のみ | 複素係数を仮定 |
| Phase 1B | 幾何学的導出 | 接続依存 | 接続で仮定 |
| Phase 2 | 情報理論的導出 | 部分的成功 | 演算数から計算 |

> **SK計算から複素数構造を「完全に導出」することはできなかった。**
> ただし、情報量から位相を「計算」する方法を発見。

## 研究目標

1. SK計算の分岐確率が Sorkin の2次非加法性条件 (I₂ ≠ 0, I₃ = 0) を満たすかどうかを検証
2. 満たす場合、複素振幅が数学的に要請されることを確認
3. 満たさない場合、SK計算が本質的に古典的であることを確認

## 参考文献

- Sorkin, R.D. (1994). Quantum mechanics as quantum measure theory. Modern Physics Letters A.
- Curry, H.B., & Feys, R. (1958). Combinatory Logic, Vol. I. North-Holland.

