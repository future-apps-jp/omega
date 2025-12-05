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
├── phase4/                    # ✅ Phase 4 完了（研究計画v2）
│   ├── reversible/           # ✅ 可逆論理ゲート (18 tests)
│   │   ├── gates.py          # ✅ Toffoli/Fredkin ゲート
│   │   ├── group_analysis.py # ✅ 群構造解析
│   │   └── symplectic.py     # ✅ シンプレクティック解析
│   └── experiments/
│       └── RESULTS_004_algebra.md # ✅ Phase 4 結果
├── phase5/                    # ✅ Phase 5 完了
│   ├── spectral/             # ✅ ハミルトニアン・量子ウォーク (13 tests)
│   │   └── hamiltonian.py    # ✅ 連続時間量子ウォーク
│   └── experiments/
│       └── RESULTS_005_spectral.md # ✅ Phase 5 結果
├── phase6/                    # ✅ Phase 6 完了
│   ├── rca/                  # ✅ 可逆セルオートマトン (17 tests)
│   │   ├── automata.py       # ✅ Rule 90/150 実装
│   │   ├── graph.py          # ✅ RCAグラフ
│   │   └── hamiltonian.py    # ✅ RCAハミルトニアン
│   ├── comparison/           # ✅ 3モデル比較 (14 tests)
│   │   ├── quantum_circuit.py # ✅ 量子回路
│   │   └── model_comparison.py # ✅ SK/RCA/Quantum比較
│   └── experiments/
│       └── RESULTS_006_comparison.md # ✅ Phase 6 結果
├── phase7/                    # ✅ Phase 7 完了
│   ├── noncommutative/       # ✅ 非可換性解析 (15 tests)
│   │   ├── operators.py      # ✅ SK演算子代数
│   │   ├── commutator.py     # ✅ 交換子解析
│   │   └── superposition.py  # ✅ 重ね合わせ解析
│   └── experiments/
│       └── RESULTS_007_noncommutative.md # ✅ Phase 7 結果
├── phase8/                    # ✅ Phase 8 完了（研究計画v3）
│   ├── axioms/               # ✅ GPTs & 公理分析 (28 tests)
│   │   ├── gpt_framework.py  # ✅ GPTsフレームワーク
│   │   ├── axiom_candidates.py # ✅ 公理候補分析
│   │   └── test_axioms.py    # ✅ テストスイート
│   └── experiments/
│       └── RESULTS_008_axioms.md # ✅ Phase 8 結果
├── phase9/                    # ✅ Phase 9 完了
│   ├── information/          # ✅ Resource Theory (31 tests)
│   │   ├── resource_theory.py # ✅ Coherenceリソース理論
│   │   └── test_resource_theory.py # ✅ テストスイート
│   └── experiments/
│       └── RESULTS_009_information.md # ✅ Phase 9 結果
├── phase10/                   # ✅ Phase 10 完了
│   ├── lambda_calculus/      # ✅ λ計算分析 (31 tests)
│   │   ├── lambda_core.py    # ✅ λ計算コア実装
│   │   ├── lambda_analysis.py # ✅ 代数構造・GPT・Resource Theory分析
│   │   └── test_lambda.py    # ✅ テストスイート
│   └── experiments/
│       └── RESULTS_010_lambda.md # ✅ Phase 10 結果
└── src/                       # 本格実装（Haskell）予定
```

## 進捗状況

### Phase 0-2: SK計算と量子構造（研究計画v1）

| Phase | 目標 | 結果 | 位相の由来 |
|-------|------|------|------------|
| Phase 0 | Sorkin公式検証 | 古典的 | N/A |
| Phase 1A | 代数的導出 | 自明解のみ | 複素係数を仮定 |
| Phase 1B | 幾何学的導出 | 接続依存 | 接続で仮定 |
| Phase 2 | 情報理論的導出 | 部分的成功 | 演算数から計算 |

> **結論**: SK計算から複素数構造を「完全に導出」することはできなかった。
> → 論文 "On the Independence of Quantum Structure from SK Combinatory Logic" として発表準備完了

### Phase 4: 可逆計算の代数構造（研究計画v2）✅

| 検証項目 | 結果 | 判定 |
|----------|------|------|
| Sp(2n,ℝ) 埋め込み | 全要素が埋め込み可能 | 古典的 |
| J² = -I | 非自明解なし | 古典的 |
| 固有値 | 全て実数（1の冪根） | 古典的 |
| 連続的位相 | 存在しない | 古典的 |

**結論**: 可逆論理ゲートは「古典的」
- シンプレクティック群 Sp(2n, ℝ) に埋め込み可能
- しかし、ユニタリ群 U(n) への拡大は生じない
- **仮説 H1 を支持**: 「可逆計算は古典的であり、Sp(2n,ℝ) に閉じる」

---

## 実験結果サマリー

### Phase 0: Sorkin公式による量子性検証 ✅

| Expression | Paths | I₂≠0 | Status |
|------------|-------|------|--------|
| `(K a b) (K c d)` | 2 | 0 | ✓ Classical |
| `S (K a) (K b) c` | 2 | 0 | ✓ Classical |
| `(K a b) (K c d) (K e f)` | 6 | 15 | 🔔 見かけの非加法性* |

### Phase 1A: 代数的構造からの複素数導出 ✅

| 検証項目 | 結果 |
|----------|------|
| J² = -I の候補 | 1250個（全て自明解） |
| Clifford構造 | なし |
| Pauli構造 | なし |

### Phase 1B: パス空間のホロノミー解析 ✅

| Expression | ループ数 | U(1) 候補 |
|------------|----------|-----------|
| `S (K a) (K b) (S c d e)` | **120** | **はい（接続依存）** |

### Phase 2: 情報理論的アプローチ ✅

| Expression | パス数 | 位相差 | 計算式 |
|------------|--------|--------|--------|
| `S (K a) (K b) (S c d e)` | 16 | **≠0** | **linear** |

### Phase 4: 可逆計算の代数構造 ✅

| ゲート | 群の位数 | Sp埋め込み | J² = -I |
|--------|----------|------------|---------|
| Toffoli | 2 | ✓ | なし |
| Fredkin | 2 | ✓ | なし |
| Toffoli+Fredkin | 6 | ✓ | なし |

**結論**: 離散的可逆計算は古典的（H1 支持）

### Phase 5: ハミルトニアンと干渉 ✅

| 式 | ノード数 | 干渉あり | Total Variation |
|----|----------|----------|-----------------|
| `S (K a) (K b) c` | 5 | ✅ | 0.85 |
| `(K a b) (K c d)` | 4 | ✅ | 0.85 |
| `(K a b) (K c d) (K e f)` | 8 | ✅ | 0.74 |
| `S (K a) (K b) (S c d e)` | 11 | ✅ | 0.58 |

**結論**: 連続時間量子ウォーク U(t) = exp(-iAt) で干渉が生じる！
→ 「離散→連続」の極限で量子性が現れる可能性

### Phase 6: 3モデル比較 ✅

| モデル | 可逆性 | 離散性 | 干渉（連続） | 重ね合わせ |
|--------|--------|--------|--------------|------------|
| SK | ✗ | ✓ | ✓ | ✗ |
| RCA (Rule 90) | ✓ | ✓ | ✓ | ✗ |
| 量子回路 | ✓ | ✗ | ✓ | ✓ |

**RCA連続時間化結果**:

| Rule | Size | 干渉あり | 量子振動 | 古典振動 | Max TVD |
|------|------|----------|----------|----------|---------|
| 90 | 3 | ✅ | 9 | 0 | 0.82 |
| 90 | 4 | ✅ | 6 | 0 | 0.84 |
| 150 | 3 | ✅ | 9 | 0 | 0.82 |
| 150 | 4 | ✅ | 9 | 0 | 0.82 |

**結論**: 
- 仮説H6「RCAも連続時間化すれば干渉を示す」→ ✅ **SUPPORTED**
- 離散計算（SK/RCA）は本質的に古典的だが、連続時間化で干渉が出現
- 真の量子性（重ね合わせ・もつれ）は量子回路のみ

### Phase 7: 非可換性と量子化 ✅

| 検証項目 | 結果 | 判定 |
|----------|------|------|
| 交換子 [Ŝ, K̂] | **= 0** | 可換（この定義では） |
| 重ね合わせ要件 | 0/4 満たす | 生成不可 |
| H2 Status | INCONCLUSIVE | 関係は定義依存 |

**重ね合わせ要件の検証**:

| 要件 | 量子力学 | SK計算 | 満たす？ |
|------|----------|--------|----------|
| 連続的振幅 | α, β ∈ ℂ | 0 or 1 | ✗ |
| 位相コヒーレンス | e^{iφ} | なし | ✗ |
| ユニタリ性 | UU† = I | 非ユニタリ | ✗ |
| 規格化 | |α|² + |β|² = 1 | なし | ✗ |

**結論**:
- 「左からの適用」演算子では [Ŝ, K̂] = 0（アプリケーションの結合性による）
- 重ね合わせの4要件をいずれも満たさない
- **量子性は計算から自動的に導出できない** → 追加公理が必要

---

## 使用方法

### SK式パーサ

```python
from sk_parser import parse, to_string, to_canonical

expr = parse("S (K a) (K b)")
print(to_canonical(expr))  # ((S (K a)) (K b))
```

### 可逆ゲート解析

```python
from phase4.reversible import TOFFOLI, GateGroup, matrix_properties

M = TOFFOLI.to_matrix()
props = matrix_properties(M)
print(props['is_orthogonal'])  # True
```

### テスト実行

```bash
# Phase 0-2
cd phase0 && python3 -m pytest

# Phase 4
cd phase4/reversible && python3 -m pytest test_gates.py -v
```

---

## SK計算の基本規則

```
S x y z → x z (y z)
K x y → x
```

## 研究計画

- **v1** (Phase 0-3): SK計算と量子構造 → 完了、論文化
- **v2** (Phase 4-7): 可逆計算と量子構造 → ✅ **完了**、論文化
- **v3** (Phase 8-12): 最小公理集合の探求 → 🔄 **進行中**

詳細: `docs/research_plan_v3.md`

### Phase 8: 最小公理集合 ✅

| 仮説 | 予測 | 結果 |
|------|------|------|
| H8.1 | A1（重ね合わせ）は根源的公理 | ✅ SUPPORTED |
| H8.2 | No-Cloningは A1+A3 から導出可能 | ✅ SUPPORTED |
| H8.3 | 非可換性だけでは不十分 | ✅ SUPPORTED |
| H8.4 | 文脈依存性は強い条件 | △ 部分支持 |

**結論**: 「量子への跳躍」に必要な最小公理は **A1（状態空間拡張/重ね合わせ）**

### Phase 9: 情報理論的アプローチ ✅

| 仮説 | 予測 | 結果 |
|------|------|------|
| H9.1 | 情報保存だけでは量子構造は出ない | ✅ SUPPORTED |
| H9.2 | No-Cloningは結果であり原因ではない | ✅ SUPPORTED |
| H9.3 | 古典計算はCoherence生成不可 | ✅ SUPPORTED |
| H9.4 | Coherence生成能力が量子性の指標 | ✅ SUPPORTED |

**結論**: 情報理論的原理（No-Cloning等）は量子構造の**結果**であり、A1を導出できない

### Phase 10: 代替計算モデル（λ計算）✅

| 仮説 | 予測 | 結果 |
|------|------|------|
| H10.1 | λ計算でもSKと同様の古典的振る舞い | ✅ SUPPORTED |

**結論**: 結果は**計算モデル非依存**（SK, RCA, λ計算で確認）

### Phase 12: 論文執筆 ✅

第3論文 "Minimal Axioms for Quantum Structure: What Computation Cannot Derive" を執筆完了。

**成果物**: `papers/minimal-axioms/main.tex`

### Phase 11: 形式的検証 (Coq) ✅

Theorem 1（置換行列 → Sp埋め込み）の形式化を実施。

| 項目 | 状態 |
|------|------|
| 定理ステートメント | ✅ 型チェック完了 |
| 証明骨格 | ✅ 完成 |
| 完全証明 | ⚠️ 一部Admitted |

**成果物**: `sk-quantum/phase11/coq/PermSymplectic.v`

## 総テスト数

- Phase 0: 89 tests
- Phase 1: 40 tests
- Phase 2: 15 tests
- Phase 4: 18 tests
- Phase 5: 13 tests
- Phase 6: 31 tests (RCA: 17, Comparison: 14)
- Phase 7: 15 tests
- Phase 8: 28 tests
- Phase 9: 31 tests
- Phase 10: 31 tests
- **合計: 311 tests**

## 参考文献

- Sorkin, R.D. (1994). Quantum mechanics as quantum measure theory. Modern Physics Letters A.
- Curry, H.B., & Feys, R. (1958). Combinatory Logic, Vol. I. North-Holland.
- Bennett, C.H. (1973). Logical reversibility of computation. IBM Journal.
- Toffoli, T. (1980). Reversible Computing. Tech Memo MIT.
