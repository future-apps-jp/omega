# Genesis-Matrix

**量子構造の進化的創発シミュレーションフレームワーク**

Genesis-Matrixは、リソース制約下でDSL（ドメイン特化言語）が進化する過程で、行列演算（量子的構造）が自然に支配的になることを実証するシミュレーションフレームワークです。

## 概要

### 研究仮説

> 物理法則（特に量子力学の公理A1）は、計算リソース制約下で「進化的に創発」する。

### 核心的洞察

1. **Scalar DSL (古典)**: スカラー演算のみ。グラフ探索には経路を1つずつ列挙する必要があり、記述長 K = O(N)
2. **Matrix DSL (量子的)**: 行列演算を持つ。A^k で全経路を一度に計算でき、記述長 K = O(1)

この記述長の差が選択圧となり、Matrix DSLが進化的に勝利します。

## インストール

```bash
# 仮想環境の作成
python3 -m venv genesis-env
source genesis-env/bin/activate

# 依存関係のインストール
pip install numpy jax[cpu] pytest
```

## クイックスタート

```bash
# 環境確認
python genesis/check_cpu.py

# テスト実行
python -m pytest genesis/tests/ -v

# "The Quantum Dawn" 実験
python genesis/experiments/quantum_dawn.py
```

## 構造

```
genesis/
├── core/                  # 基盤クラス群
│   ├── dsl.py            # DSL抽象クラス
│   ├── localhost.py      # 親宇宙（リソース管理）
│   ├── container.py      # 子宇宙（DSL + 現象）
│   └── fitness.py        # 適応度評価
├── dsl/                   # DSL実装
│   ├── scalar.py         # Species A: 古典的スカラー演算
│   └── matrix.py         # Species B: 行列演算（量子的）
├── evolution/             # 進化エンジン
│   ├── mutation.py       # AST構造変異
│   ├── selection.py      # 選択メカニズム
│   └── population.py     # 集団管理
├── tasks/                 # タスク定義
│   └── graph_walk.py     # グラフ探索タスク
├── experiments/           # 実験スクリプト
│   └── quantum_dawn.py   # Phase 25実験
└── tests/                 # テストスイート
```

## 主要な概念

### Localhost（親宇宙）

計算リソースを提供する「メタ宇宙」。目的論的意図なく、効率的なプロセスにリソースを配分します。

### Container（子宇宙）

独自のDSL（物理法則）と現象（プログラム）を持つ「子宇宙」。インフレーション速度 V_inf ∝ 1/(K × T) でリソースを獲得します。

### ホログラフィック制約

記述長 K に上限を設けることで、効率的な記述を進化的に選択する圧力を生み出します。

## 実験結果

### Phase 25: "The Quantum Dawn"

```
Task: 5ノードグラフで、ノード0→4への3ステップ経路数を計算
Target: 2（正解）

Initial: 80% Scalar, 20% Matrix
Final:   0% Scalar, 100% Matrix

→ Matrix DSL (A1) が進化的に支配！
```

## 次のフェーズ

- **Phase 26**: "Evolution of Operators" - 非行列DSLから行列操作の自発的創発を観測
- **Phase 27**: 理論統合と論文執筆

## 参考文献

- Research Plan v5: `docs/research_plan_v5.md`
- Artificial Physics Proposal: `docs/Research_Proposal_Artificial_Physics.md`

## ライセンス

MIT License

