# A1 Implementation Plan
## 「宇宙の最小公理を記述するための、最小の言語」

---

## 概要

**A1** は、Axiom A1（状態空間拡張/重ね合わせ）をネイティブに実装するScheme方言です。
このプロジェクトでは、A1言語のPythonインタプリタを実装し、AWS Braketを通じて実際の量子プロセッサ上で実行可能にします。

### 目的

1. **理論的検証**: 論文「Algorithmic Naturalness」の主張を実証
   - 量子基盤上では記述長が最小化される
   - 古典基盤との記述長比較（数桁のオーダー差）

2. **工学的実証**: AWS Braketでの実機実行
   - Bell状態生成の実演
   - IonQ, Rigetti等での量子相関検証

---

## 形式的基盤

### A1の意味論（Denotational Semantics）

**定理（Evaluation Homomorphism）**: A1インタプリタ `eval` は、S式からBraket回路への準同型写像である：

```
eval : A1-Expression → Braket-Circuit
```

すなわち、A1式の構成的評価が、量子ゲートの順次適用に対応する。

| A1 構文 | 意味論 | Braket対応 |
|---------|--------|------------|
| `(H q)` | 量子ビット `q` にHadamardを適用 | `circuit.h(q)` |
| `(CNOT c t)` | 制御cから標的tへCNOT | `circuit.cnot(c, t)` |
| `(f arg1 arg2)` | 関数適用 | 順次回路拡張 |

この性質により、A1プログラムの意味は回路の意味と一致し、
論文における「A1が量子基盤上でAxiom A1をネイティブに実現する」という主張の形式的根拠となる。

---

## 複雑度計測の形式化

### Kolmogorov複雑度の近似

**定義（A1-Complexity）**:
A1プログラム `p` の複雑度を以下で定義する：

```
K_A1(p) = |tokens(p)| × log₂(|V_A1|)
```

ここで：
- `tokens(p)`: プログラムのトークン列（括弧を除く）
- `|V_A1|`: A1の語彙サイズ

### 命令セットとコストモデル

**A1側とNumPy側で共通のルールでカウントする**ことを保証するため、
以下のコストモデルを明示的に定義する：

#### A1 命令セット（コスト = 1トークン/命令）

| カテゴリ | 命令 | コスト |
|----------|------|--------|
| 量子ゲート | H, X, Y, Z, CNOT, CZ, SWAP, RX, RY, RZ | 1 |
| 測定 | MEASURE | 1 |
| 特殊形式 | DEFINE, LAMBDA, IF, LET | 1 |
| リテラル | 数値, シンボル | 1 |

#### 古典（NumPy）命令セット（コスト = 1トークン/命令）

| カテゴリ | 命令 | コスト |
|----------|------|--------|
| 関数呼び出し | np.array, np.kron, np.dot, etc. | 1 |
| 演算子 | +, -, *, /, @ | 1 |
| リテラル | 数値, 配列要素 | 1 |
| 構文要素 | import, =, [, ], dtype | 1 |

**重要**: 両言語とも「同一タスク（例：Bell状態生成）を実現するプログラム」を対象とし、
コメント・空白・インデントはカウントしない。

### 語彙サイズの定義

| 言語 | 語彙 |V| | log₂|V| |
|------|------|---------|
| A1 | 11ゲート + 4特殊形式 + qubit index + 定数 ≈ 32 | ~5 bits |
| NumPy | 関数 + 演算子 + リテラル ≈ 256 | ~8 bits |

---

## Phase 1: Core Engine（A1カーネル）
### 期間: 1週間

### 1.1 ディレクトリ構造

```
omega/
├── a1/
│   ├── __init__.py
│   ├── core.py         # パーサーとインタプリタ (A1クラス)
│   ├── gates.py        # 量子ゲート定義 (H, CNOT, X, Y, Z, etc.)
│   ├── metrics.py      # 記述長(Complexity)計測エンジン
│   └── tests/
│       ├── test_core.py
│       ├── test_gates.py
│       └── test_metrics.py
├── experiments/
│   ├── README.md       # 実行手順・環境設定ドキュメント
│   ├── hello_world.a1  # Bell状態生成デモ
│   ├── ghz_state.a1    # GHZ状態生成
│   ├── teleport.a1     # 量子テレポーテーション
│   ├── run_local.py    # ローカルシミュレーション
│   ├── run_aws.py      # AWS Braket実行
│   └── compare.py      # A1 vs 古典 記述長比較
└── requirements.txt
```

### 1.2 パーサー実装

**入力**: S式文字列
**出力**: Pythonリスト（AST）

```python
# 入力
"(CNOT (H 0) 1)"

# 出力
['CNOT', ['H', 0], 1]
```

### 1.3 インタプリタ実装

**核心的設計**: ゲート関数は作用した量子ビットインデックスを返す

```python
(H 0)           # qubit 0 に H を適用し、0 を返す
(CNOT (H 0) 1)  # (H 0) が 0 になるので、実質 (CNOT 0 1)
```

これにより、Lispらしいネスト構造でゲート列を表現可能。

**Invariant**: `eval(expr)` は常に以下を保証する：
1. `self.circuit` が正しいBraket回路を保持
2. 返り値は最後に操作したqubitのindex（チェイン用）

### 1.4 環境（Environment）

| 特殊形式 | 機能 |
|----------|------|
| `DEFINE` | 変数・関数を環境に登録 |
| `LAMBDA` | クロージャを作成 |
| `IF` | 条件分岐 |
| `LET` | ローカル変数束縛 |

### 1.5 成果物

- [ ] `core.py`: A1クラス（パーサー + インタプリタ）
- [ ] `gates.py`: H, CNOT, X, Y, Z, RX, RY, RZ, MEASURE
- [ ] `test_core.py`: 基本テスト（15+）
- [ ] Bell状態生成が動作すること

---

## Phase 2: Complexity Metrics（記述長計測）
### 期間: 3日

### 2.1 A1 Complexity Counter

```python
class A1Metrics:
    VOCABULARY_SIZE = 32  # 固定語彙サイズ
    
    def count_tokens(self, source: str) -> int:
        """A1ソースコードのトークン数を返す（括弧除く）"""
        pass
    
    def complexity(self, source: str) -> float:
        """Kolmogorov複雑度の近似（トークン数 × log₂(語彙サイズ)）"""
        tokens = self.count_tokens(source)
        return tokens * math.log2(self.VOCABULARY_SIZE)
```

### 2.2 Classical Complexity Counter

同じBell状態生成をPython + NumPyで実装し、比較:

```python
# 古典的実装（NumPy）
import numpy as np

H = (1/np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex)
CNOT = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 0]
], dtype=complex)

# ... 約50-100行
```

### 2.3 ベンチマーク対象（3種類）

**成功基準として、以下の3ケース全てで一貫した桁差が出ること**:

| ベンチマーク | A1コード | 期待トークン数 |
|--------------|----------|---------------|
| Bell状態 | `(CNOT (H 0) 1)` | 4 |
| GHZ状態 | `(CNOT (CNOT (H 0) 1) 2)` | 7 |
| テレポーテーション | 完全プロトコル | ~20 |

### 2.4 期待される結果

| ベンチマーク | A1 トークン | NumPy トークン | 比率 |
|--------------|------------|----------------|------|
| Bell状態 | 4 | 100+ | ~25x |
| GHZ状態 | 7 | 150+ | ~20x |
| テレポーテーション | 20 | 300+ | ~15x |

**重要**: 各ベンチマークで平均・分散を計算し、統計的に有意な差を確認。

### 2.5 成果物

- [ ] `metrics.py`: 記述長計測モジュール
- [ ] `compare.py`: A1 vs 古典の比較レポート生成
- [ ] 論文用データ（表・グラフ）

---

## Phase 3: Cloud Execution（クラウド実行）
### 期間: 1週間

### 3.1 AWS Braket連携

```python
from braket.circuits import Circuit
from braket.aws import AwsDevice

class A1:
    def to_braket_circuit(self) -> Circuit:
        """A1プログラムをBraket回路に変換"""
        return self.circuit
    
    def run_on_simulator(self, shots=1000):
        """SV1シミュレータで実行"""
        pass
    
    def run_on_device(self, device_arn: str, shots=100):
        """実機（IonQ, Rigetti）で実行"""
        pass
```

### 3.2 実行対象

| バックエンド | タイプ | 用途 |
|-------------|--------|------|
| LocalSimulator | ローカル | 開発・デバッグ |
| SV1 | AWSシミュレータ | 大規模テスト |
| IonQ Harmony | イオントラップ | 実機検証 |
| Rigetti Aspen | 超伝導 | 実機検証 |

### 3.3 統計処理プロトコル

**ノイズ・統計処理を厳密に行う**ため、以下の実験設計を採用：

#### 3.3.1 実験パラメータ

| パラメータ | 値 | 理由 |
|------------|-----|------|
| ショット数 | 1000 (シミュレータ), 100 (実機) | コスト/精度バランス |
| 試行回数 | 5回/バックエンド | 統計的信頼性 |
| レポート形式 | 平均 ± 標準偏差 | 再現性確保 |

#### 3.3.2 忠実度推定法

Bell状態の理論的期待値: `{00: 0.5, 11: 0.5}`

```python
def estimate_fidelity(counts: dict, shots: int) -> float:
    """
    Total Variation Distance (TVD) ベースの忠実度推定
    
    F = 1 - TVD(p_measured, p_ideal)
    TVD = 0.5 * Σ |p_measured(x) - p_ideal(x)|
    """
    ideal = {'00': 0.5, '11': 0.5}
    tvd = 0.5 * sum(abs(counts.get(k, 0)/shots - ideal.get(k, 0)) 
                   for k in set(ideal) | set(counts))
    return 1 - tvd
```

#### 3.3.3 デバイスごとのノイズ特性

| デバイス | 期待ノイズ源 | 対策 |
|----------|-------------|------|
| IonQ | 読み出しエラー (~1%) | 複数試行で平均化 |
| Rigetti | ゲートエラー (~2%), クロストーク | エラー緩和を検討 |

### 3.4 実験プロトコル（3種類）

**全3プロトコルで成功することを成功基準とする**:

1. **Bell状態生成**: `(CNOT (H 0) 1)`
   - 検証: 00/11の確率が各50%に近い
   
2. **GHZ状態生成**: 3量子ビットエンタングルメント
   - 検証: 000/111の確率が各50%に近い
   
3. **量子テレポーテーション**: 完全プロトコル
   - 検証: 入力状態の忠実な転送

### 3.5 成果物

- [ ] `run_local.py`: ローカル実行スクリプト
- [ ] `run_aws.py`: AWS実行スクリプト
- [ ] 実験結果（忠実度、統計）
- [ ] 論文用データ

---

## Phase 4: Integration（統合）
### 期間: 3日

### 4.1 論文用出力

```python
# 自動レポート生成
def generate_paper_data():
    """論文のSection 4用データを生成"""
    # A1記述長
    # NumPy記述長
    # 比率
    # AWS実行結果（平均±標準偏差）
    pass
```

### 4.2 最終成果物

- [ ] 完全なA1インタプリタ
- [ ] AWS Braket実行結果
- [ ] 記述長比較データ
- [ ] 論文の実験セクション完成

---

## 再現性・環境設定

### experiments/README.md の内容

```markdown
# A1 Experiments - Setup Guide

## 必要条件
- Python 3.9+
- AWS アカウント（Braket有効化済み）

## 環境構築

### 1. 依存関係インストール
pip install -r requirements.txt

### 2. AWS認証設定
export AWS_PROFILE=your-profile
export AWS_DEFAULT_REGION=us-east-1

### 3. ローカル実行
python run_local.py

### 4. AWS実行
python run_aws.py --backend ionq --shots 100

## ファイル構成
- hello_world.a1: Bell状態生成
- ghz_state.a1: GHZ状態生成
- teleport.a1: 量子テレポーテーション
- run_local.py: ローカルシミュレーション
- run_aws.py: AWS Braket実行
- compare.py: 記述長比較

## トラブルシューティング
- Braket認証エラー: AWS CLIの設定確認
- デバイス不可: デバイスARNとリージョン確認
```

---

## タイムライン

```
Week 1: Phase 1（Core Engine）
  - Day 1-2: パーサー + 形式的invariant確認
  - Day 3-4: インタプリタ
  - Day 5-7: ゲート実装・テスト

Week 2: Phase 2 + 3
  - Day 1-3: 記述長計測（3ベンチマーク）
  - Day 4-7: AWS Braket連携 + 統計処理

Week 3: Phase 4
  - Day 1-3: 統合・論文データ生成
```

---

## 技術仕様

### 依存関係

```
# requirements.txt
numpy>=1.21.0
amazon-braket-sdk>=1.35.0
pytest>=7.0.0
matplotlib>=3.5.0  # 結果可視化用
```

### A1文法（EBNF）

```ebnf
program     = expression*
expression  = atom | list
list        = '(' expression* ')'
atom        = NUMBER | SYMBOL | STRING
NUMBER      = [0-9]+
SYMBOL      = [A-Za-z_][A-Za-z0-9_-]*
STRING      = '"' [^"]* '"'
```

### プリミティブ（コスト = 1トークン）

| ゲート | 説明 | A1構文 | コスト |
|--------|------|--------|--------|
| H | Hadamard | `(H q)` | 1 |
| X | Pauli-X | `(X q)` | 1 |
| Y | Pauli-Y | `(Y q)` | 1 |
| Z | Pauli-Z | `(Z q)` | 1 |
| CNOT | 制御NOT | `(CNOT c t)` | 1 |
| CZ | 制御Z | `(CZ c t)` | 1 |
| SWAP | スワップ | `(SWAP q1 q2)` | 1 |
| RX | X軸回転 | `(RX q theta)` | 1 |
| RY | Y軸回転 | `(RY q theta)` | 1 |
| RZ | Z軸回転 | `(RZ q theta)` | 1 |
| MEASURE | 測定 | `(MEASURE q)` | 1 |

---

## 成功基準

### 機能要件

| 基準 | 条件 | 検証方法 |
|------|------|----------|
| Bell状態 | AWS Braketで実行可能 | run_aws.py |
| GHZ状態 | AWS Braketで実行可能 | run_aws.py |
| テレポーテーション | AWS Braketで実行可能 | run_aws.py |

### 記述長基準

| 基準 | 条件 | 検証方法 |
|------|------|----------|
| A1トークン数 | Bell < 5, GHZ < 10, Teleport < 25 | metrics.py |
| 比率 | A1/古典 > 10倍（全3ケース） | compare.py |

### 実機検証基準

| 基準 | 条件 | 検証方法 |
|------|------|----------|
| IonQ忠実度 | > 90% (5回平均) | run_aws.py |
| Rigetti忠実度 | > 85% (5回平均) | run_aws.py |
| 統計レポート | 平均±標準偏差を記録 | compare.py |

### 一貫性基準

**3種類のベンチマーク（Bell, GHZ, Teleportation）全てで、
A1の記述長優位性が確認されること。**

---

## Limitations と今後の課題

1. **複雑度近似の限界**: 本計画のK_A1はトークンベースの近似であり、
   真のKolmogorov複雑度ではない。より厳密な理論的基盤は今後の課題。

2. **ノイズモデルの簡略化**: デバイス固有のノイズ特性を完全には
   モデル化していない。エラー緩和技術の導入は将来の拡張として検討。

3. **スケーラビリティ**: 現在のベンチマークは3量子ビット以下。
   より大規模な回路での検証は今後の課題。

---

*A1 Implementation Plan v2.0 — December 2025*
*「宇宙の最小公理を記述するための、最小の言語」*

*Revised based on academic review: formalized complexity metrics, 
expanded benchmarks, statistical protocols, and reproducibility guidelines.*
