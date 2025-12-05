# 研究計画 v3: 量子構造導出のための最小公理集合の探求

**From Computation to Quantum: Minimal Axioms for the Quantum Leap**

---

## 1. 概要

### 1.1 これまでの成果（v1-v2）

研究計画v1-v2では、計算から量子構造を導出する試みを系統的に行い、以下の否定的結果を得た：

| Phase | 対象 | 結果 | 論文 |
|-------|------|------|------|
| 0-3 | SK計算 | 複素構造は自動出現しない | `papers/sk-quantum-independence/` |
| 4 | 可逆論理ゲート | Sp(2N,ℝ)に埋め込み、古典的 | `papers/computational-quantum-limits/` |
| 5 | 連続時間発展 | 干渉は出現するが、重ね合わせは出ない | 同上 |
| 6 | モデル比較 | RCAも連続時間で干渉、量子回路のみ重ね合わせ | 同上 |
| 7 | 非可換性 | [Ŝ, K̂] = 0（適用演算子定義）、要件0/4 | 同上 |

**主要結論**: 計算から量子構造を導出するには**追加公理が必要**

### 1.2 本計画の目的

**「どのような追加公理が量子構造を生成するか？」**を明らかにする。

具体的には：
1. 計算→量子の「橋渡し」に必要な**最小公理集合**の特定
2. 情報理論的原理（no-cloning, no-deleting）の役割の解明
3. 異なる計算モデルでの普遍性の検証
4. 形式的検証による結果の厳密化

### 1.3 期待される成果

- 「計算→量子」の**Minimal Axiom Set**の提案
- 量子情報論と計算理論の**接点の明確化**
- 形式的に検証された**No-Go定理**の体系

---

## 2. 理論的背景

### 2.1 追加公理の候補（v2論文より）

論文で提案した3つの追加公理候補：

1. **Superposition Axiom**
   - 系の状態が複素ベクトル空間を形成
   - Born則による確率解釈

2. **Non-commutative Observable Structure**  
   - $[X, P] = i\hbar$ のような交換関係
   - ハイゼンベルクの不確定性原理の源

3. **Continuous Amplitude Axiom**
   - 離散値ではなく連続複素振幅
   - 重ね合わせの前提条件

### 2.2 Generalized Probabilistic Theories (GPTs)

**【v3.1追加】** 公理候補をアドホックに選ぶのではなく、GPTsの枠組みで統一的に扱う：

- **状態空間**: 凸集合として定義（古典：シンプレックス、量子：Bloch球）
- **効果 (Effects)**: 測定結果の確率を与える写像
- **変換**: 状態から状態への許容される操作

**GPTsにおける「計算モデル」の位置づけ**:
- 古典計算 → シンプレックス上の確率的操作（置換行列）
- 可逆計算 → シンプレックスの頂点間の置換
- 量子計算 → Bloch球上のユニタリ操作

**問い**: 計算モデルの状態空間を「シンプレックスからBloch球」に変形させる最小の追加構造は何か？

### 2.3 情報理論的原理

量子力学の特徴を情報理論から特徴づける試み：

- **No-Cloning Theorem**: 未知の量子状態は複製できない
- **No-Deleting Theorem**: 未知の量子状態は完全削除できない
- **No-Broadcasting Theorem**: 非可換なオブザーバブルは同時放送禁止

**【v3.1追加】循環論法への注意**:
> 古典計算では「任意の未知の状態」が存在しない（計算基底のみ）。
> したがって、No-Cloningを議論する前提として重ね合わせが必要であり、
> 「No-Cloning → 量子構造」の導出は循環する可能性がある。

**対策**: Resource Theory の視点を導入し、「量子性（Coherence）をリソースとして注入したとき、計算モデルがどう振る舞うか」を動的に解析する。

### 2.4 Resource Theory of Coherence

**【v3.1追加】** 量子性をリソースとして扱う枠組み：

- **Free States**: 古典状態（対角密度行列）
- **Free Operations**: 非coherence生成操作（古典計算に対応）
- **Resource States**: 重ね合わせ状態（非対角成分を持つ）

**問い**: 計算モデルに「1 qubitの重ね合わせ」を注入したとき、そのリソースはどう消費・増幅されるか？

### 2.5 Contextuality（文脈依存性）

**【v3.1追加】** Bell非局所性よりも基礎的とされる量子力学の非古典性：

- **Kochen-Specker定理**: 量子力学では測定結果は「文脈」（同時に測定する他のオブザーバブル）に依存する
- **計算との関連**: 計算の順序（文脈）が結果に影響するか？ = Church-Rosser性の破れ？

**問い**: 計算モデルにおいて「文脈依存性」はどのような形で現れ得るか？

### 2.6 計算モデルの普遍性

SK計算以外の計算モデルでも同様の結果が成り立つか：

- **λ計算**: Church-Rosser性との関係
- **チューリングマシン**: 構成の可逆性
- **セルオートマトン**: 可逆CAと量子CAの関係

---

## 3. 研究フェーズ

### Phase 8: 最小公理集合の探求（4週間）

#### 8.1 目的
計算から量子構造を得るための最小の追加公理セットを特定する。

#### 8.2 アプローチ

**【v3.1更新】GPTsの枠組みを導入**

**Step 1: GPTsによる計算モデルの定式化**
```
計算モデル = (Ω, Σ, T) where:
  Ω: 状態空間（計算基底の凸包 = シンプレックス Δ^{2^n-1}）
  Σ: 効果空間（測定の確率を与える写像）
  T: 変換群（置換群 S_{2^n} ⊂ GL(2^n, ℝ)）

量子モデル = (Ω_Q, Σ_Q, T_Q) where:
  Ω_Q: Bloch球（2次元の場合）/ 一般の状態空間
  Σ_Q: POVM
  T_Q: ユニタリ群 U(2^n)
```

**Step 2: 状態空間の変形条件**
- シンプレックス Δ → Bloch球 B への変形
- 何を追加すると状態空間が「膨らむ」か？
- 幾何学的条件: self-dual, bit-symmetric, etc.

**Step 3: 公理候補の形式化（GPTs言語）**
```
A1: State Space Extension — Ω が Δ より真に大きい
A2: Born Rule — 効果と状態の内積が確率を与える
A3: Reversibility — T が Ω 上で可逆（ユニタリに一般化）
A4: Non-commutativity — ∃ E, F ∈ Σ: 測定順序が結果に影響
A5: No-Cloning — ∄ T ∈ T: ρ ⊗ σ_0 → ρ ⊗ ρ for all ρ
A6: Contextuality — 測定結果が文脈に依存
```

**Step 4: 含意関係の解析**
- GPTsの枠組みで各公理の独立性を検証
- 計算モデル（シンプレックス + 置換群）から導出可能な公理の特定
- 「シンプレックス → Bloch球」の最小条件

#### 8.3 仮説

| ID | 仮説 | 予測 |
|----|------|------|
| **H8.1** | A1（状態空間拡張）は他から導出不可能 | 重ね合わせは根源的公理 |
| **H8.2** | A5（no-cloning）は A1 + A3 から導出可能 | no-cloningは派生定理 |
| **H8.3** | A4（非可換性）は A1 なしでは量子性を生成しない | 非可換だけでは不十分 |
| **H8.4** | A6（文脈依存性）は A1 + A4 を含意 | 文脈依存性は強い条件 |

#### 8.4 成果物
- GPTsでの計算モデル/量子モデルの定式化
- 公理間の含意関係グラフ（GPTs言語）
- 最小公理セットの候補リスト
- 「シンプレックス → Bloch球」の幾何学的条件

---

### Phase 9: 情報理論的アプローチ（3週間）

#### 9.1 目的
情報理論的原理から量子構造が導出されるかを検証する。

**【v3.1追加】循環論法の回避**:
古典計算には「未知の状態」が存在しないため、No-Cloningを直接議論すると循環する。
代わりに **Resource Theory** の視点を導入し、「外部からCoherenceリソースを注入したとき、計算モデルがどう振る舞うか」を解析する。

#### 9.2 アプローチ

**Step 1: Resource Theory of Coherenceの導入**
```
Free States (無料状態): 計算基底 |0⟩, |1⟩, ... （対角密度行列）
Free Operations (無料操作): 古典計算（置換、測定、古典混合）
Resource States (リソース状態): |+⟩ = (|0⟩+|1⟩)/√2 等の重ね合わせ

問い: 計算モデルに1 qubitの|+⟩を注入したとき、
      - リソースは消費されるか？増幅されるか？
      - どのような操作がリソースを生成/破壊するか？
```

**Step 2: 計算モデルへのリソース注入実験**
```python
# 概念的なコード
def inject_coherence(computation_model, resource_qubit):
    """計算モデルにCoherenceリソースを注入"""
    # 1. 古典計算（置換）をリソース qubit に適用
    # 2. リソース量（Coherence measure）の変化を測定
    # 3. 計算モデルがリソースをどう扱うか観察
```

**Step 3: 情報理論的公理の再定式化**
```
I1: No-Cloning — Coherenceリソースは複製で増幅しない
I2: No-Deleting — Coherenceリソースは自由に破壊できない
I3: No-Broadcasting — 非対角成分は分配できない
I4: Information Conservation — von Neumann entropy の保存
```

**Step 4: 量子構造への接続**
- Resource Theory の枠組みで A1-A4 を再解釈
- 「リソース生成能力」の観点から計算モデルと量子モデルを比較
- 「何がリソースを生成できるか」= 量子性の源泉

#### 9.3 仮説

| ID | 仮説 | 予測 |
|----|------|------|
| **H9.1** | 情報保存だけでは量子構造は出ない | 可逆計算が反例 |
| **H9.2** | No-cloningは量子構造の結果であり原因ではない | 逆導出は不可能 |
| **H9.3** | 古典計算はCoherenceリソースを生成できない | Free Operations の限界 |
| **H9.4** | Coherence生成能力が量子性の指標 | リソース理論的特徴づけ |

#### 9.4 成果物
- Resource Theory による計算モデルの特徴づけ
- 「Coherence生成能力」の定量的比較（SK, 可逆, 量子）
- 情報原理と量子公理の対応表（循環を回避した形式）
- 「計算モデルにリソースを注入する」実験フレームワーク

---

### Phase 10: 代替計算モデルでの検証（3週間）

#### 10.1 目的
SK計算以外の計算モデルでも同様の結果が成り立つかを検証し、結果の普遍性を確立する。

#### 10.2 対象モデル

1. **λ計算（型なし/型付き）**
   - β簡約の代数構造
   - Church-Rosser性と量子性の関係

2. **チューリングマシン**
   - 可逆チューリングマシン
   - Bennett's trick との関係

3. **量子セルオートマトン (QCA)**
   - 古典CA → QCA の遷移
   - QCA の計算能力

#### 10.3 検証項目

| モデル | Phase 4相当 | Phase 5相当 | Phase 7相当 |
|--------|-------------|-------------|-------------|
| λ計算 | 簡約の代数構造 | 連続時間β簡約？ | 演算子の可換性 |
| TM | 可逆TMの群構造 | 連続時間TM？ | 遷移の可換性 |
| CA | 可逆CAの群構造 | 量子CA | セル演算子の可換性 |

#### 10.4 仮説

| ID | 仮説 | 予測 |
|----|------|------|
| **H10.1** | λ計算でもSKと同様の結果 | 計算モデル非依存 |
| **H10.2** | 可逆TMも古典的（Sp埋め込み） | 可逆性では不十分 |
| **H10.3** | QCAは量子構造を持つ | QCAは量子公理を含む |

#### 10.5 成果物
- 各計算モデルでの代数構造分析
- 結果の普遍性の証明
- 計算モデル間の関係図

---

### Phase 11: 形式的検証（4週間）

#### 11.1 目的
主要な結果をCoq/Leanで形式的に証明し、厳密性を確保する。

**【v3.1更新】スコープの明確化**:
> 全定理の形式化は非現実的。**Theorem 1（Sp埋め込み）に焦点を絞る**。
> これは数学的に閉じており（有限群論→シンプレクティック幾何学）、
> 計算物理としての成果も高い。

#### 11.2 形式化対象（優先順位付け）

**【必須】Theorem 1: 置換行列 → Sp(2N,ℝ) 埋め込み**
```
∀ P ∈ S_{2^n}, ∃ M ∈ Sp(2·2^n, ℝ): P = π(M)
where π: Sp → GL は自然な射影
```
- 数学的に最も閉じている
- MathComp の有限群論ライブラリを活用可能
- 4週間で完遂可能な規模

**【オプション】No-Go Lemma の形式化**
- Theorem 1 が完了した場合のみ
- 「置換力学 → 重ね合わせなし」の形式的ステートメント

**【延期】その他**
- Hierarchy Theorem: 定義が複雑、形式化コスト高
- 公理間含意関係: GPTsの形式化が前提、現時点では困難

#### 11.3 使用ツール

- **Coq + MathComp**: 有限群論の形式化（主要ツール）
- **Lean 4 + Mathlib**: 代替（MathCompが合わない場合）

**不使用**:
- 量子力学ライブラリ（QWIRE, CoqQ等）: 今回のスコープ外

#### 11.4 具体的な形式化計画

**Week 1-2: 準備**
- MathComp の有限群論（perm, matrix）の習熟
- 置換行列の定義と基本性質の形式化

**Week 3: Sp埋め込みの構成**
- シンプレクティック形式 ω の定義
- 埋め込み写像 φ: S_n → Sp(2n, ℝ) の構成
- φ(P)†ω φ(P) = ω の証明

**Week 4: 検証と文書化**
- 証明の完全性チェック
- 形式化レポートの作成

#### 11.5 仮説

| ID | 仮説 | 予測 |
|----|------|------|
| **H11.1** | Theorem 1 は4週間で形式化可能 | MathCompで実現可能 |
| **H11.2** | 形式化により埋め込みの一意性に関する洞察 | 隠れた条件の発見 |

#### 11.6 成果物
- Coq証明コード（Theorem 1）
- 形式的定理のステートメントと証明概要
- 形式化レポート（何ができて何ができなかったか）

---

### Phase 12: 論文執筆と統合（3週間）

#### 12.1 目的
Phase 8-11の結果を統合し、論文としてまとめる。

#### 12.2 論文構成案

```
Title: "Minimal Axioms for Quantum Structure: 
        What Computation Cannot Derive"

1. Introduction
   - 計算から量子への問い
   - v1-v2の成果サマリー

2. Axiom Analysis
   - 候補公理の形式化
   - 含意関係と独立性

3. Information-Theoretic Perspective
   - 情報原理と量子構造
   - 導出可能性の限界

4. Universality Across Models
   - λ計算、TM、CAでの検証
   - 結果の普遍性

5. Formal Verification
   - 主要定理の形式証明
   - 証明の意義

6. Discussion
   - 最小公理セットの提案
   - 物理学・計算理論への示唆

7. Conclusion
   - 「計算→量子」の条件の完全な特定
```

#### 12.3 成果物
- 第3論文の完成原稿
- 補足資料（形式証明コード、実験データ）

---

## 4. スケジュール

| 週 | Phase | 主要タスク |
|----|-------|------------|
| 1-2 | 8 | 公理候補の形式化、含意関係の初期分析 |
| 3-4 | 8 | 最小公理セットの探求、仮説検証 |
| 5-6 | 9 | 情報理論的公理の形式化、計算モデルでの検証 |
| 7 | 9 | 情報原理と量子公理の関係分析 |
| 8-9 | 10 | λ計算、TM、CAでの検証 |
| 10 | 10 | 普遍性の証明、結果統合 |
| 11-12 | 11 | 形式的検証（Coq/Lean） |
| 13-14 | 11 | 形式証明の完成、レビュー |
| 15-16 | 12 | 論文執筆 |
| 17 | 12 | レビュー、投稿準備 |

**総期間**: 約17週間（4ヶ月）

---

## 5. 仮説一覧

### Phase 8: 最小公理集合（GPTs枠組み）
| ID | 仮説 | 予測 | 重要度 |
|----|------|------|--------|
| H8.1 | 状態空間拡張（A1）は根源的公理 | 他から導出不可能 | 高 |
| H8.2 | No-cloningは派生定理 | A1+A3から導出可能 | 中 |
| H8.3 | 非可換性だけでは不十分 | A1なしでは量子性なし | 高 |
| H8.4 | 文脈依存性（A6）は強い条件 | A1+A4を含意 | 中 |

### Phase 9: 情報理論（Resource Theory）
| ID | 仮説 | 予測 | 重要度 |
|----|------|------|--------|
| H9.1 | 情報保存では量子構造は出ない | 可逆計算が反例 | 高 |
| H9.2 | No-cloningは結果であり原因ではない | 逆導出不可能 | 中 |
| H9.3 | 古典計算はCoherence生成不可 | Free Operationsの限界 | 高 |
| H9.4 | Coherence生成能力が量子性の指標 | リソース理論的特徴づけ | 中 |

### Phase 10: 普遍性
| ID | 仮説 | 予測 | 重要度 |
|----|------|------|--------|
| H10.1 | λ計算でも同様の結果 | 計算モデル非依存 | 高 |
| H10.2 | 可逆TMも古典的 | Sp埋め込み | 中 |
| H10.3 | QCAは量子構造を持つ | 量子公理を含む | 低 |

### Phase 11: 形式化（スコープ限定）
| ID | 仮説 | 予測 | 重要度 |
|----|------|------|--------|
| H11.1 | Theorem 1は4週間で形式化可能 | MathCompで実現 | 高 |
| H11.2 | 形式化で埋め込みの一意性に洞察 | 隠れた条件の発見 | 中 |

---

## 6. リスクと対策

| リスク | 影響 | 対策 |
|--------|------|------|
| GPTs枠組みの習熟に時間 | Phase 8 遅延 | [Barrett2007], [Janotta2014] の早期レビュー |
| Resource Theory の理解不足 | Phase 9 遅延 | [Chitambar2019] のサーベイ論文を精読 |
| 公理分析が収束しない | Phase 8 遅延 | 既存文献の活用、範囲限定 |
| 形式化が困難 | Phase 11 遅延 | **Theorem 1のみに焦点**、MathComp活用 |
| λ計算の分析が複雑 | Phase 10 遅延 | 単純なケースに限定 |
| 循環論法に陥る | Phase 9 失敗 | Resource Theory で外部リソース注入の観点を維持 |
| 予想外の肯定的結果 | 計画変更 | 柔軟に対応、新発見として報告 |

---

## 7. 成功基準

### 7.1 最低基準（Phase 8-9 完了）
- 量子構造に必要な最小公理セットの候補を特定
- 情報理論的原理と量子公理の関係を明確化

### 7.2 標準基準（Phase 8-11 完了）
- 複数の計算モデルで結果の普遍性を確認
- 主要定理の形式的証明を完成

### 7.3 理想基準（全Phase完了）
- 第3論文の投稿
- 「計算→量子」の条件の完全な特定
- 形式的に検証されたNo-Go定理体系

---

## 8. 参考文献

### 量子公理論
- [Chiribella2011] Informational derivation of quantum theory
- [Hardy2001] Quantum theory from five reasonable axioms
- [Masanes2011] A derivation of quantum theory from physical requirements

### Generalized Probabilistic Theories (GPTs)【v3.1追加】
- [Barrett2007] Information processing in generalized probabilistic theories
- [Janotta2014] Generalized probabilistic theories: What determines the structure of quantum theory?
- [Muller2021] Probabilistic theories and reconstructions of quantum theory

### Resource Theory【v3.1追加】
- [Chitambar2019] Quantum resource theories (Reviews of Modern Physics)
- [Streltsov2017] Colloquium: Quantum coherence as a resource
- [Winter2016] Operational resource theory of coherence

### Contextuality【v3.1追加】
- [Kochen1967] The problem of hidden variables in quantum mechanics
- [Spekkens2005] Contextuality for preparations, transformations, and unsharp measurements
- [Abramsky2011] The sheaf-theoretic structure of non-locality and contextuality

### 情報理論的基礎
- [Wootters1982] No-cloning theorem
- [Pati2000] No-deleting theorem
- [Barnum2007] No-broadcasting theorem

### 計算と量子
- [Deutsch1985] Quantum Turing machines
- [Arrighi2012] Quantum cellular automata
- [Bennett1973] Reversible computation

### 形式的検証
- [Rand2018] Quantum verification in Coq
- [MathComp] Mathematical Components library for Coq
- [Mathlib] Lean mathematical library

---

## 変更履歴

| バージョン | 日付 | 変更内容 |
|------------|------|----------|
| v3.0 | 2024-12 | 初期計画: Phase 8-12 設計 |
| v3.1 | 2024-12 | **レビュー反映**: GPTs枠組み導入、Resource Theory追加、Contextuality追加、Phase 11スコープ縮小、循環論法への対策明記 |
| v3.2 | 2024-12 | **Phase 8 完了**: H8.1-H8.3 支持、A1（重ね合わせ）が根源的公理と特定 |
| v3.3 | 2024-12 | **Phase 9 完了**: H9.1-H9.4 全て支持、情報原理は量子構造の「結果」であり「原因」ではない |
| v3.4 | 2024-12 | **Phase 10 完了**: H10.1 支持、λ計算でも同様の結果 → 普遍性確立 |

---

## 付録

### A. v1-v2との関係

```
v1 (Phase 0-3): SK計算 → 量子? 
  結果: 独立性確認（否定的）
  
v2 (Phase 4-7): 可逆計算 → 量子?
  結果: 追加公理が必要（否定的）
  
v3 (Phase 8-12): 最小公理集合の特定
  問い: どの公理が「量子への跳躍」を可能にするか？
```

### B. 関連ファイル

| カテゴリ | パス |
|----------|------|
| 研究計画v1 | `docs/research_plan.md` |
| 研究計画v2 | `docs/research_plan_v2.md` |
| 論文1 | `papers/sk-quantum-independence/` |
| 論文2 | `papers/computational-quantum-limits/` |
| 実装 | `sk-quantum/` |

### C. 実装計画

Phase 8-12の実装は `sk-quantum/` 以下に追加:

```
sk-quantum/
├── phase8/
│   ├── axioms/
│   │   ├── formalization.py    # 公理の形式的定義
│   │   ├── implication.py      # 含意関係分析
│   │   └── minimal_set.py      # 最小セット探索
│   └── experiments/
├── phase9/
│   ├── information/
│   │   ├── no_cloning.py       # No-cloning検証
│   │   ├── no_deleting.py      # No-deleting検証
│   │   └── conservation.py     # 情報保存分析
│   └── experiments/
├── phase10/
│   ├── lambda_calculus/        # λ計算分析
│   ├── turing_machine/         # TM分析
│   └── cellular_automata/      # CA分析
├── phase11/
│   ├── coq/                    # Coq証明
│   └── lean/                   # Lean証明
└── phase12/
    └── integration/            # 結果統合
```

