# Structural Origins of Superposition

## 概要

Phase 13-17の研究成果を統合した第4論文。

## タイトル

**Structural Origins of Superposition: Symmetry and Causality Constraints on Quantum Structure**

## 主要な貢献

### 1. 対称性によるA1の必然性（Phase 13）

- 置換群はスピノル表現を持たない（J² = -I なし）
- Lorentz対称性 + スピン → SL(2,C)スピノル → A1必須
- 2π回転で-1：複素数なしでは不可能

### 2. ContextualityによるA1の必然性（Phase 14）

- KS Contextuality → 非可換観測量 → 非直交状態 → A1
- Spekkensモデル ≠ 計算制限（認識論 vs 存在論）
- 古典計算はContextualになれない

### 3. 因果構造はA1を生成しない（Phase 15）

- Multiway Graph: No-Go定理適用
- Causal Set: 経路積分でA1を明示的に導入
- 因果構造は「舞台」を提供、量子性は生成しない

### 4. 統合的結論（Phase 16）

```
A1は:
1. 物理的対称性（Lorentz + スピン）により必要とされる
2. Contextualityの前提条件である
3. 計算や因果構造から導出できない
4. 仮定されるべき原始的公理である
```

## 仮説検証

| 仮説 | 結果 |
|------|------|
| H13.1-3 | ✅ 支持 |
| H14.1, H14.3 | ✅ 支持 |
| H15.1-3 | ✅ 支持 |

**全8仮説が支持された**

## ファイル一覧

| ファイル | 説明 |
|----------|------|
| `main.tex` | 論文本文 |
| `main.pdf` | コンパイル済みPDF（7ページ） |

## ビルド

```bash
pdflatex main.tex
pdflatex main.tex  # 参照解決のため2回
```

## 関連ファイル

- Phase 13 結果: `../../sk-quantum/phase13/experiments/RESULTS_013_symmetry.md`
- Phase 14 コード: `../../sk-quantum/phase14/contextuality/spekkens_model.py`
- Phase 15 コード: `../../sk-quantum/phase15/causal/causal_sets.py`
- 統合結果: `../../sk-quantum/phase13/experiments/RESULTS_013_016_synthesis.md`
- 実験的含意: `../../sk-quantum/phase17/RESULTS_017_experimental.md`

## 先行論文

1. Paper 1: SK計算と量子構造の独立性
2. Paper 2: 可逆計算から量子構造導出の限界
3. Paper 3: 量子構造の最小公理（決定版統合論文）
4. **Paper 4**: 本論文 - 重ね合わせの構造的起源

---

*Phase 18 完了 — December 2025*

