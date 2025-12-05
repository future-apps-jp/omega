# Minimal Axioms for Quantum Structure: What Computation Cannot Derive

## 概要

この論文は、Phase 8-11の研究成果を統合した第3論文です。

## 主要な貢献

1. **A1（状態空間拡張/重ね合わせ）が唯一の根源的公理**
   - 他の公理や計算からは導出不可能

2. **情報理論的原理は量子構造の「結果」**
   - No-Cloningは重ね合わせを前提とする
   - 循環論法を回避するResource Theory分析

3. **結果の普遍性**
   - SK計算、RCA、λ計算で確認
   - 計算モデル非依存

4. **形式検証（NEW）**
   - 主定理をCoq/MathCompで完全証明
   - 補足資料としてCoqコードを添付

## 構成

1. Introduction - 問題設定と先行研究
2. Framework - GPTsとResource Theory
3. Axiom Analysis - 公理の分析（Phase 8）
4. Information-Theoretic Analysis - 情報原理（Phase 9）
5. Universality - 普遍性検証（Phase 10）
6. Discussion - 考察、形式検証、含意
7. Conclusion - 結論
8. **Appendix A: Coq Formalization** - 形式検証コード

## ビルド

```bash
pdflatex main.tex
pdflatex main.tex  # 参照解決のため2回
```

## ファイル一覧

| ファイル | 説明 |
|----------|------|
| `main.tex` | 論文本文 |
| `main.pdf` | コンパイル済みPDF（13ページ） |
| `PermSymplectic.v` | **Coq形式検証コード（補足資料）** |

## 関連ファイル

- Phase 8 結果: `../../sk-quantum/phase8/experiments/RESULTS_008_axioms.md`
- Phase 9 結果: `../../sk-quantum/phase9/experiments/RESULTS_009_information.md`
- Phase 10 結果: `../../sk-quantum/phase10/experiments/RESULTS_010_lambda.md`
- Phase 11 結果: `../../sk-quantum/phase11/RESULTS_011_formal.md`

## 先行論文

1. `../sk-quantum-independence/` - SK計算と量子構造の独立性
2. `../computational-quantum-limits/` - 可逆計算から量子構造導出の限界

