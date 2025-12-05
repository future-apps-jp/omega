# Minimal Axioms for Quantum Structure: What Computation Cannot Derive

## 論文バージョン

### 📕 統合版（ジャーナル投稿用）— `main_unified.tex`
**Minimal Axioms for Quantum Structure: What Computation Cannot Derive**

3部作を統合した「決定版」論文。12ページ。

**構成**:
1. Introduction - 問題提起
2. The Limits of Computation - SK計算と可逆ゲートの結果を統合
3. The No-Go Theorem - シンプレクティック埋め込み（形式検証付き）
4. Axiomatic Reconstruction - GPTsによるA1同定
5. Universality - 計算モデル非依存性
6. Discussion & Conclusion
7. Appendix: Coq Formalization

**ビルド**:
```bash
pdflatex main_unified.tex
pdflatex main_unified.tex  # 参照解決のため2回
```

---

### 📗 Phase III 単独版 — `main.tex`
公理解析に焦点を当てた単独論文。14ページ。

---

## 主要な貢献

1. **No-Go定理（形式検証済み）**
   - 可逆n-bitゲートはSp(2·2ⁿ,ℝ)に埋め込まれる
   - Coq/MathCompで完全証明

2. **A1（状態空間拡張/重ね合わせ）が唯一の根源的公理**
   - 他の公理や計算からは導出不可能

3. **普遍性**
   - SK、可逆ゲート、RCA、λ計算で確認
   - 計算モデル非依存

## ファイル一覧

| ファイル | 説明 |
|----------|------|
| `main_unified.tex` | **統合版論文（ジャーナル投稿用）** |
| `main_unified.pdf` | 統合版PDF（12ページ） |
| `main.tex` | Phase III単独版論文 |
| `main.pdf` | Phase III単独版PDF（14ページ） |
| `PermSymplectic.v` | Coq形式検証コード（補足資料） |
| `references.bib` | 参考文献データベース |

## 先行論文（統合版に組み込み済み）

1. **Paper I**: `../sk-quantum-independence/` - SK計算と量子構造の独立性
2. **Paper II**: `../computational-quantum-limits/` - 可逆計算から量子構造導出の限界

## 関連ファイル

- Phase 8 結果: `../../sk-quantum/phase8/experiments/RESULTS_008_axioms.md`
- Phase 9 結果: `../../sk-quantum/phase9/experiments/RESULTS_009_information.md`
- Phase 10 結果: `../../sk-quantum/phase10/experiments/RESULTS_010_lambda.md`
- Phase 11 結果: `../../sk-quantum/phase11/RESULTS_011_formal.md`
