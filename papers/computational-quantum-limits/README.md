# Limits of Deriving Quantum Structure from Reversible Computation: Symplectic Embedding of Reversible Gates and the Hierarchy of Quantum Resources

## 概要

本論文は、研究計画v2 (Phase 4-7) の成果をまとめたものです。

**主題**: 可逆計算から量子構造を導出することの限界

## 主要な発見

1. **可逆性だけでは不十分** (Phase 4)
   - Toffoli/Fredkinゲートは古典的（Sp(2n,ℝ)に埋め込み可能）
   - 複素構造 J² = -I は存在しない

2. **連続時間化で干渉が出現** (Phase 5)
   - U(t) = exp(-iAt) により干渉パターンが生じる
   - しかし重ね合わせは生成されない

3. **量子性の階層構造** (Phase 6)
   - Level 0: 非可逆計算 → 古典的
   - Level 1: 可逆計算 → 古典的（置換群）
   - Level 2: 連続時間発展 → 干渉出現
   - Level 3: 量子回路 → 重ね合わせ・もつれ

4. **非可換性も十分条件ではない** (Phase 7)
   - [Ŝ, K̂] = 0（この定義では）
   - 重ね合わせ要件: 0/4 満たす

## ファイル構成

```
computational-quantum-limits/
├── README.md          # 本ファイル
├── main.tex           # 論文本体（LaTeX）
└── references.bib     # 参考文献（BibTeX）
```

## コンパイル方法

```bash
# pdflatexでコンパイル
pdflatex main.tex
pdflatex main.tex  # 参照を解決するため2回実行
```

## 関連文書

- 第1論文: `papers/sk-quantum-independence/` - SK計算と量子構造の独立性
- 研究計画: `docs/research_plan_v2.md`
- 実装: `sk-quantum/phase4-7/`

## 実験結果

| Phase | 結果ファイル |
|-------|-------------|
| Phase 4 | `sk-quantum/phase4/experiments/RESULTS_004_algebra.md` |
| Phase 5 | `sk-quantum/phase5/experiments/RESULTS_005_spectral.md` |
| Phase 6 | `sk-quantum/phase6/experiments/RESULTS_006_comparison.md` |
| Phase 7 | `sk-quantum/phase7/experiments/RESULTS_007_noncommutative.md` |

## 統計

- 総テスト数: 221 (Phase 4: 18, Phase 5: 13, Phase 6: 31, Phase 7: 15)
- 仮説検証: H1, H5, H6 支持、H4 部分支持、H2 inconclusive

## 著者

Hiroshi Kohashiguchi  
Independent Researcher, Tokyo, Japan

