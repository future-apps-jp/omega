"""
Experiment 001: Sorkin公式によるSK計算の量子性検証
==================================================

Day 7: 最初の実験 - I₂, I₃ の体系的検証

研究の問い:
    SK計算の分岐確率は Sorkin の2次非加法性条件 (I₂ ≠ 0, I₃ = 0) を満たすか？

実験計画:
    1. 様々なSK式でMultiway graphを構築
    2. 複数の確率モデルで I₂, I₃ を計算
    3. 量子的シグネチャ（I₂ ≠ 0 かつ I₃ = 0）の有無を検証
    4. 結果の分析と考察
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sk_parser import parse, to_string, to_canonical
from multiway import build_multiway_graph, enumerate_paths
from probability import (
    UniformModel, LengthWeightedModel, 
    ComplexityWeightedModel, BranchWeightedModel
)
from sorkin import verify_sorkin, SorkinAnalyzer, compute_I2, compute_I3


def separator(title: str = "") -> None:
    """セパレータを表示"""
    print()
    print("=" * 70)
    if title:
        print(f" {title}")
        print("=" * 70)


def run_experiment():
    """メイン実験"""
    
    separator("EXPERIMENT 001: Sorkin Formula Verification")
    print("""
    研究の問い:
        SK計算の分岐確率は Sorkin の2次非加法性条件を満たすか？
        
    判定基準:
        - I₂ ≠ 0: 2次干渉あり（量子的シグネチャの候補）
        - I₂ ≠ 0 かつ I₃ = 0: 完全な量子的シグネチャ
        - I₂ = 0: 古典的（加法的確率）
    """)
    
    # ==========================================================================
    # 実験1: 基本的なSK式での検証
    # ==========================================================================
    separator("実験1: 基本的なSK式")
    
    basic_expressions = [
        ("K a b", "K-reduction のみ"),
        ("S K K a", "S K K = Identity"),
        ("S a b c", "S-reduction のみ"),
        ("K (S a b c) d", "ネストした式"),
    ]
    
    for expr_str, desc in basic_expressions:
        print(f"\n--- {expr_str} ({desc}) ---")
        graph = build_multiway_graph(parse(expr_str))
        paths = graph.get_all_paths()
        terminals = list(graph.terminals)
        
        print(f"  パス数: {len(paths)}")
        print(f"  終端数: {len(terminals)}")
        
        if len(paths) >= 2:
            model = UniformModel()
            analyzer = SorkinAnalyzer(graph, model)
            result = analyzer.find_quantum_signatures()
            print(f"  I₂≠0 のペア数: {result['quantum_pairs']}/{result['total_pairs']}")
        else:
            print(f"  (パスが1つのみ - 分析不可)")
    
    # ==========================================================================
    # 実験2: 分岐を持つSK式での検証
    # ==========================================================================
    separator("実験2: 分岐を持つSK式")
    
    branching_expressions = [
        ("(K a b) (K c d)", "2つのK-redex"),
        ("S (K a) (K b) c", "研究例"),
        ("S (K a b) c d", "ネストしたredex"),
        ("(K a b) (K c d) (K e f)", "3つのK-redex"),
        ("S (K a) (K b) (S c d e)", "複合的な分岐"),
    ]
    
    results_table = []
    
    for expr_str, desc in branching_expressions:
        print(f"\n--- {expr_str} ({desc}) ---")
        
        try:
            graph = build_multiway_graph(parse(expr_str), max_depth=30)
            paths = graph.get_all_paths()
            terminals = list(graph.terminals)
            
            print(f"  パス数: {len(paths)}")
            print(f"  終端数: {len(terminals)}")
            
            for t in terminals:
                t_paths = graph.get_paths_to(t)
                print(f"    → {to_string(t.expr)}: {len(t_paths)} パス")
            
            if len(paths) >= 2:
                model = UniformModel()
                analyzer = SorkinAnalyzer(graph, model)
                result = analyzer.find_quantum_signatures()
                
                print(f"  I₂≠0 のペア: {result['quantum_pairs']}/{result['total_pairs']}")
                
                if result['I2_values']:
                    nonzero_I2 = [v for v in result['I2_values'] if abs(v) > 1e-10]
                    if nonzero_I2:
                        print(f"  I₂ 値の例: {nonzero_I2[:3]}")
                
                if result['total_triples'] > 0:
                    print(f"  I₃=0 の3つ組: {result['quantum_triples']}/{result['total_triples']}")
                
                status = "🔔 Quantum" if result['has_quantum_signature'] else "✓ Classical"
                print(f"  判定: {status}")
                
                results_table.append({
                    "expression": expr_str,
                    "paths": len(paths),
                    "terminals": len(terminals),
                    "I2_nonzero": result['quantum_pairs'],
                    "I3_zero": result.get('quantum_triples', 0),
                    "status": status
                })
            else:
                print(f"  (パスが1つのみ - 分析不可)")
                results_table.append({
                    "expression": expr_str,
                    "paths": len(paths),
                    "terminals": len(terminals),
                    "I2_nonzero": 0,
                    "I3_zero": 0,
                    "status": "N/A"
                })
        
        except Exception as e:
            print(f"  エラー: {e}")
    
    # ==========================================================================
    # 実験3: 確率モデル間の比較
    # ==========================================================================
    separator("実験3: 確率モデル間の比較")
    
    test_expr = "(K a b) (K c d) (K e f)"
    print(f"対象式: {test_expr}")
    
    graph = build_multiway_graph(parse(test_expr))
    
    models = [
        UniformModel(),
        LengthWeightedModel(alpha=0.5),
        LengthWeightedModel(alpha=1.0),
        ComplexityWeightedModel(beta=0.1),
        ComplexityWeightedModel(beta=0.5),
        BranchWeightedModel(graph),
    ]
    
    print("\n確率モデル別の I₂ 統計:")
    print("-" * 60)
    
    for model in models:
        analyzer = SorkinAnalyzer(graph, model)
        result = analyzer.find_quantum_signatures()
        
        I2_values = result['I2_values']
        if I2_values:
            I2_nonzero = [v for v in I2_values if abs(v) > 1e-10]
            I2_max = max(I2_values) if I2_values else 0
            I2_min = min(I2_values) if I2_values else 0
            
            print(f"\n{model.name}:")
            print(f"  I₂≠0: {len(I2_nonzero)}/{len(I2_values)}")
            print(f"  I₂ range: [{I2_min:.4f}, {I2_max:.4f}]")
            print(f"  判定: {'🔔 Quantum' if result['has_quantum_signature'] else '✓ Classical'}")
    
    # ==========================================================================
    # 実験4: 詳細分析 - 研究例
    # ==========================================================================
    separator("実験4: 研究例の詳細分析")
    
    research_expr = "S (K a) (K b) c"
    print(f"研究例: {research_expr}")
    
    graph = build_multiway_graph(parse(research_expr))
    paths = graph.get_all_paths()
    
    print(f"\nMultiway Graph:")
    print(f"  パス数: {len(paths)}")
    
    for i, path in enumerate(paths, 1):
        print(f"\n  パス {i}: {path.path_id}")
        for j, node in enumerate(path.nodes):
            prefix = "    " if j == 0 else "    → "
            print(f"{prefix}{to_string(node.expr)}")
    
    if len(paths) >= 2:
        print("\n確率とI₂の分析:")
        
        for model in [UniformModel(), BranchWeightedModel(graph)]:
            print(f"\n  {model.name}:")
            analyzer = SorkinAnalyzer(graph, model)
            
            # 各パスの確率
            for path in paths:
                prob = analyzer.calc.get_path_probability(path)
                print(f"    P({path.path_id}) = {prob:.4f}")
            
            # I₂の計算
            if len(paths) >= 2:
                I2 = compute_I2(paths[0], paths[1], analyzer.calc)
                print(f"    I₂(path1, path2) = {I2:.6f}")
    
    # ==========================================================================
    # 結論
    # ==========================================================================
    separator("実験結果のまとめ")
    
    print("\n結果テーブル:")
    print("-" * 80)
    print(f"{'Expression':<30} {'Paths':>6} {'Terms':>6} {'I₂≠0':>8} {'Status':<12}")
    print("-" * 80)
    
    for r in results_table:
        print(f"{r['expression']:<30} {r['paths']:>6} {r['terminals']:>6} "
              f"{r['I2_nonzero']:>8} {r['status']:<12}")
    
    print("-" * 80)
    
    # 結論
    print("\n【結論】")
    
    quantum_found = any(r['status'] == '🔔 Quantum' for r in results_table)
    
    if quantum_found:
        print("""
    I₂ ≠ 0 となるケースが発見されました。
    
    ただし、これは以下の解釈が必要です：
    
    1. 現在の実装では、I₂ ≠ 0 は「複数パスが同じ終端に到達する」
       場合に、P(A∪B) > P(A) + P(B) となることを示しています。
    
    2. これは厳密には「量子的干渉」ではなく、確率の定義の問題です。
       - P(A∪B) = 終端への総確率
       - P(A), P(B) = 個別パスの確率
       → P(A∪B) = P(A) + P(B) が成り立つ定義を使用しているため
    
    3. 真の量子的非加法性を検証するには：
       - パスに複素振幅を割り当てる
       - 振幅の干渉効果を導入する
       必要があります。
    
    【次のステップ】
    - アプローチ A-C を探求し、複素振幅の導出を試みる
    - または、異なる確率定義で非加法性が現れるか検証する
    """)
    else:
        print("""
    全てのケースで I₂ = 0（古典的）でした。
    
    これは、現在の確率定義（パスへの重み付け）では
    古典的な加法性が保たれることを示しています。
    
    【次のステップ】
    - 異なる確率定義を試す
    - アプローチ A（代数的構造）へ移行
    """)
    
    separator("実験完了")


if __name__ == "__main__":
    run_experiment()

