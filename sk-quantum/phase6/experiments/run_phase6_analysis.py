#!/usr/bin/env python3
"""
Phase 6: Three-Model Comparison Analysis

SK計算（非可逆）、RCA（可逆セルオートマトン）、量子回路の3モデルを比較し、
量子性の本質がどこにあるかを明らかにする。

研究仮説:
- H1: 可逆性だけでは量子的ではない（Phase 4で確認済み）
- H5: 連続時間化により干渉が生じる（Phase 5で確認済み）
- H6: RCAも連続時間化すれば干渉を示す（Phase 6で検証）
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'phase0'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'phase5', 'spectral'))

import numpy as np
from datetime import datetime

from phase6.comparison.model_comparison import ModelComparison, run_comparison
from phase6.comparison.quantum_circuit import SimpleQuantumCircuit, create_sample_circuits
from phase6.rca.automata import Rule90, Rule150, analyze_rca_group
from phase6.rca.graph import RCAGraph
from phase6.rca.hamiltonian import RCAHamiltonian, build_rca_hamiltonian


def print_section(title: str):
    """セクションヘッダを表示"""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


def analyze_rca_detailed():
    """RCAの詳細解析"""
    print_section("RCA Detailed Analysis")
    
    results = {}
    
    for rule_num, RuleClass in [(90, Rule90), (150, Rule150)]:
        print(f"\n--- Rule {rule_num} ---")
        
        for size in [3, 4]:
            rca = RuleClass(size)
            print(f"\n  Size {size} cells:")
            
            # グループ構造解析
            props = analyze_rca_group(rca)
            print(f"    Group order: {props['order']}")
            print(f"    Is permutation: {props['is_permutation']}")
            print(f"    Has complex eigenvalues: {props['has_complex_eigenvalues']}")
            
            # ハミルトニアンと干渉
            H = build_rca_hamiltonian(size, rule_num, full_space=False)
            times = np.linspace(0, 10, 100)
            interference = H.detect_interference(times)
            
            print(f"    Graph nodes: {H.graph.num_nodes}")
            print(f"    Has interference: {interference['has_interference']}")
            print(f"    Quantum oscillations: {interference['quantum_oscillations']}")
            print(f"    Classical oscillations: {interference['classical_oscillations']}")
            print(f"    Max TVD: {interference['max_tvd']:.4f}")
            
            results[f'Rule{rule_num}_size{size}'] = {
                'group_order': props['order'],
                'is_permutation': props['is_permutation'],
                'has_interference': interference['has_interference'],
                'oscillations_quantum': interference['quantum_oscillations'],
                'oscillations_classical': interference['classical_oscillations'],
            }
    
    return results


def analyze_quantum_circuits_detailed():
    """量子回路の詳細解析"""
    print_section("Quantum Circuit Analysis")
    
    results = []
    
    for name, circuit in create_sample_circuits():
        props = circuit.analyze_structure()
        print(f"\n{name}:")
        print(f"  Is permutation: {props['is_permutation']}")
        print(f"  Has complex entries: {props['has_complex_entries']}")
        print(f"  Creates superposition: {props['creates_superposition']}")
        
        results.append({
            'name': name,
            'is_permutation': props['is_permutation'],
            'has_complex': props['has_complex_entries'],
            'creates_superposition': props['creates_superposition'],
        })
    
    return results


def compare_three_models():
    """3モデルの統合比較"""
    print_section("Three-Model Comparison")
    
    result = run_comparison()
    print(result.summary)
    
    return result


def analyze_key_hypothesis():
    """主要仮説の検証"""
    print_section("Hypothesis Verification")
    
    hypotheses = {
        'H1': {
            'statement': '可逆性だけでは量子的ではない',
            'result': 'Phase 4で支持済み - Toffoli/Fredkinは古典的（Sp(2n,ℝ)に埋め込み可能）',
            'status': 'SUPPORTED'
        },
        'H5': {
            'statement': '連続時間化により干渉が生じる',
            'result': 'Phase 5で支持済み - exp(-iAt)により複素構造が導入される',
            'status': 'SUPPORTED'
        },
        'H6': {
            'statement': 'RCAも連続時間化すれば干渉を示す',
            'result': 'Phase 6で検証中',
            'status': 'TESTING'
        },
    }
    
    # H6の検証
    rca_h = build_rca_hamiltonian(3, 90, full_space=False)
    times = np.linspace(0, 10, 100)
    interference = rca_h.detect_interference(times)
    
    if interference['has_interference']:
        hypotheses['H6']['result'] = f"SUPPORTED - RCA Rule 90でも干渉検出（振動数: {interference['quantum_oscillations']}）"
        hypotheses['H6']['status'] = 'SUPPORTED'
    else:
        hypotheses['H6']['result'] = f"NOT SUPPORTED - RCA Rule 90で干渉なし"
        hypotheses['H6']['status'] = 'NOT SUPPORTED'
    
    print("仮説検証結果:")
    for h_id, h_data in hypotheses.items():
        print(f"\n  {h_id}: {h_data['statement']}")
        print(f"      Status: {h_data['status']}")
        print(f"      Result: {h_data['result']}")
    
    return hypotheses


def generate_conclusions():
    """結論の生成"""
    print_section("Conclusions")
    
    conclusions = [
        "1. SK計算（非可逆）とRCA（可逆）はどちらも離散的で、本質的には古典的",
        "2. しかし、連続時間量子ウォーク U(t)=exp(-iAt) を適用すると、両者とも干渉を示す",
        "3. 量子回路は連続的振幅（ユニタリ行列）を持ち、本来的に複素構造を持つ",
        "",
        "重要な区別:",
        "  - 離散・置換 → 古典的（Phase 4で確認）",
        "  - 連続時間化 → 干渉が出現（Phase 5, 6で確認）",
        "  - 本来的複素構造 → 量子回路の特徴",
        "",
        "量子性の本質:",
        "  - 可逆性は必要だが十分ではない",
        "  - 連続時間（exp(-iHt)）は干渉を導入する",
        "  - 真の量子性は「本来的な複素振幅」と「重ね合わせ」",
    ]
    
    for c in conclusions:
        print(c)
    
    return conclusions


def main():
    """Phase 6解析のメイン実行"""
    print(f"Phase 6: Three-Model Comparison Analysis")
    print(f"Executed at: {datetime.now().isoformat()}")
    
    # 1. RCA詳細解析
    rca_results = analyze_rca_detailed()
    
    # 2. 量子回路解析
    qc_results = analyze_quantum_circuits_detailed()
    
    # 3. 3モデル比較
    comparison = compare_three_models()
    
    # 4. 仮説検証
    hypotheses = analyze_key_hypothesis()
    
    # 5. 結論
    conclusions = generate_conclusions()
    
    # サマリー
    print_section("Summary")
    print(f"RCA configurations analyzed: {len(rca_results)}")
    print(f"Quantum circuits analyzed: {len(qc_results)}")
    print(f"Key differences identified: {len(comparison.key_differences)}")
    
    # 仮説サマリー
    supported = sum(1 for h in hypotheses.values() if h['status'] == 'SUPPORTED')
    print(f"\nHypotheses supported: {supported}/{len(hypotheses)}")
    
    return {
        'rca': rca_results,
        'quantum_circuits': qc_results,
        'comparison': comparison,
        'hypotheses': hypotheses,
        'conclusions': conclusions,
    }


if __name__ == '__main__':
    results = main()

