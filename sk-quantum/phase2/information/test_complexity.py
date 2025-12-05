"""
Tests for Information-Theoretic Approach
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'phase0'))

import pytest
import numpy as np
import cmath

from sk_parser import parse, to_canonical, S, K, Var, App
from multiway import build_multiway_graph
from complexity import (
    kolmogorov_size, kolmogorov_string_length, kolmogorov_compressed,
    kolmogorov_depth_weighted, kolmogorov_variable_entropy,
    InformationChange, PathInformation,
    InformationPhaseCalculator, extract_path_information,
    run_information_analysis
)
from reduction import RedexType


# =============================================================================
# Kolmogorov Complexity Tests
# =============================================================================

class TestKolmogorovMeasures:
    
    def test_size_basic(self):
        expr = parse("S K K")
        # App(App(S, K), K) = 5 nodes (2 App + S + K + K)
        assert kolmogorov_size(expr) == 5
    
    def test_size_with_variables(self):
        expr = parse("S a b c")
        # App(App(App(S, a), b), c) = 7 nodes (3 App + S + a + b + c)
        assert kolmogorov_size(expr) == 7
    
    def test_string_length(self):
        expr = parse("S K K")
        length = kolmogorov_string_length(expr)
        assert length > 0
    
    def test_compressed_size(self):
        expr = parse("S K K")
        compressed = kolmogorov_compressed(expr)
        assert compressed > 0
    
    def test_depth_weighted(self):
        expr1 = parse("S K K")
        expr2 = parse("S (K (K K)) K")
        
        # より深い式はより大きな値を持つはず
        dw1 = kolmogorov_depth_weighted(expr1)
        dw2 = kolmogorov_depth_weighted(expr2)
        assert dw2 > dw1
    
    def test_variable_entropy(self):
        expr1 = parse("S K K")  # 変数なし
        expr2 = parse("S a b c")  # 3変数
        
        assert kolmogorov_variable_entropy(expr1) == 0
        assert kolmogorov_variable_entropy(expr2) == 3


# =============================================================================
# Information Change Tests
# =============================================================================

class TestInformationChange:
    
    def test_k_reduction_erasure(self):
        source = parse("K a b")  # K a b → a
        target = parse("a")
        
        change = InformationChange(
            source_expr=source,
            target_expr=target,
            redex_type=RedexType.K_REDEX,
            redex_path=""
        )
        
        assert change.is_information_erasing == True
        assert change.delta_size < 0  # サイズ減少
    
    def test_s_reduction_no_erasure(self):
        source = parse("S a b c")
        target = parse("(a c) (b c)")
        
        change = InformationChange(
            source_expr=source,
            target_expr=target,
            redex_type=RedexType.S_REDEX,
            redex_path=""
        )
        
        assert change.is_information_erasing == False


# =============================================================================
# Path Information Tests
# =============================================================================

class TestPathInformation:
    
    def test_summary(self):
        source = parse("K a b")
        target = parse("a")
        
        change = InformationChange(
            source_expr=source,
            target_expr=target,
            redex_type=RedexType.K_REDEX,
            redex_path=""
        )
        
        path_info = PathInformation(
            changes=[change],
            start_expr=source,
            end_expr=target
        )
        
        summary = path_info.summary()
        assert summary['num_steps'] == 1
        assert summary['num_k_reductions'] == 1
        assert summary['num_s_reductions'] == 0


# =============================================================================
# Phase Calculator Tests
# =============================================================================

class TestInformationPhaseCalculator:
    
    def test_linear_formula(self):
        calculator = InformationPhaseCalculator('linear')
        
        # ダミーのパス情報を作成
        source = parse("K a b")
        target = parse("a")
        
        change = InformationChange(
            source_expr=source,
            target_expr=target,
            redex_type=RedexType.K_REDEX,
            redex_path=""
        )
        
        path_info = PathInformation(
            changes=[change],
            start_expr=source,
            end_expr=target
        )
        
        phase = calculator.compute_phase(path_info, alpha=0.1)
        
        # 位相因子の絶対値は1であるべき
        assert abs(abs(phase) - 1.0) < 1e-10
    
    def test_landauer_formula(self):
        calculator = InformationPhaseCalculator('landauer')
        
        source = parse("K a b")
        target = parse("a")
        
        change = InformationChange(
            source_expr=source,
            target_expr=target,
            redex_type=RedexType.K_REDEX,
            redex_path=""
        )
        
        path_info = PathInformation(
            changes=[change],
            start_expr=source,
            end_expr=target
        )
        
        phase = calculator.compute_phase(path_info)
        
        # Landauer式では K簡約1回で π/4 の位相
        expected_phase = np.pi / 4
        actual_phase = cmath.phase(phase)
        
        assert abs(actual_phase - expected_phase) < 1e-10


# =============================================================================
# Path Extraction Tests
# =============================================================================

class TestPathExtraction:
    
    def test_extract_basic(self):
        expr = parse("K a b")
        graph = build_multiway_graph(expr, max_depth=5)
        path_infos = extract_path_information(graph)
        
        assert len(path_infos) >= 1
    
    def test_extract_multiple_paths(self):
        expr = parse("(K a b) (K c d)")
        graph = build_multiway_graph(expr, max_depth=5)
        path_infos = extract_path_information(graph)
        
        assert len(path_infos) >= 1


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    
    def test_run_analysis(self):
        expr = parse("S (K a) (K b) c")
        results = run_information_analysis(expr, max_depth=5, verbose=False)
        
        assert 'num_paths' in results
    
    def test_run_analysis_complex(self):
        expr = parse("(K a b) (K c d)")
        results = run_information_analysis(expr, max_depth=5, verbose=False)
        
        assert 'num_paths' in results


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

