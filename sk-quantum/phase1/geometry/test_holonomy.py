"""
Tests for Path Space Holonomy
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'phase0'))

import pytest
import numpy as np
import cmath

from sk_parser import parse, to_canonical
from multiway import build_multiway_graph
from holonomy import (
    PathOperation, ComputationPath, ComputationLoop,
    extract_computation_paths, find_loops,
    Connection, constant_phase_connection, depth_dependent_connection,
    complexity_dependent_connection, information_erasure_connection,
    HolonomyGroupAnalysis, run_holonomy_analysis
)
from reduction import RedexType


# =============================================================================
# PathOperation Tests
# =============================================================================

class TestPathOperation:
    
    def test_repr_s_redex(self):
        op = PathOperation(
            redex_type=RedexType.S_REDEX,
            redex_path="L",
            source_canonical="((S a) b)",
            target_canonical="result"
        )
        assert "S@L" in repr(op)
    
    def test_repr_k_redex(self):
        op = PathOperation(
            redex_type=RedexType.K_REDEX,
            redex_path="",
            source_canonical="((K a) b)",
            target_canonical="a"
        )
        assert "K@root" in repr(op)


# =============================================================================
# ComputationPath Tests
# =============================================================================

class TestComputationPath:
    
    def test_length(self):
        ops = [
            PathOperation(RedexType.S_REDEX, "", "a", "b"),
            PathOperation(RedexType.K_REDEX, "L", "b", "c"),
        ]
        path = ComputationPath(ops, "a", "c")
        assert path.length == 2
    
    def test_signature(self):
        ops = [
            PathOperation(RedexType.S_REDEX, "", "a", "b"),
            PathOperation(RedexType.K_REDEX, "L", "b", "c"),
        ]
        path = ComputationPath(ops, "a", "c")
        sig = path.signature
        assert "S@" in sig
        assert "K@L" in sig
    
    def test_operation_types(self):
        ops = [
            PathOperation(RedexType.S_REDEX, "", "a", "b"),
            PathOperation(RedexType.K_REDEX, "L", "b", "c"),
        ]
        path = ComputationPath(ops, "a", "c")
        types = path.operation_types()
        assert types == [RedexType.S_REDEX, RedexType.K_REDEX]


# =============================================================================
# ComputationLoop Tests
# =============================================================================

class TestComputationLoop:
    
    def test_area(self):
        ops1 = [PathOperation(RedexType.S_REDEX, "", "a", "b")]
        ops2 = [
            PathOperation(RedexType.K_REDEX, "", "a", "c"),
            PathOperation(RedexType.K_REDEX, "L", "c", "b"),
        ]
        path1 = ComputationPath(ops1, "a", "b")
        path2 = ComputationPath(ops2, "a", "b")
        
        loop = ComputationLoop(path1, path2, "a", "b")
        assert loop.area == 1
    
    def test_operation_difference(self):
        ops1 = [
            PathOperation(RedexType.S_REDEX, "", "a", "b"),
            PathOperation(RedexType.S_REDEX, "L", "b", "c"),
        ]
        ops2 = [
            PathOperation(RedexType.K_REDEX, "", "a", "d"),
            PathOperation(RedexType.K_REDEX, "L", "d", "c"),
        ]
        path1 = ComputationPath(ops1, "a", "c")
        path2 = ComputationPath(ops2, "a", "c")
        
        loop = ComputationLoop(path1, path2, "a", "c")
        diff = loop.operation_difference()
        
        assert diff['delta_S'] == 2  # path1: 2S, path2: 0S
        assert diff['delta_K'] == -2  # path1: 0K, path2: 2K


# =============================================================================
# Path Extraction Tests
# =============================================================================

class TestPathExtraction:
    
    def test_extract_paths_simple(self):
        expr = parse("K a b")
        graph = build_multiway_graph(expr, max_depth=5)
        paths = extract_computation_paths(graph)
        
        # K a b -> a は1パス
        assert len(paths) >= 1
    
    def test_extract_paths_multiple(self):
        expr = parse("(K a b) (K c d)")
        graph = build_multiway_graph(expr, max_depth=5)
        paths = extract_computation_paths(graph)
        
        # 複数パスがあるはず
        assert len(paths) >= 1


# =============================================================================
# Loop Detection Tests
# =============================================================================

class TestLoopDetection:
    
    def test_find_loops_same_endpoint(self):
        # 同じ終点を持つ2つのパスを作成
        ops1 = [PathOperation(RedexType.S_REDEX, "", "start", "end")]
        ops2 = [
            PathOperation(RedexType.K_REDEX, "", "start", "mid"),
            PathOperation(RedexType.K_REDEX, "L", "mid", "end"),
        ]
        path1 = ComputationPath(ops1, "start", "end")
        path2 = ComputationPath(ops2, "start", "end")
        
        loops = find_loops([path1, path2])
        assert len(loops) == 1
    
    def test_find_loops_different_endpoints(self):
        # 異なる終点を持つパス
        ops1 = [PathOperation(RedexType.S_REDEX, "", "start", "end1")]
        ops2 = [PathOperation(RedexType.K_REDEX, "", "start", "end2")]
        path1 = ComputationPath(ops1, "start", "end1")
        path2 = ComputationPath(ops2, "start", "end2")
        
        loops = find_loops([path1, path2])
        assert len(loops) == 0


# =============================================================================
# Connection Tests
# =============================================================================

class TestConnection:
    
    def test_constant_phase_s(self):
        conn = constant_phase_connection(np.pi/4, 0)
        op = PathOperation(RedexType.S_REDEX, "", "a", "b")
        
        ops = [op]
        path = ComputationPath(ops, "a", "b")
        
        phase = conn.parallel_transport(path)
        expected = cmath.exp(1j * np.pi/4)
        
        assert abs(phase - expected) < 1e-10
    
    def test_constant_phase_k(self):
        conn = constant_phase_connection(0, -np.pi/4)
        op = PathOperation(RedexType.K_REDEX, "", "a", "b")
        
        ops = [op]
        path = ComputationPath(ops, "a", "b")
        
        phase = conn.parallel_transport(path)
        expected = cmath.exp(-1j * np.pi/4)
        
        assert abs(phase - expected) < 1e-10
    
    def test_holonomy_trivial_loop(self):
        conn = constant_phase_connection(np.pi/4, np.pi/4)
        
        # 同じ操作の2パス → ホロノミー = 1
        ops = [PathOperation(RedexType.S_REDEX, "", "a", "b")]
        path1 = ComputationPath(ops, "a", "b")
        path2 = ComputationPath(ops, "a", "b")
        
        loop = ComputationLoop(path1, path2, "a", "b")
        h = conn.holonomy(loop)
        
        assert abs(h - 1.0) < 1e-10
    
    def test_holonomy_nontrivial(self):
        conn = constant_phase_connection(np.pi/2, 0)
        
        # path1: S操作1回
        ops1 = [PathOperation(RedexType.S_REDEX, "", "a", "b")]
        # path2: K操作2回（位相なし）
        ops2 = [
            PathOperation(RedexType.K_REDEX, "", "a", "c"),
            PathOperation(RedexType.K_REDEX, "L", "c", "b"),
        ]
        
        path1 = ComputationPath(ops1, "a", "b")
        path2 = ComputationPath(ops2, "a", "b")
        
        loop = ComputationLoop(path1, path2, "a", "b")
        h = conn.holonomy(loop)
        
        # path1: exp(iπ/2) = i
        # path2: 1
        # holonomy = i / 1 = i
        expected = 1j
        
        assert abs(h - expected) < 1e-10


# =============================================================================
# HolonomyGroupAnalysis Tests
# =============================================================================

class TestHolonomyGroupAnalysis:
    
    def test_analyze_empty_loops(self):
        conn = constant_phase_connection(0, 0)
        analyzer = HolonomyGroupAnalysis(conn)
        
        result = analyzer.analyze_loops([])
        assert result['num_loops'] == 0
        assert result['is_trivial'] == True
    
    def test_analyze_with_loops(self):
        conn = constant_phase_connection(np.pi/4, -np.pi/4)
        analyzer = HolonomyGroupAnalysis(conn)
        
        # テストループを作成
        ops1 = [PathOperation(RedexType.S_REDEX, "", "a", "b")]
        ops2 = [PathOperation(RedexType.K_REDEX, "", "a", "b")]
        
        path1 = ComputationPath(ops1, "a", "b")
        path2 = ComputationPath(ops2, "a", "b")
        
        loop = ComputationLoop(path1, path2, "a", "b")
        
        result = analyzer.analyze_loops([loop])
        assert result['num_loops'] == 1


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    
    def test_run_holonomy_analysis(self):
        expr = parse("S (K a) (K b) c")
        results = run_holonomy_analysis(expr, max_depth=5, verbose=False)
        
        assert 'num_paths' in results
        assert 'num_loops' in results
    
    def test_run_holonomy_analysis_complex(self):
        expr = parse("(K a b) (K c d)")
        results = run_holonomy_analysis(expr, max_depth=5, verbose=False)
        
        assert 'num_paths' in results


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])


