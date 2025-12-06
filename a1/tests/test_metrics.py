"""
Tests for A1 Metrics (Complexity Measurement)

Author: Hiroshi Kohashiguchi
Date: December 2025
"""

import pytest
import sys
import os
import math

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from metrics import (
    A1Metrics, ClassicalMetrics, ComplexityResult, ClassicalComplexityResult,
    compare, ComparisonResult, run_benchmarks, BENCHMARK_A1, BENCHMARK_NUMPY
)


# =============================================================================
# A1Metrics Tests
# =============================================================================

class TestA1Metrics:
    """Tests for A1 complexity measurement."""
    
    @pytest.fixture
    def metrics(self):
        """Create an A1Metrics instance."""
        return A1Metrics()
    
    def test_tokenize_simple(self, metrics):
        """Test tokenizing simple expression."""
        tokens = metrics.tokenize("(H 0)")
        assert tokens == ['H', '0']  # Parentheses excluded
    
    def test_tokenize_nested(self, metrics):
        """Test tokenizing nested expression."""
        tokens = metrics.tokenize("(CNOT (H 0) 1)")
        assert tokens == ['CNOT', 'H', '0', '1']
    
    def test_tokenize_comments_removed(self, metrics):
        """Test that comments are removed during tokenization."""
        tokens = metrics.tokenize("(H 0) ; comment")
        assert 'comment' not in tokens
        assert tokens == ['H', '0']
    
    def test_count_tokens_bell(self, metrics):
        """Test token count for Bell state."""
        count = metrics.count_tokens("(CNOT (H 0) 1)")
        assert count == 4  # CNOT, H, 0, 1
    
    def test_count_tokens_ghz(self, metrics):
        """Test token count for GHZ state."""
        count = metrics.count_tokens("(CNOT (CNOT (H 0) 1) 2)")
        assert count == 6  # CNOT, CNOT, H, 0, 1, 2
    
    def test_complexity_formula(self, metrics):
        """Test complexity calculation formula."""
        # K = tokens × log2(vocab_size)
        source = "(H 0)"  # 2 tokens
        result = metrics.analyze(source)
        
        expected_bits = 2 * math.log2(metrics.vocabulary_size)
        assert abs(result.bits - expected_bits) < 0.01
    
    def test_analyze_returns_result(self, metrics):
        """Test analyze returns ComplexityResult."""
        result = metrics.analyze("(H 0)")
        assert isinstance(result, ComplexityResult)
        assert result.token_count == 2
        assert result.vocabulary_size == metrics.vocabulary_size
        assert len(result.tokens) == 2


# =============================================================================
# ClassicalMetrics Tests
# =============================================================================

class TestClassicalMetrics:
    """Tests for classical complexity measurement."""
    
    @pytest.fixture
    def metrics(self):
        """Create a ClassicalMetrics instance."""
        return ClassicalMetrics()
    
    def test_tokenize_python(self, metrics):
        """Test tokenizing Python code."""
        code = "x = 1 + 2"
        tokens = metrics.tokenize(code)
        assert 'x' in tokens
        assert '=' in tokens
        assert '1' in tokens
    
    def test_tokenize_removes_comments(self, metrics):
        """Test that Python comments are removed."""
        code = "x = 1  # comment"
        tokens = metrics.tokenize(code)
        assert 'comment' not in tokens
    
    def test_count_tokens_numpy(self, metrics):
        """Test token count for NumPy code."""
        code = "import numpy as np"
        count = metrics.count_tokens(code)
        assert count > 0
    
    def test_larger_vocabulary(self, metrics):
        """Test that classical vocabulary is larger."""
        assert ClassicalMetrics.VOCABULARY_SIZE > 100
    
    def test_analyze_returns_result(self, metrics):
        """Test analyze returns ClassicalComplexityResult."""
        result = metrics.analyze("x = 1")
        assert isinstance(result, ClassicalComplexityResult)
        assert result.token_count > 0


# =============================================================================
# Comparison Tests
# =============================================================================

class TestComparison:
    """Tests for A1 vs Classical comparison."""
    
    def test_compare_returns_result(self):
        """Test compare returns ComparisonResult."""
        result = compare("test", "(H 0)", "x = 1")
        assert isinstance(result, ComparisonResult)
        assert result.task == "test"
    
    def test_compare_bell_state(self):
        """Test comparison for Bell state."""
        result = compare(
            "Bell",
            BENCHMARK_A1['bell'],
            BENCHMARK_NUMPY['bell']
        )
        
        # A1 should have fewer tokens
        assert result.a1_result.token_count < result.classical_result.token_count
        
        # Token ratio should be > 10x
        assert result.token_ratio > 10
    
    def test_compare_ghz_state(self):
        """Test comparison for GHZ state."""
        result = compare(
            "GHZ",
            BENCHMARK_A1['ghz'],
            BENCHMARK_NUMPY['ghz']
        )
        
        # A1 should have fewer tokens
        assert result.a1_result.token_count < result.classical_result.token_count
    
    def test_compare_teleport(self):
        """Test comparison for teleportation."""
        result = compare(
            "Teleport",
            BENCHMARK_A1['teleport'],
            BENCHMARK_NUMPY['teleport']
        )
        
        # A1 should have fewer tokens
        assert result.a1_result.token_count < result.classical_result.token_count


# =============================================================================
# Benchmark Tests
# =============================================================================

class TestBenchmarks:
    """Tests for benchmark suite."""
    
    def test_benchmarks_defined(self):
        """Test that all benchmarks are defined."""
        for name in ['bell', 'ghz', 'teleport']:
            assert name in BENCHMARK_A1
            assert name in BENCHMARK_NUMPY
    
    def test_run_benchmarks(self):
        """Test running all benchmarks."""
        results = run_benchmarks()
        
        assert len(results) == 3
        
        for result in results:
            assert isinstance(result, ComparisonResult)
            assert result.token_ratio > 1  # A1 should always be shorter
    
    def test_benchmark_consistency(self):
        """Test that A1 is consistently shorter."""
        results = run_benchmarks()
        
        for result in results:
            # A1 should have fewer tokens in all cases
            assert result.a1_result.token_count < result.classical_result.token_count
            
            # Ratio should be significant (> 5x)
            assert result.token_ratio > 5


# =============================================================================
# Success Criteria Tests (from Research Plan)
# =============================================================================

class TestSuccessCriteria:
    """Tests for research plan success criteria."""
    
    def test_bell_tokens_under_5(self):
        """Test Bell state has < 5 A1 tokens."""
        metrics = A1Metrics()
        count = metrics.count_tokens(BENCHMARK_A1['bell'])
        assert count < 5
    
    def test_ghz_tokens_under_10(self):
        """Test GHZ state has < 10 A1 tokens."""
        metrics = A1Metrics()
        count = metrics.count_tokens(BENCHMARK_A1['ghz'])
        assert count < 10
    
    def test_teleport_tokens_under_25(self):
        """Test teleportation has < 25 A1 tokens."""
        metrics = A1Metrics()
        count = metrics.count_tokens(BENCHMARK_A1['teleport'])
        assert count < 25
    
    def test_ratio_over_10x(self):
        """Test A1/classical ratio > 10x for all benchmarks."""
        results = run_benchmarks()
        
        for result in results:
            assert result.token_ratio > 10, \
                f"{result.task}: ratio {result.token_ratio:.1f}x < 10x"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

