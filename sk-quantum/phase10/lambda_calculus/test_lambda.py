"""
Tests for Phase 10: Lambda Calculus Analysis

Tests for lambda calculus implementation and comparison with SK.
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lambda_calculus.lambda_core import (
    Term, Var, Abs, App,
    parse, beta_reduce, beta_reduce_step, is_normal_form,
    I, K, S, TRUE, FALSE, church_numeral,
    LambdaMultiwayGraph,
    fresh_var,
)

from lambda_calculus.lambda_analysis import (
    analyze_reduction_algebra,
    check_symplectic_embedding,
    analyze_lambda_coherence,
    compare_lambda_sk,
    run_full_lambda_analysis,
)


# =============================================================================
# Lambda Core Tests
# =============================================================================

class TestTermConstruction:
    """Tests for term construction."""
    
    def test_variable(self):
        """Test variable construction."""
        x = Var('x')
        assert x.name == 'x'
        assert x.free_vars() == {'x'}
    
    def test_abstraction(self):
        """Test abstraction construction."""
        x = Var('x')
        abs_term = Abs('x', x)
        assert abs_term.var == 'x'
        assert abs_term.free_vars() == set()  # x is bound
    
    def test_application(self):
        """Test application construction."""
        x = Var('x')
        y = Var('y')
        app = App(x, y)
        assert app.free_vars() == {'x', 'y'}
    
    def test_free_vars_nested(self):
        """Test free variables in nested terms."""
        # λx.y has free variable y
        term = Abs('x', Var('y'))
        assert term.free_vars() == {'y'}
        
        # λx.x has no free variables
        term2 = Abs('x', Var('x'))
        assert term2.free_vars() == set()


class TestParsing:
    """Tests for parsing."""
    
    def test_parse_variable(self):
        """Test parsing variables."""
        term = parse("x")
        assert isinstance(term, Var)
        assert term.name == 'x'
    
    def test_parse_identity(self):
        """Test parsing identity function."""
        term = parse("λx.x")
        assert isinstance(term, Abs)
        assert term.var == 'x'
        assert isinstance(term.body, Var)
    
    def test_parse_application(self):
        """Test parsing application."""
        term = parse("(λx.x) y")
        assert isinstance(term, App)
        assert isinstance(term.func, Abs)
        assert isinstance(term.arg, Var)
    
    def test_parse_k_combinator(self):
        """Test parsing K combinator."""
        term = parse("λx.λy.x")
        assert isinstance(term, Abs)
        assert term.var == 'x'
        assert isinstance(term.body, Abs)
        assert term.body.var == 'y'
    
    def test_parse_backslash_syntax(self):
        """Test parsing with backslash instead of λ."""
        term = parse("\\x.x")
        assert isinstance(term, Abs)


class TestSubstitution:
    """Tests for substitution."""
    
    def test_substitute_variable(self):
        """Test substituting in a variable."""
        x = Var('x')
        result = x.substitute('x', Var('y'))
        assert result == Var('y')
    
    def test_substitute_different_variable(self):
        """Test substituting different variable."""
        x = Var('x')
        result = x.substitute('y', Var('z'))
        assert result == Var('x')
    
    def test_substitute_in_abstraction_bound(self):
        """Test substitution doesn't affect bound variables."""
        term = Abs('x', Var('x'))
        result = term.substitute('x', Var('y'))
        # x is bound, so no substitution
        assert result == term
    
    def test_substitute_in_application(self):
        """Test substitution in application."""
        term = App(Var('x'), Var('y'))
        result = term.substitute('x', Var('z'))
        assert result == App(Var('z'), Var('y'))


class TestBetaReduction:
    """Tests for β-reduction."""
    
    def test_identity_application(self):
        """Test (λx.x) y → y"""
        term = App(I, Var('y'))
        result, _ = beta_reduce(term)
        assert result == Var('y')
    
    def test_k_combinator(self):
        """Test K a b → a"""
        term = App(App(K, Var('a')), Var('b'))
        result, _ = beta_reduce(term)
        assert result == Var('a')
    
    def test_normal_form_detection(self):
        """Test detection of normal forms."""
        assert is_normal_form(Var('x'))
        assert is_normal_form(Abs('x', Var('x')))
        assert not is_normal_form(App(I, Var('y')))
    
    def test_s_combinator_partial(self):
        """Test partial application of S."""
        # S a b c → a c (b c)
        s_abc = App(App(App(S, Var('a')), Var('b')), Var('c'))
        result, _ = beta_reduce(s_abc)
        
        # Result should be a c (b c)
        expected = App(App(Var('a'), Var('c')), App(Var('b'), Var('c')))
        assert result == expected
    
    def test_church_numerals(self):
        """Test Church numeral construction."""
        zero = church_numeral(0)
        one = church_numeral(1)
        two = church_numeral(2)
        
        # zero = λf.λx.x
        assert is_normal_form(zero)
        
        # one = λf.λx.f x
        assert is_normal_form(one)


class TestMultiwayGraph:
    """Tests for multiway graph construction."""
    
    def test_simple_graph(self):
        """Test graph for simple term."""
        term = parse("(λx.x) y")
        graph = LambdaMultiwayGraph(term, max_depth=3)
        
        assert graph.n_nodes >= 2  # At least start and result
    
    def test_graph_with_choices(self):
        """Test graph with multiple reduction choices."""
        # (λx.x x) ((λy.y) z) has multiple redexes
        term = parse("(λx.x x) ((λy.y) z)")
        graph = LambdaMultiwayGraph(term, max_depth=5)
        
        # Should have multiple paths
        assert graph.n_edges > 1


class TestStandardCombinators:
    """Tests for standard combinators."""
    
    def test_i_combinator(self):
        """I x = x"""
        term = App(I, Var('x'))
        result, _ = beta_reduce(term)
        assert result == Var('x')
    
    def test_k_combinator_two_args(self):
        """K x y = x"""
        term = App(App(K, Var('x')), Var('y'))
        result, _ = beta_reduce(term)
        assert result == Var('x')
    
    def test_s_k_k_is_identity(self):
        """S K K x = x (S K K is identity)"""
        skk = App(App(S, K), K)
        skk_x = App(skk, Var('x'))
        result, _ = beta_reduce(skk_x, max_steps=20)
        assert result == Var('x')
    
    def test_true_false(self):
        """Test boolean encodings."""
        # TRUE a b → a
        true_ab = App(App(TRUE, Var('a')), Var('b'))
        result, _ = beta_reduce(true_ab)
        assert result == Var('a')
        
        # FALSE a b → b
        false_ab = App(App(FALSE, Var('a')), Var('b'))
        result, _ = beta_reduce(false_ab)
        assert result == Var('b')


# =============================================================================
# Lambda Analysis Tests
# =============================================================================

class TestAlgebraicAnalysis:
    """Tests for algebraic analysis."""
    
    def test_reduction_algebra(self):
        """Test basic algebraic analysis."""
        terms = [parse("λx.x"), parse("(λx.x) y")]
        result = analyze_reduction_algebra(terms)
        
        assert result.n_basis_terms > 0
        assert result.has_confluence  # Church-Rosser
    
    def test_symplectic_check(self):
        """Test symplectic embedding check."""
        # Create a simple transition matrix
        matrix = np.array([[0, 1], [1, 0]])  # Permutation
        result = check_symplectic_embedding(matrix)
        
        assert result['is_permutation']
        assert result['can_embed_in_sp']


class TestCoherenceAnalysis:
    """Tests for coherence analysis."""
    
    def test_lambda_coherence(self):
        """Test that λ-calculus cannot generate coherence."""
        result = analyze_lambda_coherence()
        
        # Lambda calculus is classical
        assert result['coherence_injection_test']['coherence_destroyed']
        assert 'CLASSICAL' in result['conclusion'].upper()


class TestSKComparison:
    """Tests for SK comparison."""
    
    def test_compare_lambda_sk(self):
        """Test λ and SK are both classical."""
        result = compare_lambda_sk()
        
        assert result['are_equivalent']
        assert result['both_classical']
        assert result['neither_generates_superposition']


# =============================================================================
# Hypothesis Tests (H10.x)
# =============================================================================

class TestHypothesisH10:
    """Tests for Phase 10 hypotheses."""
    
    def test_h10_1_lambda_same_as_sk(self):
        """
        H10.1: Lambda calculus exhibits the same classical behavior as SK.
        
        Evidence: Both are deterministic, neither generates superposition.
        """
        comparison = compare_lambda_sk()
        
        # Both classical
        assert comparison['lambda_properties']['is_classical']
        assert comparison['sk_properties']['is_classical']
        
        # Neither generates superposition
        assert not comparison['lambda_properties']['has_superposition']
        assert not comparison['sk_properties']['has_superposition']
        
        # Neither generates coherence
        assert not comparison['lambda_properties']['coherence_generation']
        assert not comparison['sk_properties']['coherence_generation']
    
    def test_h10_1_full_analysis(self):
        """Test full analysis supports H10.1."""
        results = run_full_lambda_analysis()
        
        assert results['hypothesis_h10_1']['supported']
    
    def test_church_rosser_implies_determinism(self):
        """
        Church-Rosser property implies deterministic normal forms.
        
        Even though there are multiple reduction paths,
        they all lead to the same normal form (if it exists).
        This is classical confluence, not quantum superposition.
        """
        # Simple test: (λx.x) y → y (deterministic)
        term = parse("(λx.x) y")
        result, history = beta_reduce(term, max_steps=10)
        
        # Should reach normal form y
        assert result == Var('y')
        assert is_normal_form(result)
        
        # Another test: K a b → a (deterministic)
        term2 = parse("(λx.λy.x) a b")
        result2, _ = beta_reduce(term2, max_steps=10)
        assert result2 == Var('a')
        
        # Key point: Both reductions are DETERMINISTIC
        # Even with multiple redexes, Church-Rosser guarantees
        # convergence to unique normal form (if it exists).
        # This is classical confluence, NOT quantum superposition.


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

