"""
Tests for A1 Core (Parser and Interpreter)

Author: Hiroshi Kohashiguchi
Date: December 2025
"""

import pytest
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import (
    tokenize, parse, parse_atom, ParseError,
    A1, Lambda, Environment, execute
)


# =============================================================================
# Tokenizer Tests
# =============================================================================

class TestTokenizer:
    """Tests for the tokenizer."""
    
    def test_simple_expression(self):
        """Test tokenizing a simple expression."""
        tokens = tokenize("(H 0)")
        assert tokens == ['(', 'H', '0', ')']
    
    def test_nested_expression(self):
        """Test tokenizing nested expressions."""
        tokens = tokenize("(CNOT (H 0) 1)")
        assert tokens == ['(', 'CNOT', '(', 'H', '0', ')', '1', ')']
    
    def test_comments_removed(self):
        """Test that comments are removed."""
        tokens = tokenize("(H 0) ; this is a comment")
        assert tokens == ['(', 'H', '0', ')']
    
    def test_multiline(self):
        """Test tokenizing multiline source."""
        source = """
        (DEFINE x 1)
        (DEFINE y 2)
        """
        tokens = tokenize(source)
        assert 'DEFINE' in tokens
        assert 'x' in tokens
    
    def test_whitespace_handling(self):
        """Test that various whitespace is handled."""
        tokens = tokenize("(  H   0  )")
        assert tokens == ['(', 'H', '0', ')']


# =============================================================================
# Parser Tests
# =============================================================================

class TestParser:
    """Tests for the parser."""
    
    def test_parse_simple(self):
        """Test parsing a simple expression."""
        result = parse("(H 0)")
        assert result == [['H', 0]]
    
    def test_parse_nested(self):
        """Test parsing nested expressions."""
        result = parse("(CNOT (H 0) 1)")
        assert result == [['CNOT', ['H', 0], 1]]
    
    def test_parse_multiple_expressions(self):
        """Test parsing multiple expressions."""
        result = parse("(H 0) (X 1)")
        assert result == [['H', 0], ['X', 1]]
    
    def test_parse_atom_integer(self):
        """Test parsing integer atoms."""
        assert parse_atom("42") == 42
        assert parse_atom("0") == 0
        assert parse_atom("-5") == -5
    
    def test_parse_atom_float(self):
        """Test parsing float atoms."""
        assert parse_atom("3.14") == 3.14
    
    def test_parse_atom_symbol(self):
        """Test parsing symbol atoms."""
        assert parse_atom("H") == "H"
        assert parse_atom("CNOT") == "CNOT"
    
    def test_parse_define(self):
        """Test parsing DEFINE expression."""
        result = parse("(DEFINE x 5)")
        assert result == [['DEFINE', 'X', 5]]  # Note: symbols are uppercased
    
    def test_parse_lambda(self):
        """Test parsing LAMBDA expression."""
        result = parse("(LAMBDA (x) x)")
        assert result == [['LAMBDA', ['X'], 'X']]
    
    def test_unmatched_paren_raises(self):
        """Test that unmatched parentheses raise error."""
        with pytest.raises(ParseError):
            parse("(H 0")
    
    def test_extra_close_paren_raises(self):
        """Test that extra closing paren raises error."""
        with pytest.raises(ParseError):
            parse(")")


# =============================================================================
# Environment Tests
# =============================================================================

class TestEnvironment:
    """Tests for the Environment class."""
    
    def test_set_and_get(self):
        """Test setting and getting variables."""
        env = Environment()
        env.set("x", 42)
        assert env.get("x") == 42
    
    def test_undefined_raises(self):
        """Test that undefined variables raise NameError."""
        env = Environment()
        with pytest.raises(NameError):
            env.get("undefined")
    
    def test_parent_lookup(self):
        """Test lookup in parent environment."""
        parent = Environment()
        parent.set("x", 10)
        
        child = Environment(parent=parent)
        assert child.get("x") == 10
    
    def test_child_shadows_parent(self):
        """Test that child can shadow parent variables."""
        parent = Environment()
        parent.set("x", 10)
        
        child = Environment(parent=parent)
        child.set("x", 20)
        
        assert child.get("x") == 20
        assert parent.get("x") == 10
    
    def test_extend(self):
        """Test environment extension."""
        env = Environment()
        env.set("x", 1)
        
        extended = env.extend(["y", "z"], [2, 3])
        
        assert extended.get("x") == 1
        assert extended.get("y") == 2
        assert extended.get("z") == 3


# =============================================================================
# Interpreter Tests - Basic Gates
# =============================================================================

class TestInterpreterGates:
    """Tests for gate execution."""
    
    def test_hadamard(self):
        """Test Hadamard gate."""
        a1 = A1()
        circuit = a1.run("(H 0)")
        assert ('H', 0) in circuit.instructions
    
    def test_pauli_x(self):
        """Test Pauli-X gate."""
        a1 = A1()
        circuit = a1.run("(X 1)")
        assert ('X', 1) in circuit.instructions
    
    def test_pauli_y(self):
        """Test Pauli-Y gate."""
        a1 = A1()
        circuit = a1.run("(Y 0)")
        assert ('Y', 0) in circuit.instructions
    
    def test_pauli_z(self):
        """Test Pauli-Z gate."""
        a1 = A1()
        circuit = a1.run("(Z 2)")
        assert ('Z', 2) in circuit.instructions
    
    def test_cnot(self):
        """Test CNOT gate."""
        a1 = A1()
        circuit = a1.run("(CNOT 0 1)")
        assert ('CNOT', 0, 1) in circuit.instructions
    
    def test_cz(self):
        """Test CZ gate."""
        a1 = A1()
        circuit = a1.run("(CZ 0 1)")
        assert ('CZ', 0, 1) in circuit.instructions
    
    def test_swap(self):
        """Test SWAP gate."""
        a1 = A1()
        circuit = a1.run("(SWAP 0 1)")
        assert ('SWAP', 0, 1) in circuit.instructions
    
    def test_rx(self):
        """Test RX rotation gate."""
        a1 = A1()
        circuit = a1.run("(RX 0 3.14)")
        assert circuit.instructions[0][0] == 'RX'
        assert circuit.instructions[0][1] == 0
    
    def test_measure(self):
        """Test measurement."""
        a1 = A1()
        circuit = a1.run("(MEASURE 0)")
        assert ('MEASURE', 0) in circuit.instructions


# =============================================================================
# Interpreter Tests - Gate Chaining
# =============================================================================

class TestGateChaining:
    """Tests for gate chaining (gates return qubit index)."""
    
    def test_bell_state(self):
        """Test Bell state preparation: (CNOT (H 0) 1)."""
        a1 = A1()
        circuit = a1.run("(CNOT (H 0) 1)")
        
        # H should be applied first, then CNOT
        assert circuit.instructions[0] == ('H', 0)
        assert circuit.instructions[1] == ('CNOT', 0, 1)
    
    def test_ghz_state(self):
        """Test GHZ state preparation."""
        a1 = A1()
        circuit = a1.run("(CNOT (CNOT (H 0) 1) 2)")
        
        assert circuit.instructions[0] == ('H', 0)
        assert circuit.instructions[1] == ('CNOT', 0, 1)
        assert circuit.instructions[2] == ('CNOT', 1, 2)
    
    def test_gate_returns_qubit(self):
        """Test that gates return the qubit they acted on."""
        a1 = A1()
        # This should work because (H 0) returns 0
        result = a1.eval(['H', 0])
        assert result == 0


# =============================================================================
# Interpreter Tests - Special Forms
# =============================================================================

class TestSpecialForms:
    """Tests for special forms (DEFINE, LAMBDA, IF, LET)."""
    
    def test_define_variable(self):
        """Test DEFINE for variables."""
        a1 = A1()
        a1.run("(DEFINE X 5)")
        assert a1.env.get("X") == 5
    
    def test_define_function(self):
        """Test DEFINE for functions."""
        a1 = A1()
        a1.run("(DEFINE (SQUARE X) (* X X))")
        # Function should be defined
        assert isinstance(a1.env.get("SQUARE"), Lambda)
    
    def test_lambda_creation(self):
        """Test LAMBDA creates a closure."""
        a1 = A1()
        a1.run("(DEFINE F (LAMBDA (X) X))")
        assert isinstance(a1.env.get("F"), Lambda)
    
    def test_lambda_call(self):
        """Test calling a lambda."""
        a1 = A1()
        a1.run("""
            (DEFINE make-bell
                (LAMBDA (q0 q1)
                    (CNOT (H q0) q1)))
            (make-bell 0 1)
        """)
        
        circuit = a1.circuit
        assert ('H', 0) in circuit.instructions
        assert ('CNOT', 0, 1) in circuit.instructions
    
    def test_if_true_branch(self):
        """Test IF with true condition."""
        a1 = A1()
        result = a1.eval(parse("(IF 1 2 3)")[0])
        assert result == 2
    
    def test_if_false_branch(self):
        """Test IF with false condition."""
        a1 = A1()
        result = a1.eval(parse("(IF 0 2 3)")[0])
        assert result == 3
    
    def test_let_binding(self):
        """Test LET local binding."""
        a1 = A1()
        a1.run("""
            (LET ((Q 0))
                (H Q))
        """)
        assert ('H', 0) in a1.circuit.instructions
    
    def test_begin_sequence(self):
        """Test BEGIN for sequencing."""
        a1 = A1()
        a1.run("(BEGIN (H 0) (X 1) (Y 2))")
        
        assert ('H', 0) in a1.circuit.instructions
        assert ('X', 1) in a1.circuit.instructions
        assert ('Y', 2) in a1.circuit.instructions
    
    def test_quote(self):
        """Test QUOTE returns unevaluated."""
        a1 = A1()
        result = a1.eval(parse("(QUOTE (H 0))")[0])
        assert result == ['H', 0]


# =============================================================================
# Interpreter Tests - Complex Programs
# =============================================================================

class TestComplexPrograms:
    """Tests for complex A1 programs."""
    
    def test_full_bell_program(self):
        """Test complete Bell state program with function."""
        source = """
        ; Bell state generator
        (DEFINE make-bell
            (LAMBDA (q0 q1)
                (CNOT (H q0) q1)))
        
        ; Create Bell state on qubits 0 and 1
        (make-bell 0 1)
        """
        
        a1 = A1()
        circuit = a1.run(source)
        
        assert len(circuit.instructions) == 2
        assert circuit.instructions[0] == ('H', 0)
        assert circuit.instructions[1] == ('CNOT', 0, 1)
    
    def test_multiple_gate_sequence(self):
        """Test sequence of multiple gates."""
        a1 = A1()
        circuit = a1.run("""
            (H 0)
            (H 1)
            (CNOT 0 1)
            (MEASURE 0)
            (MEASURE 1)
        """)
        
        assert len(circuit.instructions) == 5
    
    def test_reset_circuit(self):
        """Test that reset clears the circuit."""
        a1 = A1()
        a1.run("(H 0)")
        assert len(a1.circuit.instructions) == 1
        
        a1.reset()
        assert len(a1.circuit.instructions) == 0
    
    def test_run_fresh(self):
        """Test run_fresh creates new circuit."""
        a1 = A1()
        a1.run("(H 0)")
        
        circuit = a1.run_fresh("(X 1)")
        assert len(circuit.instructions) == 1
        assert circuit.instructions[0] == ('X', 1)


# =============================================================================
# Convenience Function Tests
# =============================================================================

class TestConvenienceFunctions:
    """Tests for convenience functions."""
    
    def test_execute(self):
        """Test execute function."""
        circuit = execute("(CNOT (H 0) 1)")
        
        assert len(circuit.instructions) == 2
        assert circuit.instructions[0] == ('H', 0)
        assert circuit.instructions[1] == ('CNOT', 0, 1)


# =============================================================================
# Error Handling Tests
# =============================================================================

class TestErrorHandling:
    """Tests for error handling."""
    
    def test_undefined_variable(self):
        """Test that undefined variable raises NameError."""
        a1 = A1()
        with pytest.raises(NameError):
            a1.run("(UNDEFINED 0)")
    
    def test_wrong_arity(self):
        """Test that wrong number of arguments raises TypeError."""
        a1 = A1()
        with pytest.raises(TypeError):
            a1.run("(H 0 1)")  # H takes 1 argument
    
    def test_call_non_callable(self):
        """Test that calling non-callable raises TypeError."""
        a1 = A1()
        a1.run("(DEFINE X 5)")
        with pytest.raises(TypeError):
            a1.run("(X 0)")  # X is not callable


if __name__ == "__main__":
    pytest.main([__file__, "-v"])



