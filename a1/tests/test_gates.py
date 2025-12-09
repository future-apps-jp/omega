"""
Tests for A1 Gates

Author: Hiroshi Kohashiguchi
Date: December 2025
"""

import pytest
import sys
import os
import math

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gates import (
    QUANTUM_GATES, GateDefinition, VOCABULARY_SIZE,
    get_all_gates, get_single_qubit_gates, get_two_qubit_gates,
    get_rotation_gates, get_gate_matrix, total_gate_cost
)


# =============================================================================
# Gate Definition Tests
# =============================================================================

class TestGateDefinitions:
    """Tests for gate definitions."""
    
    def test_all_gates_defined(self):
        """Test that all expected gates are defined."""
        expected = ['H', 'X', 'Y', 'Z', 'S', 'T', 'CNOT', 'CZ', 'SWAP', 
                   'RX', 'RY', 'RZ', 'MEASURE']
        for gate in expected:
            assert gate in QUANTUM_GATES
    
    def test_gate_definition_structure(self):
        """Test that gate definitions have correct structure."""
        for name, gate in QUANTUM_GATES.items():
            assert isinstance(gate, GateDefinition)
            assert gate.name == name
            assert gate.arity >= 1
            assert gate.cost == 1  # All primitives cost 1
    
    def test_single_qubit_gates_arity(self):
        """Test single-qubit gates have arity 1."""
        single = ['H', 'X', 'Y', 'Z', 'S', 'T', 'MEASURE']
        for name in single:
            assert QUANTUM_GATES[name].arity == 1
    
    def test_two_qubit_gates_arity(self):
        """Test two-qubit gates have arity 2."""
        two_qubit = ['CNOT', 'CZ', 'SWAP']
        for name in two_qubit:
            assert QUANTUM_GATES[name].arity == 2
    
    def test_rotation_gates_have_angle(self):
        """Test rotation gates are marked with has_angle."""
        rotation = ['RX', 'RY', 'RZ']
        for name in rotation:
            assert QUANTUM_GATES[name].has_angle is True
    
    def test_non_rotation_gates_no_angle(self):
        """Test non-rotation gates don't have has_angle."""
        non_rotation = ['H', 'X', 'Y', 'Z', 'CNOT', 'CZ', 'SWAP', 'MEASURE']
        for name in non_rotation:
            assert QUANTUM_GATES[name].has_angle is False


# =============================================================================
# Gate List Functions Tests
# =============================================================================

class TestGateListFunctions:
    """Tests for gate list helper functions."""
    
    def test_get_all_gates(self):
        """Test get_all_gates returns all gate names."""
        all_gates = get_all_gates()
        assert len(all_gates) == len(QUANTUM_GATES)
        for name in all_gates:
            assert name in QUANTUM_GATES
    
    def test_get_single_qubit_gates(self):
        """Test get_single_qubit_gates returns correct gates."""
        single = get_single_qubit_gates()
        assert 'H' in single
        assert 'X' in single
        assert 'CNOT' not in single  # Two-qubit
        assert 'RX' not in single    # Has angle
    
    def test_get_two_qubit_gates(self):
        """Test get_two_qubit_gates returns correct gates."""
        two = get_two_qubit_gates()
        assert 'CNOT' in two
        assert 'CZ' in two
        assert 'SWAP' in two
        assert 'H' not in two
    
    def test_get_rotation_gates(self):
        """Test get_rotation_gates returns correct gates."""
        rotation = get_rotation_gates()
        assert 'RX' in rotation
        assert 'RY' in rotation
        assert 'RZ' in rotation
        assert 'H' not in rotation


# =============================================================================
# Gate Matrix Tests
# =============================================================================

class TestGateMatrices:
    """Tests for gate matrix definitions."""
    
    def test_hadamard_matrix(self):
        """Test Hadamard gate matrix."""
        import numpy as np
        H = get_gate_matrix('H')
        
        # H should be unitary
        assert np.allclose(H @ H.conj().T, np.eye(2))
        
        # H^2 = I
        assert np.allclose(H @ H, np.eye(2))
    
    def test_pauli_x_matrix(self):
        """Test Pauli-X gate matrix."""
        import numpy as np
        X = get_gate_matrix('X')
        
        # X^2 = I
        assert np.allclose(X @ X, np.eye(2))
        
        # X |0> = |1>
        zero = np.array([1, 0])
        one = np.array([0, 1])
        assert np.allclose(X @ zero, one)
    
    def test_pauli_y_matrix(self):
        """Test Pauli-Y gate matrix."""
        import numpy as np
        Y = get_gate_matrix('Y')
        
        # Y^2 = I
        assert np.allclose(Y @ Y, np.eye(2))
    
    def test_pauli_z_matrix(self):
        """Test Pauli-Z gate matrix."""
        import numpy as np
        Z = get_gate_matrix('Z')
        
        # Z^2 = I
        assert np.allclose(Z @ Z, np.eye(2))
        
        # Z |0> = |0>, Z |1> = -|1>
        assert np.allclose(Z @ np.array([1, 0]), np.array([1, 0]))
        assert np.allclose(Z @ np.array([0, 1]), np.array([0, -1]))
    
    def test_cnot_matrix(self):
        """Test CNOT gate matrix."""
        import numpy as np
        CNOT = get_gate_matrix('CNOT')
        
        # CNOT^2 = I
        assert np.allclose(CNOT @ CNOT, np.eye(4))
        
        # CNOT |00> = |00>, CNOT |10> = |11>
        assert np.allclose(CNOT @ np.array([1, 0, 0, 0]), np.array([1, 0, 0, 0]))
        assert np.allclose(CNOT @ np.array([0, 0, 1, 0]), np.array([0, 0, 0, 1]))
    
    def test_swap_matrix(self):
        """Test SWAP gate matrix."""
        import numpy as np
        SWAP = get_gate_matrix('SWAP')
        
        # SWAP^2 = I
        assert np.allclose(SWAP @ SWAP, np.eye(4))
        
        # SWAP |01> = |10>
        assert np.allclose(SWAP @ np.array([0, 1, 0, 0]), np.array([0, 0, 1, 0]))
    
    def test_rx_matrix(self):
        """Test RX rotation matrix."""
        import numpy as np
        
        # RX(0) = I
        RX_0 = get_gate_matrix('RX', 0)
        assert np.allclose(RX_0, np.eye(2))
        
        # RX(pi) = -iX (up to global phase)
        RX_pi = get_gate_matrix('RX', math.pi)
        X = get_gate_matrix('X')
        # Check that RX(pi) |0> = -i |1>
        assert np.allclose(np.abs(RX_pi @ np.array([1, 0])), [0, 1])
    
    def test_ry_matrix(self):
        """Test RY rotation matrix."""
        import numpy as np
        
        # RY(0) = I
        RY_0 = get_gate_matrix('RY', 0)
        assert np.allclose(RY_0, np.eye(2))
    
    def test_rz_matrix(self):
        """Test RZ rotation matrix."""
        import numpy as np
        
        # RZ(0) = I
        RZ_0 = get_gate_matrix('RZ', 0)
        assert np.allclose(RZ_0, np.eye(2))


# =============================================================================
# Cost Calculation Tests
# =============================================================================

class TestCostCalculation:
    """Tests for gate cost calculations."""
    
    def test_total_gate_cost_single(self):
        """Test cost for single gate."""
        gates = [('H', 0)]
        assert total_gate_cost(gates) == 1
    
    def test_total_gate_cost_multiple(self):
        """Test cost for multiple gates."""
        gates = [('H', 0), ('CNOT', 0, 1), ('MEASURE', 0)]
        assert total_gate_cost(gates) == 3
    
    def test_total_gate_cost_empty(self):
        """Test cost for empty gate list."""
        assert total_gate_cost([]) == 0


# =============================================================================
# Vocabulary Tests
# =============================================================================

class TestVocabulary:
    """Tests for vocabulary size."""
    
    def test_vocabulary_size_reasonable(self):
        """Test vocabulary size is reasonable (~32)."""
        assert 20 <= VOCABULARY_SIZE <= 50
    
    def test_vocabulary_size_matches_expected(self):
        """Test vocabulary size matches our design."""
        # 13 gates + 6 special forms + 1 constant + 10 digits + 2 parens = 32
        assert VOCABULARY_SIZE == 32


if __name__ == "__main__":
    pytest.main([__file__, "-v"])



