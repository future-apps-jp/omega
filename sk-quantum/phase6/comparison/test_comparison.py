"""
Tests for model comparison framework.
"""

import pytest
import numpy as np

from .quantum_circuit import (
    SimpleQuantumCircuit,
    QuantumState,
    HADAMARD,
    SIGMA_X,
    create_sample_circuits,
)
from .model_comparison import ModelComparison, run_comparison


class TestQuantumState:
    """Tests for quantum state."""
    
    def test_state_creation(self):
        """Test quantum state creation."""
        psi = QuantumState(np.array([1, 0], dtype=np.complex128))
        assert np.isclose(np.sum(psi.probabilities), 1.0)
    
    def test_superposition_state(self):
        """Test superposition state."""
        psi = QuantumState(np.array([1, 1], dtype=np.complex128) / np.sqrt(2))
        assert np.allclose(psi.probabilities, [0.5, 0.5])


class TestQuantumCircuit:
    """Tests for quantum circuit."""
    
    def test_hadamard_creates_superposition(self):
        """Test that Hadamard creates superposition."""
        qc = SimpleQuantumCircuit(1)
        qc.h(0)
        
        psi_0 = QuantumState(np.array([1, 0], dtype=np.complex128))
        psi_1 = qc.apply(psi_0)
        
        assert np.allclose(psi_1.probabilities, [0.5, 0.5])
    
    def test_x_gate_flips(self):
        """Test that X gate flips qubit."""
        qc = SimpleQuantumCircuit(1)
        qc.x(0)
        
        psi_0 = QuantumState(np.array([1, 0], dtype=np.complex128))
        psi_1 = qc.apply(psi_0)
        
        assert np.allclose(psi_1.probabilities, [0, 1])
    
    def test_circuit_is_unitary(self):
        """Test that circuit unitary is unitary."""
        qc = SimpleQuantumCircuit(2)
        qc.h(0).cx(0, 1).s(1)
        
        U = qc.get_unitary()
        assert np.allclose(U @ U.conj().T, np.eye(4))
    
    def test_bell_state(self):
        """Test Bell state creation."""
        qc = SimpleQuantumCircuit(2)
        qc.h(0).cx(0, 1)
        
        psi_0 = QuantumState(np.array([1, 0, 0, 0], dtype=np.complex128))
        psi_1 = qc.apply(psi_0)
        
        # Bell state: (|00> + |11>) / sqrt(2)
        expected = np.array([1, 0, 0, 1], dtype=np.complex128) / np.sqrt(2)
        assert np.allclose(np.abs(psi_1.amplitudes), np.abs(expected))
    
    def test_x_gate_is_permutation(self):
        """Test that X gates alone create permutation circuit."""
        qc = SimpleQuantumCircuit(2)
        qc.x(0).x(1)
        
        props = qc.analyze_structure()
        assert props['is_permutation'] is True
    
    def test_hadamard_not_permutation(self):
        """Test that Hadamard makes circuit non-permutation."""
        qc = SimpleQuantumCircuit(1)
        qc.h(0)
        
        props = qc.analyze_structure()
        assert props['is_permutation'] is False
        assert props['creates_superposition'] is True


class TestModelComparison:
    """Tests for model comparison."""
    
    def test_sk_analysis(self):
        """Test SK computation analysis."""
        mc = ModelComparison()
        props = mc.analyze_sk_computation('SKK')
        
        assert props.name == 'SK Computation'
        assert props.is_reversible is False  # SK is irreversible
        assert props.is_discrete is True
    
    def test_rca_analysis(self):
        """Test RCA analysis."""
        mc = ModelComparison()
        props = mc.analyze_rca(3, 90)
        
        assert 'RCA' in props.name
        assert props.is_reversible is True
        assert props.is_discrete is True
        assert props.matrix_type == 'permutation'
    
    def test_quantum_analysis(self):
        """Test quantum circuit analysis."""
        mc = ModelComparison()
        
        qc = SimpleQuantumCircuit(2)
        qc.h(0).cx(0, 1)
        props = mc.analyze_quantum_circuit(qc)
        
        assert props.is_reversible is True
        assert props.is_discrete is False  # Continuous amplitudes
        assert props.matrix_type == 'unitary'
        assert props.has_superposition is True
    
    def test_full_comparison(self):
        """Test full three-model comparison."""
        result = run_comparison()
        
        assert result.sk_props is not None
        assert result.rca_props is not None
        assert result.quantum_props is not None
        assert len(result.key_differences) > 0
        assert len(result.summary) > 0


class TestSampleCircuits:
    """Tests for sample circuits."""
    
    def test_create_sample_circuits(self):
        """Test sample circuit creation."""
        circuits = create_sample_circuits()
        
        assert len(circuits) >= 5
        
        for name, circuit in circuits:
            U = circuit.get_unitary()
            # All should be unitary
            assert np.allclose(U @ U.conj().T, np.eye(len(U)))
    
    def test_classical_vs_quantum_circuits(self):
        """Test distinction between classical and quantum circuits."""
        circuits = create_sample_circuits()
        
        classical_count = 0
        quantum_count = 0
        
        for name, circuit in circuits:
            props = circuit.analyze_structure()
            if props['is_permutation']:
                classical_count += 1
            else:
                quantum_count += 1
        
        # Should have both types
        assert classical_count >= 1
        assert quantum_count >= 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

