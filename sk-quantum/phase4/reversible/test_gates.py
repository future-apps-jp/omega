"""
Tests for Reversible Logic Gates
"""

import pytest
import numpy as np
from gates import (
    NOT, CNOT, TOFFOLI, FREDKIN, SWAP,
    NOTGate, CNOTGate, ToffoliGate, FredkinGate, SWAPGate,
    IdentityGate, CompositeGate, EmbeddedGate, GateGroup,
    verify_reversibility, verify_self_inverse, matrix_properties
)


class TestBasicGates:
    """基本ゲートのテスト"""
    
    def test_not_gate(self):
        """NOT ゲートのテスト"""
        assert NOT.apply((0,)) == (1,)
        assert NOT.apply((1,)) == (0,)
        assert NOT.n_bits == 1
    
    def test_cnot_gate(self):
        """CNOT ゲートのテスト"""
        # control=0 のとき target は変化しない
        assert CNOT.apply((0, 0)) == (0, 0)
        assert CNOT.apply((0, 1)) == (0, 1)
        # control=1 のとき target は反転
        assert CNOT.apply((1, 0)) == (1, 1)
        assert CNOT.apply((1, 1)) == (1, 0)
        assert CNOT.n_bits == 2
    
    def test_toffoli_gate(self):
        """Toffoli ゲートのテスト"""
        # control bits が両方 1 のときのみ target が反転
        assert TOFFOLI.apply((0, 0, 0)) == (0, 0, 0)
        assert TOFFOLI.apply((0, 0, 1)) == (0, 0, 1)
        assert TOFFOLI.apply((0, 1, 0)) == (0, 1, 0)
        assert TOFFOLI.apply((0, 1, 1)) == (0, 1, 1)
        assert TOFFOLI.apply((1, 0, 0)) == (1, 0, 0)
        assert TOFFOLI.apply((1, 0, 1)) == (1, 0, 1)
        assert TOFFOLI.apply((1, 1, 0)) == (1, 1, 1)  # 反転
        assert TOFFOLI.apply((1, 1, 1)) == (1, 1, 0)  # 反転
        assert TOFFOLI.n_bits == 3
    
    def test_fredkin_gate(self):
        """Fredkin ゲートのテスト"""
        # control=0 のときスワップしない
        assert FREDKIN.apply((0, 0, 0)) == (0, 0, 0)
        assert FREDKIN.apply((0, 0, 1)) == (0, 0, 1)
        assert FREDKIN.apply((0, 1, 0)) == (0, 1, 0)
        assert FREDKIN.apply((0, 1, 1)) == (0, 1, 1)
        # control=1 のときスワップ
        assert FREDKIN.apply((1, 0, 0)) == (1, 0, 0)
        assert FREDKIN.apply((1, 0, 1)) == (1, 1, 0)  # スワップ
        assert FREDKIN.apply((1, 1, 0)) == (1, 0, 1)  # スワップ
        assert FREDKIN.apply((1, 1, 1)) == (1, 1, 1)
        assert FREDKIN.n_bits == 3
    
    def test_swap_gate(self):
        """SWAP ゲートのテスト"""
        assert SWAP.apply((0, 0)) == (0, 0)
        assert SWAP.apply((0, 1)) == (1, 0)
        assert SWAP.apply((1, 0)) == (0, 1)
        assert SWAP.apply((1, 1)) == (1, 1)
        assert SWAP.n_bits == 2


class TestReversibility:
    """可逆性のテスト"""
    
    def test_all_gates_reversible(self):
        """全ゲートが可逆"""
        for gate in [NOT, CNOT, TOFFOLI, FREDKIN, SWAP]:
            assert verify_reversibility(gate), f"{gate.name} is not reversible"
    
    def test_all_gates_self_inverse(self):
        """全ゲートが自己逆"""
        for gate in [NOT, CNOT, TOFFOLI, FREDKIN, SWAP]:
            assert verify_self_inverse(gate), f"{gate.name} is not self-inverse"


class TestMatrixRepresentation:
    """行列表現のテスト"""
    
    def test_not_matrix(self):
        """NOT の行列表現"""
        M = NOT.to_matrix()
        expected = np.array([[0, 1], [1, 0]], dtype=float)
        assert np.allclose(M, expected)
    
    def test_cnot_matrix(self):
        """CNOT の行列表現"""
        M = CNOT.to_matrix()
        # 状態順: (0,0), (1,0), (0,1), (1,1)
        # CNOT: (a,b) -> (a, a⊕b)
        # (0,0)->0, (1,0)->1, (0,1)->2, (1,1)->3
        # (0,0)->(0,0)=0, (1,0)->(1,1)=3, (0,1)->(0,1)=2, (1,1)->(1,0)=1
        expected = np.array([
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
            [0, 1, 0, 0]
        ], dtype=float)
        assert np.allclose(M, expected)
    
    def test_toffoli_matrix_is_permutation(self):
        """Toffoli 行列は置換行列"""
        M = TOFFOLI.to_matrix()
        # 各行/列に 1 が1つだけ
        assert np.allclose(M.sum(axis=0), np.ones(8))
        assert np.allclose(M.sum(axis=1), np.ones(8))
    
    def test_matrices_are_orthogonal(self):
        """全行列が直交行列"""
        for gate in [NOT, CNOT, TOFFOLI, FREDKIN, SWAP]:
            M = gate.to_matrix()
            I = np.eye(M.shape[0])
            assert np.allclose(M.T @ M, I), f"{gate.name} matrix is not orthogonal"


class TestCompositeGates:
    """合成ゲートのテスト"""
    
    def test_composition(self):
        """ゲート合成"""
        # NOT ∘ NOT = I
        not_not = NOT.compose(NOT)
        for state in [(0,), (1,)]:
            assert not_not.apply(state) == state
    
    def test_composite_matrix(self):
        """合成ゲートの行列"""
        not_not = NOT.compose(NOT)
        M = not_not.to_matrix()
        I = np.eye(2)
        assert np.allclose(M, I)


class TestEmbeddedGates:
    """埋め込みゲートのテスト"""
    
    def test_cnot_in_3bit(self):
        """3-bit 空間での CNOT"""
        cnot_01 = EmbeddedGate(CNOT, [0, 1], 3)
        
        # bit 2 は変化しない
        assert cnot_01.apply((0, 0, 0)) == (0, 0, 0)
        assert cnot_01.apply((0, 0, 1)) == (0, 0, 1)
        assert cnot_01.apply((1, 0, 0)) == (1, 1, 0)  # CNOT 作用
        assert cnot_01.apply((1, 0, 1)) == (1, 1, 1)  # CNOT 作用


class TestGateGroup:
    """ゲート群のテスト"""
    
    def test_not_group(self):
        """NOT の生成する群は Z_2"""
        group = GateGroup([NOT])
        order = group.group_order()
        assert order == 2  # {I, NOT}
    
    def test_swap_group(self):
        """SWAP の生成する群は Z_2"""
        group = GateGroup([SWAP])
        order = group.group_order()
        assert order == 2  # {I, SWAP}
    
    def test_toffoli_group_order(self):
        """Toffoli 群の位数"""
        group = GateGroup([TOFFOLI])
        order = group.group_order(max_depth=20)
        # Toffoli は S_8 全体を生成しない
        assert order < 40320  # 8! = 40320


class TestMatrixProperties:
    """行列性質のテスト"""
    
    def test_orthogonal_matrices(self):
        """全行列が直交行列"""
        for gate in [TOFFOLI, FREDKIN]:
            M = gate.to_matrix()
            props = matrix_properties(M)
            assert props['is_orthogonal']
            assert props['det_is_pm1']


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

