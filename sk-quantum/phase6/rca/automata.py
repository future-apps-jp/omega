"""
Reversible Cellular Automata (RCA) implementation.

Key properties:
- Bijective global map (each configuration has unique predecessor)
- Can be made from combining linear rules
- Creates reversible discrete dynamics
"""

import numpy as np
from typing import List, Tuple, Optional, Set
from dataclasses import dataclass


@dataclass
class RCAState:
    """State of an RCA."""
    cells: Tuple[int, ...]
    
    def __hash__(self):
        return hash(self.cells)
    
    def __eq__(self, other):
        if isinstance(other, RCAState):
            return self.cells == other.cells
        return False
    
    def __str__(self):
        return ''.join(str(c) for c in self.cells)
    
    def __repr__(self):
        return f"RCAState({self.cells})"


class ReversibleCellularAutomaton:
    """
    Base class for reversible cellular automata.
    
    A second-order RCA using history for reversibility:
    new_state = rule(current_state) XOR previous_state
    
    This makes any rule reversible by storing one step of history.
    """
    
    def __init__(self, size: int, rule_number: int = 90):
        """
        Initialize RCA.
        
        Args:
            size: Number of cells
            rule_number: Elementary CA rule number (default: 90)
        """
        self.size = size
        self.rule_number = rule_number
        self.rule_table = self._build_rule_table()
    
    def _build_rule_table(self) -> dict:
        """Build lookup table for the rule."""
        table = {}
        for neighborhood in range(8):  # 3 bits: left, center, right
            output = (self.rule_number >> neighborhood) & 1
            table[neighborhood] = output
        return table
    
    def apply_rule(self, state: RCAState) -> RCAState:
        """Apply the elementary CA rule (first-order, not yet reversible)."""
        cells = list(state.cells)
        new_cells = []
        
        for i in range(self.size):
            left = cells[(i - 1) % self.size]
            center = cells[i]
            right = cells[(i + 1) % self.size]
            neighborhood = (left << 2) | (center << 1) | right
            new_cells.append(self.rule_table[neighborhood])
        
        return RCAState(tuple(new_cells))
    
    def evolve(self, current: RCAState, previous: RCAState) -> Tuple[RCAState, RCAState]:
        """
        Evolve one step using second-order reversible dynamics.
        
        new = rule(current) XOR previous
        This is reversible: previous = rule(current) XOR new
        
        Returns:
            (new_state, current_state) - new becomes current, current becomes previous
        """
        rule_result = self.apply_rule(current)
        new_cells = tuple(r ^ p for r, p in zip(rule_result.cells, previous.cells))
        new_state = RCAState(new_cells)
        return new_state, current
    
    def reverse(self, current: RCAState, next_state: RCAState) -> Tuple[RCAState, RCAState]:
        """
        Reverse one step.
        
        previous = rule(current) XOR next
        """
        rule_result = self.apply_rule(current)
        prev_cells = tuple(r ^ n for r, n in zip(rule_result.cells, next_state.cells))
        prev_state = RCAState(prev_cells)
        return current, prev_state
    
    def generate_orbit(self, initial: RCAState, previous: RCAState, 
                       max_steps: int = 100) -> List[Tuple[RCAState, RCAState]]:
        """
        Generate the orbit (trajectory) starting from initial state.
        
        Returns list of (current, previous) pairs.
        Stops when returning to initial state or max_steps reached.
        """
        orbit = [(initial, previous)]
        current, prev = initial, previous
        
        for _ in range(max_steps):
            new_state, new_prev = self.evolve(current, prev)
            if new_state == initial and new_prev == previous:
                break  # Found cycle
            orbit.append((new_state, new_prev))
            current, prev = new_state, new_prev
        
        return orbit
    
    def get_transition_matrix(self, 
                              include_history: bool = True) -> np.ndarray:
        """
        Get the transition matrix for this RCA.
        
        Args:
            include_history: If True, state space is (current, previous) pairs
                           If False, just current state transitions
        
        Returns:
            Permutation matrix of the state transitions
        """
        if include_history:
            # State space: all (current, previous) pairs
            n = 2 ** self.size
            state_space_size = n * n
            matrix = np.zeros((state_space_size, state_space_size), dtype=np.float64)
            
            for curr_idx in range(n):
                for prev_idx in range(n):
                    current = RCAState(tuple(int(b) for b in format(curr_idx, f'0{self.size}b')))
                    previous = RCAState(tuple(int(b) for b in format(prev_idx, f'0{self.size}b')))
                    
                    new_state, new_prev = self.evolve(current, previous)
                    
                    new_curr_idx = int(''.join(str(b) for b in new_state.cells), 2)
                    new_prev_idx = int(''.join(str(b) for b in new_prev.cells), 2)
                    
                    from_idx = curr_idx * n + prev_idx
                    to_idx = new_curr_idx * n + new_prev_idx
                    matrix[to_idx, from_idx] = 1.0
        else:
            # First-order (non-reversible) transitions
            n = 2 ** self.size
            matrix = np.zeros((n, n), dtype=np.float64)
            
            for idx in range(n):
                state = RCAState(tuple(int(b) for b in format(idx, f'0{self.size}b')))
                new_state = self.apply_rule(state)
                new_idx = int(''.join(str(b) for b in new_state.cells), 2)
                matrix[new_idx, idx] = 1.0
        
        return matrix


class Rule90(ReversibleCellularAutomaton):
    """
    Rule 90 cellular automaton.
    
    Rule 90: a' = left XOR right
    This is linear and produces Sierpinski triangle patterns.
    """
    
    def __init__(self, size: int):
        super().__init__(size, rule_number=90)


class Rule150(ReversibleCellularAutomaton):
    """
    Rule 150 cellular automaton.
    
    Rule 150: a' = left XOR center XOR right
    Also linear, different pattern from Rule 90.
    """
    
    def __init__(self, size: int):
        super().__init__(size, rule_number=150)


def analyze_rca_group(rca: ReversibleCellularAutomaton) -> dict:
    """
    Analyze the group structure of an RCA.
    
    Returns:
        Dictionary with group properties
    """
    matrix = rca.get_transition_matrix(include_history=True)
    
    # Check if permutation matrix
    is_permutation = (
        np.allclose(matrix @ matrix.T, np.eye(len(matrix))) and
        np.allclose(np.sum(matrix, axis=0), 1) and
        np.allclose(np.sum(matrix, axis=1), 1)
    )
    
    # Eigenvalue analysis
    eigenvalues = np.linalg.eigvals(matrix)
    
    # Check for complex eigenvalues with non-trivial imaginary part
    has_complex_eigenvalues = any(
        abs(ev.imag) > 1e-10 for ev in eigenvalues
    )
    
    # Find order (how many applications to return to identity)
    current = matrix.copy()
    order = 1
    max_order = 1000
    while order < max_order:
        if np.allclose(current, np.eye(len(matrix))):
            break
        current = current @ matrix
        order += 1
    
    if order >= max_order:
        order = None  # Did not find finite order
    
    return {
        'matrix_size': len(matrix),
        'is_permutation': is_permutation,
        'is_orthogonal': np.allclose(matrix @ matrix.T, np.eye(len(matrix))),
        'determinant': np.linalg.det(matrix),
        'eigenvalues': eigenvalues,
        'has_complex_eigenvalues': has_complex_eigenvalues,
        'order': order,
        'is_involutory': np.allclose(matrix @ matrix, np.eye(len(matrix))),
    }


if __name__ == '__main__':
    # Quick test
    rca = Rule90(4)
    initial = RCAState((1, 0, 0, 0))
    previous = RCAState((0, 0, 0, 0))
    
    print(f"Initial: {initial}")
    print(f"Previous: {previous}")
    
    orbit = rca.generate_orbit(initial, previous, max_steps=20)
    print(f"\nOrbit (length {len(orbit)}):")
    for i, (curr, prev) in enumerate(orbit[:10]):
        print(f"  Step {i}: current={curr}, previous={prev}")
    
    print("\n--- Group Analysis ---")
    props = analyze_rca_group(rca)
    print(f"Matrix size: {props['matrix_size']}")
    print(f"Is permutation: {props['is_permutation']}")
    print(f"Order: {props['order']}")
    print(f"Has complex eigenvalues: {props['has_complex_eigenvalues']}")

