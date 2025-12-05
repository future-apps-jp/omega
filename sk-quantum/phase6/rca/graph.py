"""
RCA Graph representation for comparison with SK multiway graphs.

This creates a transition graph from RCA dynamics,
analogous to the multiway graph in SK computation.
"""

import numpy as np
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass, field
from .automata import ReversibleCellularAutomaton, RCAState


@dataclass
class RCANode:
    """Node in RCA transition graph."""
    state: Tuple[RCAState, RCAState]  # (current, previous) pair
    index: int
    
    def __hash__(self):
        return hash((self.state[0], self.state[1]))
    
    def __eq__(self, other):
        if isinstance(other, RCANode):
            return self.state == other.state
        return False


class RCAGraph:
    """
    Graph representation of RCA state transitions.
    
    Unlike SK multiway graphs which have branching,
    RCA graphs are deterministic and reversible (bijective).
    """
    
    def __init__(self, rca: ReversibleCellularAutomaton):
        """
        Initialize RCA graph.
        
        Args:
            rca: The reversible cellular automaton
        """
        self.rca = rca
        self.nodes: Dict[Tuple[RCAState, RCAState], RCANode] = {}
        self.edges: List[Tuple[int, int]] = []
        self._node_counter = 0
    
    def _get_or_create_node(self, current: RCAState, previous: RCAState) -> RCANode:
        """Get existing node or create new one."""
        key = (current, previous)
        if key not in self.nodes:
            node = RCANode(state=key, index=self._node_counter)
            self.nodes[key] = node
            self._node_counter += 1
        return self.nodes[key]
    
    def build_from_initial(self, initial: RCAState, previous: RCAState,
                           max_steps: int = 100) -> 'RCAGraph':
        """
        Build graph starting from initial state.
        
        Args:
            initial: Initial current state
            previous: Initial previous state
            max_steps: Maximum evolution steps
        
        Returns:
            self for chaining
        """
        current, prev = initial, previous
        
        for _ in range(max_steps):
            node = self._get_or_create_node(current, prev)
            new_state, new_prev = self.rca.evolve(current, prev)
            new_node = self._get_or_create_node(new_state, new_prev)
            
            self.edges.append((node.index, new_node.index))
            
            # Check for cycle
            if new_state == initial and new_prev == previous:
                break
            
            current, prev = new_state, new_prev
        
        return self
    
    def build_full_state_space(self) -> 'RCAGraph':
        """
        Build complete graph of all possible states.
        
        Warning: Exponential in RCA size!
        Only use for small RCA sizes (≤ 4 cells).
        """
        n = 2 ** self.rca.size
        
        # Create all nodes
        for curr_idx in range(n):
            for prev_idx in range(n):
                current = RCAState(tuple(int(b) for b in format(curr_idx, f'0{self.rca.size}b')))
                previous = RCAState(tuple(int(b) for b in format(prev_idx, f'0{self.rca.size}b')))
                self._get_or_create_node(current, previous)
        
        # Create all edges
        for (curr, prev), node in self.nodes.items():
            new_state, new_prev = self.rca.evolve(curr, prev)
            new_node = self.nodes[(new_state, new_prev)]
            self.edges.append((node.index, new_node.index))
        
        return self
    
    def get_adjacency_matrix(self) -> np.ndarray:
        """Get adjacency matrix of the graph."""
        n = len(self.nodes)
        if n == 0:
            return np.array([[]])
        
        adj = np.zeros((n, n), dtype=np.float64)
        for i, j in self.edges:
            adj[i, j] = 1.0
            adj[j, i] = 1.0  # Undirected for quantum walk
        return adj
    
    def get_directed_adjacency(self) -> np.ndarray:
        """Get directed adjacency (transition matrix)."""
        n = len(self.nodes)
        if n == 0:
            return np.array([[]])
        
        adj = np.zeros((n, n), dtype=np.float64)
        for i, j in self.edges:
            adj[j, i] = 1.0  # Column stochastic: from i to j
        return adj
    
    @property
    def num_nodes(self) -> int:
        return len(self.nodes)
    
    @property
    def num_edges(self) -> int:
        return len(self.edges)
    
    def get_statistics(self) -> dict:
        """Get graph statistics."""
        if self.num_nodes == 0:
            return {'num_nodes': 0, 'num_edges': 0}
        
        adj = self.get_adjacency_matrix()
        degrees = np.sum(adj, axis=1)
        
        return {
            'num_nodes': self.num_nodes,
            'num_edges': self.num_edges,
            'avg_degree': np.mean(degrees),
            'max_degree': np.max(degrees),
            'min_degree': np.min(degrees),
            'is_connected': self._is_connected(adj),
        }
    
    def _is_connected(self, adj: np.ndarray) -> bool:
        """Check if graph is connected using BFS."""
        if len(adj) == 0:
            return True
        
        visited = set([0])
        queue = [0]
        
        while queue:
            node = queue.pop(0)
            for neighbor in range(len(adj)):
                if adj[node, neighbor] > 0 and neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        
        return len(visited) == len(adj)


if __name__ == '__main__':
    from .automata import Rule90
    
    rca = Rule90(3)
    
    # Build from single initial state
    initial = RCAState((1, 0, 0))
    previous = RCAState((0, 0, 0))
    
    graph = RCAGraph(rca).build_from_initial(initial, previous)
    
    print(f"Graph from single orbit:")
    print(f"  Nodes: {graph.num_nodes}")
    print(f"  Edges: {graph.num_edges}")
    print(f"  Stats: {graph.get_statistics()}")
    
    # Build full state space
    graph_full = RCAGraph(rca).build_full_state_space()
    print(f"\nFull state space graph:")
    print(f"  Nodes: {graph_full.num_nodes}")
    print(f"  Edges: {graph_full.num_edges}")

