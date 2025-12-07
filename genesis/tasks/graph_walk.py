"""
Graph Walk Task

The primary task for Phase 25 "The Quantum Dawn":
Count the number of paths from node A to node B in exactly k steps.

This task has a natural "quantum advantage":
- Classical (Scalar DSL): Must enumerate paths, O(k) or O(N) description
- Quantum (Matrix DSL): A^k gives all paths at once, O(1) description

This is the fitness landscape where matrix DSLs should dominate.
"""

from typing import Tuple, Optional, Callable
from dataclasses import dataclass
import numpy as np

from genesis.core.container import Container
from genesis.core.fitness import FitnessEvaluator, FitnessConfig


@dataclass
class GraphWalkTask:
    """
    Graph walk task configuration.
    
    Attributes:
        adjacency_matrix: The graph's adjacency matrix
        steps: Number of steps (k)
        start_node: Starting node index
        end_node: Ending node index
        target_value: The correct answer (computed from A^k)
    """
    adjacency_matrix: np.ndarray
    steps: int
    start_node: int
    end_node: int
    target_value: float
    
    @property
    def n_nodes(self) -> int:
        return self.adjacency_matrix.shape[0]
    
    def __str__(self) -> str:
        return (f"GraphWalkTask(N={self.n_nodes}, k={self.steps}, "
                f"{self.start_node}→{self.end_node}, target={self.target_value})")


def create_graph_walk_task(
    n_nodes: int = 5,
    steps: int = 3,
    start_node: int = 0,
    end_node: int = 4,
    adjacency_matrix: Optional[np.ndarray] = None,
) -> GraphWalkTask:
    r"""
    Create a graph walk task.
    
    Args:
        n_nodes: Number of nodes in the graph
        steps: Number of steps for the walk
        start_node: Starting node
        end_node: Ending node
        adjacency_matrix: Optional custom adjacency matrix
    
    Returns:
        GraphWalkTask instance
    """
    if adjacency_matrix is None:
        # Create a default connected graph
        adjacency_matrix = create_default_graph(n_nodes)
    
    # Compute target value
    target = np.linalg.matrix_power(adjacency_matrix, steps)[start_node, end_node]
    
    return GraphWalkTask(
        adjacency_matrix=adjacency_matrix,
        steps=steps,
        start_node=start_node,
        end_node=end_node,
        target_value=float(target),
    )


def create_default_graph(n_nodes: int) -> np.ndarray:
    """
    Create a default connected graph.
    
    For n=5, creates:
    0 -- 1 -- 3 -- 4
     \  /|  /
      2--+
    """
    if n_nodes == 5:
        return np.array([
            [0, 1, 1, 0, 0],
            [1, 0, 1, 1, 0],
            [1, 1, 0, 1, 0],
            [0, 1, 1, 0, 1],
            [0, 0, 0, 1, 0]
        ], dtype=float)
    
    # For other sizes, create a simple chain with some extra edges
    adj = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes - 1):
        adj[i, i+1] = 1
        adj[i+1, i] = 1
    
    # Add some extra edges for connectivity
    for i in range(0, n_nodes - 2, 2):
        adj[i, i+2] = 1
        adj[i+2, i] = 1
    
    return adj


def create_graph_walk_evaluator(task: GraphWalkTask) -> Callable[[Container], Tuple[float, float]]:
    """
    Create an evaluator function for the graph walk task.
    
    Args:
        task: The graph walk task
    
    Returns:
        Evaluator function that takes a Container and returns (accuracy, error)
    """
    def evaluator(container: Container) -> Tuple[float, float]:
        # Provide context
        context = {
            "A": task.adjacency_matrix,
            "steps": task.steps,
            "start": task.start_node,
            "end": task.end_node,
            "k": task.steps,  # Alias
        }
        
        try:
            result = container.evaluate(context)
        except Exception as e:
            return 0.0, float('inf')
        
        # Handle different result types
        if isinstance(result, np.ndarray):
            try:
                # If result is a matrix, extract the relevant element
                result = result[task.start_node, task.end_node]
            except (IndexError, TypeError):
                # Matrix but wrong shape
                result = np.linalg.norm(result)
        
        # Calculate error
        try:
            error = abs(task.target_value - float(result))
        except (ValueError, TypeError):
            error = float('inf')
        
        # Accuracy: 1 if exact, decreasing with error
        if error == 0:
            accuracy = 1.0
        else:
            accuracy = 1.0 / (1.0 + error)
        
        return accuracy, error
    
    return evaluator


def create_graph_walk_fitness_evaluator(
    task: GraphWalkTask,
    config: Optional[FitnessConfig] = None,
) -> FitnessEvaluator:
    """
    Create a fitness evaluator for the graph walk task.
    
    Args:
        task: The graph walk task
        config: Optional fitness configuration
    
    Returns:
        FitnessEvaluator instance
    """
    evaluator_fn = create_graph_walk_evaluator(task)
    return FitnessEvaluator(evaluator_fn, config)


# Predefined tasks for experiments
TASK_SMALL = create_graph_walk_task(n_nodes=5, steps=3, start_node=0, end_node=4)
TASK_MEDIUM = create_graph_walk_task(n_nodes=10, steps=5, start_node=0, end_node=9)
TASK_LARGE = create_graph_walk_task(n_nodes=20, steps=10, start_node=0, end_node=19)

