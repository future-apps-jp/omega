"""
Fitness Evaluation

Defines how containers are evaluated for survival.
The fitness function embodies the "selection pressure" of the artificial universe.

Key principle: Fitness = Accuracy / (1 + K * penalty)
- Accuracy: How well the program solves the task
- K: Description length (Kolmogorov complexity proxy)
- penalty: Weighting for description length
"""

from typing import Any, Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np

from genesis.core.container import Container


@dataclass
class FitnessConfig:
    """Configuration for fitness evaluation."""
    k_penalty: float = 0.5  # Penalty coefficient for description length
    time_penalty: float = 0.1  # Penalty coefficient for execution time
    accuracy_weight: float = 1000.0  # Weight for accuracy component
    max_error: float = 1e10  # Maximum error before fitness = 0


class FitnessEvaluator:
    """
    Evaluates container fitness based on task performance and efficiency.
    
    The evaluator embodies the "holographic screen" that selects
    efficient descriptions over verbose ones.
    """
    
    def __init__(
        self,
        task_evaluator: Callable[[Container], Tuple[float, float]],
        config: Optional[FitnessConfig] = None,
    ):
        """
        Args:
            task_evaluator: Function that takes a Container and returns
                           (accuracy, error) tuple
            config: Fitness configuration
        """
        self.task_evaluator = task_evaluator
        self.config = config or FitnessConfig()
    
    def evaluate(self, container: Container) -> float:
        """
        Evaluate a single container's fitness.
        
        Returns:
            Fitness score (higher is better)
        """
        if not container.is_alive:
            return 0.0
        
        try:
            accuracy, error = self.task_evaluator(container)
            
            # Check for catastrophic error
            if error > self.config.max_error:
                container.mark_dead("Catastrophic error")
                return 0.0
            
            # Calculate fitness
            k = container.description_length()
            t = container.execution_time_ms()
            
            # Fitness = accuracy_weight / (1 + error + K*k_penalty + T*time_penalty)
            penalty = (
                error +
                k * self.config.k_penalty +
                t * self.config.time_penalty
            )
            
            fitness = self.config.accuracy_weight / (1 + penalty)
            container.fitness = fitness
            
            return fitness
            
        except Exception as e:
            container.mark_dead(f"Evaluation failed: {e}")
            return 0.0
    
    def evaluate_population(self, containers: List[Container]) -> Dict[str, float]:
        """
        Evaluate all containers in a population.
        
        Returns:
            Dict mapping container IDs to fitness scores
        """
        results = {}
        for container in containers:
            results[container.id] = self.evaluate(container)
        return results
    
    def rank_population(self, containers: List[Container]) -> List[Container]:
        """
        Rank containers by fitness (highest first).
        
        Also evaluates containers that haven't been evaluated yet.
        """
        for container in containers:
            if container.is_alive and container.fitness == 0.0:
                self.evaluate(container)
        
        return sorted(
            containers,
            key=lambda c: c.fitness,
            reverse=True
        )


def create_graph_walk_evaluator(
    adjacency_matrix: np.ndarray,
    steps: int,
    start_node: int,
    end_node: int,
) -> Callable[[Container], Tuple[float, float]]:
    """
    Create an evaluator for the graph walk task.
    
    The task is to compute the number of paths from start_node to end_node
    in exactly `steps` steps.
    
    Args:
        adjacency_matrix: Graph adjacency matrix
        steps: Number of steps
        start_node: Starting node index
        end_node: Ending node index
    
    Returns:
        Evaluator function
    """
    # Compute the ground truth
    target = np.linalg.matrix_power(adjacency_matrix, steps)[start_node, end_node]
    
    def evaluator(container: Container) -> Tuple[float, float]:
        """Evaluate a container on the graph walk task."""
        # Provide the adjacency matrix in context
        context = {
            "A": adjacency_matrix,
            "steps": steps,
            "start": start_node,
            "end": end_node,
        }
        
        result = container.evaluate(context)
        
        # Handle different result types
        if isinstance(result, np.ndarray):
            # If result is a matrix, try to extract the relevant element
            try:
                result = result[start_node, end_node]
            except (IndexError, TypeError):
                # If extraction fails, use matrix norm as a rough estimate
                result = np.linalg.norm(result)
        
        # Calculate error
        try:
            error = abs(float(target) - float(result))
        except (ValueError, TypeError):
            error = float('inf')
        
        # Accuracy is 1 if error is 0, decreasing as error increases
        accuracy = 1.0 / (1.0 + error)
        
        return accuracy, error
    
    return evaluator

