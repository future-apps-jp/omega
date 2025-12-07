"""
Task Definitions

Defines the tasks (fitness landscapes) that DSLs must solve.
"""

from genesis.tasks.graph_walk import GraphWalkTask, create_graph_walk_task

__all__ = ["GraphWalkTask", "create_graph_walk_task"]

