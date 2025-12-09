"""
A1: The Minimal Language for the Minimal Axiom

A1 is a homoiconic Scheme dialect that natively implements Axiom A1 
(state space extension / superposition). It is designed to measure 
Kolmogorov complexity on a quantum substrate.

Usage:
    from a1 import A1
    
    interpreter = A1()
    circuit = interpreter.run("(CNOT (H 0) 1)")
    print(circuit)

Author: Hiroshi Kohashiguchi
Date: December 2025
"""

from .core import A1
from .gates import QUANTUM_GATES
from .metrics import A1Metrics

__version__ = "0.1.0"
__all__ = ["A1", "QUANTUM_GATES", "A1Metrics"]



