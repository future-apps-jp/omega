# Genesis-Matrix

**Evolutionary Emergence of Quantum Structure Simulation Framework**

Genesis-Matrix is a simulation framework that demonstrates how matrix operations (quantum-like structures) naturally become dominant when Domain-Specific Languages (DSLs) evolve under resource constraints.

## Overview

### Research Hypothesis

> Physical laws (particularly the quantum axiom A1) emerge evolutionarily under computational resource constraints.

### Core Insight

1. **Scalar DSL (Classical)**: Only scalar operations. Graph traversal requires enumerating paths one by one, with description length K = O(N)
2. **Matrix DSL (Quantum-like)**: Has matrix operations. A^k computes all paths simultaneously, with description length K = O(1)

This difference in description length creates selection pressure, causing Matrix DSL to win evolutionarily.

## Installation

```bash
# Create virtual environment
python3 -m venv genesis-env
source genesis-env/bin/activate

# Install dependencies
pip install numpy jax[cpu] pytest
```

## Quick Start

```bash
# Check environment
python genesis/check_cpu.py

# Run tests
python -m pytest genesis/tests/ -v

# Run "The Quantum Dawn" experiment
python genesis/experiments/quantum_dawn.py
```

## Structure

```
genesis/
├── core/                  # Foundation classes
│   ├── dsl.py            # DSL abstract class
│   ├── localhost.py      # Parent universe (resource management)
│   ├── container.py      # Child universe (DSL + phenomena)
│   └── fitness.py        # Fitness evaluation
├── dsl/                   # DSL implementations
│   ├── scalar.py         # Species A: Classical scalar operations
│   └── matrix.py         # Species B: Matrix operations (quantum-like)
├── evolution/             # Evolution engine
│   ├── mutation.py       # AST structural mutation
│   ├── selection.py      # Selection mechanisms
│   └── population.py     # Population management
├── tasks/                 # Task definitions
│   └── graph_walk.py     # Graph traversal task
├── experiments/           # Experiment scripts
│   └── quantum_dawn.py   # Phase 25 experiment
└── tests/                 # Test suite
```

## Key Concepts

### Localhost (Parent Universe)

A "meta-universe" that provides computational resources. It allocates resources to efficient processes without teleological intent.

### Container (Child Universe)

A "child universe" with its own DSL (physical laws) and phenomena (programs). It acquires resources at an inflation rate V_inf ∝ 1/(K × T).

### Holographic Constraint

By imposing an upper limit on description length K, we create evolutionary pressure that selects for efficient descriptions.

## Experimental Results

### Phase 25: "The Quantum Dawn"

```
Task: Calculate the number of 3-step paths from node 0 to node 4 in a 5-node graph
Target: 2 (correct answer)

Initial: 80% Scalar, 20% Matrix
Final:   0% Scalar, 100% Matrix

→ Matrix DSL (A1) dominates evolutionarily!
```

## Related Work

- Paper: "Artificial Physics: Evolutionary Emergence of Quantum Structures"
- [PhilPapers](https://philpapers.org/rec/KOHAPE)

## License

MIT License
