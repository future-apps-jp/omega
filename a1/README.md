# A1: The Minimal Language for the Minimal Axiom

**A1** is a homoiconic Scheme dialect that natively implements **Axiom A1** (state space extension / superposition). It is designed to measure Kolmogorov complexity on a quantum substrate.

---

## Overview

A1 is part of the research project "Algorithmic Naturalness on a Quantum Substrate", which investigates whether quantum mechanics is algorithmically natural on a quantum computational substrate.

### Key Features

- **Minimal syntax**: Scheme-like S-expressions
- **Quantum-native**: Gates are primitives, not library calls
- **Measurable complexity**: Token count ≈ Kolmogorov complexity
- **AWS Braket integration**: Execute on real quantum hardware

### Description Length Comparison

| Program | A1 Tokens | NumPy Tokens | Ratio |
|---------|-----------|--------------|-------|
| Bell State | 4 | 52 | 13x |
| GHZ State | 6 | 78 | 13x |
| Teleportation | 18 | 234 | 13x |

---

## Quick Start

### Installation

```bash
cd /home/hkohashi/research
pip install amazon-braket-sdk  # For AWS execution
```

### Hello Quantum World

```scheme
; bell_state.a1 - Create Bell state (|00⟩ + |11⟩)/√2
(CNOT (H 0) 1)
```

### Python API

```python
from a1 import A1

# Create interpreter
interpreter = A1()

# Run A1 code
circuit = interpreter.run("(CNOT (H 0) 1)")
print(circuit)
```

---

## Language Reference

### Quantum Gates (Primitives)

| Gate | Syntax | Description |
|------|--------|-------------|
| H | `(H q)` | Hadamard gate |
| X | `(X q)` | Pauli-X (NOT) |
| Y | `(Y q)` | Pauli-Y |
| Z | `(Z q)` | Pauli-Z |
| CNOT | `(CNOT c t)` | Controlled-NOT |
| CZ | `(CZ c t)` | Controlled-Z |
| SWAP | `(SWAP q1 q2)` | Swap qubits |
| RX | `(RX q θ)` | X-rotation |
| RY | `(RY q θ)` | Y-rotation |
| RZ | `(RZ q θ)` | Z-rotation |

### Special Forms

```scheme
; Variable definition
(DEFINE x 5)

; Function definition
(DEFINE (bell q0 q1)
    (CNOT (H q0) q1))

; Lambda expression
(LAMBDA (x) (H x))

; Sequential execution
(BEGIN (H 0) (CNOT 0 1))

; Conditional
(IF condition then-expr else-expr)
```

### Examples

#### Bell State
```scheme
(CNOT (H 0) 1)
```

#### GHZ State (3-qubit)
```scheme
(CNOT (CNOT (H 0) 1) 2)
```

#### Quantum Teleportation
```scheme
(BEGIN
  ; Create Bell pair
  (CNOT (H 1) 2)
  ; Bell measurement
  (H 0)
  (CNOT 0 1)
  ; Corrections (classical control simulated)
  (X 2)
  (Z 2))
```

---

## Complexity Measurement

```python
from a1.metrics import A1Metrics, ClassicalMetrics, ComplexityComparison

# Measure A1 complexity
a1_code = "(CNOT (H 0) 1)"
a1_metrics = A1Metrics(a1_code)
print(f"Tokens: {a1_metrics.token_count}")
print(f"Bits: {a1_metrics.bit_length}")

# Compare with classical
comparison = ComplexityComparison(a1_code, classical_code)
print(f"Ratio: {comparison.token_ratio}x")
```

---

## AWS Braket Integration

A1 programs can be executed on AWS Braket quantum hardware and simulators.

### Prerequisites

1. AWS Account with Braket access
2. IAM user with appropriate permissions
3. S3 bucket for results

### Setup

#### 1. Environment Configuration

Copy `env.example` to `.env` and configure:

```bash
cp env.example .env
```

Edit `.env`:
```bash
# AWS Credentials (Method A: Profile - Recommended)
AWS_PROFILE=your-profile-name
AWS_DEFAULT_REGION=us-east-1

# OR (Method B: Direct credentials - Less secure)
AWS_ACCESS_KEY_ID=your_access_key_id
AWS_SECRET_ACCESS_KEY=your_secret_access_key
AWS_DEFAULT_REGION=us-east-1

# S3 Bucket for results
BRAKET_S3_BUCKET=amazon-braket-your-bucket-name
BRAKET_S3_PREFIX=a1-experiments

# Safety: Disable QPU by default
ENABLE_QPU_EXECUTION=false
```

#### 2. Create S3 Bucket

```bash
aws s3 mb s3://amazon-braket-your-bucket-name --region us-east-1
```

#### 3. IAM Permissions

Required IAM policy:

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "BraketQuantumTaskManagement",
            "Effect": "Allow",
            "Action": [
                "braket:CreateQuantumTask",
                "braket:GetQuantumTask",
                "braket:CancelQuantumTask",
                "braket:SearchDevices"
            ],
            "Resource": "*"
        },
        {
            "Sid": "BraketS3Access",
            "Effect": "Allow",
            "Action": [
                "s3:PutObject",
                "s3:GetObject",
                "s3:ListBucket"
            ],
            "Resource": [
                "arn:aws:s3:::amazon-braket-*",
                "arn:aws:s3:::amazon-braket-*/*"
            ]
        },
        {
            "Sid": "S3BucketManagement",
            "Effect": "Allow",
            "Action": [
                "s3:CreateBucket",
                "s3:ListAllMyBuckets",
                "s3:ListBucket",
                "s3:PutObject",
                "s3:GetObject",
                "s3:DeleteObject"
            ],
            "Resource": [
                "arn:aws:s3:::amazon-braket-*",
                "arn:aws:s3:::amazon-braket-*/*"
            ]
        },
        {
            "Sid": "BraketFullAccess",
            "Effect": "Allow",
            "Action": [
                "braket:*"
            ],
            "Resource": "*"
        }
    ]
}
```

#### 4. Verify Configuration

```bash
python3 -c "from a1.config import config; config.print_status()"
```

### Run Experiments

```bash
# Local simulator (free)
python3 sk-quantum/phase21/experiments/run_aws.py --backend local

# AWS SV1 simulator (~$0.01)
python3 sk-quantum/phase21/experiments/run_aws.py --backend sv1

# Real QPU (requires ENABLE_QPU_EXECUTION=true)
python3 sk-quantum/phase21/experiments/run_aws.py --backend ionq
```

### Pricing

| Backend | Cost |
|---------|------|
| Local Simulator | Free |
| SV1 (Cloud Simulator) | ~$0.01/experiment |
| IonQ | $0.30/task + $0.01/shot |
| Rigetti | $0.30/task + $0.00035/shot |

---

## Troubleshooting

### "Unable to locate credentials"

```bash
# Check profile
aws configure list --profile your-profile-name

# Or check environment
echo $AWS_ACCESS_KEY_ID
```

### "Access Denied"

Ensure IAM user has `braket:*` and `s3:*` permissions for `amazon-braket-*` resources.

### "Bucket does not exist"

```bash
# List buckets
aws s3 ls | grep amazon-braket

# Create if missing
aws s3 mb s3://amazon-braket-your-bucket --region us-east-1
```

---

## Project Structure

```
a1/
├── __init__.py      # Package initialization
├── core.py          # Parser and interpreter
├── gates.py         # Quantum gate definitions
├── metrics.py       # Complexity measurement
├── config.py        # AWS configuration loader
├── env.example      # Environment template
├── examples/        # Example programs
│   ├── hello_world.a1
│   ├── ghz_state.a1
│   └── teleport.a1
└── tests/           # Test suite
    ├── test_core.py
    ├── test_gates.py
    └── test_metrics.py
```

---

## Running Tests

```bash
cd /home/hkohashi/research
python3 -m pytest a1/tests/ -v
```

Current status: **96 tests passing** ✅

---

## Related Work

- **Algorithmic Naturalness on a Quantum Substrate** (Phase 19-23)  
  [PhilPapers](https://philpapers.org/rec/KOHANO-8)

- **Minimal Axioms for Quantum Structure: What Computation Cannot Derive** (Phase 8-18)  
  [PhilPapers](https://philpapers.org/rec/KOHMAF)

---

## References

- [AWS Braket Documentation](https://docs.aws.amazon.com/braket/)

---

## Author

**Hiroshi Kohashiguchi**  
December 2025

---

*A1: The language where quantum is natural.*

