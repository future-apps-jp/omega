# Omega Project: Computational Foundations of Quantum Structure

This repository contains the source code, formal verification files, and research documentation for a series of papers investigating **whether quantum structure can be derived from computation**.

## Research Overview

The Omega Project addresses a fundamental question in the foundations of physics and computer science:

> **Can the distinctive features of quantum mechanics—superposition, interference, entanglement—arise from computation alone?**

Through systematic investigation across multiple phases, we have established:

| Phase | Paper | Key Finding |
|-------|-------|-------------|
| 0-3 | SK Combinatory Logic | SK computation is fundamentally classical; no quantum structure emerges |
| 4-7 | Reversible Computation | Reversible gates embed into Sp(2N,ℝ), not U(N) |
| 8-18 | Minimal Axioms | A1 (superposition) is the unique primitive axiom — **No-Go theorem formally verified in Coq** |
| 19-23 | Algorithmic Naturalness | Quantum mechanics has minimal description length on a quantum substrate |
| genesis | Artificial Physics | Matrix operations confer evolutionary advantage but do not spontaneously emerge |

**Central Conclusion**: Quantum structure cannot be derived from computation, simulation, or classical logic. The superposition axiom A1 must be postulated as a primitive extension of state space.

## Quick Guide for Reviewers

> **Note for Reviewers of "Minimal Axioms for Quantum Structure"**:  
> The core results discussed in the paper are located in:
> - **`sk-quantum/phase11/coq/PermSymplectic.v`** — No-Go Theorem (Coq formal verification)
> - **`sk-quantum/phase8/`** — GPT framework and axiom analysis
> 
> Other directories contain supplementary experiments from related papers and are not essential for reviewing the main claims.

## Repository Structure

```
omega/
├── sk-quantum/              # Computational experiments and formal verification
│   │
│   │ # Paper 1: SK Combinatory Logic (Phase 0-3)
│   ├── phase0/              # SK parser, reduction, multiway graph
│   ├── phase1/              # Algebraic structure and holonomy analysis
│   ├── phase2/              # Information-theoretic approach
│   │
│   │ # Paper 2: Reversible Computation (Phase 4-7)
│   ├── phase4/              # Reversible gates (Toffoli/Fredkin)
│   ├── phase5/              # Hamiltonian and quantum walk
│   ├── phase6/              # RCA and model comparison
│   ├── phase7/              # Non-commutativity and quantization
│   │
│   │ # Paper 3: Minimal Axioms (Phase 8-18) ← MAIN PAPER UNDER REVIEW
│   ├── phase8/              # GPT framework and axiom analysis (Main Result)
│   ├── phase9/              # Implication structure
│   ├── phase10/             # Model independence verification
│   ├── phase11/
│   │   └── coq/             # Coq formal verification (Main Result)
│   │       └── PermSymplectic.v   # ← Main verification file
│   │
│   │ # Paper 3 Extended: Structural Origins (Phase 13-18) (Supplementary)
│   ├── phase13/             # Symmetry analysis
│   ├── phase14/             # Contextuality analysis
│   ├── phase15/             # Causal structure analysis
│   ├── phase17/             # Additional analysis
│   │
│   │ # Paper 4: Algorithmic Naturalness (Phase 19-23)
│   ├── phase19/             # A1 language design
│   ├── phase20/             # Complexity metrics
│   ├── phase21/             # AWS Braket experiments
│   └── phase22/             # Theoretical deepening
│
├── a1/                      # A1 language implementation (Paper 4)
├── genesis/                 # Evolution simulation (Paper 5: Artificial Physics)
└── genesis-env/             # Genesis environment configuration
```

> **Note**: LaTeX source files and detailed research plans will be made available after paper acceptance.

## Formal Verification (Coq)

The No-Go Theorem has been formally verified using the Coq Proof Assistant with the Mathematical Components library.

- **File**: [`sk-quantum/phase11/coq/PermSymplectic.v`](sk-quantum/phase11/coq/PermSymplectic.v)
- **Requirements**: Coq 8.15+, MathComp 1.14+

### How to Verify the Proofs

To compile and verify the Coq proofs, follow these steps:

1. Install Coq and MathComp (e.g., via opam):
   ```bash
   opam install coq-mathcomp-ssreflect
   ```

2. Navigate to the verification directory:
   ```bash
   cd sk-quantum/phase11/coq/
   ```

3. Compile the proof file:
   ```bash
   coqc -Q . PermSymplectic PermSymplectic.v
   ```

If the command completes without output/error, the proofs are formally verified.

### Key Verified Theorems

```coq
(* Permutation matrices are orthogonal *)
Lemma perm_matrix_orthogonal (s : {perm 'I_n}) : 
  (perm_mx s)^T *m (perm_mx s) = 1%:M.

(* Main theorem: orthogonal => symplectic embedding *)
(* This establishes that reversible computation lives in Sp(2N,ℝ), not U(N) *)
Theorem embed_orthogonal_is_symplectic (P : 'M_N) :
  P^T *m P = 1%:M -> is_symplectic (embed P).

(* Corollary: all permutations embed in Sp *)
Corollary perm_embeds_in_Sp (s : 'S_N) :
  is_symplectic (embed (perm_mx s)).

(* No-Go: permutations preserve basis states *)
(* This theorem corresponds to the negation of Axiom A1 (Superposition):
   it proves that classical permutation dynamics cannot generate superposition,
   establishing that A1 must be postulated as a primitive axiom. *)
Theorem no_superposition_from_perm (s : 'S_n) (i : 'I_n) :
  exists j : 'I_n, perm_mx s *m basis_vec i = basis_vec j.
```

## Related Papers

1. **On the Independence of Quantum Structure from SK Combinatory Logic** (Phase 0-3)  
   [PhilPapers](https://philpapers.org/rec/KOHOTI)

2. **On the Limits of Deriving Quantum Structure from Reversible Computation** (Phase 4-7)  
   [PhilPapers](https://philpapers.org/rec/KOHLOD)

3. **Minimal Axioms for Quantum Structure: What Computation Cannot Derive** (Phase 8-18)  
   [PhilPapers](https://philpapers.org/rec/KOHMAF) | Under review at *Foundations of Physics*

4. **Algorithmic Naturalness on a Quantum Substrate** (Phase 19-23)  
   [PhilPapers](https://philpapers.org/rec/KOHANO-8)

5. **Artificial Physics: Evolutionary Emergence of Quantum Structures** (genesis/)  
   [PhilPapers](https://philpapers.org/rec/KOHAPE)

## License

This research is made available for academic purposes. Please cite appropriately if you use any part of this work.

## Contact

Hiroshi Kohashiguchi  
Independent Researcher, Tokyo, Japan  
hiroshi.kohashiguchi@future-apps.jp

