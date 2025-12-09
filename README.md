# Omega Project: Computational Foundations of Quantum Structure

This repository contains the source code, formal verification files, and research documentation for a series of papers investigating **whether quantum structure can be derived from computation**.

## Research Overview

The Omega Project addresses a fundamental question in the foundations of physics and computer science:

> **Can the distinctive features of quantum mechanics—superposition, interference, entanglement—arise from computation alone?**

Through systematic investigation across multiple phases, we have established:

| Phase | Paper | Key Finding |
|-------|-------|-------------|
| 1 | SK Combinatory Logic | SK computation is fundamentally classical; no quantum structure emerges |
| 2 | Reversible Computation | Reversible gates embed into Sp(2N,ℝ), not U(N) — **formally verified in Coq** |
| 3 | Minimal Axioms | A1 (superposition) is the unique primitive axiom that cannot be derived |
| 4 | Algorithmic Naturalness | Quantum mechanics has minimal description length on a quantum substrate |
| 5 | Artificial Physics | Matrix operations confer evolutionary advantage but do not spontaneously emerge |

**Central Conclusion**: Quantum structure cannot be derived from computation, simulation, or classical logic. The superposition axiom A1 must be postulated as a primitive extension of state space.

## Repository Structure

```
omega/
├── sk-quantum/          # Computational experiments and formal verification
│   ├── phase0/          # SK parser, reduction, multiway graph
│   ├── phase1/          # Algebraic structure and holonomy analysis
│   ├── phase2/          # Information-theoretic approach
│   ├── phase4/          # Reversible gates (Toffoli/Fredkin)
│   ├── phase5/          # Hamiltonian and quantum walk
│   ├── phase6/          # RCA and model comparison
│   └── phase11/
│       └── coq/         # Coq formal verification
│           └── PermSymplectic.v   # ← Main verification file
├── a1/                  # A1 language implementation
├── genesis/             # Experimental genesis environment
└── genesis-env/         # Genesis environment configuration
```

> **Note**: LaTeX source files and detailed research plans will be made available after paper acceptance.

## Formal Verification (Coq)

The No-Go Theorem has been formally verified using the Coq Proof Assistant with the Mathematical Components library.

- **File**: [`sk-quantum/phase11/coq/PermSymplectic.v`](sk-quantum/phase11/coq/PermSymplectic.v)
- **Requirements**: Coq 8.15+, MathComp 1.14+

### Key Verified Theorems

```coq
(* Permutation matrices are orthogonal *)
Lemma perm_matrix_orthogonal (s : {perm 'I_n}) : 
  (perm_mx s)^T *m (perm_mx s) = 1%:M.

(* Main theorem: orthogonal => symplectic embedding *)
Theorem embed_orthogonal_is_symplectic (P : 'M_N) :
  P^T *m P = 1%:M -> is_symplectic (embed P).

(* Corollary: all permutations embed in Sp *)
Corollary perm_embeds_in_Sp (s : 'S_N) :
  is_symplectic (embed (perm_mx s)).

(* No-Go: permutations preserve basis states *)
Theorem no_superposition_from_perm (s : 'S_n) (i : 'I_n) :
  exists j : 'I_n, perm_mx s *m basis_vec i = basis_vec j.
```

## Related Papers

1. **On the Independence of Quantum Structure from SK Combinatory Logic** (Phase 1)  
   [PhilPapers](https://philpapers.org/rec/KOHOTI)

2. **On the Limits of Deriving Quantum Structure from Reversible Computation** (Phase 2)  
   [PhilPapers](https://philpapers.org/rec/KOHLOD)

3. **Minimal Axioms for Quantum Structure: What Computation Cannot Derive** (Phase 3)  
   [PhilPapers](https://philpapers.org/rec/KOHMAF) | Under review at *Foundations of Physics*

4. **Algorithmic Naturalness on a Quantum Substrate** (Phase 4)  
   [PhilPapers](https://philpapers.org/rec/KOHANO-8)

5. **Artificial Physics: Evolutionary Emergence of Quantum Structures** (Phase 5)  
   [PhilPapers](https://philpapers.org/rec/KOHAPE)

## License

This research is made available for academic purposes. Please cite appropriately if you use any part of this work.

## Contact

Hiroshi Kohashiguchi  
Independent Researcher, Tokyo, Japan

