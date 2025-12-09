# Minimal Axioms for Quantum Structure: What Computation Cannot Derive

This repository contains the source code, formal verification files, and research documentation for the paper **"Minimal Axioms for Quantum Structure: What Computation Cannot Derive"** by Hiroshi Kohashiguchi.

## Abstract

Recent developments in quantum foundations have strengthened the necessity of identifying the minimal departure point between classical and quantum theories. Through systematic analysis of multiple computational models—SK combinatory logic, reversible logic gates, reversible cellular automata, and lambda calculus—we establish that no form of computation can generate quantum structure without the superposition axiom A1.

Our main contributions are:

1. **The No-Go Theorem**: Reversible n-bit gates embed into the symplectic group Sp(2·2ⁿ, ℝ), not the unitary group U(2ⁿ). This theorem has been **formally verified in Coq**.
2. **Axiom Identification**: Using Generalized Probabilistic Theories (GPTs), we show that A1 (state-space extension/superposition) is the *unique primitive axiom* that cannot be derived from computation.
3. **Universality**: Results hold across all Turing-complete computation models tested, establishing computation-model independence.

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
├── papers/              # LaTeX source for papers
│   ├── minimal-axioms/  # This paper
│   ├── sk-quantum-independence/
│   └── computational-quantum-limits/
├── docs/                # Research plans and documentation
│   ├── research_plan.md      # Original plan (Phase 0-3)
│   ├── research_plan_v2.md   # Phase 4-7
│   ├── research_plan_v3.md   # Phase 8-12
│   ├── research_plan_v3.1.md # Phase 13-18
│   ├── research_plan_v4.md   # Phase 19-23
│   └── research_plan_v5.md   # Current
└── genesis/             # Experimental genesis environment
```

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

1. **On the Independence of Quantum Structure from SK Combinatory Logic**  
   [PhilPapers](https://philpapers.org/rec/KOHOTI)

2. **On the Limits of Deriving Quantum Structure from Reversible Computation**  
   [PhilPapers](https://philpapers.org/rec/KOHLOD)

3. **Minimal Axioms for Quantum Structure** (this paper)  
   Available in `papers/minimal-axioms/`

## License

This research is made available for academic purposes. Please cite appropriately if you use any part of this work.

## Contact

Hiroshi Kohashiguchi  
Independent Researcher, Tokyo, Japan

