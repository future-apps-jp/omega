"""
Axiom Candidates for Quantum Structure

This module formalizes the axiom candidates that might bridge
computation and quantum mechanics.

Axioms (in GPT language):
A1: State Space Extension — Ω is strictly larger than Δ (simplex)
A2: Born Rule — Inner product of effect and state gives probability
A3: Reversibility — Transformations are reversible (unitary generalization)
A4: Non-commutativity — Measurement order affects results
A5: No-Cloning — Universal state copying is impossible
A6: Contextuality — Measurement results depend on context

Key Question: Which axioms are independent? Which can be derived from computation?
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Set, Tuple, Optional
from enum import Enum, auto
from abc import ABC, abstractmethod

from .gpt_framework import (
    GPT, StateSpace, StateSpaceType, 
    Measurement, Effect, LinearTransformation
)


class AxiomID(Enum):
    """Identifiers for axiom candidates."""
    A1_STATE_EXTENSION = auto()      # State space is larger than simplex
    A2_BORN_RULE = auto()            # Probability = |⟨effect|state⟩|²
    A3_REVERSIBILITY = auto()        # Transformations are reversible
    A4_NON_COMMUTATIVITY = auto()    # Measurement order matters
    A5_NO_CLONING = auto()           # Cannot clone unknown states
    A6_CONTEXTUALITY = auto()        # Results depend on measurement context


@dataclass
class AxiomStatus:
    """Status of an axiom in a given theory."""
    axiom_id: AxiomID
    satisfied: bool
    derivable: bool  # Can be derived from other axioms
    evidence: str    # Explanation or proof sketch
    
    def __repr__(self) -> str:
        status = "✓" if self.satisfied else "✗"
        deriv = "(derivable)" if self.derivable else "(primitive)"
        return f"{self.axiom_id.name}: {status} {deriv}"


class AxiomChecker(ABC):
    """Abstract base class for axiom checkers."""
    
    @abstractmethod
    def check(self, gpt: GPT) -> AxiomStatus:
        """Check if the axiom holds in the given GPT."""
        pass


class StateExtensionChecker(AxiomChecker):
    """
    A1: Check if state space extends beyond simplex.
    
    A theory satisfies this if its state space contains states
    that cannot be represented as probability distributions over
    pure classical states.
    """
    
    def check(self, gpt: GPT) -> AxiomStatus:
        is_simplex = gpt.state_space.space_type == StateSpaceType.SIMPLEX
        
        if is_simplex:
            return AxiomStatus(
                axiom_id=AxiomID.A1_STATE_EXTENSION,
                satisfied=False,
                derivable=False,
                evidence="State space is a simplex (classical)"
            )
        else:
            return AxiomStatus(
                axiom_id=AxiomID.A1_STATE_EXTENSION,
                satisfied=True,
                derivable=False,
                evidence=f"State space type: {gpt.state_space.space_type.value}"
            )


class BornRuleChecker(AxiomChecker):
    """
    A2: Check if probabilities follow Born rule structure.
    
    In GPTs, this means effects are linear functionals on states.
    Classical theories trivially satisfy this.
    """
    
    def check(self, gpt: GPT) -> AxiomStatus:
        # In our GPT framework, Born rule is built-in via Effect.probability
        # The key question is whether it has the specific form |⟨·|·⟩|²
        
        # Check if there's a Hilbert space structure
        # For now, we check if state space is ball-like (has inner product)
        
        has_inner_product = gpt.state_space.space_type == StateSpaceType.BALL
        
        return AxiomStatus(
            axiom_id=AxiomID.A2_BORN_RULE,
            satisfied=True,  # Linear probabilities always hold in GPT
            derivable=True,  # Derivable from GPT axioms
            evidence="Linear probability rule holds (GPT axiom); "
                    f"Inner product structure: {has_inner_product}"
        )


class ReversibilityChecker(AxiomChecker):
    """
    A3: Check if all transformations are reversible.
    
    Classical computation: permutation matrices (reversible)
    Quantum computation: unitary matrices (reversible)
    """
    
    def check(self, gpt: GPT) -> AxiomStatus:
        n_reversible = sum(1 for t in gpt.transformations if t.is_reversible())
        n_total = len(gpt.transformations)
        
        all_reversible = (n_reversible == n_total)
        
        return AxiomStatus(
            axiom_id=AxiomID.A3_REVERSIBILITY,
            satisfied=all_reversible,
            derivable=False,
            evidence=f"{n_reversible}/{n_total} transformations are reversible"
        )


class NonCommutativityChecker(AxiomChecker):
    """
    A4: Check if measurements can be non-commutative.
    
    Non-commutativity means: measuring A then B gives different
    results than measuring B then A.
    
    This is related to whether the theory has incompatible observables.
    """
    
    def check(self, gpt: GPT) -> AxiomStatus:
        if len(gpt.measurements) < 2:
            return AxiomStatus(
                axiom_id=AxiomID.A4_NON_COMMUTATIVITY,
                satisfied=False,
                derivable=False,
                evidence="Only one measurement available"
            )
        
        # Check if any pair of measurements is incompatible
        # Two measurements are compatible if they can be refined
        # into a common measurement
        
        # Simplified check: look at effect vectors
        has_non_commuting = False
        evidence_pairs = []
        
        for i, m1 in enumerate(gpt.measurements):
            for j, m2 in enumerate(gpt.measurements):
                if i >= j:
                    continue
                
                # Check if effect vectors span different subspaces
                # (simplified: check if they're not parallel)
                for e1 in m1.effects:
                    for e2 in m2.effects:
                        v1 = e1.vector / (np.linalg.norm(e1.vector) + 1e-10)
                        v2 = e2.vector / (np.linalg.norm(e2.vector) + 1e-10)
                        
                        # Check orthogonality
                        dot = abs(np.dot(v1, v2))
                        if 0.1 < dot < 0.9:  # Neither parallel nor orthogonal
                            has_non_commuting = True
                            evidence_pairs.append((m1.name, m2.name))
        
        if has_non_commuting:
            evidence = f"Non-commuting measurements: {evidence_pairs[:3]}"
        else:
            evidence = "All measurements appear compatible"
        
        return AxiomStatus(
            axiom_id=AxiomID.A4_NON_COMMUTATIVITY,
            satisfied=has_non_commuting,
            derivable=False,
            evidence=evidence
        )


class NoCloningChecker(AxiomChecker):
    """
    A5: Check if no-cloning holds.
    
    No-cloning: There is no transformation that clones arbitrary states.
    
    Classical theories: VIOLATE no-cloning (can copy classical bits)
    Quantum theories: SATISFY no-cloning
    """
    
    def check(self, gpt: GPT) -> AxiomStatus:
        # In classical theories (simplex), we can always copy
        # by measuring and preparing the same state
        
        is_classical = gpt.state_space.space_type == StateSpaceType.SIMPLEX
        
        if is_classical:
            return AxiomStatus(
                axiom_id=AxiomID.A5_NO_CLONING,
                satisfied=False,
                derivable=True,  # Derivable from simplex structure
                evidence="Classical states can be cloned (measure & prepare)"
            )
        else:
            # For non-classical theories, no-cloning typically holds
            # due to existence of non-orthogonal states
            return AxiomStatus(
                axiom_id=AxiomID.A5_NO_CLONING,
                satisfied=True,
                derivable=True,  # Derivable from non-orthogonal states
                evidence="Non-orthogonal states exist; no-cloning follows"
            )


class ContextualityChecker(AxiomChecker):
    """
    A6: Check if the theory exhibits contextuality.
    
    Contextuality: Measurement outcomes depend on which other
    measurements are performed simultaneously.
    
    Classical: Non-contextual (outcomes predetermined)
    Quantum: Contextual (Kochen-Specker)
    """
    
    def check(self, gpt: GPT) -> AxiomStatus:
        # Contextuality is deeply related to the geometry of state space
        # Simplex → non-contextual
        # Ball → potentially contextual
        
        is_classical = gpt.state_space.space_type == StateSpaceType.SIMPLEX
        
        if is_classical:
            return AxiomStatus(
                axiom_id=AxiomID.A6_CONTEXTUALITY,
                satisfied=False,
                derivable=True,
                evidence="Simplex state space admits hidden variable model"
            )
        else:
            # Non-simplex may exhibit contextuality
            # Full check would require Kochen-Specker analysis
            return AxiomStatus(
                axiom_id=AxiomID.A6_CONTEXTUALITY,
                satisfied=True,  # Assumed for ball-like spaces
                derivable=False,
                evidence="Non-simplex state space; contextuality possible"
            )


# =============================================================================
# Axiom Implication Analysis
# =============================================================================

class AxiomImplicationGraph:
    """
    Analyze implication relationships between axioms.
    
    An axiom A implies axiom B if whenever A holds, B must hold.
    """
    
    def __init__(self):
        self.checkers = {
            AxiomID.A1_STATE_EXTENSION: StateExtensionChecker(),
            AxiomID.A2_BORN_RULE: BornRuleChecker(),
            AxiomID.A3_REVERSIBILITY: ReversibilityChecker(),
            AxiomID.A4_NON_COMMUTATIVITY: NonCommutativityChecker(),
            AxiomID.A5_NO_CLONING: NoCloningChecker(),
            AxiomID.A6_CONTEXTUALITY: ContextualityChecker(),
        }
        
        # Known implications (from theory)
        # Format: (antecedent, consequent)
        self.theoretical_implications: List[Tuple[AxiomID, AxiomID]] = [
            # A1 + A3 → A5 (state extension + reversibility → no-cloning)
            # This is the no-cloning theorem
            
            # A1 + A4 → A6 (state extension + non-commutativity → contextuality)
            # This is Kochen-Specker-like
            
            # A6 → A4 (contextuality → non-commutativity)
            (AxiomID.A6_CONTEXTUALITY, AxiomID.A4_NON_COMMUTATIVITY),
        ]
    
    def check_all(self, gpt: GPT) -> Dict[AxiomID, AxiomStatus]:
        """Check all axioms for a given GPT."""
        return {aid: checker.check(gpt) for aid, checker in self.checkers.items()}
    
    def analyze_implications(self, gpt: GPT) -> Dict[str, any]:
        """
        Analyze which axioms hold and their implications.
        
        Returns:
            Dictionary with axiom status and implication analysis
        """
        statuses = self.check_all(gpt)
        
        # Count satisfied axioms
        satisfied = [aid for aid, status in statuses.items() if status.satisfied]
        not_satisfied = [aid for aid, status in statuses.items() if not status.satisfied]
        derivable = [aid for aid, status in statuses.items() if status.derivable]
        primitive = [aid for aid, status in statuses.items() if not status.derivable]
        
        return {
            'gpt_name': gpt.name,
            'statuses': statuses,
            'satisfied': [a.name for a in satisfied],
            'not_satisfied': [a.name for a in not_satisfied],
            'derivable': [a.name for a in derivable],
            'primitive': [a.name for a in primitive],
            'n_satisfied': len(satisfied),
            'n_total': len(statuses),
            'theoretical_implications': [
                (a.name, b.name) for a, b in self.theoretical_implications
            ]
        }


def compare_axiom_status(gpts: List[GPT]) -> Dict[str, any]:
    """
    Compare axiom status across multiple GPTs.
    
    Args:
        gpts: List of GPTs to compare
        
    Returns:
        Comparison table
    """
    graph = AxiomImplicationGraph()
    
    results = {}
    for gpt in gpts:
        analysis = graph.analyze_implications(gpt)
        results[gpt.name] = analysis
    
    # Create comparison table
    all_axioms = list(AxiomID)
    comparison_table = {}
    
    for axiom in all_axioms:
        comparison_table[axiom.name] = {
            gpt.name: results[gpt.name]['statuses'][axiom].satisfied
            for gpt in gpts
        }
    
    return {
        'detailed_results': results,
        'comparison_table': comparison_table,
        'axiom_names': [a.name for a in all_axioms],
        'gpt_names': [g.name for g in gpts]
    }


# =============================================================================
# Minimal Axiom Set Search
# =============================================================================

def find_minimal_axiom_set_for_quantum() -> Dict[str, any]:
    """
    Attempt to find the minimal set of axioms that characterizes
    quantum theory vs classical computation.
    
    Returns:
        Analysis of minimal axiom sets
    """
    # Based on theoretical analysis
    
    minimal_sets = [
        {
            'name': 'Chiribella-D\'Ariano-Perinotti',
            'axioms': ['A1', 'A2', 'A3', 'Purification'],
            'reference': 'Chiribella et al. 2011',
            'notes': 'Information-theoretic derivation; Purification is key'
        },
        {
            'name': 'Hardy',
            'axioms': ['A1', 'A2', 'A3', 'A4', 'Continuity'],
            'reference': 'Hardy 2001',
            'notes': 'Five reasonable axioms; Continuity distinguishes from boxworld'
        },
        {
            'name': 'Masanes-Müller',
            'axioms': ['A1', 'A3', 'Tomographic locality'],
            'reference': 'Masanes & Müller 2011',
            'notes': 'Physical requirements approach'
        },
        {
            'name': 'This work (conjecture)',
            'axioms': ['A1'],
            'reference': 'Phase 8 hypothesis',
            'notes': 'State extension (superposition) may be the only primitive; '
                    'others derivable'
        }
    ]
    
    return {
        'minimal_sets': minimal_sets,
        'key_insight': 'A1 (state space extension / superposition) appears in ALL '
                      'minimal axiom sets, suggesting it is truly primitive.',
        'derivable_from_A1': ['A5 (No-Cloning)', 'possibly A4, A6 with structure'],
        'not_derivable_from_computation': ['A1 (superposition)'],
        'computation_provides': ['A3 (reversibility for reversible computation)'],
    }


if __name__ == "__main__":
    from .gpt_framework import create_classical_bit_gpt, create_qubit_gpt
    
    print("=" * 60)
    print("Axiom Analysis")
    print("=" * 60)
    
    # Create theories
    classical = create_classical_bit_gpt(1)
    quantum = create_qubit_gpt()
    
    # Compare axiom status
    comparison = compare_axiom_status([classical, quantum])
    
    print("\nAxiom Comparison Table:")
    print("-" * 40)
    for axiom_name, gpt_status in comparison['comparison_table'].items():
        status_str = " | ".join(
            f"{gpt}: {'✓' if sat else '✗'}" 
            for gpt, sat in gpt_status.items()
        )
        print(f"{axiom_name}: {status_str}")
    
    print("\n" + "=" * 60)
    print("Minimal Axiom Sets for Quantum")
    print("=" * 60)
    
    minimal = find_minimal_axiom_set_for_quantum()
    for ms in minimal['minimal_sets']:
        print(f"\n{ms['name']}:")
        print(f"  Axioms: {ms['axioms']}")
        print(f"  Notes: {ms['notes']}")
    
    print(f"\nKey Insight: {minimal['key_insight']}")


