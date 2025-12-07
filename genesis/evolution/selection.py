"""
Selection Mechanisms

Implements various selection strategies for evolutionary computation.
"""

from typing import List, Tuple
import random

from genesis.core.container import Container


class Selection:
    """
    Selection strategies for evolutionary computation.
    """
    
    @staticmethod
    def tournament(
        population: List[Container],
        tournament_size: int = 3,
        n_select: int = 1,
    ) -> List[Container]:
        """
        Tournament selection.
        
        Args:
            population: List of containers to select from
            tournament_size: Number of contestants per tournament
            n_select: Number of winners to select
        
        Returns:
            List of selected containers
        """
        alive = [c for c in population if c.is_alive]
        if not alive:
            return []
        
        selected = []
        for _ in range(n_select):
            # Select random contestants
            contestants = random.sample(
                alive,
                min(tournament_size, len(alive))
            )
            
            # Winner is the one with highest fitness
            winner = max(contestants, key=lambda c: c.fitness)
            selected.append(winner)
        
        return selected
    
    @staticmethod
    def roulette(
        population: List[Container],
        n_select: int = 1,
    ) -> List[Container]:
        """
        Roulette wheel selection (fitness-proportionate).
        
        Args:
            population: List of containers
            n_select: Number to select
        
        Returns:
            List of selected containers
        """
        alive = [c for c in population if c.is_alive]
        if not alive:
            return []
        
        # Calculate selection probabilities
        total_fitness = sum(c.fitness for c in alive)
        if total_fitness == 0:
            # Equal probability if all fitness is 0
            return random.sample(alive, min(n_select, len(alive)))
        
        probabilities = [c.fitness / total_fitness for c in alive]
        
        # Select
        selected = []
        for _ in range(n_select):
            r = random.random()
            cumulative = 0
            for i, prob in enumerate(probabilities):
                cumulative += prob
                if r <= cumulative:
                    selected.append(alive[i])
                    break
        
        return selected
    
    @staticmethod
    def elite(
        population: List[Container],
        n_elite: int = 5,
    ) -> List[Container]:
        """
        Elite selection (top N by fitness).
        
        Args:
            population: List of containers
            n_elite: Number of elite to select
        
        Returns:
            List of elite containers
        """
        alive = [c for c in population if c.is_alive]
        sorted_pop = sorted(alive, key=lambda c: c.fitness, reverse=True)
        return sorted_pop[:n_elite]
    
    @staticmethod
    def truncation(
        population: List[Container],
        survival_rate: float = 0.2,
    ) -> List[Container]:
        """
        Truncation selection (keep top percentage).
        
        Args:
            population: List of containers
            survival_rate: Fraction to keep
        
        Returns:
            List of surviving containers
        """
        alive = [c for c in population if c.is_alive]
        sorted_pop = sorted(alive, key=lambda c: c.fitness, reverse=True)
        n_keep = max(1, int(len(sorted_pop) * survival_rate))
        return sorted_pop[:n_keep]
    
    @staticmethod
    def species_balanced(
        population: List[Container],
        n_select: int = 10,
    ) -> List[Container]:
        """
        Species-balanced selection.
        
        Ensures diversity by selecting top individuals from each species.
        
        Args:
            population: List of containers
            n_select: Total number to select
        
        Returns:
            List of selected containers
        """
        alive = [c for c in population if c.is_alive]
        
        # Group by species
        species_groups = {}
        for c in alive:
            if c.species not in species_groups:
                species_groups[c.species] = []
            species_groups[c.species].append(c)
        
        # Select top from each species
        selected = []
        n_species = len(species_groups)
        per_species = max(1, n_select // n_species)
        
        for species, members in species_groups.items():
            sorted_members = sorted(members, key=lambda c: c.fitness, reverse=True)
            selected.extend(sorted_members[:per_species])
        
        # If we need more, add from overall top
        if len(selected) < n_select:
            remaining = sorted(
                [c for c in alive if c not in selected],
                key=lambda c: c.fitness,
                reverse=True
            )
            selected.extend(remaining[:n_select - len(selected)])
        
        return selected[:n_select]

