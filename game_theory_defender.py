"""
Game Theory Defender: Payoff matrix and game mechanics
Defines the strategic interaction between Defender and Hacker
"""

import numpy as np
from typing import Tuple, Dict


class DefenseGame:
    """
    Represents the game between Defender and Hacker.
    Manages payoff matrices and game rules.
    """
    
    # Defense levels (Defender's actions)
    DEFENSES = {
        0: "Simple",
        1: "Complex",
        2: "Double Factor"
    }
    
    # Attack types (Hacker's actions)
    ATTACKS = {
        0: "Brute Force",
        1: "Dictionary",
        2: "Phishing"
    }
    
    def __init__(self):
        """Initialize the game with payoff matrices."""
        # Payoff matrix for Defender [defender_action, hacker_action]
        # Higher values = better for Defender
        self.defender_payoff = np.array([
            [40, 30, 20],    # Simple defense vs [BF, Dict, Phishing]
            [50, 60, 40],    # Complex defense vs [BF, Dict, Phishing]
            [90, 80, 70]     # Double Factor vs [BF, Dict, Phishing]
        ])
        
        # Payoff matrix for Hacker [defender_action, hacker_action]
        # Higher values = better for Hacker
        self.hacker_payoff = np.array([
            [60, 70, 80],    # Brute Force vs [Simple, Complex, Double Factor]
            [70, 40, 60],    # Dictionary vs [Simple, Complex, Double Factor]
            [80, 60, 30]     # Phishing vs [Simple, Complex, Double Factor]
        ])
    
    def get_payoffs(self, defender_action: int, hacker_action: int) -> Tuple[float, float]:
        """
        Get payoffs for both players given their actions.
        
        Args:
            defender_action: Defense level (0, 1, or 2)
            hacker_action: Attack type (0, 1, or 2)
        
        Returns:
            Tuple of (defender_payoff, hacker_payoff)
        """
        if not (0 <= defender_action < 3 and 0 <= hacker_action < 3):
            raise ValueError("Actions must be between 0 and 2")
        
        defender_reward = self.defender_payoff[defender_action, hacker_action]
        hacker_reward = self.hacker_payoff[defender_action, hacker_action]
        
        return float(defender_reward), float(hacker_reward)
    
    def get_defender_payoff_matrix(self) -> np.ndarray:
        """Return the complete payoff matrix for the Defender."""
        return self.defender_payoff.copy()
    
    def get_hacker_payoff_matrix(self) -> np.ndarray:
        """Return the complete payoff matrix for the Hacker."""
        return self.hacker_payoff.copy()
    
    def get_defense_name(self, action: int) -> str:
        """Get the name of a defense level."""
        return self.DEFENSES.get(action, "Unknown")
    
    def get_attack_name(self, action: int) -> str:
        """Get the name of an attack type."""
        return self.ATTACKS.get(action, "Unknown")
    
    def display_payoff_matrices(self):
        """Display both payoff matrices in human-readable format."""
        print("\n" + "="*60)
        print("DEFENDER PAYOFF MATRIX")
        print("="*60)
        print("         Brute Force  Dictionary  Phishing")
        for i in range(3):
            print(f"{self.get_defense_name(i):15} {self.defender_payoff[i, 0]:>10} {self.defender_payoff[i, 1]:>10} {self.defender_payoff[i, 2]:>10}")
        
        print("\n" + "="*60)
        print("HACKER PAYOFF MATRIX")
        print("="*60)
        print("         Brute Force  Dictionary  Phishing")
        for i in range(3):
            print(f"{self.get_defense_name(i):15} {self.hacker_payoff[i, 0]:>10} {self.hacker_payoff[i, 1]:>10} {self.hacker_payoff[i, 2]:>10}")
        print("="*60 + "\n")


if __name__ == "__main__":
    game = DefenseGame()
    game.display_payoff_matrices()
    
    # Test payoff retrieval
    print("Test: Defender chooses Simple (0), Hacker chooses Brute Force (0)")
    d_payoff, h_payoff = game.get_payoffs(0, 0)
    print(f"Defender gets: {d_payoff}, Hacker gets: {h_payoff}\n")
