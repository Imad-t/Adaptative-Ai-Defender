"""
Utility functions for game simulation and analysis
"""

import numpy as np
from typing import Dict, Tuple, List
from collections import defaultdict


class GameStatistics:
    """Tracks and analyzes game statistics."""
    
    def __init__(self):
        """Initialize statistics tracker."""
        self.rounds = []
        self.defender_rewards = []
        self.hacker_rewards = []
        self.defender_actions = []
        self.hacker_actions = []
        self.defender_payoff_by_action = defaultdict(list)
        self.hacker_payoff_by_action = defaultdict(list)
    
    def record_round(self, defender_action: int, hacker_action: int, 
                    defender_reward: float, hacker_reward: float):
        """Record a round of the game."""
        self.defender_actions.append(defender_action)
        self.hacker_actions.append(hacker_action)
        self.defender_rewards.append(defender_reward)
        self.hacker_rewards.append(hacker_reward)
        
        self.defender_payoff_by_action[defender_action].append(defender_reward)
        self.hacker_payoff_by_action[hacker_action].append(hacker_reward)
    
    def get_action_frequencies(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get frequency of actions for both players."""
        def_counts = np.bincount(self.defender_actions, minlength=3)
        hack_counts = np.bincount(self.hacker_actions, minlength=3)
        return def_counts, hack_counts
    
    def get_average_rewards(self) -> Tuple[float, float]:
        """Get average rewards for both players."""
        def_avg = np.mean(self.defender_rewards) if self.defender_rewards else 0
        hack_avg = np.mean(self.hacker_rewards) if self.hacker_rewards else 0
        return def_avg, hack_avg
    
    def get_cumulative_rewards(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get cumulative rewards over time."""
        def_cumsum = np.cumsum(self.defender_rewards)
        hack_cumsum = np.cumsum(self.hacker_rewards)
        return def_cumsum, hack_cumsum
    
    def get_average_payoff_by_action(self) -> Tuple[Dict[int, float], Dict[int, float]]:
        """Get average payoff for each action."""
        def_payoffs = {action: np.mean(rewards) 
                       for action, rewards in self.defender_payoff_by_action.items()}
        hack_payoffs = {action: np.mean(rewards) 
                        for action, rewards in self.hacker_payoff_by_action.items()}
        return def_payoffs, hack_payoffs
    
    def get_round_count(self) -> int:
        """Get total number of rounds played."""
        return len(self.defender_actions)


def calculate_nash_equilibrium_approximation(defender_strategy: np.ndarray, 
                                            hacker_strategy: np.ndarray,
                                            defender_payoff: np.ndarray,
                                            hacker_payoff: np.ndarray) -> Tuple[float, float]:
    """
    Calculate expected payoffs given mixed strategies.
    This approximates how close we are to Nash equilibrium.
    
    Args:
        defender_strategy: Probability distribution over defender actions
        hacker_strategy: Probability distribution over hacker actions
        defender_payoff: Payoff matrix for defender
        hacker_payoff: Payoff matrix for hacker
    
    Returns:
        Tuple of (defender_expected_payoff, hacker_expected_payoff)
    """
    # Expected payoff = strategy^T * payoff_matrix * strategy
    def_expected = np.dot(defender_strategy, np.dot(defender_payoff, hacker_strategy))
    hack_expected = np.dot(hacker_strategy, np.dot(hacker_payoff.T, defender_strategy))
    
    return float(def_expected), float(hack_expected)


def calculate_best_response_payoff(opponent_strategy: np.ndarray, 
                                   payoff_matrix: np.ndarray) -> Tuple[float, int]:
    """
    Calculate the payoff of playing best response to opponent's strategy.
    
    Args:
        opponent_strategy: Opponent's mixed strategy
        payoff_matrix: Player's payoff matrix
    
    Returns:
        Tuple of (best_response_payoff, best_response_action)
    """
    # Expected payoff for each action against opponent's strategy
    expected_payoffs = np.dot(payoff_matrix, opponent_strategy)
    
    best_action = np.argmax(expected_payoffs)
    best_payoff = expected_payoffs[best_action]
    
    return float(best_payoff), int(best_action)


def measure_exploitability(player_strategy: np.ndarray,
                          opponent_payoff_matrix: np.ndarray) -> float:
    """
    Measure how much an opponent can exploit this strategy.
    Higher values mean more exploitable.
    
    Args:
        player_strategy: Mixed strategy to evaluate
        opponent_payoff_matrix: Opponent's payoff matrix
    
    Returns:
        Maximum payoff opponent can get against this strategy
    """
    expected_payoffs = np.dot(opponent_payoff_matrix, player_strategy)
    return float(np.max(expected_payoffs))


def compute_strategy_similarity(strategy1: np.ndarray, strategy2: np.ndarray) -> float:
    """
    Compute similarity between two strategies using cosine similarity.
    1.0 = identical, 0.0 = completely different
    
    Args:
        strategy1: First mixed strategy
        strategy2: Second mixed strategy
    
    Returns:
        Similarity score (0 to 1)
    """
    norm1 = np.linalg.norm(strategy1)
    norm2 = np.linalg.norm(strategy2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    similarity = np.dot(strategy1, strategy2) / (norm1 * norm2)
    return float(np.clip(similarity, 0, 1))


def print_game_statistics(stats: GameStatistics, defender_name: str = "Defender",
                         hacker_name: str = "Hacker"):
    """Print formatted game statistics."""
    def_counts, hack_counts = stats.get_action_frequencies()
    def_avg, hack_avg = stats.get_average_rewards()
    def_payoffs, hack_payoffs = stats.get_average_payoff_by_action()
    
    print("\n" + "="*70)
    print(f"GAME STATISTICS (Total rounds: {stats.get_round_count()})")
    print("="*70)
    
    print(f"\n{defender_name.upper()} STATISTICS:")
    print(f"  Average reward: {def_avg:.2f}")
    print(f"  Action frequencies: Simple={def_counts[0]}, Complex={def_counts[1]}, Double Factor={def_counts[2]}")
    print(f"  Payoff by action:")
    for action, payoff in sorted(def_payoffs.items()):
        print(f"    Action {action}: {payoff:.2f}")
    
    print(f"\n{hacker_name.upper()} STATISTICS:")
    print(f"  Average reward: {hack_avg:.2f}")
    print(f"  Action frequencies: BruteForce={hack_counts[0]}, Dictionary={hack_counts[1]}, Phishing={hack_counts[2]}")
    print(f"  Payoff by action:")
    for action, payoff in sorted(hack_payoffs.items()):
        print(f"    Action {action}: {payoff:.2f}")
    
    print("="*70 + "\n")


if __name__ == "__main__":
    # Test statistics tracking
    stats = GameStatistics()
    
    for i in range(100):
        d_action = np.random.randint(0, 3)
        h_action = np.random.randint(0, 3)
        d_reward = np.random.randint(20, 100)
        h_reward = np.random.randint(20, 100)
        
        stats.record_round(d_action, h_action, d_reward, h_reward)
    
    print_game_statistics(stats)
    
    # Test Nash equilibrium calculation
    def_strategy = np.array([0.2, 0.3, 0.5])
    hack_strategy = np.array([0.4, 0.3, 0.3])
    
    def_payoff = np.array([[40, 30, 20], [50, 60, 40], [90, 80, 70]])
    hack_payoff = np.array([[60, 70, 80], [70, 40, 60], [80, 60, 30]])
    
    def_exp, hack_exp = calculate_nash_equilibrium_approximation(
        def_strategy, hack_strategy, def_payoff, hack_payoff
    )
    print(f"Expected payoffs: Defender={def_exp:.2f}, Hacker={hack_exp:.2f}")
