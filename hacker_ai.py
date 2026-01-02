"""
Hacker AI Agent
Two variants: Reactive (exploits patterns) and Adaptive (uses Q-learning)
"""

import numpy as np
from typing import List


class ReactiveHackerAI:
    """
    Reactive Hacker AI that adapts to the Defender's strategy.
    Uses best response dynamics.
    """
    
    def __init__(self, num_actions: int = 3):
        """
        Initialize Reactive Hacker AI.
        
        Args:
            num_actions: Number of attack types
        """
        self.num_actions = num_actions
        
        # Track defender's action frequency
        self.defender_action_counts = np.zeros(num_actions)
        
        # Payoff matrix for Hacker [defender_action, hacker_action]
        self.payoff_matrix = np.array([
            [60, 70, 80],    # Brute Force vs [Simple, Complex, Double Factor]
            [70, 40, 60],    # Dictionary vs [Simple, Complex, Double Factor]
            [80, 60, 30]     # Phishing vs [Simple, Complex, Double Factor]
        ])
        
        self.reward_history = []
        self.action_history = []
        self.last_reward = 0
    
    def select_action(self, defender_action_counts: np.ndarray = None) -> int:
        """
        Select attack based on best response to defender's strategy.
        
        Args:
            defender_action_counts: Frequency of defender's actions observed
        
        Returns:
            Attack type (0, 1, or 2)
        """
        if defender_action_counts is not None:
            self.defender_action_counts = np.array(defender_action_counts)
        
        # Estimate defender's mixed strategy
        total_actions = np.sum(self.defender_action_counts)
        if total_actions == 0:
            # No information yet, choose randomly
            return np.random.randint(self.num_actions)
        
        defender_strategy = self.defender_action_counts / total_actions
        
        # Calculate expected payoff for each attack
        expected_payoffs = np.zeros(self.num_actions)
        for attack in range(self.num_actions):
            expected_payoffs[attack] = np.dot(defender_strategy, self.payoff_matrix[attack])
        
        # Best response: choose attack with highest expected payoff
        return np.argmax(expected_payoffs)
    
    def update(self, defender_action: int, reward: float):
        """
        Update hacker's knowledge with new information.
        
        Args:
            defender_action: Defender's action
            reward: Reward for hacker
        """
        self.defender_action_counts[defender_action] += 1
        self.reward_history.append(reward)
        self.last_reward = reward
    
    def get_average_reward(self) -> float:
        """Get average reward received."""
        if not self.reward_history:
            return 0.0
        return np.mean(self.reward_history)
    
    def reset(self):
        """Reset for a new episode."""
        self.defender_action_counts = np.zeros(self.num_actions)


class AdaptiveHackerAI:
    """
    Adaptive Hacker using Q-learning.
    Can learn complex patterns in defender's behavior.
    """
    
    def __init__(self, learning_rate: float = 0.1, discount_factor: float = 0.95,
                 epsilon: float = 0.1, num_actions: int = 3):
        """
        Initialize Adaptive Hacker AI.
        
        Args:
            learning_rate: Q-learning parameter
            discount_factor: Discount factor for future rewards
            epsilon: Exploration rate
            num_actions: Number of attack types
        """
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.num_actions = num_actions
        
        # Q-table: state x action (state = defender's last action)
        # States: 0 (initial), 1 (Simple), 2 (Complex), 3 (Double Factor)
        self.q_table = np.zeros((4, num_actions))
        
        # Mixed strategy
        self.strategy = np.ones(num_actions) / num_actions
        
        self.last_action = None
        self.last_state = 0
        self.reward_history = []
    
    def select_action(self, defender_action: int = None) -> int:
        """
        Select attack using epsilon-greedy strategy.
        
        Args:
            defender_action: Defender's last action (0, 1, or 2)
        
        Returns:
            Attack type
        """
        state = 0 if defender_action is None else defender_action + 1
        self.last_state = state
        
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.num_actions)
        else:
            action = np.argmax(self.q_table[state])
        
        self.last_action = action
        return action
    
    def update_q_table(self, reward: float, defender_action: int = None):
        """
        Update Q-table using Q-learning.
        
        Args:
            reward: Reward received
            defender_action: Defender's next action
        """
        if self.last_action is None:
            return
        
        next_state = 0 if defender_action is None else defender_action + 1
        
        current_q = self.q_table[self.last_state, self.last_action]
        max_next_q = np.max(self.q_table[next_state])
        
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        self.q_table[self.last_state, self.last_action] = new_q
        
        self._update_strategy()
        self.reward_history.append(reward)
    
    def _update_strategy(self):
        """Update mixed strategy using softmax."""
        avg_q_values = np.mean(self.q_table[1:], axis=0)
        
        exp_q = np.exp(avg_q_values - np.max(avg_q_values))
        self.strategy = exp_q / np.sum(exp_q)
    
    def get_mixed_strategy(self) -> np.ndarray:
        """Return mixed strategy."""
        return self.strategy.copy()
    
    def get_average_reward(self) -> float:
        """Get average reward."""
        if not self.reward_history:
            return 0.0
        return np.mean(self.reward_history)
    
    def set_epsilon(self, epsilon: float):
        """Set exploration rate."""
        self.epsilon = max(0.01, epsilon)
    
    def reset(self):
        """Reset for new episode."""
        self.last_action = None
        self.last_state = 0
    
    def display_strategy(self):
        """Display mixed strategy."""
        attack_names = ["Brute Force", "Dictionary", "Phishing"]
        print("\n" + "="*50)
        print("HACKER MIXED STRATEGY")
        print("="*50)
        for i, (name, prob) in enumerate(zip(attack_names, self.strategy)):
            print(f"{name:15} : {prob:.2%}")
        print("="*50 + "\n")


if __name__ == "__main__":
    # Test reactive hacker
    print("REACTIVE HACKER TEST")
    reactive = ReactiveHackerAI()
    
    defender_actions = np.array([10, 5, 15])  # Defender's action frequencies
    action = reactive.select_action(defender_actions)
    print(f"Hacker chooses attack type: {action}")
    
    # Test adaptive hacker
    print("\nADAPTIVE HACKER TEST")
    adaptive = AdaptiveHackerAI()
    
    for i in range(50):
        action = adaptive.select_action(defender_action=i % 3)
        reward = np.random.randint(30, 90)
        adaptive.update_q_table(reward, defender_action=(i + 1) % 3)
    
    adaptive.display_strategy()
    print(f"Average reward: {adaptive.get_average_reward():.2f}")
