"""
Q-Learning based Defender AI
Learns optimal mixed strategies through reinforcement learning
"""

import numpy as np
from typing import Tuple


class DefenderAI:
    """
    Q-Learning agent for the Defender.
    Learns mixed strategy probabilities over defense levels.
    """
    
    def __init__(self, learning_rate: float = 0.1, discount_factor: float = 0.95, 
                 epsilon: float = 0.1, num_actions: int = 3):
        """
        Initialize the Defender AI with Q-learning parameters.
        
        Args:
            learning_rate: Alpha parameter for Q-learning (0 to 1)
            discount_factor: Gamma parameter for discounting future rewards
            epsilon: Exploration rate for epsilon-greedy strategy
            num_actions: Number of possible defense levels
        """
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.num_actions = num_actions
        
        # Q-table: state x action (state = hacker's last action)
        # States: 0 (initial), 1 (Brute Force), 2 (Dictionary), 3 (Phishing)
        self.q_table = np.zeros((4, num_actions))
        
        # Mixed strategy: probability of choosing each defense
        self.strategy = np.ones(num_actions) / num_actions
        
        # Track action history
        self.action_history = []
        self.reward_history = []
        self.last_action = None
        self.last_state = 0
    
    def select_action(self, hacker_action: int = None) -> int:
        """
        Select a defense action using epsilon-greedy strategy.
        
        Args:
            hacker_action: Previous hacker's action (0, 1, or 2) or None for initial action
        
        Returns:
            Defense level (0, 1, or 2)
        """
        state = 0 if hacker_action is None else hacker_action + 1
        self.last_state = state
        
        # Epsilon-greedy: explore with probability epsilon, exploit with probability 1-epsilon
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.num_actions)
        else:
            # Choose action with highest Q-value
            action = np.argmax(self.q_table[state])
        
        self.last_action = action
        return action
    
    def select_action_with_strategy(self) -> int:
        """
        Select action based on current mixed strategy (probability distribution).
        Used during exploitation phase.
        
        Returns:
            Defense level sampled from mixed strategy
        """
        return np.random.choice(self.num_actions, p=self.strategy)
    
    def update_q_table(self, reward: float, hacker_action: int = None):
        """
        Update Q-table using Q-learning rule.
        Q(s,a) = Q(s,a) + α[r + γ*max(Q(s',a')) - Q(s,a)]
        
        Args:
            reward: Reward received for the last action
            hacker_action: Next hacker's action (for next state)
        """
        if self.last_action is None:
            return
        
        next_state = 0 if hacker_action is None else hacker_action + 1
        
        # Current Q-value
        current_q = self.q_table[self.last_state, self.last_action]
        
        # Maximum Q-value for next state
        max_next_q = np.max(self.q_table[next_state])
        
        # Q-learning update
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        self.q_table[self.last_state, self.last_action] = new_q
        
        # Update mixed strategy based on Q-values (softmax)
        self._update_strategy()
        
        self.reward_history.append(reward)
    
    def _update_strategy(self):
        """
        Update mixed strategy using softmax of Q-values.
        Represents probability distribution over actions.
        """
        # Use average Q-values across all states
        avg_q_values = np.mean(self.q_table[1:], axis=0)  # Exclude initial state
        
        # Softmax function: P(a) = exp(Q(a)) / sum(exp(Q(a')))
        exp_q = np.exp(avg_q_values - np.max(avg_q_values))  # Subtract max for numerical stability
        self.strategy = exp_q / np.sum(exp_q)
    
    def get_mixed_strategy(self) -> np.ndarray:
        """Return current mixed strategy (probability distribution)."""
        return self.strategy.copy()
    
    def get_q_table(self) -> np.ndarray:
        """Return current Q-table."""
        return self.q_table.copy()
    
    def reset(self):
        """Reset the agent for a new episode."""
        self.last_action = None
        self.last_state = 0
    
    def get_average_reward(self) -> float:
        """Get average reward received so far."""
        if not self.reward_history:
            return 0.0
        return np.mean(self.reward_history)
    
    def set_epsilon(self, epsilon: float):
        """Set exploration rate (for epsilon decay over time)."""
        self.epsilon = max(0.01, epsilon)  # Keep minimum exploration
    
    def display_strategy(self):
        """Display current mixed strategy in human-readable format."""
        defense_names = ["Simple", "Complex", "Double Factor"]
        print("\n" + "="*50)
        print("DEFENDER MIXED STRATEGY")
        print("="*50)
        for i, (name, prob) in enumerate(zip(defense_names, self.strategy)):
            print(f"{name:15} : {prob:.2%}")
        print("="*50 + "\n")


if __name__ == "__main__":
    defender = DefenderAI()
    
    # Simulate some interactions
    print("Initial strategy:")
    defender.display_strategy()
    
    # Simulate learning
    for episode in range(100):
        for round_num in range(5):
            # Defender chooses action
            action = defender.select_action(hacker_action=round_num % 3)
            
            # Simulate reward
            reward = np.random.randint(20, 100)
            
            # Update Q-table
            defender.update_q_table(reward, hacker_action=(round_num + 1) % 3)
    
    print(f"After 100 episodes:")
    defender.display_strategy()
    print(f"Average reward: {defender.get_average_reward():.2f}")
