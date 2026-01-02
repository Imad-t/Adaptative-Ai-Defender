"""
Main simulation script for Adaptive Defense AI Game
Runs the game between Defender and Hacker AIs
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
import sys

from game_theory_defender import DefenseGame
from defender_ai import DefenderAI
from hacker_ai import AdaptiveHackerAI, ReactiveHackerAI
from utils import GameStatistics, calculate_nash_equilibrium_approximation
from utils import measure_exploitability, compute_strategy_similarity
from utils import print_game_statistics
from visualization import GameVisualizer


class GameSimulation:
    """Main simulation orchestrator."""
    
    def __init__(self, hacker_type: str = "adaptive", num_episodes: int = 500,
                 rounds_per_episode: int = 100):
        """
        Initialize game simulation.
        
        Args:
            hacker_type: "adaptive" or "reactive"
            num_episodes: Number of episodes to run
            rounds_per_episode: Rounds per episode
        """
        self.game = DefenseGame()
        self.defender = DefenderAI(learning_rate=0.1, discount_factor=0.95, epsilon=0.15)
        
        if hacker_type.lower() == "adaptive":
            self.hacker = AdaptiveHackerAI(learning_rate=0.1, discount_factor=0.95, epsilon=0.15)
        else:
            self.hacker = ReactiveHackerAI()
        
        self.num_episodes = num_episodes
        self.rounds_per_episode = rounds_per_episode
        self.hacker_type = hacker_type
        
        self.stats = GameStatistics()
        
        # History for analysis
        self.defender_strategy_history = []
        self.hacker_strategy_history = []
        self.defender_payoff_history = []
        self.hacker_payoff_history = []
        self.exploitability_history = []
    
    def run_episode(self, episode_num: int) -> Tuple[float, float]:
        """
        Run one episode of the game.
        
        Args:
            episode_num: Episode number (for epsilon decay)
        
        Returns:
            Tuple of (defender_episode_reward, hacker_episode_reward)
        """
        self.defender.reset()
        if hasattr(self.hacker, 'reset'):
            self.hacker.reset()
        
        episode_def_reward = 0
        episode_hack_reward = 0
        
        defender_last_action = None
        hacker_last_action = None
        
        for round_num in range(self.rounds_per_episode):
            # Defender chooses action
            defender_action = self.defender.select_action(defender_last_action)
            
            # Hacker chooses action
            if isinstance(self.hacker, AdaptiveHackerAI):
                hacker_action = self.hacker.select_action(defender_action)
            else:
                # Reactive hacker uses defender's action frequencies
                def_counts, _ = self.stats.get_action_frequencies()
                hacker_action = self.hacker.select_action(def_counts)
            
            # Get payoffs
            defender_payoff, hacker_payoff = self.game.get_payoffs(defender_action, hacker_action)
            
            # Update AI agents
            self.defender.update_q_table(defender_payoff, hacker_action)
            
            if isinstance(self.hacker, AdaptiveHackerAI):
                self.hacker.update_q_table(hacker_payoff, defender_action)
            else:
                self.hacker.update(defender_action, hacker_payoff)
            
            # Record statistics
            self.stats.record_round(defender_action, hacker_action, defender_payoff, hacker_payoff)
            
            episode_def_reward += defender_payoff
            episode_hack_reward += hacker_payoff
            
            defender_last_action = defender_action
            hacker_last_action = hacker_action
        
        # Epsilon decay
        decay_rate = 0.995
        new_epsilon = max(0.01, self.defender.epsilon * decay_rate)
        self.defender.set_epsilon(new_epsilon)
        
        if isinstance(self.hacker, AdaptiveHackerAI):
            new_epsilon = max(0.01, self.hacker.epsilon * decay_rate)
            self.hacker.set_epsilon(new_epsilon)
        
        return episode_def_reward, episode_hack_reward
    
    def run_simulation(self):
        """Run complete simulation."""
        print("\n" + "="*70)
        print(f"GAME SIMULATION: Defender vs {self.hacker_type.capitalize()} Hacker")
        print("="*70)
        print(f"Episodes: {self.num_episodes}, Rounds per episode: {self.rounds_per_episode}")
        print("="*70 + "\n")
        
        for episode in range(self.num_episodes):
            def_reward, hack_reward = self.run_episode(episode)
            
            # Store strategy histories
            if isinstance(self.hacker, AdaptiveHackerAI):
                self.defender_strategy_history.append(self.defender.get_mixed_strategy())
                self.hacker_strategy_history.append(self.hacker.get_mixed_strategy())
            
            # Calculate expected payoffs
            if isinstance(self.hacker, AdaptiveHackerAI):
                def_strat = self.defender.get_mixed_strategy()
                hack_strat = self.hacker.get_mixed_strategy()
                
                def_expected, hack_expected = calculate_nash_equilibrium_approximation(
                    def_strat, hack_strat,
                    self.game.get_defender_payoff_matrix(),
                    self.game.get_hacker_payoff_matrix()
                )
            else:
                def_counts, hack_counts = self.stats.get_action_frequencies()
                total = np.sum(def_counts)
                if total > 0:
                    def_strat = def_counts / total
                else:
                    def_strat = np.ones(3) / 3
                
                hack_counts_norm = hack_counts / (np.sum(hack_counts) + 1e-6)
                
                def_expected, hack_expected = calculate_nash_equilibrium_approximation(
                    def_strat, hack_counts_norm,
                    self.game.get_defender_payoff_matrix(),
                    self.game.get_hacker_payoff_matrix()
                )
            
            self.defender_payoff_history.append(def_expected)
            self.hacker_payoff_history.append(hack_expected)
            
            # Calculate exploitability
            if isinstance(self.hacker, AdaptiveHackerAI):
                def_strat = self.defender.get_mixed_strategy()
                hack_payoff_matrix = self.game.get_hacker_payoff_matrix()
                exploit = measure_exploitability(def_strat, hack_payoff_matrix)
                self.exploitability_history.append(exploit)
            
            # Progress update
            if (episode + 1) % max(1, self.num_episodes // 10) == 0:
                print(f"Episode {episode + 1}/{self.num_episodes} completed")
                if isinstance(self.hacker, AdaptiveHackerAI):
                    print(f"  Defender avg reward: {self.defender.get_average_reward():.2f}")
                    print(f"  Hacker avg reward: {self.hacker.get_average_reward():.2f}")
        
        print("\nSimulation completed!\n")
    
    def print_final_results(self):
        """Print final game results."""
        print_game_statistics(self.stats, "Defender", self.hacker_type.capitalize() + " Hacker")
        
        print("\n" + "="*70)
        print("FINAL STRATEGY ANALYSIS")
        print("="*70)
        
        if isinstance(self.hacker, AdaptiveHackerAI):
            print("\nDEFENDER FINAL MIXED STRATEGY:")
            self.defender.display_strategy()
            
            print("HACKER FINAL MIXED STRATEGY:")
            self.hacker.display_strategy()
            
            # Nash equilibrium analysis
            def_strat = self.defender.get_mixed_strategy()
            hack_strat = self.hacker.get_mixed_strategy()
            
            def_expected, hack_expected = calculate_nash_equilibrium_approximation(
                def_strat, hack_strat,
                self.game.get_defender_payoff_matrix(),
                self.game.get_hacker_payoff_matrix()
            )
            
            print(f"\nExpected payoffs at learned strategies:")
            print(f"  Defender: {def_expected:.2f}")
            print(f"  Hacker: {hack_expected:.2f}")
            
            # Exploitability
            exploit = measure_exploitability(def_strat, self.game.get_hacker_payoff_matrix())
            print(f"\nDefender's strategy exploitability: {exploit:.2f}")
            print(f"  (How much hacker can gain with best response)")
        
        print("="*70 + "\n")
    
    def visualize_results(self, save_path: str = "."):
        """Create and save visualizations."""
        print(f"Creating visualizations...\n")
        
        viz = GameVisualizer()
        
        # 1. Reward evolution
        fig1 = viz.plot_reward_evolution(
            self.stats.defender_rewards,
            self.stats.hacker_rewards,
            window_size=50
        )
        fig1.savefig(f"{save_path}/01_reward_evolution.png", dpi=150, bbox_inches='tight')
        print("✓ Saved: 01_reward_evolution.png")
        plt.close(fig1)
        
        # 2. Cumulative rewards
        fig2 = viz.plot_cumulative_rewards(
            self.stats.defender_rewards,
            self.stats.hacker_rewards
        )
        fig2.savefig(f"{save_path}/02_cumulative_rewards.png", dpi=150, bbox_inches='tight')
        print("✓ Saved: 02_cumulative_rewards.png")
        plt.close(fig2)
        
        # 3. Action frequencies
        def_counts, hack_counts = self.stats.get_action_frequencies()
        fig3 = viz.plot_action_frequencies(def_counts, hack_counts)
        fig3.savefig(f"{save_path}/03_action_frequencies.png", dpi=150, bbox_inches='tight')
        print("✓ Saved: 03_action_frequencies.png")
        plt.close(fig3)
        
        # 4. Payoff matrices
        fig4 = viz.plot_payoff_matrix(
            self.game.get_defender_payoff_matrix(),
            "Defender Payoff Matrix",
            is_defender=True
        )
        fig4.savefig(f"{save_path}/04_payoff_matrix_defender.png", dpi=150, bbox_inches='tight')
        print("✓ Saved: 04_payoff_matrix_defender.png")
        plt.close(fig4)
        
        fig5 = viz.plot_payoff_matrix(
            self.game.get_hacker_payoff_matrix(),
            "Hacker Payoff Matrix",
            is_defender=False
        )
        fig5.savefig(f"{save_path}/05_payoff_matrix_hacker.png", dpi=150, bbox_inches='tight')
        print("✓ Saved: 05_payoff_matrix_hacker.png")
        plt.close(fig5)
        
        # 5. Strategy evolution (if available)
        if self.defender_strategy_history:
            fig6 = viz.plot_strategy_evolution(self.defender_strategy_history, is_defender=True)
            fig6.savefig(f"{save_path}/06_strategy_evolution_defender.png", dpi=150, bbox_inches='tight')
            print("✓ Saved: 06_strategy_evolution_defender.png")
            plt.close(fig6)
            
            fig7 = viz.plot_strategy_evolution(self.hacker_strategy_history, is_defender=False)
            fig7.savefig(f"{save_path}/07_strategy_evolution_hacker.png", dpi=150, bbox_inches='tight')
            print("✓ Saved: 07_strategy_evolution_hacker.png")
            plt.close(fig7)
        
        # 6. Convergence analysis
        if self.defender_payoff_history:
            fig8 = viz.plot_convergence_analysis(
                self.defender_payoff_history,
                self.hacker_payoff_history,
                window_size=50
            )
            fig8.savefig(f"{save_path}/08_convergence_analysis.png", dpi=150, bbox_inches='tight')
            print("✓ Saved: 08_convergence_analysis.png")
            plt.close(fig8)
        
        # 7. Final mixed strategies
        if isinstance(self.hacker, AdaptiveHackerAI):
            fig9 = viz.plot_mixed_strategies(
                self.defender.get_mixed_strategy(),
                self.hacker.get_mixed_strategy()
            )
            fig9.savefig(f"{save_path}/09_final_mixed_strategies.png", dpi=150, bbox_inches='tight')
            print("✓ Saved: 09_final_mixed_strategies.png")
            plt.close(fig9)
        
        print("\nAll visualizations created successfully!\n")


def main():
    """Main execution function."""
    # Configuration
    NUM_EPISODES = 500
    ROUNDS_PER_EPISODE = 100
    HACKER_TYPE = "adaptive"  # or "reactive"
    
    # Run simulation
    simulation = GameSimulation(
        hacker_type=HACKER_TYPE,
        num_episodes=NUM_EPISODES,
        rounds_per_episode=ROUNDS_PER_EPISODE
    )
    
    # Run the game
    simulation.run_simulation()
    
    # Print results
    simulation.print_final_results()
    
    # Create visualizations
    simulation.visualize_results(".")
    
    print("Simulation and analysis complete!")


if __name__ == "__main__":
    main()
