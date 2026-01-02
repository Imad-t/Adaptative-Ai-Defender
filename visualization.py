"""
Visualization module for game results and strategy evolution
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import List, Tuple
import warnings
warnings.filterwarnings('ignore')


class GameVisualizer:
    """Visualizes game statistics and strategy evolution."""
    
    def __init__(self, figsize: Tuple[int, int] = (14, 10)):
        """Initialize visualizer."""
        self.figsize = figsize
        self.defense_names = ["Simple", "Complex", "Double Factor"]
        self.attack_names = ["Brute Force", "Dictionary", "Phishing"]
    
    def plot_reward_evolution(self, defender_rewards: List[float], 
                             hacker_rewards: List[float],
                             window_size: int = 50):
        """
        Plot reward evolution over time with moving average.
        
        Args:
            defender_rewards: List of defender rewards
            hacker_rewards: List of hacker rewards
            window_size: Moving average window size
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        rounds = range(len(defender_rewards))
        
        # Plot raw rewards
        ax.plot(rounds, defender_rewards, alpha=0.3, label='Defender (raw)', color='blue')
        ax.plot(rounds, hacker_rewards, alpha=0.3, label='Hacker (raw)', color='red')
        
        # Plot moving average
        if len(defender_rewards) >= window_size:
            def_ma = np.convolve(defender_rewards, np.ones(window_size)/window_size, mode='valid')
            hack_ma = np.convolve(hacker_rewards, np.ones(window_size)/window_size, mode='valid')
            
            ma_rounds = range(window_size-1, len(defender_rewards))
            ax.plot(ma_rounds, def_ma, linewidth=2, label=f'Defender (MA-{window_size})', color='darkblue')
            ax.plot(ma_rounds, hack_ma, linewidth=2, label=f'Hacker (MA-{window_size})', color='darkred')
        
        ax.set_xlabel('Round', fontsize=12)
        ax.set_ylabel('Reward', fontsize=12)
        ax.set_title('Reward Evolution Over Time', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def plot_cumulative_rewards(self, defender_rewards: List[float],
                               hacker_rewards: List[float]):
        """
        Plot cumulative rewards over time.
        
        Args:
            defender_rewards: List of defender rewards
            hacker_rewards: List of hacker rewards
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        def_cumsum = np.cumsum(defender_rewards)
        hack_cumsum = np.cumsum(hacker_rewards)
        
        rounds = range(len(defender_rewards))
        
        ax.plot(rounds, def_cumsum, linewidth=2, label='Defender', color='blue')
        ax.plot(rounds, hack_cumsum, linewidth=2, label='Hacker', color='red')
        
        ax.set_xlabel('Round', fontsize=12)
        ax.set_ylabel('Cumulative Reward', fontsize=12)
        ax.set_title('Cumulative Rewards Over Time', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def plot_action_frequencies(self, defender_counts: np.ndarray,
                               hacker_counts: np.ndarray):
        """
        Plot action frequency comparison.
        
        Args:
            defender_counts: Count of each defense action
            hacker_counts: Count of each attack type
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        x_pos = np.arange(3)
        width = 0.35
        
        # Defender actions
        ax1.bar(x_pos, defender_counts, width, color='steelblue', alpha=0.8)
        ax1.set_ylabel('Frequency', fontsize=11)
        ax1.set_title('Defender Action Frequencies', fontsize=12, fontweight='bold')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(self.defense_names)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add count labels on bars
        for i, v in enumerate(defender_counts):
            ax1.text(i, v + 0.5, str(int(v)), ha='center', fontweight='bold')
        
        # Hacker actions
        ax2.bar(x_pos, hacker_counts, width, color='coral', alpha=0.8)
        ax2.set_ylabel('Frequency', fontsize=11)
        ax2.set_title('Hacker Action Frequencies', fontsize=12, fontweight='bold')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(self.attack_names)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add count labels on bars
        for i, v in enumerate(hacker_counts):
            ax2.text(i, v + 0.5, str(int(v)), ha='center', fontweight='bold')
        
        plt.tight_layout()
        return fig
    
    def plot_mixed_strategies(self, defender_strategy: np.ndarray,
                             hacker_strategy: np.ndarray):
        """
        Plot mixed strategies as bar charts.
        
        Args:
            defender_strategy: Probability distribution for defenses
            hacker_strategy: Probability distribution for attacks
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        x_pos = np.arange(3)
        colors_def = ['#1f77b4', '#ff7f0e', '#2ca02c']
        colors_hack = ['#d62728', '#9467bd', '#8c564b']
        
        # Defender mixed strategy
        bars1 = ax1.bar(x_pos, defender_strategy, color=colors_def, alpha=0.8)
        ax1.set_ylabel('Probability', fontsize=11)
        ax1.set_title('Defender Mixed Strategy', fontsize=12, fontweight='bold')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(self.defense_names)
        ax1.set_ylim([0, 1])
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add percentage labels
        for i, (bar, prob) in enumerate(zip(bars1, defender_strategy)):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{prob:.1%}', ha='center', fontweight='bold', fontsize=10)
        
        # Hacker mixed strategy
        bars2 = ax2.bar(x_pos, hacker_strategy, color=colors_hack, alpha=0.8)
        ax2.set_ylabel('Probability', fontsize=11)
        ax2.set_title('Hacker Mixed Strategy', fontsize=12, fontweight='bold')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(self.attack_names)
        ax2.set_ylim([0, 1])
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add percentage labels
        for i, (bar, prob) in enumerate(zip(bars2, hacker_strategy)):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{prob:.1%}', ha='center', fontweight='bold', fontsize=10)
        
        plt.tight_layout()
        return fig
    
    def plot_payoff_matrix(self, payoff_matrix: np.ndarray, 
                          title: str, is_defender: bool = True):
        """
        Plot payoff matrix as heatmap.
        
        Args:
            payoff_matrix: 3x3 payoff matrix
            title: Title for the plot
            is_defender: Whether this is defender's payoff
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        
        im = ax.imshow(payoff_matrix, cmap='RdYlGn', aspect='auto')
        
        # Labels
        row_labels = self.defense_names if is_defender else self.attack_names
        col_labels = self.attack_names if is_defender else self.defense_names
        
        ax.set_xticks(np.arange(3))
        ax.set_yticks(np.arange(3))
        ax.set_xticklabels(col_labels)
        ax.set_yticklabels(row_labels)
        
        # Rotate labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add text annotations
        for i in range(3):
            for j in range(3):
                text = ax.text(j, i, int(payoff_matrix[i, j]),
                             ha="center", va="center", color="black", fontweight='bold')
        
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Payoff', rotation=270, labelpad=20)
        
        plt.tight_layout()
        return fig
    
    def plot_strategy_evolution(self, strategy_history: List[np.ndarray],
                               is_defender: bool = True):
        """
        Plot evolution of mixed strategy over episodes.
        
        Args:
            strategy_history: List of mixed strategies over time
            is_defender: Whether this is defender's strategy
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        strategy_history = np.array(strategy_history)
        episodes = range(len(strategy_history))
        
        labels = self.defense_names if is_defender else self.attack_names
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c'] if is_defender else ['#d62728', '#9467bd', '#8c564b']
        
        for i in range(3):
            ax.plot(episodes, strategy_history[:, i], linewidth=2.5, 
                   label=labels[i], color=colors[i], marker='o', markersize=3, alpha=0.7)
        
        player_name = "Defender" if is_defender else "Hacker"
        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel('Strategy Probability', fontsize=12)
        ax.set_title(f'{player_name} Mixed Strategy Evolution', fontsize=14, fontweight='bold')
        ax.set_ylim([0, 1])
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_convergence_analysis(self, defender_payoffs: List[float],
                                 hacker_payoffs: List[float],
                                 window_size: int = 100):
        """
        Plot convergence to equilibrium analysis.
        
        Args:
            defender_payoffs: Expected payoffs over time
            hacker_payoffs: Expected payoffs over time
            window_size: Window for averaging
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        rounds = range(len(defender_payoffs))
        
        if len(defender_payoffs) >= window_size:
            def_ma = np.convolve(defender_payoffs, np.ones(window_size)/window_size, mode='valid')
            hack_ma = np.convolve(hacker_payoffs, np.ones(window_size)/window_size, mode='valid')
            
            ma_rounds = range(window_size-1, len(defender_payoffs))
            ax.plot(ma_rounds, def_ma, linewidth=2.5, label='Defender', color='blue')
            ax.plot(ma_rounds, hack_ma, linewidth=2.5, label='Hacker', color='red')
        else:
            ax.plot(rounds, defender_payoffs, linewidth=2.5, label='Defender', color='blue')
            ax.plot(rounds, hacker_payoffs, linewidth=2.5, label='Hacker', color='red')
        
        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel('Expected Payoff', fontsize=12)
        ax.set_title('Convergence to Equilibrium', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig


if __name__ == "__main__":
    # Example usage
    viz = GameVisualizer()
    
    # Generate synthetic data
    np.random.seed(42)
    n_rounds = 500
    defender_rewards = 50 + 20*np.sin(np.arange(n_rounds)/50) + np.random.normal(0, 10, n_rounds)
    hacker_rewards = 50 + 20*np.cos(np.arange(n_rounds)/50) + np.random.normal(0, 10, n_rounds)
    
    # Create plots
    fig1 = viz.plot_reward_evolution(defender_rewards, hacker_rewards)
    fig1.savefig('reward_evolution.png', dpi=150, bbox_inches='tight')
    print("Saved: reward_evolution.png")
    
    fig2 = viz.plot_cumulative_rewards(defender_rewards, hacker_rewards)
    fig2.savefig('cumulative_rewards.png', dpi=150, bbox_inches='tight')
    print("Saved: cumulative_rewards.png")
    
    defender_counts = np.array([150, 200, 150])
    hacker_counts = np.array([180, 160, 160])
    
    fig3 = viz.plot_action_frequencies(defender_counts, hacker_counts)
    fig3.savefig('action_frequencies.png', dpi=150, bbox_inches='tight')
    print("Saved: action_frequencies.png")
    
    defender_strategy = np.array([0.25, 0.35, 0.40])
    hacker_strategy = np.array([0.40, 0.30, 0.30])
    
    fig4 = viz.plot_mixed_strategies(defender_strategy, hacker_strategy)
    fig4.savefig('mixed_strategies.png', dpi=150, bbox_inches='tight')
    print("Saved: mixed_strategies.png")
    
    print("\nAll example visualizations created successfully!")
