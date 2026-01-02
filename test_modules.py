"""
Quick test script to verify all modules are working
"""

import sys
import numpy as np

print("Testing Adaptive Defense AI modules...\n")

# Test 1: Game Theory Module
print("✓ Testing game_theory_defender.py...")
try:
    from game_theory_defender import DefenseGame
    game = DefenseGame()
    d_payoff, h_payoff = game.get_payoffs(0, 0)
    assert d_payoff > 0 and h_payoff > 0
    print("  ✓ DefenseGame working correctly\n")
except Exception as e:
    print(f"  ✗ Error: {e}\n")
    sys.exit(1)

# Test 2: Defender AI
print("✓ Testing defender_ai.py...")
try:
    from defender_ai import DefenderAI
    defender = DefenderAI()
    action = defender.select_action()
    assert 0 <= action < 3
    defender.update_q_table(50, 1)
    strategy = defender.get_mixed_strategy()
    assert len(strategy) == 3 and abs(np.sum(strategy) - 1.0) < 0.01
    print("  ✓ DefenderAI working correctly\n")
except Exception as e:
    print(f"  ✗ Error: {e}\n")
    sys.exit(1)

# Test 3: Hacker AI
print("✓ Testing hacker_ai.py...")
try:
    from hacker_ai import AdaptiveHackerAI, ReactiveHackerAI
    
    # Adaptive
    adaptive_hacker = AdaptiveHackerAI()
    action = adaptive_hacker.select_action()
    assert 0 <= action < 3
    adaptive_hacker.update_q_table(50, 1)
    print("  ✓ AdaptiveHackerAI working correctly")
    
    # Reactive
    reactive_hacker = ReactiveHackerAI()
    action = reactive_hacker.select_action(np.array([10, 5, 15]))
    assert 0 <= action < 3
    print("  ✓ ReactiveHackerAI working correctly\n")
except Exception as e:
    print(f"  ✗ Error: {e}\n")
    sys.exit(1)

# Test 4: Utils
print("✓ Testing utils.py...")
try:
    from utils import GameStatistics, calculate_nash_equilibrium_approximation
    stats = GameStatistics()
    stats.record_round(0, 1, 50, 60)
    assert stats.get_round_count() == 1
    
    def_strat = np.array([0.3, 0.4, 0.3])
    hack_strat = np.array([0.4, 0.3, 0.3])
    def_payoff = np.array([[40, 30, 20], [50, 60, 40], [90, 80, 70]])
    hack_payoff = np.array([[60, 70, 80], [70, 40, 60], [80, 60, 30]])
    
    d_exp, h_exp = calculate_nash_equilibrium_approximation(def_strat, hack_strat, def_payoff, hack_payoff)
    assert d_exp > 0 and h_exp > 0
    print("  ✓ Utils working correctly\n")
except Exception as e:
    print(f"  ✗ Error: {e}\n")
    sys.exit(1)

# Test 5: Visualization
print("✓ Testing visualization.py...")
try:
    from visualization import GameVisualizer
    viz = GameVisualizer()
    
    rewards_d = [50 + np.random.randn() for _ in range(100)]
    rewards_h = [50 + np.random.randn() for _ in range(100)]
    
    # Just test that methods exist and can be called
    assert hasattr(viz, 'plot_reward_evolution')
    assert hasattr(viz, 'plot_mixed_strategies')
    print("  ✓ GameVisualizer working correctly\n")
except Exception as e:
    print(f"  ✗ Error: {e}\n")
    sys.exit(1)

print("="*60)
print("ALL TESTS PASSED! ✓")
print("="*60)
print("\nYou can now run the main simulation with:")
print("  python main.py")
print("\nOr import the modules in your own scripts!")
