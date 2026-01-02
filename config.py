"""
Configuration settings for the game simulation
"""

# Simulation Parameters
NUM_EPISODES = 500
ROUNDS_PER_EPISODE = 100
HACKER_TYPE = "adaptive"  # Options: "adaptive" or "reactive"

# Learning Parameters
DEFENDER_LEARNING_RATE = 0.1
DEFENDER_DISCOUNT_FACTOR = 0.95
DEFENDER_INITIAL_EPSILON = 0.15

HACKER_LEARNING_RATE = 0.1
HACKER_DISCOUNT_FACTOR = 0.95
HACKER_INITIAL_EPSILON = 0.15

# Epsilon Decay
EPSILON_DECAY_RATE = 0.995
EPSILON_MINIMUM = 0.01

# Visualization Parameters
PLOT_MOVING_AVERAGE_WINDOW = 50
VISUALIZATION_DPI = 150
SAVE_PLOTS = True
PLOT_FORMAT = "png"

# Game Parameters
NUM_DEFENSE_LEVELS = 3
NUM_ATTACK_TYPES = 3

# Display Parameters
PRINT_INTERVAL = 50  # Print progress every N episodes
VERBOSE = True

# Random Seed (for reproducibility)
RANDOM_SEED = 42
