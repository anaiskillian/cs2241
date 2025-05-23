# src/config.py
"""
Configuration settings for the request routing system.
"""

# Environment settings
ENV_CONFIG = {
    "num_servers": 4,
    "history_length": 5,
    "time_step": 0.1,
    "max_steps": 1000
}

# Server configurations (heterogeneous cluster)
SERVER_CONFIGS = [
    # Server 1: Balanced
    {
        "cpu_speed": 1.0,
        "ram_size": 16,
        "processing_capacity": 8
    },
    # Server 2: CPU optimized
    {
        "cpu_speed": 1.5,
        "ram_size": 8,
        "processing_capacity": 10
    },
    # Server 3: Memory optimized
    {
        "cpu_speed": 0.8,
        "ram_size": 32,
        "processing_capacity": 6
    },
    # Server 4: High capacity
    {
        "cpu_speed": 1.2,
        "ram_size": 16,
        "processing_capacity": 12
    }
]

# Multi-Armed Bandit agent default settings
MAB_CONFIG = {
    "epsilon_greedy": {
        "epsilon": 0.05,     # Reduced epsilon for less exploration
        "alpha": 0.2,        # Increased alpha for faster learning
        "throughput_weight": 0.6  # Weight throughput more than latency
    },
    "ucb": {
        "ucb_c": 0.8,        # Reduced exploration coefficient
        "alpha": 0.2,        # Increased alpha for faster learning
        "throughput_weight": 0.7  # Higher weight on throughput for UCB
    },
    "thompson": {
        "alpha": 0.2,        # Increased alpha for faster learning
        "throughput_weight": 0.6  # Weight throughput more than latency
    }
}

# PPO agent default settings
PPO_CONFIG = {
    "learning_rate": 3e-4,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_ratio": 0.2,
    "value_coef": 0.5,
    "entropy_coef": 0.01,
    "hidden_dim": 64,
    "max_grad_norm": 0.5
}

# Workload generator settings
WORKLOAD_CONFIG = {
    "uniform": {
        "size_range": (0.8, 1.2)
    },
    "bursty": {
        "burst_factor": 3.0,
        "burst_prob": 0.1,
        "size_range": (0.8, 1.2)
    },
    "diurnal": {
        "period": 24.0,
        "peak_factor": 2.0,
        "size_range": (0.8, 1.2)
    },
    "skewed": {
        "skew_factor": 0.8,
        "size_range": (0.8, 1.2)
    }
}
