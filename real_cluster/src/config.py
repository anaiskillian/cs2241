# src/config.py
"""
Configuration settings for the request routing system.
"""

HOST_IPS = ["13.59.230.137", "18.191.248.228", "18.191.169.241", "3.19.59.13"]

HOSTS = [
    f"ec2-{ip.replace(".", "-")}.us-east-2.compute.amazonaws.com" for ip in HOST_IPS
]


DB_TABLES_CONFIG = [
    # {"name": "main_tbl", "columns": ["col1", "col2", "col3", "col4"]},
    # {"name": "main_tbl_small", "columns": ["col1", "col2", "col3", "col4"]},
    {"name": "main_tbl_tiny", "columns": ["col1", "col2", "col3", "col4"]},
    {"name": "join_tbl", "columns": ["col1", "col2", "col3", "col4"]},
]

# Environment settings
ENV_CONFIG = {
    "num_servers": 4,
    "history_length": 5,
    "time_step": 0.1,
    "max_steps": 1000,
}

# Server configurations (heterogeneous cluster)
# SERVER_CONFIGS = [
#     # Server 1: Balanced
#     {"cpu_speed": 1.0, "ram_size": 16, "processing_capacity": 8},
#     # Server 2: CPU optimized
#     {"cpu_speed": 1.5, "ram_size": 8, "processing_capacity": 10},
#     # Server 3: Memory optimized
#     {"cpu_speed": 0.8, "ram_size": 32, "processing_capacity": 6},
#     # Server 4: High capacity
#     {"cpu_speed": 1.2, "ram_size": 16, "processing_capacity": 12},
# ]

# Multi-Armed Bandit agent default settings
# Optimized settings to make MAB perform better
MAB_CONFIG = {
    "epsilon_greedy": {
        "epsilon": 0.05,  # Reduced epsilon for less exploration
        "alpha": 0.2,  # Increased alpha for faster learning
    },
    "ucb": {
        "ucb_c": 1.0,  # Reduced exploration coefficient for better exploitation
        "alpha": 0.2,  # Increased alpha for faster learning
    },
    "thompson": {"alpha": 0.2},  # Increased alpha for faster learning
}

# PPO agent default settings
# PPO_CONFIG = {
#     "learning_rate": 3e-4,
#     "gamma": 0.99,
#     "gae_lambda": 0.95,
#     "clip_ratio": 0.2,
#     "value_coef": 0.5,
#     "entropy_coef": 0.01,
#     "hidden_dim": 64,
#     "max_grad_norm": 0.5
# }

# Workload generator settings
# WORKLOAD_CONFIG = {
#     "uniform": {
#         "size_range": (0.8, 1.2)
#     },
#     "bursty": {
#         "burst_factor": 3.0,
#         "burst_prob": 0.1,
#         "size_range": (0.8, 1.2)
#     },
#     "diurnal": {
#         "period": 24.0,
#         "peak_factor": 2.0,
#         "size_range": (0.8, 1.2)
#     },
#     "skewed": {
#         "skew_factor": 0.8,
#         "size_range": (0.8, 1.2)
#     }
# }
