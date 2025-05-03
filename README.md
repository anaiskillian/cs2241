# Multi-Armed Bandit for Adaptive Request Routing

This repository implements a reinforcement learning approach using multi-armed bandits for adaptively routing requests to heterogeneous servers. It compares the performance of different algorithms, including traditional load balancing methods like round-robin, random, and least-loaded allocation.

## Project Overview

In modern distributed systems with heterogeneous hardware and workloads, efficient request routing is critical for optimizing performance. This project explores whether reinforcement learning techniques, specifically multi-armed bandits, can outperform traditional load balancing methods by dynamically adapting to changing server conditions.

### Key Features

- Simulated cluster of heterogeneous servers with different CPU speeds and capacities
- Various request types with different resource requirements
- Multiple routing strategies implementation:
  - Traditional methods (Round Robin, Random, Least Loaded)
  - Multi-Armed Bandits (Epsilon-Greedy, UCB, Thompson Sampling)
  - Proximal Policy Optimization (PPO) for comparison
- Comprehensive performance metrics and visualization
- Support for different workload patterns (uniform, bursty, diurnal, skewed)

## Getting Started

### Prerequisites

- Python 3.7+
- PyTorch
- NumPy
- Matplotlib
- Gym
- Pandas

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/bandit-request-routing.git
   cd bandit-request-routing
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

### Usage

#### Running Baseline Comparison

To compare all routing strategies with default settings:

```
python main.py --experiment baseline --episodes 5 --steps 1000
```

#### Testing Different Workloads

To evaluate performance on different workload patterns:

```
python main.py --experiment workload --workload_type all
```

Or test a specific workload type:

```
python main.py --experiment workload --workload_type bursty
```

#### Hyperparameter Tuning

To tune hyperparameters for the MAB algorithms:

```
python main.py --experiment hyperparameter --mab_strategy ucb
```

#### PPO Comparison

To train and evaluate a PPO agent:

```
python main.py --experiment ppo --episodes 5 --steps 1000
```

## Experiment Results

Results are saved in the `results/` directory, organized by experiment type:

- `results/baseline/`: Baseline comparison of all routing methods
- `results/workload/`: Analysis of performance under different workload patterns
- `results/hyperparameter/`: Hyperparameter tuning results
- `results/ppo/`: PPO training and evaluation results

Each directory contains JSON files with detailed metrics and PNG visualizations.

## Project Structure

```
bandit-request-routing/
├── src/
│   ├── environment/          # Cluster and server environment
│   ├── agents/               # Routing agent implementations
│   ├── utils/                # Utilities for metrics and visualization
│   └── config.py             # Configuration settings
├── experiments/              # Experiment scripts
├── results/                  # Results and visualizations
├── main.py                   # Main runner script
├── requirements.txt          # Package dependencies
└── README.md                 # This file
```

python main.py --experiment baseline --episodes 5 --steps 1000

## Methodology

The project follows this methodology:

1. **Environment Simulation**: Models a cluster of servers with heterogeneous capabilities
2. **Request Generation**: Creates different types of requests with varying resource needs
3. **State Representation**: Tracks server utilization, request types, and routing history
4. **Decision Making**: Maps states to routing decisions via different algorithms
5. **Metrics Collection**: Measures latency, throughput, success rate, and fairness
6. **Performance Analysis**: Compares routing strategies across different scenarios

## Multi-Armed Bandit Approach

The multi-armed bandit approach treats each server as an "arm" with unknown reward distributions. The algorithms learn to select servers that minimize latency over time:

- **Epsilon-Greedy**: Balances exploration and exploitation with random probability
- **Upper Confidence Bound (UCB)**: Selects actions based on uncertainty and expected reward
- **Thompson Sampling**: Uses Bayesian probabilistic modeling of reward distributions

## Reference Papers

This work is inspired by research in reinforcement learning for resource allocation:

- Proximal Policy Optimization (PPO) by Schulman et al. (2017)
- A Deep Reinforcement Learning Approach for Traffic Signal Control Optimization by Li et al. (2021)
- A Hybrid Reinforcement Learning Approach to Autonomic Resource Allocation by Tesauro et al. (2006)

## License

This project is licensed under the MIT License - see the LICENSE file for details.
