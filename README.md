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

## What the RL Agent is Deciding:
The RL agent (particularly the MAB and PPO implementations) is making request routing decisions - specifically, which server to send each incoming request to.
In the code, this is implemented in the select_action method of each agent, which returns a server index (an integer from 0 to num_servers-1). This decision is then used in the step method of the ServerCluster environment to assign the current pending request to the selected server.
State Space the RL Agent is Exploring
The state space (observation) provided to the agent includes:
1. Server Utilization: For each server, its CPU utilization quantized to a number between 1-10, exactly as Utku suggested.
  server_utils = np.array([server.quantized_cpu_util() for server in self.servers])
2. Latency History: The latency values of recently completed requests (stored in a fixed-length queue).
  'latency_history': np.array(self.latency_history)
3. Decision History: The recent server selections made by the agent (also stored in a fixed-length queue).
  'decision_history': np.array(self.decision_history)
4. Request Type: A one-hot encoded representation of the current request type.
  request_type_onehot = np.zeros(len(RequestType))
  if self.pending_requests:
    req_type_idx = self.pending_requests[0].request_type.value - 1
	  request_type_onehot[req_type_idx] = 1

This matches what Utku suggested - the agent receives information about current server loads and the type of request being routed, and must learn which servers perform best for which request types without being explicitly told the hardware specifications.
The reward function is based on the latency of completed requests:

# Reward is inverse of latency (lower latency = higher reward)
reward = 1.0 / (1.0 + avg_latency)
With penalties for rejected requests:
# Penalty for rejected requests
if not success:
    reward -= 0.5
This setup allows the RL agent to learn which servers best handle which types of requests through experience, without needing to explicitly know the hardware specifications. The agent discovers the performance characteristics of each server by observing how quickly they process different request types.


## Getting Started

### Prerequisites

- Python 3.7+
- PyTorch
- NumPy
- Matplotlib
- Gym
- Pandas

### Setting Up a Virtual Environment (Recommended)

1. Create a virtual environment:
   ```bash
   # Using venv
   python -m venv venv
   
   # Activate on macOS/Linux
   source venv/bin/activate
   
   # Activate on Windows
   bandit_env\Scripts\activate
   ```

2. Clone the repository:
   ```bash
   git clone https://github.com/anaiskillian/cs2241_finalproj.git
   cd cs2241_finalproj
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Installation Without Virtual Environment

If you prefer not to use a virtual environment:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/bandit-request-routing.git
   cd bandit-request-routing
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Code

To run the code properly, you need to ensure the Python interpreter can find all the modules:

#### Option 1: Using PYTHONPATH (Recommended)

```bash
# Set PYTHONPATH for Unix/macOS
PYTHONPATH=. python main.py --experiment baseline --episodes 5 --steps 1000

# Set PYTHONPATH for Windows (PowerShell)
$env:PYTHONPATH = "."; python main.py --experiment baseline --episodes 5 --steps 1000

# Set PYTHONPATH for Windows (Command Prompt)
set PYTHONPATH=. && python main.py --experiment baseline --episodes 5 --steps 1000
```

#### Option 2: Installing as a local package

```bash
pip install -e .
python main.py --experiment baseline --episodes 5 --steps 1000
```

### Quick Start: Optimized MAB Experiment

For convenience, use the provided script to run an optimized Multi-Armed Bandit experiment:

```bash
# Make the script executable (Unix/macOS)
chmod +x run_mab_optimized.sh

# Run the script
./run_mab_optimized.sh
```

Or manually:

```bash
# Run with UCB strategy (performs best with heterogeneous servers)
PYTHONPATH=. python main.py --experiment baseline --episodes 10 --steps 1000 --mab_strategy ucb
```

## Usage Options

### Running Baseline Comparison

To compare all routing strategies with default settings:

```bash
PYTHONPATH=. python main.py --experiment baseline --episodes 5 --steps 1000
```

### Testing Different Workloads

To evaluate performance on different workload patterns:

```bash
PYTHONPATH=. python main.py --experiment workload --workload_type all
```

Or test a specific workload type:

```bash
PYTHONPATH=. python main.py --experiment workload --workload_type bursty
```

### Hyperparameter Tuning

To tune hyperparameters for the MAB algorithms:

```bash
PYTHONPATH=. python main.py --experiment hyperparameter --mab_strategy ucb
```

### PPO Comparison

To train and evaluate a PPO agent:

```bash
PYTHONPATH=. python main.py --experiment ppo --episodes 5 --steps 1000
```

## Experiment Results

Results are saved in timestamped directories in the `results/` folder:

- `results/baseline_{timestamp}/`: Baseline comparison of all routing methods
- `results/workload_{timestamp}/`: Analysis of performance under different workload patterns
- `results/hyperparameter_{timestamp}/`: Hyperparameter tuning results
- `results/ppo_{timestamp}/`: PPO training and evaluation results

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
├── run_mab_optimized.sh      # Convenience script for running optimized MAB
├── requirements.txt          # Package dependencies
└── README.md                 # This file
```

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

## Troubleshooting

### ModuleNotFoundError

If you encounter `ModuleNotFoundError: No module named 'src.agents.round_robin'`, ensure you're running the code with the correct PYTHONPATH:

```bash
PYTHONPATH=. python main.py --experiment baseline --episodes 5 --steps 1000
```

### Missing Agent Files

If you encounter errors about missing agent files, verify that all required files are in the correct locations:
- `src/agents/base_agent.py`
- `src/agents/round_robin.py` 
- `src/agents/random_agent.py`
- `src/agents/least_loaded.py`
- `src/agents/mab_agent.py`

## Reference Papers

This work is inspired by research in reinforcement learning for resource allocation:

- Proximal Policy Optimization (PPO) by Schulman et al. (2017)
- A Deep Reinforcement Learning Approach for Traffic Signal Control Optimization by Li et al. (2021)
- A Hybrid Reinforcement Learning Approach to Autonomic Resource Allocation by Tesauro et al. (2006)

## License

This project is licensed under the MIT License - see the LICENSE file for details.
