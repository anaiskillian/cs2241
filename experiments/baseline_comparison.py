# experiments/baseline_comparison.py
import numpy as np
import matplotlib.pyplot as plt
import argparse
import time
import os
import json

import sys
sys.path.append('.')

from src.environment.cluster import ServerCluster
from src.agents.round_robin import RoundRobinAgent
from src.agents.random_agent import RandomAgent
from src.agents.least_loaded import LeastLoadedAgent
from src.agents.mab_agent import MultiArmedBanditAgent, BanditStrategy
from src.utils.metrics import PerformanceMetrics
from src.utils.visualization import (
    plot_latency_comparison,
    plot_throughput_comparison,
    plot_server_utilization,
    plot_comparative_metrics
)

def run_experiment(agent_name, agent, env, num_episodes=5, max_steps=500):
    """
    Run experiment with the given agent and environment.
    
    Args:
        agent_name: Name of the agent
        agent: Agent instance
        env: Environment instance
        num_episodes: Number of episodes to run
        max_steps: Maximum steps per episode
        
    Returns:
        PerformanceMetrics object with results
    """
    print(f"Running experiment with {agent_name}...")
    metrics = PerformanceMetrics()
    
    for episode in range(num_episodes):
        print(f"  Episode {episode+1}/{num_episodes}")
        observation = env.reset()
        agent.reset()
        done = False
        step = 0
        
        while not done and step < max_steps:
            action = agent.select_action(observation)
            next_observation, reward, done, info = env.step(action)
            
            agent.update(observation, action, reward, next_observation, done)
            
            # Record metrics
            server_utils = [server.cpu_utilization for server in env.servers]
            metrics.update(info, server_utils)
            
            observation = next_observation
            step += 1
            
            if step % 100 == 0:
                print(f"    Step {step}, Latency: {info['latency']:.4f}, Throughput: {info['throughput']:.4f}")
    
    avg_metrics = metrics.get_average_metrics()
    fairness = metrics.get_fairness_index()
    
    print(f"Results for {agent_name}:")
    print(f"  Avg Latency: {avg_metrics['avg_latency']:.4f}")
    print(f"  Avg Throughput: {avg_metrics['avg_throughput']:.4f}")
    print(f"  Success Rate: {avg_metrics['avg_success_rate']*100:.2f}%")
    print(f"  Fairness Index: {fairness:.4f}")
    print("")
    
    return metrics

def main(args):
    # Create results directory
    os.makedirs("results", exist_ok=True)
    
    # Configure environment
    server_configs = [
        {"cpu_speed": 1.0, "ram_size": 16, "processing_capacity": 8},
        {"cpu_speed": 1.2, "ram_size": 32, "processing_capacity": 10},
        {"cpu_speed": 0.8, "ram_size": 8, "processing_capacity": 6},
        {"cpu_speed": 1.5, "ram_size": 64, "processing_capacity": 12}
    ]
    
    env = ServerCluster(
        num_servers=len(server_configs),
        server_configs=server_configs,
        history_length=args.history_length,
        max_steps=args.max_steps
    )
    
    # Initialize agents
    agents = {
        "Round Robin": RoundRobinAgent(num_servers=env.num_servers),
        "Random": RandomAgent(num_servers=env.num_servers),
        "Least Loaded": LeastLoadedAgent(num_servers=env.num_servers),
        "MAB (Epsilon-Greedy)": MultiArmedBanditAgent(
            num_servers=env.num_servers,
            strategy=BanditStrategy.EPSILON_GREEDY,
            epsilon=0.1
        ),
        "MAB (UCB)": MultiArmedBanditAgent(
            num_servers=env.num_servers,
            strategy=BanditStrategy.UCB,
            ucb_c=2.0
        ),
        "MAB (Thompson)": MultiArmedBanditAgent(
            num_servers=env.num_servers,
            strategy=BanditStrategy.THOMPSON_SAMPLING
        )
    }
    
    # Run experiments
    all_metrics = {}
    comparative_metrics = {}
    
    for agent_name, agent in agents.items():
        metrics = run_experiment(
            agent_name=agent_name,
            agent=agent,
            env=env,
            num_episodes=args.num_episodes,
            max_steps=args.max_steps
        )
        
        all_metrics[agent_name] = metrics
        comparative_metrics[agent_name] = metrics.get_average_metrics()
        
        # Plot server utilization for this agent
        plot_server_utilization(metrics.server_utilizations, agent_name)
    
    # Extract data for comparison plots
    latency_comparison = {name: metrics.latencies for name, metrics in all_metrics.items()}
    throughput_comparison = {name: metrics.throughputs for name, metrics in all_metrics.items()}
    
    # Generate comparison plots
    plot_latency_comparison(latency_comparison)
    plot_throughput_comparison(throughput_comparison)
    plot_comparative_metrics(comparative_metrics)
    
    # Save results to JSON
    results = {
        name: {
            "avg_metrics": metrics.get_average_metrics(),
            "fairness": metrics.get_fairness_index()
        }
        for name, metrics in all_metrics.items()
    }
    
    with open("results/baseline_comparison_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("Experiments complete. Results saved to 'results/' directory.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare baseline load balancing algorithms")
    parser.add_argument("--num_episodes", type=int, default=5, help="Number of episodes to run")
    parser.add_argument("--max_steps", type=int, default=500, help="Maximum steps per episode")
    parser.add_argument("--history_length", type=int, default=5, help="Length of history for state representation")
    
    args = parser.parse_args()
    main(args)


# experiments/workload_analysis.py
import numpy as np
import matplotlib.pyplot as plt
import argparse
import time
import os
import json

import sys
sys.path.append('.')

from src.environment.cluster import ServerCluster
from src.agents.round_robin import RoundRobinAgent
from src.agents.least_loaded import LeastLoadedAgent
from src.agents.mab_agent import MultiArmedBanditAgent, BanditStrategy
from src.utils.metrics import PerformanceMetrics
from src.utils.workload_gen import WorkloadGenerator
from src.utils.visualization import (
    plot_latency_comparison,
    plot_throughput_comparison,
    plot_comparative_metrics
)

def run_workload_experiment(env, agents, workload_name, workload_requests, max_steps=1000):
    """
    Run experiment with different agents on the same workload.
    
    Args:
        env: Environment instance
        agents: Dictionary of agent_name -> agent_instance
        workload_name: Name of the workload type
        workload_requests: List of Request objects for the workload
        max_steps: Maximum steps to run
        
    Returns:
        Dictionary of agent_name -> metrics
    """
    print(f"Running {workload_name} workload experiment...")
    results = {}
    
    for agent_name, agent in agents.items():
        print(f"  Using {agent_name} agent...")
        metrics = PerformanceMetrics()
        
        # Reset environment with specific workload
        observation = env.reset()
        agent.reset()
        
        # Override the environment's pending requests with our workload
        # (Note: In a real implementation, you'd modify the environment to accept a workload)
        env.pending_requests = workload_requests.copy()
        
        done = False
        step = 0
        
        while not done and step < max_steps and env.pending_requests:
            action = agent.select_action(observation)
            next_observation, reward, done, info = env.step(action)
            
            agent.update(observation, action, reward, next_observation, done)
            
            # Record metrics
            server_utils = [server.cpu_utilization for server in env.servers]
            metrics.update(info, server_utils)
            
            observation = next_observation
            step += 1
            
            if step % 100 == 0:
                print(f"    Step {step}, Latency: {info['latency']:.4f}, Throughput: {info['throughput']:.4f}")
        
        avg_metrics = metrics.get_average_metrics()
        fairness = metrics.get_fairness_index()
        
        print(f"  Results for {agent_name} on {workload_name}:")
        print(f"    Avg Latency: {avg_metrics['avg_latency']:.4f}")
        print(f"    Avg Throughput: {avg_metrics['avg_throughput']:.4f}")
        print(f"    Success Rate: {avg_metrics['avg_success_rate']*100:.2f}%")
        print(f"    Fairness Index: {fairness:.4f}")
        print("")
        
        results[agent_name] = metrics
    
    return results

def main(args):
    # Create results directory
    os.makedirs("results/workload_analysis", exist_ok=True)
    
    # Configure environment with heterogeneous servers
    server_configs = [
        {"cpu_speed": 1.0, "ram_size": 16, "processing_capacity": 8},
        {"cpu_speed": 1.2, "ram_size": 32, "processing_capacity": 10},
        {"cpu_speed": 0.8, "ram_size": 8, "processing_capacity": 6},
        {"cpu_speed": 1.5, "ram_size": 64, "processing_capacity": 12}
    ]
    
    env = ServerCluster(
        num_servers=len(server_configs),
        server_configs=server_configs,
        history_length=args.history_length,
        max_steps=args.max_steps
    )
    
    # Initialize agents
    agents = {
        "Round Robin": RoundRobinAgent(num_servers=env.num_servers),
        "Least Loaded": LeastLoadedAgent(num_servers=env.num_servers),
        "MAB (Epsilon-Greedy)": MultiArmedBanditAgent(
            num_servers=env.num_servers,
            strategy=BanditStrategy.EPSILON_GREEDY,
            epsilon=0.1
        ),
        "MAB (UCB)": MultiArmedBanditAgent(
            num_servers=env.num_servers,
            strategy=BanditStrategy.UCB,
            ucb_c=2.0
        ),
        "MAB (Thompson)": MultiArmedBanditAgent(
            num_servers=env.num_servers,
            strategy=BanditStrategy.THOMPSON_SAMPLING
        )
    }
    
    # Generate different workloads
    workload_gen = WorkloadGenerator(seed=args.seed)
    
    workloads = {
        "Uniform": workload_gen.uniform_workload(num_requests=args.workload_size),
        "Bursty": workload_gen.bursty_workload(num_requests=args.workload_size),
        "Diurnal": workload_gen.diurnal_workload(num_requests=args.workload_size),
        "Skewed": workload_gen.skewed_workload(num_requests=args.workload_size)
    }
    
    # Run experiments for each workload
    all_results = {}
    
    for workload_name, workload_requests in workloads.items():
        results = run_workload_experiment(
            env=env,
            agents=agents,
            workload_name=workload_name,
            workload_requests=workload_requests,
            max_steps=args.max_steps
        )
        
        all_results[workload_name] = results
        
        # Extract data for comparison plots
        latency_comparison = {name: metrics.latencies for name, metrics in results.items()}
        throughput_comparison = {name: metrics.throughputs for name, metrics in results.items()}
        
        # Generate comparison plots
        plot_latency_comparison(
            latency_comparison,
            title=f"Latency Comparison - {workload_name} Workload"
        )
        plot_throughput_comparison(
            throughput_comparison,
            title=f"Throughput Comparison - {workload_name} Workload"
        )
        
        # Save plots to workload-specific directory
        os.system(f"mv *{workload_name.lower()}* results/workload_analysis/")
    
    # Generate summary metrics across all workloads
    summary = {}
    
    for workload_name, results in all_results.items():
        summary[workload_name] = {
            agent_name: {
                "avg_latency": metrics.get_average_metrics()["avg_latency"],
                "avg_throughput": metrics.get_average_metrics()["avg_throughput"],
                "success_rate": metrics.get_average_metrics()["avg_success_rate"],
                "fairness_index": metrics.get_fairness_index()
            }
            for agent_name, metrics in results.items()
        }
    
    # Save summary to JSON
    with open("results/workload_analysis/summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print("Workload analysis complete. Results saved to 'results/workload_analysis/' directory.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze agent performance on different workloads")
    parser.add_argument("--max_steps", type=int, default=1000, help="Maximum steps per experiment")
    parser.add_argument("--history_length", type=int, default=5, help="Length of history for state representation")
    parser.add_argument("--workload_size", type=int, default=500, help="Number of requests in workload")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    main(args)


# experiments/hyperparameter_tuning.py
import numpy as np
import matplotlib.pyplot as plt
import argparse
import time
import os
import json
from itertools import product

import sys
sys.path.append('.')

from src.environment.cluster import ServerCluster
from src.agents.mab_agent import MultiArmedBanditAgent, BanditStrategy
from src.utils.metrics import PerformanceMetrics

def run_hyperparameter_experiment(env, strategy, params, num_episodes=3, max_steps=300):
    """
    Run experiment with different hyperparameter configurations.
    
    Args:
        env: Environment instance
        strategy: BanditStrategy to use
        params: Dict of parameter values to try
        num_episodes: Number of episodes per configuration
        max_steps: Maximum steps per episode
        
    Returns:
        Dictionary of param_config -> metrics
    """
    print(f"Tuning hyperparameters for {strategy.name}...")
    results = {}
    
    # Generate all combinations of parameters
    param_names = list(params.keys())
    param_values = list(params.values())
    
    for values in product(*param_values):
        config = dict(zip(param_names, values))
        config_str = ", ".join(f"{k}={v}" for k, v in config.items())
        print(f"  Testing configuration: {config_str}")
        
        # Create agent with this configuration
        agent = MultiArmedBanditAgent(
            num_servers=env.num_servers,
            strategy=strategy,
            **config
        )
        
        # Run experiment
        metrics = PerformanceMetrics()
        
        for episode in range(num_episodes):
            observation = env.reset()
            agent.reset()
            done = False
            step = 0
            
            while not done and step < max_steps:
                action = agent.select_action(observation)
                next_observation, reward, done, info = env.step(action)
                
                agent.update(observation, action, reward, next_observation, done)
                
                # Record metrics
                server_utils = [server.cpu_utilization for server in env.servers]
                metrics.update(info, server_utils)
                
                observation = next_observation
                step += 1
        
        # Calculate average metrics
        avg_metrics = metrics.get_average_metrics()
        
        print(f"    Avg Latency: {avg_metrics['avg_latency']:.4f}")
        print(f"    Avg Throughput: {avg_metrics['avg_throughput']:.4f}")
        print("")
        
        # Store results
        results[config_str] = {
            "config": config,
            "metrics": avg_metrics,
            "fairness": metrics.get_fairness_index()
        }
    
    return results

def main(args):
    # Create results directory
    os.makedirs("results/hyperparameter_tuning", exist_ok=True)
    
    # Configure environment
    server_configs = [
        {"cpu_speed": 1.0, "ram_size": 16, "processing_capacity": 8},
        {"cpu_speed": 1.2, "ram_size": 32, "processing_capacity": 10},
        {"cpu_speed": 0.8, "ram_size": 8, "processing_capacity": 6},
        {"cpu_speed": 1.5, "ram_size": 64, "processing_capacity": 12}
    ]
    
    env = ServerCluster(
        num_servers=len(server_configs),
        server_configs=server_configs,
        history_length=args.history_length,
        max_steps=args.max_steps
    )
    
    # Define hyperparameter ranges to test
    hyperparams = {
        BanditStrategy.EPSILON_GREEDY: {
            "epsilon": [0.05, 0.1, 0.2, 0.3],
            "alpha": [0.05, 0.1, 0.2]
        },
        BanditStrategy.UCB: {
            "ucb_c": [0.5, 1.0, 2.0, 4.0],
            "alpha": [0.05, 0.1, 0.2]
        },
        BanditStrategy.THOMPSON_SAMPLING: {
            "alpha": [0.05, 0.1, 0.2]
        }
    }
    
    # Run experiments for each strategy
    all_results = {}
    
    for strategy, params in hyperparams.items():
        results = run_hyperparameter_experiment(
            env=env,
            strategy=strategy,
            params=params,
            num_episodes=args.num_episodes,
            max_steps=args.max_steps
        )
        
        all_results[strategy.name] = results
        
        # Find best configuration
        best_config = min(results.items(), key=lambda x: x[1]["metrics"]["avg_latency"])
        print(f"Best configuration for {strategy.name}:")
        print(f"  {best_config[0]}")
        print(f"  Avg Latency: {best_config[1]['metrics']['avg_latency']:.4f}")
        print(f"  Avg Throughput: {best_config[1]['metrics']['avg_throughput']:.4f}")
        print("")
    
    # Save results to JSON
    with open("results/hyperparameter_tuning/results.json", "w") as f:
        json.dump(
            {k: {k2: v2 for k2, v2 in v.items() if k2 != "config"} for k, v in all_results.items()},
            f,
            indent=2
        )
    
    # Plot results for each strategy
    for strategy_name, results in all_results.items():
        # Extract relevant data for plotting
        configs = list(results.keys())
        latencies = [results[c]["metrics"]["avg_latency"] for c in configs]
        throughputs = [results[c]["metrics"]["avg_throughput"] for c in configs]
        
        # Plot latency comparison
        plt.figure(figsize=(10, 6))
        plt.bar(configs, latencies)
        plt.xlabel("Configuration")
        plt.ylabel("Average Latency (s)")
        plt.title(f"Latency by Configuration - {strategy_name}")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(f"results/hyperparameter_tuning/latency_{strategy_name.lower()}.png")
        plt.close()
        
        # Plot throughput comparison
        plt.figure(figsize=(10, 6))
        plt.bar(configs, throughputs)
        plt.xlabel("Configuration")
        plt.ylabel("Average Throughput (req/s)")
        plt.title(f"Throughput by Configuration - {strategy_name}")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(f"results/hyperparameter_tuning/throughput_{strategy_name.lower()}.png")
        plt.close()
    
    print("Hyperparameter tuning complete. Results saved to 'results/hyperparameter_tuning/' directory.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tune hyperparameters for MAB agents")
    parser.add_argument("--num_episodes", type=int, default=3, help="Number of episodes per configuration")
    parser.add_argument("--max_steps", type=int, default=300, help="Maximum steps per episode")
    parser.add_argument("--history_length", type=int, default=5, help="Length of history for state representation")
    
    args = parser.parse_args()
    main(args)
