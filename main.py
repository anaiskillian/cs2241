# main.py
import argparse
import os
import sys
import time
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from src.environment.cluster import ServerCluster
from src.environment.request import RequestType
from src.agents.round_robin import RoundRobinAgent
from src.agents.random_agent import RandomAgent
from src.agents.least_loaded import LeastLoadedAgent
from src.agents.mab_agent import MultiArmedBanditAgent, BanditStrategy
from src.agents.enhanced_mab_agent import EnhancedMultiArmedBanditAgent, EnhancedBanditStrategy
from src.agents.ppo_agent import PPOAgent
from src.utils.metrics import PerformanceMetrics
from src.utils.workload_gen import WorkloadGenerator
from src.utils.visualization import (
    plot_latency_comparison,
    plot_throughput_comparison,
    plot_server_utilization,
    plot_comparative_metrics
)
from src.config import (
    ENV_CONFIG,
    SERVER_CONFIGS,
    MAB_CONFIG,
    PPO_CONFIG
)

def parse_args():
    parser = argparse.ArgumentParser(description="Multi-Armed Bandit Request Routing")
    
    # Experiment selection
    parser.add_argument("--experiment", type=str, default="baseline",
                        choices=["baseline", "workload", "hyperparameter", "ppo"],
                        help="Type of experiment to run")
    
    # Common parameters
    parser.add_argument("--episodes", type=int, default=5,
                        help="Number of episodes to run")
    parser.add_argument("--steps", type=int, default=1000,
                        help="Maximum steps per episode")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    
    # Environment parameters
    parser.add_argument("--servers", type=int, default=4,
                        help="Number of servers (if not using configs)")
    parser.add_argument("--history", type=int, default=5,
                        help="History length for RL state")
    
    # Agent parameters
    parser.add_argument("--mab_strategy", type=str, default="ucb",
                        choices=["epsilon_greedy", "ucb", "thompson"],
                        help="Strategy for MAB agent")
    
    # Workload parameters
    parser.add_argument("--workload_type", type=str, default="all",
                        choices=["uniform", "bursty", "diurnal", "skewed", "all"],
                        help="Type of workload to test")
    parser.add_argument("--workload_size", type=int, default=500,
                        help="Number of requests in workload")
    
    return parser.parse_args()

def create_environment(args):
    """Create and configure the environment."""
    # Use server configs if available, otherwise create homogeneous servers
    if os.path.exists("server_configs.json"):
        with open("server_configs.json", "r") as f:
            configs = json.load(f)
    else:
        configs = SERVER_CONFIGS if args.servers == 4 else None
    
    env = ServerCluster(
        num_servers=args.servers if configs is None else len(configs),
        server_configs=configs,
        history_length=args.history,
        max_steps=args.steps
    )
    
    return env

def create_agents(env, args):
    """Create all agents for experimentation."""
    # Calculate state dimension for PPO
    state_dim = (
        env.num_servers +  # Server utilization
        env.history_length +  # Latency history
        env.history_length +  # Decision history
        len(RequestType)  # Request type one-hot encoding
    )
    
    # Create all agents
    agents = {
        "Round Robin": RoundRobinAgent(num_servers=env.num_servers),
        "Random": RandomAgent(num_servers=env.num_servers),
        "Least Loaded": LeastLoadedAgent(num_servers=env.num_servers),
    }
    
    # Add MAB agent with specified strategy
    if args.mab_strategy == "epsilon_greedy":
        agents["MAB (Epsilon-Greedy)"] = MultiArmedBanditAgent(
            num_servers=env.num_servers,
            strategy=BanditStrategy.EPSILON_GREEDY,
            **MAB_CONFIG["epsilon_greedy"]
        )
    elif args.mab_strategy == "ucb":
        agents["MAB (UCB)"] = MultiArmedBanditAgent(
            num_servers=env.num_servers,
            strategy=BanditStrategy.UCB,
            **MAB_CONFIG["ucb"]
        )
    elif args.mab_strategy == "thompson":
        agents["MAB (Thompson)"] = MultiArmedBanditAgent(
            num_servers=env.num_servers,
            strategy=BanditStrategy.THOMPSON_SAMPLING,
            **MAB_CONFIG["thompson"]
        )
    else:
        # Add all MAB variants for comprehensive comparison
        agents["MAB (Epsilon-Greedy)"] = MultiArmedBanditAgent(
            num_servers=env.num_servers,
            strategy=BanditStrategy.EPSILON_GREEDY,
            **MAB_CONFIG["epsilon_greedy"]
        )
        agents["MAB (UCB)"] = MultiArmedBanditAgent(
            num_servers=env.num_servers,
            strategy=BanditStrategy.UCB,
            **MAB_CONFIG["ucb"]
        )
        agents["MAB (Thompson)"] = MultiArmedBanditAgent(
            num_servers=env.num_servers,
            strategy=BanditStrategy.THOMPSON_SAMPLING,
            **MAB_CONFIG["thompson"]
        )
    
    # Add PPO agent for reinforcement learning comparison
    if args.experiment == "ppo":
        agents["PPO"] = PPOAgent(
            num_servers=env.num_servers,
            state_dim=state_dim,
            **PPO_CONFIG
        )
    
    # Add Enhanced MAB agents
    agents["Enhanced MAB (Epsilon-Greedy)"] = EnhancedMultiArmedBanditAgent(
        num_servers=env.num_servers,
        strategy=EnhancedBanditStrategy.EPSILON_GREEDY
    )
    agents["Enhanced MAB (UCB)"] = EnhancedMultiArmedBanditAgent(
        num_servers=env.num_servers,
        strategy=EnhancedBanditStrategy.UCB
    )
    agents["Enhanced MAB (Thompson)"] = EnhancedMultiArmedBanditAgent(
        num_servers=env.num_servers,
        strategy=EnhancedBanditStrategy.THOMPSON_SAMPLING
    )
    
    return agents

def run_episode(env, agent, max_steps=1000, render=False):
    """Run a single episode with the given agent."""
    observation = env.reset()
    agent.reset()
    
    done = False
    step = 0
    
    metrics = PerformanceMetrics()
    
    while not done and step < max_steps:
        action = agent.select_action(observation)
        next_observation, reward, done, info = env.step(action)
        
        agent.update(observation, action, reward, next_observation, done)
        
        # Record metrics
        server_utils = [server.cpu_utilization for server in env.servers]
        metrics.update(info, server_utils)
        
        observation = next_observation
        step += 1
        
        if render and step % 100 == 0:
            env.render()
            print(f"Step {step}, Latency: {info['latency']:.4f}, Throughput: {info['throughput']:.4f}")
    
    return metrics

def baseline_experiment(env, agents, args):
    """Run baseline comparison of all agents."""
    print("Running baseline comparison experiment...")
    
    # Create timestamped directory for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"results/baseline_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)

    save_command_info(args, results_dir)
    
    all_metrics = {}
    comparative_metrics = {}
    mismatch_rates = {}
    
    for agent_name, agent in agents.items():
        print(f"Testing {agent_name} agent...")
        
        agent_metrics = PerformanceMetrics()
        
        for episode in range(args.episodes):
            print(f"  Episode {episode+1}/{args.episodes}")
            episode_metrics = run_episode(env, agent, max_steps=args.steps, render=True)
            
            # Merge episode metrics
            agent_metrics.latencies.extend(episode_metrics.latencies)
            agent_metrics.throughputs.extend(episode_metrics.throughputs)
            agent_metrics.success_rates.extend(episode_metrics.success_rates)
            agent_metrics.server_utilizations.extend(episode_metrics.server_utilizations)
            agent_metrics.step_info.extend(episode_metrics.step_info)
            
            # Track mismatch rates for MAB agents
            if "MAB" in agent_name and hasattr(agent, "get_mismatch_rate"):
                if agent_name not in mismatch_rates:
                    mismatch_rates[agent_name] = []
                mismatch_rates[agent_name].append(agent.get_mismatch_rate())
        
        avg_metrics = agent_metrics.get_average_metrics()
        fairness = agent_metrics.get_fairness_index()
        
        print(f"Results for {agent_name}:")
        print(f"  Avg Latency: {avg_metrics['avg_latency']:.4f}")
        print(f"  Avg Throughput: {avg_metrics['avg_throughput']:.4f}")
        print(f"  Success Rate: {avg_metrics['avg_success_rate']*100:.2f}%")
        print(f"  Fairness Index: {fairness:.4f}")
        if "MAB" in agent_name and hasattr(agent, "get_mismatch_rate"):
            print(f"  Final Mismatch Rate: {agent.get_mismatch_rate():.4f}")
        print("")
        
        all_metrics[agent_name] = agent_metrics
        comparative_metrics[agent_name] = avg_metrics
        
        # Plot server utilization for this agent
        plot_server_utilization(agent_metrics.server_utilizations, agent_name)
    
    # Extract data for comparison plots
    latency_comparison = {name: metrics.latencies for name, metrics in all_metrics.items()}
    throughput_comparison = {name: metrics.throughputs for name, metrics in all_metrics.items()}
    
    # Generate comparison plots
    plot_latency_comparison(latency_comparison)
    plot_throughput_comparison(throughput_comparison)
    plot_comparative_metrics(comparative_metrics)
    
    # Plot mismatch rates if we have any
    if mismatch_rates:
        plt.figure(figsize=(10, 6))
        for agent_name, rates in mismatch_rates.items():
            plt.plot(rates, label=agent_name)
        plt.xlabel("Episode")
        plt.ylabel("Mismatch Rate with Least Loaded")
        plt.title("MAB vs Least Loaded Mismatch Rates")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{results_dir}/mismatch_rates.png")
        plt.close()
    
    # Save results to JSON
    results = {
        name: {
            "avg_metrics": metrics.get_average_metrics(),
            "fairness": metrics.get_fairness_index()
        }
        for name, metrics in all_metrics.items()
    }
    
    # Add mismatch rates to results
    for agent_name, rates in mismatch_rates.items():
        if agent_name in results:
            results[agent_name]["mismatch_rates"] = rates
            results[agent_name]["final_mismatch_rate"] = rates[-1] if rates else 0
    
    with open(f"{results_dir}/results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Move plots to results directory
    os.system(f"mv *.png {results_dir}/")
    
    print(f"Baseline experiment complete. Results saved to '{results_dir}/' directory.")

def workload_experiment(env, agents, args):
    """Run workload analysis experiment."""
    print(f"Running workload analysis experiment...")
    
    # Create timestamped directory for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"results/workload_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)

    save_command_info(args, results_dir)
    
    # Initialize workload generator
    workload_gen = WorkloadGenerator(seed=args.seed)
    
    # Define workloads to test
    if args.workload_type == "all":
        workloads = {
            "Uniform": workload_gen.uniform_workload(num_requests=args.workload_size),
            "Bursty": workload_gen.bursty_workload(num_requests=args.workload_size),
            "Diurnal": workload_gen.diurnal_workload(num_requests=args.workload_size),
            "Skewed": workload_gen.skewed_workload(num_requests=args.workload_size)
        }
    else:
        # Create only the specified workload
        if args.workload_type == "uniform":
            workloads = {"Uniform": workload_gen.uniform_workload(num_requests=args.workload_size)}
        elif args.workload_type == "bursty":
            workloads = {"Bursty": workload_gen.bursty_workload(num_requests=args.workload_size)}
        elif args.workload_type == "diurnal":
            workloads = {"Diurnal": workload_gen.diurnal_workload(num_requests=args.workload_size)}
        elif args.workload_type == "skewed":
            workloads = {"Skewed": workload_gen.skewed_workload(num_requests=args.workload_size)}
    
    # Run experiments for each workload
    all_results = {}
    
    for workload_name, workload_requests in workloads.items():
        print(f"Testing {workload_name} workload...")
        workload_results = {}
        
        for agent_name, agent in agents.items():
            print(f"  Using {agent_name} agent...")
            
            # Reset environment and agent
            env.reset()
            agent.reset()
            
            # Override pending requests with our workload
            env.pending_requests = workload_requests.copy()
            
            # Run episode
            metrics = run_episode(env, agent, max_steps=args.steps, render=False)
            
            avg_metrics = metrics.get_average_metrics()
            fairness = metrics.get_fairness_index()
            
            print(f"  Results for {agent_name} on {workload_name}:")
            print(f"    Avg Latency: {avg_metrics['avg_latency']:.4f}")
            print(f"    Avg Throughput: {avg_metrics['avg_throughput']:.4f}")
            print(f"    Success Rate: {avg_metrics['avg_success_rate']*100:.2f}%")
            print(f"    Fairness Index: {fairness:.4f}")
            print("")
            
            workload_results[agent_name] = metrics
        
        all_results[workload_name] = workload_results
        
        # Extract data for comparison plots
        latency_comparison = {name: metrics.latencies for name, metrics in workload_results.items()}
        throughput_comparison = {name: metrics.throughputs for name, metrics in workload_results.items()}
        
        # Generate comparison plots
        plot_latency_comparison(
            latency_comparison,
            title=f"Latency Comparison - {workload_name} Workload"
        )
        plot_throughput_comparison(
            throughput_comparison,
            title=f"Throughput Comparison - {workload_name} Workload"
        )
    
    # Save results to JSON
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
    
    with open(f"{results_dir}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    # Move plots to results directory
    os.system(f"mv *.png {results_dir}/")
    
    print(f"Workload analysis complete. Results saved to '{results_dir}/' directory.")

def hyperparameter_experiment(env, args):
    """Run hyperparameter tuning experiment."""
    print("Running hyperparameter tuning experiment...")
    
    # Create timestamped directory for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"results/hyperparameter_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Save command line information
    save_command_info(args, results_dir)
    
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
    
    # Select strategy to tune
    if args.mab_strategy == "epsilon_greedy":
        strategies = [BanditStrategy.EPSILON_GREEDY]
    elif args.mab_strategy == "ucb":
        strategies = [BanditStrategy.UCB]
    elif args.mab_strategy == "thompson":
        strategies = [BanditStrategy.THOMPSON_SAMPLING]
    else:
        strategies = list(hyperparams.keys())
    
    # Run hyperparameter tuning for selected strategies
    results = {}
    
    for strategy in strategies:
        strategy_name = strategy.name
        params = hyperparams[strategy]
        
        print(f"Tuning {strategy_name}...")
        
        # Generate all parameter combinations
        param_names = list(params.keys())
        param_values = list(params.values())
        
        from itertools import product
        param_combinations = list(product(*param_values))
        
        strategy_results = {}
        best_latency = float('inf')
        best_config = None
        
        for values in param_combinations:
            config = dict(zip(param_names, values))
            config_str = ", ".join(f"{k}={v}" for k, v in config.items())
            print(f"  Testing configuration: {config_str}")
            
            # Create agent with this configuration
            agent = MultiArmedBanditAgent(
                num_servers=env.num_servers,
                strategy=strategy,
                **config
            )
            
            # Run episodes
            agent_metrics = PerformanceMetrics()
            
            for episode in range(args.episodes):
                episode_metrics = run_episode(env, agent, max_steps=args.steps, render=False)
                
                # Merge episode metrics
                agent_metrics.latencies.extend(episode_metrics.latencies)
                agent_metrics.throughputs.extend(episode_metrics.throughputs)
                agent_metrics.success_rates.extend(episode_metrics.success_rates)
                agent_metrics.server_utilizations.extend(episode_metrics.server_utilizations)
            
            avg_metrics = agent_metrics.get_average_metrics()
            fairness = agent_metrics.get_fairness_index()
            
            print(f"    Avg Latency: {avg_metrics['avg_latency']:.4f}")
            print(f"    Avg Throughput: {avg_metrics['avg_throughput']:.4f}")
            print(f"    Success Rate: {avg_metrics['avg_success_rate']*100:.2f}%")
            print("")
            
            # Save results
            strategy_results[config_str] = {
                "config": config,
                "metrics": avg_metrics,
                "fairness": fairness
            }
            
            # Check if this is the best configuration
            if avg_metrics["avg_latency"] < best_latency:
                best_latency = avg_metrics["avg_latency"]
                best_config = config_str
        
        results[strategy_name] = {
            "results": strategy_results,
            "best_config": best_config,
            "best_latency": best_latency
        }
        
        print(f"Best configuration for {strategy_name}: {best_config}")
        print(f"Best latency: {best_latency:.4f}")
        print("")
    
    # Save results to JSON
    with open("results/hyperparameter/results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("Hyperparameter tuning complete. Results saved to 'results/hyperparameter/' directory.")

def ppo_experiment(env, agents, args):
    """Run PPO comparison experiment."""
    print("Running PPO comparison experiment...")
    
    # Create timestamped directory for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"results/ppo_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Save command line information
    save_command_info(args, results_dir)
    
    # Ensure we have the PPO agent
    if "PPO" not in agents:
        # Calculate state dimension
        state_dim = (
            env.num_servers +  # Server utilization
            env.history_length +  # Latency history
            env.history_length +  # Decision history
            len(RequestType)  # Request type one-hot encoding
        )
        
        agents["PPO"] = PPOAgent(
            num_servers=env.num_servers,
            state_dim=state_dim,
            **PPO_CONFIG
        )
    
    # Select agents for comparison
    compare_agents = {
        "Round Robin": agents["Round Robin"],
        "Least Loaded": agents["Least Loaded"],
        "MAB (UCB)": agents["MAB (UCB)"],
        "PPO": agents["PPO"]
    }
    
    # Train PPO agent
    print("Training PPO agent...")
    ppo_agent = agents["PPO"]
    
    # Training loop
    training_episodes = 50  # More episodes for training
    training_metrics = PerformanceMetrics()
    training_latencies = []
    
    for episode in range(training_episodes):
        print(f"  Training episode {episode+1}/{training_episodes}")
        episode_metrics = run_episode(env, ppo_agent, max_steps=args.steps, render=False)
        
        # Track average latency for this episode
        avg_latency = np.mean(episode_metrics.latencies) if episode_metrics.latencies else 0
        training_latencies.append(avg_latency)
        
        if episode % 5 == 0:
            print(f"    Episode {episode+1}, Avg Latency: {avg_latency:.4f}")
    
    # Save trained model
    ppo_agent.save("results/ppo/ppo_model.pt")
    
    # Plot training progress
    plt.figure(figsize=(10, 6))
    plt.plot(training_latencies)
    plt.xlabel("Episode")
    plt.ylabel("Average Latency")
    plt.title("PPO Training Progress")
    plt.grid(True, alpha=0.3)
    plt.savefig("results/ppo/training_progress.png")
    plt.close()
    
    # Evaluate all agents
    all_metrics = {}
    
    for agent_name, agent in compare_agents.items():
        print(f"Evaluating {agent_name}...")
        
        agent_metrics = PerformanceMetrics()
        
        for episode in range(args.episodes):
            episode_metrics = run_episode(env, agent, max_steps=args.steps, render=(episode == 0))
            
            # Merge episode metrics
            agent_metrics.latencies.extend(episode_metrics.latencies)
            agent_metrics.throughputs.extend(episode_metrics.throughputs)
            agent_metrics.success_rates.extend(episode_metrics.success_rates)
            agent_metrics.server_utilizations.extend(episode_metrics.server_utilizations)
        
        all_metrics[agent_name] = agent_metrics
        
        avg_metrics = agent_metrics.get_average_metrics()
        fairness = agent_metrics.get_fairness_index()
        
        print(f"Results for {agent_name}:")
        print(f"  Avg Latency: {avg_metrics['avg_latency']:.4f}")
        print(f"  Avg Throughput: {avg_metrics['avg_throughput']:.4f}")
        print(f"  Success Rate: {avg_metrics['avg_success_rate']*100:.2f}%")
        print(f"  Fairness Index: {fairness:.4f}")
        print("")
    
    # Extract data for comparison plots
    latency_comparison = {name: metrics.latencies for name, metrics in all_metrics.items()}
    throughput_comparison = {name: metrics.throughputs for name, metrics in all_metrics.items()}
    
    # Generate comparison plots
    plot_latency_comparison(latency_comparison, title="Latency Comparison with PPO")
    plot_throughput_comparison(throughput_comparison, title="Throughput Comparison with PPO")
    
    # Calculate comparative metrics
    comparative_metrics = {
        name: metrics.get_average_metrics() for name, metrics in all_metrics.items()
    }
    plot_comparative_metrics(comparative_metrics)
    
    # Save results to JSON
    results = {
        name: {
            "avg_metrics": metrics.get_average_metrics(),
            "fairness": metrics.get_fairness_index()
        }
        for name, metrics in all_metrics.items()
    }
    
    with open("results/ppo/results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Move plots
    os.system("mv latency_comparison_with_ppo.png results/ppo/")
    os.system("mv throughput_comparison_with_ppo.png results/ppo/")
    os.system("mv comparative_metrics.png results/ppo/")
    os.system("mv server_utilization_*.png results/ppo/")
    
    print("PPO experiment complete. Results saved to 'results/ppo/' directory.")

def save_command_info(args, results_dir, agents=None):
    """
    Save the command line parameters, agent configurations, and MAB_CONFIG used to run the experiment.
    
    Args:
        args: Parsed command line arguments
        results_dir: Directory where results are saved
        agents: Dictionary of agent instances (optional)
    """
    import sys
    import os
    from src.config import MAB_CONFIG
    
    # Get the original command
    command = f"PYTHONPATH=. python {' '.join(sys.argv)}"
    
    # Convert args to dictionary
    args_dict = vars(args)
    
    # Create a dictionary with command and arguments
    command_info = {
        "command": command,
        "arguments": args_dict,
        "mab_config": MAB_CONFIG  # Include MAB_CONFIG from config.py
    }
    
    # If MAB agents are provided, capture their configurations
    if agents:
        agent_configs = {}
        for name, agent in agents.items():
            if "MAB" in name and hasattr(agent, "strategy"):
                # Extract MAB agent configuration
                config = {
                    "strategy": agent.strategy.name,
                    "num_servers": agent.num_servers,
                }
                
                # Add strategy-specific parameters
                if hasattr(agent, "epsilon"):
                    config["epsilon"] = agent.epsilon
                if hasattr(agent, "alpha"):
                    config["alpha"] = agent.alpha
                if hasattr(agent, "ucb_c"):
                    config["ucb_c"] = agent.ucb_c
                if hasattr(agent, "throughput_weight"):
                    config["throughput_weight"] = agent.throughput_weight
                
                agent_configs[name] = config
                
        # Add agent configurations to command info
        if agent_configs:
            command_info["agent_configs"] = agent_configs
    
    # Save to JSON
    import json
    with open(os.path.join(results_dir, "command_info.json"), "w") as f:
        json.dump(command_info, f, indent=2)
    
    # Also save as plain text for easy reading
    with open(os.path.join(results_dir, "command.txt"), "w") as f:
        f.write(f"Command: {command}\n\n")
        f.write("Arguments:\n")
        for arg, value in args_dict.items():
            f.write(f"  --{arg}: {value}\n")
        
        # Add MAB_CONFIG to plain text file
        f.write("\nMAB_CONFIG from config.py:\n")
        for strategy, params in MAB_CONFIG.items():
            f.write(f"  {strategy}:\n")
            for param, value in params.items():
                f.write(f"    {param}: {value}\n")
        
        # Add agent configurations to plain text file too
        if agents and any("MAB" in name for name in agents):
            f.write("\nMAB Agent Configurations:\n")
            for name, agent in agents.items():
                if "MAB" in name and hasattr(agent, "strategy"):
                    f.write(f"  {name}:\n")
                    f.write(f"    Strategy: {agent.strategy.name}\n")
                    if hasattr(agent, "epsilon"):
                        f.write(f"    Epsilon: {agent.epsilon}\n")
                    if hasattr(agent, "alpha"):
                        f.write(f"    Alpha: {agent.alpha}\n")
                    if hasattr(agent, "ucb_c"):
                        f.write(f"    UCB exploration coefficient: {agent.ucb_c}\n")
                    if hasattr(agent, "throughput_weight"):
                        f.write(f"    Throughput weight: {agent.throughput_weight}\n")
                        
def main():
    # Parse command line arguments
    args = parse_args()
    
    # Set random seeds for reproducibility
    np.random.seed(args.seed)
    
    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)
    
    # Create environment
    env = create_environment(args)
    
    # Create agents
    agents = create_agents(env, args)
    
    # Run selected experiment
    if args.experiment == "baseline":
        baseline_experiment(env, agents, args)
    elif args.experiment == "workload":
        workload_experiment(env, agents, args)
    elif args.experiment == "hyperparameter":
        hyperparameter_experiment(env, args)
    elif args.experiment == "ppo":
        ppo_experiment(env, agents, args)
    else:
        print(f"Unknown experiment type: {args.experiment}")

if __name__ == "__main__":
    main()
