# src/utils/visualization.py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import List, Dict, Any


def plot_latency_comparison(
    agent_metrics: Dict[str, List[float]],
    title: str = "Latency Comparison",
    subtitle: str = "",
):
    """
    Plot latency comparison between different agents.
    
    Args:
        agent_metrics: Dictionary of agent_name -> latency_list
        title: Plot title
    """
    plt.figure(figsize=(10, 6))

    for agent_name, latencies in agent_metrics.items():
        plt.plot(latencies, label=f"{agent_name} (avg: {np.mean(latencies):.4f})")

    plt.xlabel("Time Step")
    plt.ylabel("Latency (s)")
    plt.title(title)
    if subtitle:
        plt.suptitle(subtitle, fontsize=10, fontweight="light")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_').lower()}.png")
    plt.close()


def plot_throughput_comparison(agent_metrics: Dict[str, List[float]], title: str = "Throughput Comparison"):
    """
    Plot throughput comparison between different agents.
    
    Args:
        agent_metrics: Dictionary of agent_name -> throughput_list
        title: Plot title
    """
    plt.figure(figsize=(10, 6))
    
    for agent_name, throughputs in agent_metrics.items():
        plt.plot(throughputs, label=f"{agent_name} (avg: {np.mean(throughputs):.4f})")
    
    plt.xlabel("Time Step")
    plt.ylabel("Throughput (req/s)")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_').lower()}.png")
    plt.close()

def plot_server_utilization(utilizations: List[List[float]], agent_name: str):
    """
    Plot server utilization over time.
    
    Args:
        utilizations: List of server utilization lists per time step
        agent_name: Name of the agent for the title
    """
    utilizations = np.array(utilizations)
    num_servers = utilizations.shape[1]
    time_steps = np.arange(utilizations.shape[0])
    
    plt.figure(figsize=(12, 6))
    
    for i in range(num_servers):
        plt.plot(time_steps, utilizations[:, i], label=f"Server {i}")
    
    plt.xlabel("Time Step")
    plt.ylabel("Utilization")
    plt.title(f"Server Utilization - {agent_name}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"server_utilization_{agent_name.lower().replace(' ', '_')}.png")
    plt.close()

def plot_request_distribution(request_counts: Dict[str, Dict[str, int]], title: str = "Request Type Distribution"):
    """
    Plot distribution of request types across servers.
    
    Args:
        request_counts: Dict of server_name -> (request_type -> count)
        title: Plot title
    """
    # Convert to DataFrame for easier plotting
    data = []
    for server, counts in request_counts.items():
        for req_type, count in counts.items():
            data.append({
                'Server': server,
                'Request Type': req_type,
                'Count': count
            })
    
    df = pd.DataFrame(data)
    
    # Pivot for plotting
    pivot_df = df.pivot(index='Server', columns='Request Type', values='Count')
    
    # Plot
    plt.figure(figsize=(12, 8))
    pivot_df.plot(kind='bar', stacked=True)
    
    plt.xlabel("Server")
    plt.ylabel("Request Count")
    plt.title(title)
    plt.legend(title="Request Type")
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_').lower()}.png")
    plt.close()

def plot_comparative_metrics(metrics: Dict[str, Dict[str, float]]):
    """
    Create a bar chart comparing key metrics across agents.
    
    Args:
        metrics: Dict of agent_name -> (metric_name -> value)
    """
    # Extract metrics for comparison
    agent_names = list(metrics.keys())
    metric_names = [
        "avg_latency",
        "avg_throughput",
        # , "avg_success_rate"
    ]
    metric_name_map = {
        "avg_latency": "Average Latency (s)",
        "avg_throughput": "Average Throughput (req/s)",
        # "avg_success_rate": "Average Success Rate (%)"
    }

    # Create subplots
    fig, axes = plt.subplots(1, len(metric_names), figsize=(15, 5))

    # Plot each metric
    for i, metric in enumerate(metric_names):
        values = [metrics[agent][metric] for agent in agent_names]

        axes[i].bar(agent_names, values)
        axes[i].set_ylabel(metric_name_map[metric])
        axes[i].set_xlabel("Agents")
        axes[i].set_title(metric.replace('_', ' ').title())
        axes[i].grid(True, alpha=0.3, axis='y')

        # Rotate x-labels if needed
        if max(len(name) for name in agent_names) > 10:
            axes[i].set_xticks(np.arange(len(agent_names)))
            axes[i].set_xticklabels(agent_names, rotation=45, ha='right')
        # if max(len(name) for name in agent_names) > 10:
        #     axes[i].set_xticklabels(agent_names, rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig("comparative_metrics.png")
    plt.close()
