# src/utils/metrics.py
import numpy as np
from typing import List, Dict, Any

class PerformanceMetrics:
    """
    Calculates and tracks performance metrics for load balancing.
    """
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.latencies = []
        self.throughputs = []
        self.success_rates = []
        self.server_utilizations = []
        self.step_info = []
    
    def update(self, info: Dict[str, Any], server_utils: List[float]):
        """
        Update metrics with new information.
        
        Args:
            info: Info dict returned from environment step
            server_utils: List of server utilizations
        """
        self.latencies.append(info['latency'])
        self.throughputs.append(info['throughput'])
        self.success_rates.append(info['success_rate'])
        self.server_utilizations.append(server_utils)
        self.step_info.append(info)
    
    def get_average_metrics(self):
        """Get average of all metrics."""
        return {
            'avg_latency': np.mean(self.latencies) if self.latencies else 0,
            'avg_throughput': np.mean(self.throughputs) if self.throughputs else 0,
            'avg_success_rate': np.mean(self.success_rates) if self.success_rates else 0,
            'avg_server_util': np.mean(self.server_utilizations) if self.server_utilizations else 0,
            'std_latency': np.std(self.latencies) if len(self.latencies) > 1 else 0,
            'tail_latency_95': np.percentile(self.latencies, 95) if self.latencies else 0
        }
    
    def get_fairness_index(self):
        """
        Calculate Jain's fairness index for server utilization.
        Value ranges from 1/n (unfair) to 1 (fair).
        """
        if not self.server_utilizations:
            return 0
            
        # Average utilization across all timesteps for each server
        server_avg_utils = np.mean(self.server_utilizations, axis=0)
        
        # Calculate Jain's fairness index
        numerator = np.sum(server_avg_utils) ** 2
        denominator = len(server_avg_utils) * np.sum(server_avg_utils ** 2)
        
        if denominator == 0:
            return 0
            
        return numerator / denominator
    
    def summarize(self):
        """Print a summary of performance metrics."""
        metrics = self.get_average_metrics()
        fairness = self.get_fairness_index()
        
        print("Performance Summary:")
        print(f"  Average Latency: {metrics['avg_latency']:.4f} s")
        print(f"  95th Percentile Latency: {metrics['tail_latency_95']:.4f} s")
        print(f"  Average Throughput: {metrics['avg_throughput']:.4f} req/s")
        print(f"  Success Rate: {metrics['avg_success_rate']*100:.2f}%")
        print(f"  Server Utilization: {metrics['avg_server_util']*100:.2f}%")
        print(f"  Fairness Index: {fairness:.4f}")


# src/utils/workload_gen.py
import numpy as np
from typing import List, Dict, Callable
import collections

from src.environment.request import Request, RequestType

class WorkloadGenerator:
    """
    Generates workload patterns for testing load balancing algorithms.
    """
    def __init__(self, seed=None):
        """
        Initialize workload generator.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.rng = np.random.RandomState(seed)
    
    def uniform_workload(self, num_requests, size_range=(0.8, 1.2)):
        """
        Generate a uniform workload with randomly distributed request types.
        
        Args:
            num_requests: Number of requests to generate
            size_range: Range of request sizes (multiplier)
            
        Returns:
            List of Request objects
        """
        requests = []
        request_types = list(RequestType)
        
        for i in range(num_requests):
            req_type = self.rng.choice(request_types)
            size = self.rng.uniform(size_range[0], size_range[1])
            arrival_time = i * 0.1  # Simple uniform arrival
            
            request = Request.create(
                request_type=req_type,
                arrival_time=arrival_time,
                size=size
            )
            requests.append(request)
            
        return requests
    
    def bursty_workload(self, num_requests, burst_factor=3.0, burst_prob=0.1, size_range=(0.8, 1.2)):
        """
        Generate a bursty workload with periods of high request frequency.
        
        Args:
            num_requests: Base number of requests to generate
            burst_factor: Multiplier for number of requests during bursts
            burst_prob: Probability of a burst occurring
            size_range: Range of request sizes (multiplier)
            
        Returns:
            List of Request objects
        """
        requests = []
        request_types = list(RequestType)
        arrival_time = 0.0
        
        i = 0
        while i < num_requests:
            # Determine if this is a burst period
            is_burst = self.rng.random() < burst_prob
            
            # Number of requests in this period
            period_requests = int(burst_factor * num_requests / 10) if is_burst else int(num_requests / 10)
            
            # Generate requests for this period
            for j in range(period_requests):
                req_type = self.rng.choice(request_types)
                size = self.rng.uniform(size_range[0], size_range[1])
                
                # Bunched arrival times during burst, spread out otherwise
                if is_burst:
                    period_arrival = arrival_time + self.rng.exponential(0.01)
                else:
                    period_arrival = arrival_time + self.rng.exponential(0.1)
                
                request = Request.create(
                    request_type=req_type,
                    arrival_time=period_arrival,
                    size=size
                )
                requests.append(request)
            
            i += period_requests
            arrival_time += 1.0  # Move to next time period
            
        # Sort by arrival time
        requests.sort(key=lambda r: r.arrival_time)
        return requests
    
    def diurnal_workload(self, num_requests, period=24.0, peak_factor=2.0, size_range=(0.8, 1.2)):
        """
        Generate a workload with diurnal (day/night) pattern.
        
        Args:
            num_requests: Base number of requests to generate
            period: Time period for one complete cycle
            peak_factor: Multiplier for number of requests at peak
            size_range: Range of request sizes (multiplier)
            
        Returns:
            List of Request objects
        """
        requests = []
        request_types = list(RequestType)
        
        for i in range(num_requests):
            # Calculate position in cycle (0 to 1)
            cycle_pos = (i % period) / period
            
            # Calculate rate multiplier based on sine wave
            rate_multiplier = 1.0 + (peak_factor - 1.0) * (np.sin(2 * np.pi * cycle_pos) + 1) / 2
            
            # Adjust arrival time based on rate
            if i > 0:
                interval = 1.0 / rate_multiplier
                arrival_time = requests[-1].arrival_time + self.rng.exponential(interval)
            else:
                arrival_time = 0.0
            
            # Select request type with bias toward more complex requests during low periods
            if rate_multiplier < 1.5:  # Low period
                type_weights = [0.1, 0.2, 0.3, 0.2, 0.2]  # More complex queries
            else:  # High period
                type_weights = [0.4, 0.3, 0.1, 0.1, 0.1]  # More simple queries
                
            req_type = self.rng.choice(request_types, p=type_weights)
            size = self.rng.uniform(size_range[0], size_range[1])
            
            request = Request.create(
                request_type=req_type,
                arrival_time=arrival_time,
                size=size
            )
            requests.append(request)
            
        return requests
    
    def skewed_workload(self, num_requests, skew_factor=0.8, size_range=(0.8, 1.2)):
        """
        Generate a workload with skewed distribution of request types.
        
        Args:
            num_requests: Number of requests to generate
            skew_factor: Zipf parameter for request type distribution
            size_range: Range of request sizes (multiplier)
            
        Returns:
            List of Request objects
        """
        requests = []
        request_types = list(RequestType)
        
        # Calculate Zipf probabilities
        n = len(request_types)
        zipf_probs = np.array([1.0/((i+1)**skew_factor) for i in range(n)])
        zipf_probs /= zipf_probs.sum()
        
        for i in range(num_requests):
            # Select request type according to Zipf distribution
            req_type = self.rng.choice(request_types, p=zipf_probs)
            size = self.rng.uniform(size_range[0], size_range[1])
            arrival_time = i * 0.1  # Simple uniform arrival
            
            request = Request.create(
                request_type=req_type,
                arrival_time=arrival_time,
                size=size
            )
            requests.append(request)
            
        return requests


# src/utils/visualization.py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import List, Dict, Any

def plot_latency_comparison(agent_metrics: Dict[str, List[float]], title: str = "Latency Comparison"):
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
    metric_names = ['avg_latency', 'avg_throughput', 'avg_success_rate']
    
    # Create subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot each metric
    for i, metric in enumerate(metric_names):
        values = [metrics[agent][metric] for agent in agent_names]
        
        axes[i].bar(agent_names, values)
        axes[i].set_title(metric.replace('_', ' ').title())
        axes[i].grid(True, alpha=0.3, axis='y')
        
        # Rotate x-labels if needed
        if max(len(name) for name in agent_names) > 10:
            axes[i].set_xticklabels(agent_names, rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig("comparative_metrics.png")
    plt.close()
