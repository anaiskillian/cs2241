# src/agents/enhanced_mab_agent.py
import numpy as np
from enum import Enum, auto
from .base_agent import BaseAgent
from ..environment.request import RequestType

class EnhancedBanditStrategy(Enum):
    """Enum for different enhanced multi-armed bandit strategies."""
    EPSILON_GREEDY = auto()
    UCB = auto()
    THOMPSON_SAMPLING = auto()

class EnhancedMultiArmedBanditAgent(BaseAgent):
    """
    Enhanced Multi-Armed Bandit agent for request routing.
    Learns to route requests based on observed rewards and server resource utilization.
    Considers both CPU and RAM utilization in decision making.
    """
    
    def __init__(
        self,
        num_servers,
        strategy=EnhancedBanditStrategy.EPSILON_GREEDY,
        epsilon=0.1,
        alpha=0.1,
        ucb_c=2.0,
        throughput_weight=0.6,
        num_request_types=5,
        **kwargs
    ):
        """
        Initialize the Enhanced Multi-Armed Bandit agent.
        
        Args:
            num_servers: Number of servers to route requests to
            strategy: EnhancedBanditStrategy to use
            epsilon: Exploration rate for epsilon-greedy
            alpha: Learning rate
            ucb_c: Exploration coefficient for UCB
            throughput_weight: Weight given to throughput vs latency (0-1)
            num_request_types: Number of different request types
        """
        super(EnhancedMultiArmedBanditAgent, self).__init__(num_servers, **kwargs)
        
        self.strategy = strategy
        self.epsilon = epsilon
        self.alpha = alpha
        self.ucb_c = ucb_c
        self.throughput_weight = throughput_weight
        self.num_request_types = num_request_types
        
        # Initialize Q-values: request_type x server
        self.q_values = np.zeros((num_request_types, num_servers))
        
        # For UCB: action counts
        self.action_counts = np.zeros((num_request_types, num_servers))
        
        # For Thompson Sampling: success and failure counts
        self.success_counts = np.ones((num_request_types, num_servers))
        self.failure_counts = np.ones((num_request_types, num_servers))
        
        # Track last request for updates
        self.last_request_type = None
        self.last_action = None
        
        # Track throughput metrics per server
        self.server_completions = np.zeros(num_servers)
        self.server_processing_times = np.zeros(num_servers) + 1e-6
        
        # Server capacity estimates (adjusted over time)
        self.capacity_estimates = np.ones(num_servers)  # Start with equal capacity
        
        # Processing time estimates per request type and server
        self.processing_times = np.ones((num_request_types, num_servers))
        
        # RAM utilization tracking
        self.ram_utilization_history = np.zeros((num_servers, 10))  # Track last 10 steps
        self.ram_utilization_idx = 0
        
        # Resource utilization weights
        self.cpu_weight = 0.6  # Weight for CPU utilization
        self.ram_weight = 0.4  # Weight for RAM utilization
    
    def select_action(self, observation):
        """
        Select a server using the configured bandit strategy.
        Considers both CPU and RAM utilization in decision making.
        
        Args:
            observation: Environment observation
            
        Returns:
            int: Selected server index
        """
        # Extract request type and server utilization from observation
        request_type_onehot = observation['request_type']
        request_type = np.argmax(request_type_onehot)
        server_utils = observation['server_utils']
        
        # Get server states to access RAM utilization
        server_states = observation.get('server_states', [])
        if not server_states:
            # If server states not provided, use default values
            ram_utils = np.zeros(self.num_servers)
        else:
            ram_utils = np.array([state.get('ram_utilization', 0.0) for state in server_states])
        
        # Update RAM utilization history
        self.ram_utilization_history[:, self.ram_utilization_idx] = ram_utils
        self.ram_utilization_idx = (self.ram_utilization_idx + 1) % 10
        
        # Calculate average RAM utilization over history
        avg_ram_utils = np.mean(self.ram_utilization_history, axis=1)
        
        self.last_request_type = request_type
        
        # Factor in server utilization and capacity when making decisions
        # We create a modified Q-value that favors less utilized servers with higher capacity
        modified_q_values = self.q_values[request_type].copy()
        
        # Get RAM requirement for this request type
        ram_requirement = RequestType.get_ram_requirement(RequestType(request_type + 1))
        
        # Adjust for server utilization (penalize highly utilized servers)
        for i in range(self.num_servers):
            # Normalize server utilization to [0,1]
            cpu_util_factor = (server_utils[i] - 1) / 9.0  # server_utils is 1-10
            
            # Calculate throughput potential based on capacity and current utilization
            throughput_potential = self.capacity_estimates[i] * (1 - cpu_util_factor)
            
            # Calculate processing time estimate for this request type on this server
            proc_time_estimate = self.processing_times[request_type, i]
            
            # Check if server has enough RAM capacity for this request
            available_ram = 1.0 - avg_ram_utils[i]
            ram_sufficient = available_ram >= ram_requirement
            
            # Combined score balancing throughput and latency
            # Higher values are better
            if self.throughput_weight > 0:
                modified_q_values[i] *= (1 - self.throughput_weight)  # Latency component
                modified_q_values[i] += self.throughput_weight * throughput_potential  # Throughput component
            
            # Calculate combined resource utilization score
            resource_score = (
                self.cpu_weight * (1 - cpu_util_factor) +
                self.ram_weight * (1 - avg_ram_utils[i])
            )
            
            # Apply resource utilization penalties
            if cpu_util_factor > 0.9:  # Server is near CPU capacity
                modified_q_values[i] *= 0.5  # Significant penalty
                
            if not ram_sufficient:
                modified_q_values[i] *= 0.3  # Even larger penalty for insufficient RAM
                
            if avg_ram_utils[i] > 0.9:  # Server is near RAM capacity
                modified_q_values[i] *= 0.7  # Moderate penalty
            
            # Apply resource score to Q-value
            modified_q_values[i] *= resource_score
        
        # Select action based on strategy using the modified Q-values
        if self.strategy == EnhancedBanditStrategy.EPSILON_GREEDY:
            action = self._epsilon_greedy(request_type, modified_q_values)
        elif self.strategy == EnhancedBanditStrategy.UCB:
            action = self._ucb(request_type, modified_q_values)
        elif self.strategy == EnhancedBanditStrategy.THOMPSON_SAMPLING:
            action = self._thompson_sampling(request_type, modified_q_values)
        else:
            action = np.random.randint(0, self.num_servers)
        
        self.last_action = action
        return action
    
    def _epsilon_greedy(self, request_type, modified_q_values=None):
        """Epsilon-greedy action selection."""
        if modified_q_values is None:
            modified_q_values = self.q_values[request_type]
            
        if np.random.random() < self.epsilon:
            # Explore: random action
            return np.random.randint(0, self.num_servers)
        else:
            # Exploit: best action
            return np.argmax(modified_q_values)
    
    def _ucb(self, request_type, modified_q_values=None):
        """Upper Confidence Bound action selection."""
        if modified_q_values is None:
            modified_q_values = self.q_values[request_type]
            
        # Total number of actions for this request type
        total_count = np.sum(self.action_counts[request_type]) + 1
        
        # Avoid division by zero by adding 1
        counts = self.action_counts[request_type] + 1
        
        # UCB formula: Q(a) + c * sqrt(log(total_count) / count(a))
        exploration = self.ucb_c * np.sqrt(np.log(total_count) / counts)
        ucb_values = modified_q_values + exploration
        
        return np.argmax(ucb_values)
    
    def _thompson_sampling(self, request_type, modified_q_values=None):
        """Thompson Sampling action selection."""
        # Sample from Beta distribution for each arm
        samples = np.random.beta(
            self.success_counts[request_type],
            self.failure_counts[request_type]
        )
        
        # If we have modified Q-values, adjust the samples
        if modified_q_values is not None:
            # Normalize both to combine
            norm_q = (modified_q_values - np.min(modified_q_values))
            if np.max(norm_q) > 0:
                norm_q = norm_q / np.max(norm_q)
                
            norm_samples = (samples - np.min(samples))
            if np.max(norm_samples) > 0:
                norm_samples = norm_samples / np.max(norm_samples)
            
            # Weighted combination
            combined = 0.7 * norm_samples + 0.3 * norm_q
            return np.argmax(combined)
        
        return np.argmax(samples)
    
    def update(self, observation, action, reward, next_observation, done):
        """
        Update Q-values and counts based on observed reward.
        
        Args:
            observation: State before action
            action: The action taken
            reward: Reward received
            next_observation: State after action
            done: Whether the episode is done
        """
        if self.last_request_type is None or self.last_action is None:
            return
        
        request_type = self.last_request_type
        server = action
        
        # Extract throughput info from next_observation
        throughput = next_observation.get('throughput', 0)
        latency = next_observation.get('latency', 0)
        
        # Update throughput metrics
        info = next_observation.get('info', {})
        if 'completed' in info and server < len(self.server_completions):
            # Track completions and processing time
            self.server_completions[server] += info.get('completed', 0)
            self.server_processing_times[server] += info.get('processing_time', 0.1)
            
            # Update capacity estimate
            self.capacity_estimates[server] = (
                self.server_completions[server] / self.server_processing_times[server]
            )
        
        # Update processing time estimates for this request type and server
        if latency > 0:
            # Running average of observed latencies
            self.processing_times[request_type, server] = (
                0.95 * self.processing_times[request_type, server] + 0.05 * latency
            )
        
        # Modified reward that weights throughput more heavily
        modified_reward = reward
        if 'throughput' in info and 'latency' in info:
            # Normalize throughput and latency
            norm_throughput = info['throughput'] / (info['throughput'] + 1)  # Higher is better
            norm_latency = 1 / (1 + info['latency'])  # Transformed so higher is better
            
            # Combined reward
            modified_reward = (
                self.throughput_weight * norm_throughput + 
                (1 - self.throughput_weight) * norm_latency
            )
        
        # Update Q-value with running average
        self.q_values[request_type, server] += self.alpha * (modified_reward - self.q_values[request_type, server])
        
        # Update counts for UCB
        self.action_counts[request_type, server] += 1
        
        # Update counts for Thompson Sampling
        # For this domain, we convert the reward to a success/failure observation
        if reward > 0:
            self.success_counts[request_type, server] += reward
        else:
            self.failure_counts[request_type, server] += abs(reward)
    
    def reset(self):
        """Reset the agent's state for a new episode."""
        # Keep Q-values and counts as they are learned across episodes
        self.last_request_type = None
        self.last_action = None
        
        # Reset throughput tracking
        self.server_completions = np.zeros(self.num_servers)
        self.server_processing_times = np.zeros(self.num_servers) + 1e-6
        
        # Reset RAM utilization history
        self.ram_utilization_history = np.zeros((self.num_servers, 10))
        self.ram_utilization_idx = 0 
