# src/agents/least_loaded.py
import numpy as np
from .base_agent import BaseAgent

class LeastLoadedAgent(BaseAgent):
    """
    Improved Least Loaded load balancing agent.
    Routes requests based on server utilization and estimated server capabilities.
    """
    
    def __init__(self, num_servers, **kwargs):
        super(LeastLoadedAgent, self).__init__(num_servers, **kwargs)
        
        # Define default server speeds if not available directly
        # In a real implementation, these would be passed or learned
        self.server_cpu_speeds = kwargs.get('server_cpu_speeds', [1.0] * num_servers)
        
    def select_action(self, observation):
        """
        Select the server with the lowest estimated load considering server capabilities.
        
        Args:
            observation: Environment observation
            
        Returns:
            int: Selected server index
        """
        # Server utilization is quantized from 1-10
        server_utils = observation['server_utils']
        
        # Get request type (assuming one-hot encoding)
        request_type_onehot = observation['request_type']
        request_type_idx = np.argmax(request_type_onehot)
        
        # Define base processing times for different request types
        # These would typically come from RequestType.get_processing_time
        base_processing_times = [0.8, 3.0, 2.5, 1.2, 5.0]  # Values from the code
        base_time = base_processing_times[request_type_idx]
        
        # Calculate normalized load considering server speed
        estimated_times = []
        for i in range(self.num_servers):
            # Convert quantized utilization (1-10) back to 0-1 range
            utilization = (server_utils[i] - 1) / 9.0
            
            # Calculate estimated completion time
            # Small epsilon to avoid division by zero
            est_time = base_time / (self.server_cpu_speeds[i] * (1.0 - utilization + 1e-6))
            estimated_times.append(est_time)
        
        # Return server with minimum estimated completion time
        return np.argmin(estimated_times)
    
    def update(self, observation, action, reward, next_observation, done):
        """
        Update agent state (no-op for Least Loaded).
        
        Args:
            observation: State before action
            action: The action taken
            reward: Reward received
            next_observation: State after action
            done: Whether the episode is done
        """
        pass  # No need to update anything
    
    def reset(self):
        """Reset agent state."""
        pass  # Nothing to reset