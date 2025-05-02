# src/agents/least_loaded.py
import numpy as np
from .base_agent import BaseAgent

class LeastLoadedAgent(BaseAgent):
    """
    Least Loaded load balancing agent.
    Routes requests to the server with the lowest current load.
    """
    
    def __init__(self, num_servers, **kwargs):
        super(LeastLoadedAgent, self).__init__(num_servers, **kwargs)
    
    def select_action(self, observation):
        """
        Select the server with the lowest load.
        
        Args:
            observation: Environment observation
            
        Returns:
            int: Selected server index
        """
        # Server utilization is quantized from 1-10
        server_utils = observation['server_utils']
        
        # Return the index of the server with minimum utilization
        # If there are multiple servers with the same minimum utilization,
        # numpy's argmin returns the first one
        return np.argmin(server_utils)
    
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
        pass  # No need to update anything for Least Loaded
    
    def reset(self):
        """Reset agent state."""
        pass  # Nothing to reset for Least Loaded agent
