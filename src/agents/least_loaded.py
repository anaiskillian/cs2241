# src/agents/least_loaded.py
import numpy as np
from .base_agent import BaseAgent

class LeastLoadedAgent(BaseAgent):
    """
    Least Loaded agent for request routing.
    Routes requests to the server with the lowest current utilization.
    """
    
    def __init__(self, num_servers, **kwargs):
        """
        Initialize the Least Loaded agent.
        
        Args:
            num_servers: Number of servers to route requests to
        """
        super(LeastLoadedAgent, self).__init__(num_servers, **kwargs)
        self.action_history = []
        
        # Define default server speeds if not available directly
        # In a real implementation, these would be passed or learned
        self.server_cpu_speeds = kwargs.get('server_cpu_speeds', [1.0] * num_servers)
        
    def select_action(self, observation):
        """
        Select the least loaded server.
        
        Args:
            observation: Environment observation
            
        Returns:
            int: Selected server index
        """
        # Extract server utilization from observation
        server_utils = observation['server_utils']
        
        # Select server with lowest utilization
        action = np.argmin(server_utils)
        
        # Track action history
        self.action_history.append(action)
        
        return action
    
    def get_action_history(self):
        """Get the history of actions taken by the agent."""
        return self.action_history
    
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
        """Reset the agent's state for a new episode."""
        self.action_history = []
