# src/agents/random_agent.py
import numpy as np
from .base_agent import BaseAgent

class RandomAgent(BaseAgent):
    """
    Random load balancing agent.
    Randomly selects a server for each request.
    """
    
    def __init__(self, num_servers, **kwargs):
        super(RandomAgent, self).__init__(num_servers, **kwargs)
    
    def select_action(self, observation):
        """
        Select a random server.
        
        Args:
            observation: Environment observation (unused)
            
        Returns:
            int: Selected server index
        """
        return np.random.randint(0, self.num_servers)
    
    def update(self, observation, action, reward, next_observation, done):
        """
        Update agent state (no-op for Random).
        
        Args:
            observation: State before action
            action: The action taken
            reward: Reward received
            next_observation: State after action
            done: Whether the episode is done
        """
        pass  # No need to update anything for Random agent
    
    def reset(self):
        """Reset agent state."""
        pass  # Nothing to reset for Random agent
