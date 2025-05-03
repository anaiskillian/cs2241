# src/agents/round_robin.py
from .base_agent import BaseAgent

class RoundRobinAgent(BaseAgent):
    """
    Round Robin load balancing agent.
    Simply cycles through servers sequentially.
    """
    
    def __init__(self, num_servers, **kwargs):
        super(RoundRobinAgent, self).__init__(num_servers, **kwargs)
        self.current_server = -1
    
    def select_action(self, observation):
        """
        Select the next server in the rotation.
        
        Args:
            observation: Environment observation (unused)
            
        Returns:
            int: Selected server index
        """
        self.current_server = (self.current_server + 1) % self.num_servers
        return self.current_server
    
    def update(self, observation, action, reward, next_observation, done):
        """
        Update agent state (no-op for Round Robin).
        
        Args:
            observation: State before action
            action: The action taken
            reward: Reward received
            next_observation: State after action
            done: Whether the episode is done
        """
        pass  # No need to update anything for Round Robin
    
    def reset(self):
        """Reset agent state."""
        self.current_server = -1
