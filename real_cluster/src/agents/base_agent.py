# src/agents/base_agent.py

class BaseAgent:
    """
    Base class for request routing agents.
    All routing agents should inherit from this class.
    """

    def __init__(self, num_servers, **kwargs):
        """
        Initialize the agent.
        
        Args:
            num_servers: Number of servers to route requests to
            **kwargs: Additional agent-specific parameters
        """
        self.num_servers = num_servers

    def select_action(self, observation):
        """
        Select a server to route the current request to.
        
        Args:
            observation: Dictionary with environment observation
            
        Returns:
            int: Index of the selected server
        """
        raise NotImplementedError("Subclasses must implement select_action method")

    def update(self, observation, action, reward, next_observation, done):
        """
        Update the agent's internal state based on the observed transition.
        
        Args:
            observation: State before action
            action: The action taken
            reward: Reward received
            next_observation: State after action
            done: Whether the episode is done
        """
        raise NotImplementedError("Subclasses must implement update method")

    def batch_update(self, data: list[tuple[int, int, float]]):
        """
        Batch update the agent's internal state based on a list of transitions.

        Args:
            data: List of (request_type, action, reward) tuples
        """

        raise NotImplementedError("Subclasses must implement batch_update method")

    def reset(self):
        """Reset the agent's internal state."""
        raise NotImplementedError("Subclasses must implement reset method")
