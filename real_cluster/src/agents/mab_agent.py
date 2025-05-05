# src/agents/mab_agent.py
import numpy as np
from enum import Enum, auto
from .base_agent import BaseAgent
from ..environment.request import QueryType

class BanditStrategy(Enum):
    """Enum for different multi-armed bandit strategies."""
    EPSILON_GREEDY = auto()
    UCB = auto()
    THOMPSON_SAMPLING = auto()

class MultiArmedBanditAgent(BaseAgent):
    """
    Multi-Armed Bandit agent for request routing.
    Learns to route requests based on observed rewards.
    """

    def __init__(
        self,
        num_servers,
        strategy=BanditStrategy.EPSILON_GREEDY,
        epsilon=0.1,
        alpha=0.1,
        ucb_c=2.0,
        num_request_types=len(QueryType),
        **kwargs
    ):
        """
        Initialize the Multi-Armed Bandit agent.
        
        Args:
            num_servers: Number of servers to route requests to
            strategy: BanditStrategy to use
            epsilon: Exploration rate for epsilon-greedy
            alpha: Learning rate
            ucb_c: Exploration coefficient for UCB
            num_request_types: Number of different request types
        """
        super(MultiArmedBanditAgent, self).__init__(num_servers, **kwargs)
        self.trainable = True

        self.strategy = strategy
        self.epsilon = epsilon
        self.alpha = alpha
        self.ucb_c = ucb_c
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

    def select_action(self, observation):
        """
        Select a server using the configured bandit strategy.
        
        Args:
            observation: Environment observation
            
        Returns:
            int: Selected server index
        """
        # Extract request type from observation
        request_type_onehot = observation['request_type']
        request_type = np.argmax(request_type_onehot)

        self.last_request_type = request_type

        # Select action based on strategy
        if self.strategy == BanditStrategy.EPSILON_GREEDY:
            action = self._epsilon_greedy(request_type)
        elif self.strategy == BanditStrategy.UCB:
            action = self._ucb(request_type)
        elif self.strategy == BanditStrategy.THOMPSON_SAMPLING:
            action = self._thompson_sampling(request_type)
        else:
            action = np.random.randint(0, self.num_servers)

        self.last_action = action
        return action

    def _epsilon_greedy(self, request_type):
        """Epsilon-greedy action selection."""
        if np.random.random() < self.epsilon:
            # Explore: random action
            return np.random.randint(0, self.num_servers)
        else:
            # Exploit: best action
            return np.argmax(self.q_values[request_type])

    def _ucb(self, request_type):
        """Upper Confidence Bound action selection."""
        total_count = np.sum(self.action_counts[request_type])
        counts = self.action_counts[request_type]

        ucb_values = np.zeros_like(self.q_values[request_type])
        for i in range(self.num_servers):
            if counts[i] == 0:
                ucb_values[i] = float("inf")  # Force exploration
            else:
                bonus = self.ucb_c * np.sqrt(np.log(total_count + 1) / counts[i])
                ucb_values[i] = self.q_values[request_type, i] + bonus

        max_ucb = np.max(ucb_values)
        return np.random.choice(np.flatnonzero(ucb_values == max_ucb))

    def _thompson_sampling(self, request_type):
        """Thompson Sampling action selection."""
        # Sample from Beta distribution for each arm
        samples = np.random.beta(
            self.success_counts[request_type],
            self.failure_counts[request_type]
        )

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

        # Update Q-value with running average
        self.q_values[request_type, server] += self.alpha * (reward - self.q_values[request_type, server])

        # Update counts for UCB
        self.action_counts[request_type, server] += 1

        # Update counts for Thompson Sampling
        # For this domain, we convert the reward to a success/failure observation
        if reward > 0:
            self.success_counts[request_type, server] += reward
        else:
            self.failure_counts[request_type, server] += abs(reward)

    def batch_update(self, data: list[tuple[int, int, float]]):
        """
        Batch update from delayed training data.

        Args:
            data: List of (request_type, action, reward) tuples
        """
        for req_type, action, reward in data:
            self.q_values[req_type, action] += self.alpha * (
                reward - self.q_values[req_type, action]
            )
            self.action_counts[req_type, action] += 1

            if reward > 0:
                self.success_counts[req_type, action] += reward
            else:
                self.failure_counts[req_type, action] += abs(reward)

    def reset(self):
        """Reset the agent's state for a new episode."""
        # No need to reset Q-values or counts as they are learned across episodes
        self.last_request_type = None
        self.last_action = None


# # Simple linear model for contextual bandit updates
# class LinearModel:
#     """Simple linear model for contextual bandit."""

#     def __init__(self, input_dim, lr=0.01):
#         self.weights = np.zeros(input_dim)
#         self.lr = lr

#     def predict(self, x):
#         return np.dot(self.weights, x)

#     def update(self, x, reward):
#         error = reward - self.predict(x)
#         self.weights += self.lr * error * x


# class ContextualBanditAgent(BaseAgent):
#     """
#     Contextual Bandit agent for request routing using context features.
#     """

#     def __init__(self, num_servers, num_request_types, lr=0.01, epsilon=0.1, **kwargs):
#         super(ContextualBanditAgent, self).__init__(num_servers, **kwargs)
#         self.num_servers = num_servers
#         self.num_request_types = num_request_types
#         self.epsilon = epsilon
#         # input dimension: one-hot request type + cpu + ram
#         self.input_dim = num_request_types + 2
#         # one linear model per server
#         self.models = [LinearModel(self.input_dim, lr) for _ in range(num_servers)]
#     def _build_context(self, observation, server_idx):
#         # Extract request type one-hot and resource utilizations
#         cpu_util = observation["server_utils"][server_idx, 0]
#         ram_util = observation["server_utils"][server_idx, 1]
#         request_type_onehot = observation["request_type"]
#         return np.concatenate([request_type_onehot, [cpu_util, ram_util]])

#     def select_action(self, observation):
#         # Build context for each server and predict reward
#         contexts = [
#             self._build_context(observation, i) for i in range(self.num_servers)
#         ]
#         estimates = [model.predict(ctx) for model, ctx in zip(self.models, contexts)]
#         # Epsilon-greedy selection
#         if np.random.random() < self.epsilon:
#             return np.random.randint(0, self.num_servers)
#         return int(np.argmax(estimates))

#     def update(self, observation, action, reward, next_observation=None, done=None):
#         # Update the selected model with observed reward
#         context = self._build_context(observation, action)
#         self.models[action].update(context, reward)

#     def reset(self):
#         """Reset any per-episode state if needed (none for contextual)."""
#         pass
