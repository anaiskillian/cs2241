# src/agents/ppo_agent.py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import collections

from .base_agent import BaseAgent

class PPONetwork(nn.Module):
    """
    Neural network for the PPO agent with both actor and critic heads.
    """
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(PPONetwork, self).__init__()
        
        # Shared feature extraction layers
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Policy head (actor)
        self.policy = nn.Sequential(
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Value head (critic)
        self.value = nn.Sequential(
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input state tensor
            
        Returns:
            tuple: (action_probs, state_value)
        """
        features = self.shared(x)
        action_probs = self.policy(features)
        state_value = self.value(features)
        
        return action_probs, state_value
    
    def get_action(self, state, deterministic=False):
        """
        Sample an action from the policy.
        
        Args:
            state: Input state tensor
            deterministic: If True, take the most probable action
            
        Returns:
            tuple: (action, log_prob, entropy, state_value)
        """
        action_probs, state_value = self(state)
        
        if deterministic:
            action = torch.argmax(action_probs, dim=-1)
        else:
            dist = Categorical(action_probs)
            action = dist.sample()
            
        log_prob = torch.log(action_probs + 1e-10).gather(1, action.unsqueeze(-1))
        entropy = -torch.sum(action_probs * torch.log(action_probs + 1e-10), dim=-1)
        
        return action, log_prob, entropy, state_value


class PPOAgent(BaseAgent):
    """
    Proximal Policy Optimization agent for request routing.
    Implements the PPO algorithm from Schulman et al. 2017.
    """
    
    def __init__(
        self,
        num_servers,
        state_dim,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_ratio=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        **kwargs
    ):
        super(PPOAgent, self).__init__(num_servers, **kwargs)
        
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        
        # Initialize PPO network
        self.network = PPONetwork(state_dim, num_servers).to(device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        
        # Initialize experience buffer
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        
        # Initialize trackers
        self.episode_rewards = []
        self.last_state = None
    
    def _preprocess_observation(self, observation):
        """
        Convert observation dict to tensor for network input.
        
        Args:
            observation: Environment observation
            
        Returns:
            torch.Tensor: Flattened state tensor
        """
        # Flatten the observation
        server_utils = observation['server_utils']
        latency_history = observation['latency_history']
        decision_history = observation['decision_history']
        request_type = observation['request_type']
        
        # Normalize latency history
        if np.max(latency_history) > 0:
            latency_history = latency_history / np.max(latency_history)
        
        # Normalize decision history
        if self.num_servers > 1:
            decision_history = decision_history / (self.num_servers - 1)
        
        # Combine all parts
        flat_state = np.concatenate([
            server_utils / 10.0,  # Normalize to [0,1]
            latency_history,
            decision_history,
            request_type
        ])
        
        # Convert to tensor
        state_tensor = torch.FloatTensor(flat_state).unsqueeze(0).to(self.device)
        return state_tensor
    
    def select_action(self, observation):
        """
        Select action using the current policy.
        
        Args:
            observation: Environment observation
            
        Returns:
            int: Selected server index
        """
        with torch.no_grad():
            state = self._preprocess_observation(observation)
            self.last_state = state
            
            action, log_prob, _, value = self.network.get_action(state)
            
            self.states.append(state)
            self.actions.append(action)
            self.log_probs.append(log_prob)
            self.values.append(value)
            
            return action.item()
    
    def update(self, observation, action, reward, next_observation, done):
        """
        Store experience for later PPO update.
        
        Args:
            observation: State before action
            action: The action taken
            reward: Reward received
            next_observation: State after action
            done: Whether the episode is done
        """
        # Store experience
        self.rewards.append(reward)
        self.dones.append(done)
        
        # If done, perform update using collected experiences
        if done and len(self.states) >= 1:
            self._update_policy()
    
    def _compute_returns(self):
        """
        Compute returns and advantages using Generalized Advantage Estimation (GAE).
        
        Returns:
            tuple: (returns, advantages)
        """
        # Get final value for bootstrapping
        with torch.no_grad():
            if self.last_state is not None:
                _, last_value = self.network(self.last_state)
            else:
                last_value = torch.zeros(1, 1).to(self.device)
        
        # Convert lists to tensors
        rewards = torch.FloatTensor(self.rewards).unsqueeze(1).to(self.device)
        dones = torch.FloatTensor(self.dones).unsqueeze(1).to(self.device)
        values = torch.cat(self.values)
        
        # Initialize returns and advantages
        returns = torch.zeros_like(rewards).to(self.device)
        advantages = torch.zeros_like(rewards).to(self.device)
        
        # Initialize for GAE
        next_value = last_value
        next_advantage = 0
        
        # Compute returns and advantages from back to front
        for t in reversed(range(len(rewards))):
            # Calculate return (discounted sum of rewards)
            next_non_terminal = 1.0 - dones[t]
            returns[t] = rewards[t] + self.gamma * next_value * next_non_terminal
            
            # Calculate advantage using GAE
            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
            advantages[t] = delta + self.gamma * self.gae_lambda * next_advantage * next_non_terminal
            
            # Update for next iteration
            next_value = values[t]
            next_advantage = advantages[t]
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return returns, advantages
    
    def _update_policy(self):
        """
        Update policy and value network using PPO algorithm.
        """
        # Compute returns and advantages
        returns, advantages = self._compute_returns()
        
        # Convert lists to tensors
        old_states = torch.cat(self.states)
        old_actions = torch.cat(self.actions).unsqueeze(1)
        old_log_probs = torch.cat(self.log_probs)
        
        # PPO update loop (typically multiple epochs, but simplified here)
        # Get current policy and value predictions
        action_probs, values = self.network(old_states)
        
        # Get log probabilities of actions
        log_probs = torch.log(action_probs + 1e-10).gather(1, old_actions)
        
        # Calculate entropy for exploration
        entropy = -torch.sum(action_probs * torch.log(action_probs + 1e-10), dim=-1).mean()
        
        # Calculate ratio for PPO clipping
        ratio = torch.exp(log_probs - old_log_probs)
        
        # Calculate surrogate objectives
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * advantages
        
        # Calculate losses
        policy_loss = -torch.min(surr1, surr2).mean()
        value_loss = nn.MSELoss()(values, returns)
        
        # Combined loss
        loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        
        # Clip gradients
        nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
        
        self.optimizer.step()
        
        # Clear buffers
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
    
    def reset(self):
        """Reset the agent's buffers."""
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        self.last_state = None
        
    def save(self, path):
        """Save the model to a file."""
        torch.save(self.network.state_dict(), path)
        
    def load(self, path):
        """Load the model from a file."""
        self.network.load_state_dict(torch.load(path, map_location=self.device))
