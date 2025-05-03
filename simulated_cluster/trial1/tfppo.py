import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import gym

# Environment setup
env = gym.make("CartPole-v1")
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# Hyperparameters
gamma = 0.99  # Discount factor
lr_actor = 0.001  # Actor learning rate
lr_critic = 0.001  # Critic learning rate
clip_ratio = 0.2  # PPO clip ratio
epochs = 10  # Number of optimization epochs
batch_size = 64  # Batch size for optimization


# Actor and Critic networks
class ActorCritic(nn.Module):
    def __init__(self, state_size, action_size):
        super(ActorCritic, self).__init__()
        self.dense1 = nn.Linear(state_size, 64)
        self.policy_logits = nn.Linear(64, action_size)
        self.dense2 = nn.Linear(state_size, 64)
        self.value = nn.Linear(64, 1)

    def forward(self, state):
        x = F.relu(self.dense1(state))
        logits = self.policy_logits(x)
        value = self.value(x)
        return logits, value


# PPO algorithm
def ppo_loss(old_logits, old_values, advantages, states, actions, returns):
    def compute_loss(logits, values, actions, returns):
        actions_onehot = F.one_hot(actions, num_classes=action_size).float()
        policy = F.softmax(logits, dim=-1)
        action_probs = torch.sum(actions_onehot * policy, dim=1)
        old_policy = F.softmax(old_logits, dim=-1)
        old_action_probs = torch.sum(actions_onehot * old_policy, dim=1)

        # Policy loss
        ratio = torch.exp(
            torch.log(action_probs + 1e-10) - torch.log(old_action_probs + 1e-10)
        )
        clipped_ratio = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio)
        policy_loss = -torch.mean(
            torch.min(ratio * advantages, clipped_ratio * advantages)
        )

        # Value loss
        value_loss = torch.mean((values - returns) ** 2)

        # Entropy bonus (optional)
        entropy_bonus = torch.mean(policy * torch.log(policy + 1e-10))

        total_loss = (
            policy_loss + 0.5 * value_loss - 0.01 * entropy_bonus
        )  # Entropy regularization
        return total_loss

    def get_advantages(returns, values):
        advantages = returns - values
        return (advantages - torch.mean(advantages)) / (torch.std(advantages) + 1e-8)

    def train_step(states, actions, returns, old_logits, old_values):
        model.train()
        optimizer.zero_grad()
        logits, values = model(states)
        loss = compute_loss(logits, values, actions, returns)
        loss.backward()
        optimizer.step()
        return loss

    advantages = get_advantages(returns, old_values)
    for _ in range(epochs):
        loss = train_step(states, actions, returns, old_logits, old_values)
    return loss


# Initialize actor-critic model and optimizer
model = ActorCritic(state_size, action_size)
optimizer = optim.Adam(model.parameters(), lr=lr_actor)

# Main training loop
max_episodes = 1000
max_steps_per_episode = 1000

for episode in range(max_episodes):
    states, actions, rewards, values, returns = [], [], [], [], []
    state = env.reset()
    print(state)
    for step in range(max_steps_per_episode):
        state = torch.tensor(state, dtype=torch.float32).view(1, -1)
        logits, value = model(state)

        # Sample action from the policy distribution
        action = torch.distributions.Categorical(logits=logits).sample().item()
        next_state, reward, done, _ = env.step(action)

        states.append(state)
        actions.append(action)
        rewards.append(reward)
        values.append(value)

        state = next_state

        if done:
            returns_batch = []
            discounted_sum = 0
            for r in rewards[::-1]:
                discounted_sum = r + gamma * discounted_sum
                returns_batch.append(discounted_sum)
            returns_batch.reverse()

            states = torch.cat(states, dim=0)
            actions = torch.tensor(actions, dtype=torch.int64)
            values = torch.cat(values, dim=0)
            returns_batch = torch.tensor(returns_batch, dtype=torch.float32)
            old_logits, _ = model(states)

            loss = ppo_loss(
                old_logits,
                values,
                returns_batch - values,
                states,
                actions,
                returns_batch,
            )
            print(f"Episode: {episode + 1}, Loss: {loss.item()}")

            break
