# RL structure
# environment
# agent (actor-critic network)
# on-policy trajectory collection (no replay memory)
# training loop with advantage-based updates
# evaluation metrics (rewards, losses, episode length)
# plots (training curve of rewards)

import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = os.path.dirname(__file__)
PLOT_DIR = os.path.join(BASE_DIR, "plots")
LOG_DIR = os.path.join(BASE_DIR, "tensorboard_logs")

os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# TensorBoard writer for logging training metrics
writer = SummaryWriter(LOG_DIR)



# CNN Actor-Critic Network
# This network extracts visual features from the CarRacing image
# and outputs both a policy (action probabilities) and a value estimate.

class ActorCritic(nn.Module):

    def __init__(self, obs_shape, action_dim):

        super().__init__()

        c, h, w = obs_shape

        # Convolutional encoder to extract spatial features from the image
        self.encoder = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        # Determine the flattened feature size automatically
        # by passing a dummy tensor through the CNN
        with torch.no_grad():
            sample = torch.zeros(1, c, h, w)
            enc_out = self.encoder(sample)
            flat_size = enc_out.view(1, -1).shape[1]

        # Shared fully connected layer before splitting into policy and value heads
        self.fc = nn.Sequential(
            nn.Linear(flat_size, 512),
            nn.ReLU()
        )

        # Policy head outputs action logits
        self.policy = nn.Linear(512, action_dim)

        # Value head predicts the expected return of the current state
        self.value = nn.Linear(512, 1)

    def forward(self, x):

        # Extract visual features
        x = self.encoder(x)

        # Flatten features before passing to dense layers
        x = torch.flatten(x, 1)

        x = self.fc(x)

        # Compute policy logits and state value
        logits = self.policy(x)
        value = self.value(x)

        return logits, value



# A2C Agent
# Implements Advantage Actor-Critic for training the policy and value network.

class A2CAgent:

    def __init__(self, env):

        self.env = env

        obs_shape = env.observation_space.shape  # (96,96,3)

        # Convert observation format from HWC to CHW for PyTorch
        self.obs_shape = (obs_shape[2], obs_shape[0], obs_shape[1])

        # Discretized action space (CarRacing normally uses continuous actions)
        self.actions = [
            np.array([-1,0,0],dtype=np.float32),  # steer left
            np.array([1,0,0],dtype=np.float32),   # steer right
            np.array([0,1,0],dtype=np.float32),   # accelerate
            np.array([0,0,0.8],dtype=np.float32), # brake
            np.array([0,0,0],dtype=np.float32)    # no action
        ]

        self.action_dim = len(self.actions)

        # Discount factor
        self.gamma = 0.99

        # Entropy coefficient encourages exploration
        self.entropy_beta = 0.01

        # Actor-Critic neural network
        self.model = ActorCritic(self.obs_shape, self.action_dim).to(device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=3e-4)


    def preprocess(self, obs):

        # Normalize pixel values to [0,1]
        obs = np.asarray(obs, dtype=np.float32) / 255.0

        # Convert observation from HWC (Gym) to CHW (PyTorch)
        obs = np.transpose(obs, (2,0,1))

        return torch.tensor(obs).unsqueeze(0).to(device)


    def select_action(self, obs):

        # Convert observation to tensor
        obs_t = self.preprocess(obs)

        # Forward pass through Actor-Critic network
        logits, value = self.model(obs_t)

        # Convert logits to probabilities
        probs = torch.softmax(logits, dim=1)

        # Create categorical distribution for sampling actions
        dist = torch.distributions.Categorical(probs)

        # Sample action from policy
        action = dist.sample()

        # Return action and information needed for training
        return action.item(), dist.log_prob(action), dist.entropy(), value


    def compute_returns(self, rewards, dones, last_value):

        # Compute discounted returns using the Bellman equation
        R = last_value

        returns = []

        for step in reversed(range(len(rewards))):

            done_val = float(dones[step])

            R = rewards[step] + self.gamma * R * (1.0 - done_val)

            returns.insert(0, R)

        return torch.tensor(returns, dtype=torch.float32, device=device)


    def update(self, log_probs, values, rewards, dones, entropies, next_state):

        # Estimate value of the next state for bootstrapping
        next_state_t = self.preprocess(next_state)

        _, next_value = self.model(next_state_t)

        returns = self.compute_returns(
            rewards,
            dones,
            next_value.detach().item()
        )

        values = torch.cat(values).squeeze()

        # Advantage estimates how much better the action was compared to expected value
        advantage = returns - values

        # Policy gradient loss (actor)
        actor_loss = -(torch.stack(log_probs) * advantage.detach()).mean()

        # Value function loss (critic)
        critic_loss = advantage.pow(2).mean()

        # Entropy term encourages exploration
        entropy_loss = -torch.stack(entropies).mean()

        # Total loss combines actor, critic, and entropy regularization
        loss = actor_loss + 0.5 * critic_loss + self.entropy_beta * entropy_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item(), actor_loss.item(), critic_loss.item()



# Training Loop
# The agent interacts with the environment and updates the policy after each episode.

env = gym.make("CarRacing-v3", render_mode="rgb_array")

agent = A2CAgent(env)

episodes = 150

reward_history = []
recent_rewards = []


for episode in range(episodes):

    obs,_ = env.reset()

    done = False

    # Storage for trajectory information during the episode
    log_probs=[]
    values=[]
    rewards=[]
    dones=[]
    entropies=[]

    total_reward=0
    steps=0

    while not done:

        # Select action using the current policy
        action_idx,log_prob,entropy,value = agent.select_action(obs)

        action = agent.actions[action_idx]

        next_obs,reward,terminated,truncated,_ = env.step(action)

        done = terminated or truncated

        # Store trajectory data for training
        log_probs.append(log_prob)
        values.append(value)
        rewards.append(reward)
        dones.append(bool(done))
        entropies.append(entropy)

        obs = next_obs

        total_reward += reward
        steps += 1


    # Update actor and critic using the collected episode trajectory
    loss,actor_loss,critic_loss = agent.update(
        log_probs,
        values,
        rewards,
        dones,
        entropies,
        obs
    )

    reward_history.append(total_reward)

    # Track recent rewards for moving average performance
    recent_rewards.append(total_reward)
    if len(recent_rewards) > 10:
        recent_rewards.pop(0)

    avg10 = np.mean(recent_rewards)

    # Log metrics to TensorBoard
    writer.add_scalar("reward/episode", total_reward, episode)
    writer.add_scalar("reward/avg10", avg10, episode)
    writer.add_scalar("episode/length", steps, episode)

    writer.add_scalar("loss/total", loss, episode)
    writer.add_scalar("loss/actor", actor_loss, episode)
    writer.add_scalar("loss/critic", critic_loss, episode)

    print("Episode:",episode,"Reward:",total_reward)


env.close()
writer.close()



# Plot training curve

plt.figure()

plt.plot(reward_history)

plt.title("A2C Training Curve")

plt.xlabel("Episode")
plt.ylabel("Reward")

plt.savefig(os.path.join(PLOT_DIR,"reward_curve.png"))

print("Training complete.")
print("Plots saved in /plots")
print("TensorBoard logs saved in /tensorboard_logs")