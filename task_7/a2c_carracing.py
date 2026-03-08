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

writer = SummaryWriter(LOG_DIR)



# CNN Actor-Critic Network (fix for vision input)

class ActorCritic(nn.Module):

    def __init__(self, obs_shape, action_dim):

        super().__init__()

        c, h, w = obs_shape

        self.encoder = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        with torch.no_grad():
            sample = torch.zeros(1, c, h, w)
            enc_out = self.encoder(sample)
            flat_size = enc_out.view(1, -1).shape[1]

        self.fc = nn.Sequential(
            nn.Linear(flat_size, 512),
            nn.ReLU()
        )

        self.policy = nn.Linear(512, action_dim)
        self.value = nn.Linear(512, 1)

    def forward(self, x):

        x = self.encoder(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        logits = self.policy(x)
        value = self.value(x)

        return logits, value



# A2C Agent

class A2CAgent:

    def __init__(self, env):

        self.env = env

        obs_shape = env.observation_space.shape  # (96,96,3)

        # Convert to channel-first
        self.obs_shape = (obs_shape[2], obs_shape[0], obs_shape[1])

        # Internal discrete action set
        self.actions = [
            np.array([-1,0,0],dtype=np.float32),
            np.array([1,0,0],dtype=np.float32),
            np.array([0,1,0],dtype=np.float32),
            np.array([0,0,0.8],dtype=np.float32),
            np.array([0,0,0],dtype=np.float32)
        ]

        self.action_dim = len(self.actions)

        self.gamma = 0.99
        self.entropy_beta = 0.01

        self.model = ActorCritic(self.obs_shape, self.action_dim).to(device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=3e-4)


    def preprocess(self, obs):

        obs = np.asarray(obs, dtype=np.float32) / 255.0

        obs = np.transpose(obs, (2,0,1))  # HWC -> CHW

        return torch.tensor(obs).unsqueeze(0).to(device)


    def select_action(self, obs):

        obs_t = self.preprocess(obs)

        logits, value = self.model(obs_t)

        probs = torch.softmax(logits, dim=1)

        dist = torch.distributions.Categorical(probs)

        action = dist.sample()

        return action.item(), dist.log_prob(action), dist.entropy(), value


    def compute_returns(self, rewards, dones, last_value):

        R = last_value

        returns = []

        for step in reversed(range(len(rewards))):

            done_val = float(dones[step])

            R = rewards[step] + self.gamma * R * (1.0 - done_val)

            returns.insert(0, R)

        return torch.tensor(returns, dtype=torch.float32, device=device)


    def update(self, log_probs, values, rewards, dones, entropies, next_state):

        next_state_t = self.preprocess(next_state)

        _, next_value = self.model(next_state_t)

        returns = self.compute_returns(
            rewards,
            dones,
            next_value.detach().item()
        )

        values = torch.cat(values).squeeze()

        advantage = returns - values

        actor_loss = -(torch.stack(log_probs) * advantage.detach()).mean()

        critic_loss = advantage.pow(2).mean()

        entropy_loss = -torch.stack(entropies).mean()

        loss = actor_loss + 0.5 * critic_loss + self.entropy_beta * entropy_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item(), actor_loss.item(), critic_loss.item()



# Training Loop

env = gym.make("CarRacing-v3", render_mode="rgb_array")

agent = A2CAgent(env)

episodes = 150

reward_history = []
recent_rewards = []


for episode in range(episodes):

    obs,_ = env.reset()

    done = False

    log_probs=[]
    values=[]
    rewards=[]
    dones=[]
    entropies=[]

    total_reward=0
    steps=0

    while not done:

        action_idx,log_prob,entropy,value = agent.select_action(obs)

        action = agent.actions[action_idx]

        next_obs,reward,terminated,truncated,_ = env.step(action)

        done = terminated or truncated

        log_probs.append(log_prob)
        values.append(value)
        rewards.append(reward)
        dones.append(bool(done))
        entropies.append(entropy)

        obs = next_obs

        total_reward += reward
        steps += 1


    loss,actor_loss,critic_loss = agent.update(
        log_probs,
        values,
        rewards,
        dones,
        entropies,
        obs
    )

    reward_history.append(total_reward)

    recent_rewards.append(total_reward)
    if len(recent_rewards) > 10:
        recent_rewards.pop(0)

    avg10 = np.mean(recent_rewards)

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