# Structure:
# environment
# agent
# replay memory
# training loop
# evaluation metrics
# plots


import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import deque
from torch.utils.tensorboard import SummaryWriter
import os

os.makedirs("plots", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

writer = SummaryWriter("tensorboard_logs")



# Discretized Action Space

actions = [
    np.array([-1,0,0], dtype=np.float32),
    np.array([1,0,0], dtype=np.float32),
    np.array([0,1,0], dtype=np.float32),
    np.array([0,0,0.8], dtype=np.float32),
    np.array([0,0,0], dtype=np.float32),
]



# Transition Object

class Transition:

    def __init__(self, state, action, reward, next_state, done):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.done = done



# Q Network

class QNetwork(nn.Module):

    def __init__(self, state_dim, action_dim):
        super().__init__()

        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(state_dim,512),
            nn.ReLU(),
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Linear(256,action_dim)
        )

    def forward(self,x):
        return self.model(x)



# DQN Agent

class DQNAgent:

    def __init__(self, env):

        self.env = env

        self.state_dim = 96*96*3
        self.action_dim = len(actions)

        self.gamma = 0.99

        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.05

        self.batch_size = 32

        self.memory = deque(maxlen=50000)

        self.online_net = QNetwork(self.state_dim, self.action_dim).to(device)
        self.target_net = QNetwork(self.state_dim, self.action_dim).to(device)

        self.sync_networks()

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=1e-4)
        self.loss_fn = nn.SmoothL1Loss()

        self.loss_history = []


    def sync_networks(self):
        self.target_net.load_state_dict(self.online_net.state_dict())


    def store(self, state, action, reward, next_state, done):

        transition = Transition(state, action, reward, next_state, done)
        self.memory.append(transition)


    def select_action(self, state):

        if random.random() < self.epsilon:
            return random.randint(0,self.action_dim-1)

        state_t = torch.tensor(state).float().unsqueeze(0).to(device)

        with torch.no_grad():
            q_values = self.online_net(state_t)

        return int(torch.argmax(q_values).item())


    def train(self, global_step):

        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory,self.batch_size)

        states = np.array([t.state for t in batch])
        actions_b = np.array([t.action for t in batch])
        rewards = np.array([t.reward for t in batch])
        next_states = np.array([t.next_state for t in batch])
        dones = np.array([t.done for t in batch])

        states = torch.tensor(states).float().to(device)
        next_states = torch.tensor(next_states).float().to(device)
        rewards = torch.tensor(rewards).float().to(device)
        dones = torch.tensor(dones).float().to(device)
        actions_b = torch.tensor(actions_b).to(device)

        q_values = self.online_net(states).gather(1, actions_b.unsqueeze(1)).squeeze()

        next_q = self.target_net(next_states).max(1)[0]

        target = rewards + self.gamma * next_q * (1 - dones)

        loss = self.loss_fn(q_values, target.detach())

        self.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(),10.0)

        self.optimizer.step()

        self.loss_history.append(loss.item())

        writer.add_scalar("train/loss", loss.item(), global_step)


    def decay_epsilon(self):
        self.epsilon = max(self.epsilon*self.epsilon_decay,self.epsilon_min)




# Training Loop

env = gym.make("CarRacing-v3", render_mode="rgb_array")

agent = DQNAgent(env)

episodes = 150

reward_history = []
epsilon_history = []

global_step = 0
recent_rewards = []

for episode in range(episodes):

    state,_ = env.reset()

    done = False
    total_reward = 0
    episode_steps = 0

    while not done:

        action_index = agent.select_action(state)
        action = actions[action_index]

        next_state,reward,terminated,truncated,_ = env.step(action)

        done = terminated or truncated

        agent.store(state,action_index,reward,next_state,done)

        agent.train(global_step)

        state = next_state

        total_reward += reward
        episode_steps += 1
        global_step += 1

    reward_history.append(total_reward)

    agent.decay_epsilon()

    epsilon_history.append(agent.epsilon)

    if episode % 5 == 0:
        agent.sync_networks()

    recent_rewards.append(total_reward)
    if len(recent_rewards) > 10:
        recent_rewards.pop(0)

    avg_reward = np.mean(recent_rewards)

    writer.add_scalar("episode/reward", total_reward, episode)
    writer.add_scalar("episode/length", episode_steps, episode)
    writer.add_scalar("episode/epsilon", agent.epsilon, episode)
    writer.add_scalar("episode/reward_avg10", avg_reward, episode)

    print("Episode:",episode,"Reward:",total_reward,"Epsilon:",agent.epsilon)

env.close()

writer.close()



# Plotting

plt.figure()
plt.plot(reward_history)
plt.title("Episode Reward")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.savefig("plots/reward_curve.png")

plt.figure()
plt.plot(agent.loss_history)
plt.title("Training Loss")
plt.xlabel("Training Step")
plt.ylabel("Loss")
plt.savefig("plots/loss_curve.png")

plt.figure()
plt.plot(epsilon_history)
plt.title("Exploration Rate")
plt.xlabel("Episode")
plt.ylabel("Epsilon")
plt.savefig("plots/exploration_curve.png")

window = 5
moving_avg = np.convolve(reward_history, np.ones(window)/window, mode='valid')

plt.figure()
plt.plot(moving_avg)
plt.title("Moving Average Reward")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.savefig("plots/moving_average_reward.png")

print("Training complete.")
print("Plots saved in /plots")
print("TensorBoard logs saved in /tensorboard_logs")