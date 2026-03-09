# Improved DQN Implementation
#
# I improved my previous DQN algorithm with some refinements:
# - CNN feature extractor for image observations
# - Double DQN target calculation to reduce Q-value overestimation
# - Linear epsilon decay schedule for more stable exploration

# RL Structure
# 1. Environment
# 2. Agent
# 3. Replay Memory
# 4. Training Loop
# 5. Evaluation Metrics
# 6. Plots


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

# Create directory for saving plots if it does not already exist
os.makedirs("plots", exist_ok=True)

# Automatically use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# TensorBoard logger for training metrics
writer = SummaryWriter("tensorboard_logs")



# Discretized Action Space

# CarRacing uses continuous actions: [steering, gas, brake]
# Since DQN requires discrete actions, we approximate the space
# using a small set of representative actions.
actions = [
    np.array([-1,0,0], dtype=np.float32),   # steer left
    np.array([1,0,0], dtype=np.float32),    # steer right
    np.array([0,1,0], dtype=np.float32),    # accelerate
    np.array([0,0,0.8], dtype=np.float32),  # brake
    np.array([0,0,0], dtype=np.float32),    # no action
]



# Transition Object (Replay Memory Entry)

# Stores one step of interaction with the environment
class Transition:

    def __init__(self, state, action, reward, next_state, done):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.done = done



# CNN Q-Network

# Convolutional layers extract spatial features from the image
# Fully connected layers then predict Q-values for each action.
class QNetwork(nn.Module):

    def __init__(self, input_shape, action_dim):

        super().__init__()

        c,h,w = input_shape

        # Convolutional feature extractor
        # DeepMind Atari architecture
        self.conv = nn.Sequential(
            nn.Conv2d(c,32,8,stride=4),
            nn.ReLU(),
            nn.Conv2d(32,64,4,stride=2),
            nn.ReLU(),
            nn.Conv2d(64,64,3,stride=1),
            nn.ReLU()
        )

        # Determine flattened feature size automatically
        # by passing a dummy tensor through the CNN
        with torch.no_grad():
            sample = torch.zeros(1,c,h,w)
            conv_out = self.conv(sample)
            conv_size = conv_out.view(1,-1).shape[1]

        # Fully connected layers for Q-value prediction
        self.fc = nn.Sequential(
            nn.Linear(conv_size,512),
            nn.ReLU(),
            nn.Linear(512,action_dim)
        )

    def forward(self,x):

        # Extract features with CNN
        x = self.conv(x)

        # Flatten features before feeding to FC layers
        x = torch.flatten(x,1)

        return self.fc(x)




# DQN Agent

class DQNAgent:

    def __init__(self, env):

        self.env = env

        # Observation shape from environment (96x96x3)
        obs_shape = env.observation_space.shape

        # Convert to PyTorch format (C,H,W)
        self.input_shape = (obs_shape[2], obs_shape[0], obs_shape[1])

        self.action_dim = len(actions)

        # Discount factor
        self.gamma = 0.99

        # Improved epsilon exploration schedule
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay_steps = 50000

        self.batch_size = 32

        # Experience replay buffer
        self.memory = deque(maxlen=50000)

        # Online network learns Q-values
        self.online_net = QNetwork(self.input_shape, self.action_dim).to(device)

        # Target network stabilizes training
        self.target_net = QNetwork(self.input_shape, self.action_dim).to(device)

        self.sync_networks()

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=1e-4)

        # Huber loss (SmoothL1) is commonly used in DQN
        self.loss_fn = nn.SmoothL1Loss()

        self.loss_history = []


    def preprocess(self, state):

        # Convert image to float and normalize to [0,1]
        state = np.asarray(state, dtype=np.float32)/255.0

        # Convert from HWC (Gym format) to CHW (PyTorch format)
        state = np.transpose(state,(2,0,1))

        return torch.tensor(state).unsqueeze(0).to(device)


    def sync_networks(self):

        # Copy weights from online network to target network
        self.target_net.load_state_dict(self.online_net.state_dict())


    def store(self, state, action, reward, next_state, done):

        # Save transition in replay memory
        transition = Transition(state, action, reward, next_state, done)
        self.memory.append(transition)


    def select_action(self, state, global_step):

        # Epsilon-greedy exploration
        if random.random() < self.epsilon:
            return random.randint(0,self.action_dim-1)

        state_t = self.preprocess(state)

        # Choose action with highest predicted Q-value
        with torch.no_grad():
            q_values = self.online_net(state_t)

        return int(torch.argmax(q_values).item())


    def train(self, global_step):

        # Wait until replay buffer has enough samples
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory,self.batch_size)

        # Convert batch to arrays
        states = np.array([t.state for t in batch])
        actions_b = np.array([t.action for t in batch])
        rewards = np.array([t.reward for t in batch])
        next_states = np.array([t.next_state for t in batch])
        dones = np.array([t.done for t in batch])

        # Convert to tensors and normalize images
        states = torch.tensor(states).float().permute(0,3,1,2).to(device)/255.0
        next_states = torch.tensor(next_states).float().permute(0,3,1,2).to(device)/255.0

        rewards = torch.tensor(rewards).float().to(device)
        dones = torch.tensor(dones).float().to(device)
        actions_b = torch.tensor(actions_b).to(device)


        # Current Q-values for selected actions
        q_values = self.online_net(states).gather(1, actions_b.unsqueeze(1)).squeeze()


        
        # Double DQN target calculation
        

        # Online network chooses the best next action
        next_actions = self.online_net(next_states).argmax(1)

        # Target network evaluates that action
        next_q = self.target_net(next_states).gather(
            1,
            next_actions.unsqueeze(1)
        ).squeeze()

        # Bellman target
        target = rewards + self.gamma * next_q * (1 - dones)

        loss = self.loss_fn(q_values, target.detach())

        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping helps prevent unstable updates
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(),10.0)

        self.optimizer.step()

        self.loss_history.append(loss.item())

        writer.add_scalar("train/loss", loss.item(), global_step)


    def decay_epsilon(self, global_step):

        # Linear epsilon decay over training steps
        self.epsilon = max(
            self.epsilon_min,
            1.0 - global_step / self.epsilon_decay_steps
        )




# Training Loop


env = gym.make("CarRacing-v3", render_mode="rgb_array")

agent = DQNAgent(env)

episodes = 1000

reward_history = []
epsilon_history = []

global_step = 0
recent_rewards = []

for episode in range(episodes):

    state,_ = env.reset()

    done = False
    total_reward = 0
    episode_steps = 0

    # Run one episode
    while not done:

        action_index = agent.select_action(state, global_step)
        action = actions[action_index]

        next_state,reward,terminated,truncated,_ = env.step(action)

        done = terminated or truncated

        # Store transition in replay memory
        agent.store(state,action_index,reward,next_state,done)

        # Train the Q-network
        agent.train(global_step)

        # Update exploration rate
        agent.decay_epsilon(global_step)

        state = next_state

        total_reward += reward
        episode_steps += 1
        global_step += 1

    reward_history.append(total_reward)

    epsilon_history.append(agent.epsilon)

    # Periodically update target network
    if global_step % 5000 == 0:
        agent.sync_networks()

    # Track recent performance
    recent_rewards.append(total_reward)
    if len(recent_rewards) > 10:
        recent_rewards.pop(0)

    avg_reward = np.mean(recent_rewards)

    # Log metrics to TensorBoard
    writer.add_scalar("episode/reward", total_reward, episode)
    writer.add_scalar("episode/length", episode_steps, episode)
    writer.add_scalar("episode/epsilon", agent.epsilon, episode)
    writer.add_scalar("episode/reward_avg10", avg_reward, episode)

    print("Episode:",episode,"Reward:",total_reward,"Epsilon:",agent.epsilon)

env.close()
writer.close()




# Plotting Results


# Episode reward over time
plt.figure()
plt.plot(reward_history)
plt.title("Episode Reward")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.savefig("plots/reward_curve.png")

# Training loss
plt.figure()
plt.plot(agent.loss_history)
plt.title("Training Loss")
plt.xlabel("Training Step")
plt.ylabel("Loss")
plt.savefig("plots/loss_curve.png")

# Exploration rate decay
plt.figure()
plt.plot(epsilon_history)
plt.title("Exploration Rate")
plt.xlabel("Episode")
plt.ylabel("Epsilon")
plt.savefig("plots/exploration_curve.png")

# Moving average reward to smooth noise
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