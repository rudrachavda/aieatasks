# From the DQN (Deep Q Network) paper I learned:
# The paper uses a convolutional neural network whose input is raw pixels.
# I need to implement a preprocessing step that converts images to grayscale and stacks the last 4 frames of history to produce the input to the Q-Function.

# Structure:
# 1. Environment (CNN Preprocessing, Actions, and Frame Stacking)
# 2. Replay Memory
# 3. Agent (Neural Network and DQN Logic)
# 4. Training Loop
# 5. Metrics & Plots

# Car Racing only gives the agent a 96 x 96 pixel video feed, so if I try to force CarRacing through a linear network without using a CNN, I have to flatten the image into a 1D line of pixels.
# This would not be good for the spatial awareness of the agent and probably cause either a capacity collapse (no reward curve), and the agent would never learn how to drive.
# Other simulation environments like Cart-Pole would give the agent a simple array of 4 numbers (cart position, cart velocity, pole angle, pole velocity), so I would need to make up for these variables.

# I took inspiration from this https://github.com/wiitt/DQN-Car-Racing to write the code for my DQN agent. 

import gymnasium as gym
import random
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from collections import deque
from torch.utils.tensorboard import SummaryWriter
import os

BASE_DIR = os.path.dirname(__file__)
PLOT_DIR = os.path.join(BASE_DIR, "plots")
LOG_DIR = os.path.join(BASE_DIR, "tensorboard_logs")
os.makedirs("plots", exist_ok=True)
writer = SummaryWriter(log_dir="tensorboard_logs")

# Actions for Car Racing Environment
car_actions = [
    np.array([-1, 0, 0], dtype=np.float32),   # steer left
    np.array([1, 0, 0], dtype=np.float32),    # steer right
    np.array([0, 1, 0], dtype=np.float32),    # accelerate
    np.array([0, 0, 0.8], dtype=np.float32),  # brake
    np.array([0, 0, 0], dtype=np.float32),    # no action
]

# Image preprocessing for Car Racing Environment
class ImagePreprocessingWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(4, 84, 84), dtype=np.uint8)
        self.frames = deque(maxlen=4)

    def reset(self, **kwargs):
        state, info = self.env.reset(**kwargs)
        processed_state = self._process_image(state)
        for _ in range(4):
            self.frames.append(processed_state)
        return np.stack(self.frames, axis=0), info

    def step(self, action):
        state, reward, terminated, truncated, info = self.env.step(action)
        processed_state = self._process_image(state)
        self.frames.append(processed_state)
        return np.stack(self.frames, axis=0), reward, terminated, truncated, info

    def _process_image(self, image):
        grayscale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) # Grayscale the images
        resized = cv2.resize(grayscale, (84, 84), interpolation=cv2.INTER_AREA)
        return resized

class Transition:
    def __init__(self, state, action, reward, next_state, done):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.done = done

class ConvQNetwork(nn.Module):
    def __init__(self, action_dim):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 9 * 9, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )

    def forward(self, input):
        return self.conv_layers(input)

class DQNAgent:
    def __init__(
        self, 
        env: gym.Env, 
        lr: float, 
        start_epsilon: float, 
        epsilon_decay: float, 
        min_epsilon: float, 
        gamma: float = 0.99, 
        retrain_frequency: int = 1, 
        batch_size: int = 32, 
        target_sync_rate: int = 2000, 
        max_memory_size: int = 50000
    ):
        
        self.gamma = gamma
        self.epsilon = start_epsilon
        self.epsilon_decay = epsilon_decay
        self.env = env 
        self.min_epsilon = min_epsilon
        self.lr = lr
        
        self.retrain_frequency = retrain_frequency
        self.batch_size = batch_size
        self.target_sync_rate = target_sync_rate

        self.experience_buffer = []
        self.max_memory_size = max_memory_size
        self.training_error = []
        self.step_counter = 0

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_dim = len(car_actions)

        self.online_net = ConvQNetwork(self.action_dim).to(self.device)
        self.target_net = ConvQNetwork(self.action_dim).to(self.device)

        self.sync_networks()
        self.target_net.eval()

        self.optimizer = optim.RMSprop(self.online_net.parameters(), lr=self.lr)
        self.loss_fn = nn.MSELoss() 

    def memorize(self, state, action, reward, next_state, done):
        if len(self.experience_buffer) >= self.max_memory_size:
            self.experience_buffer.pop(0)
        self.experience_buffer.append(Transition(state, action, reward, next_state, done))

    def state_to_tensor(self, state):
        return torch.tensor(np.asarray(state, dtype=np.float32), device=self.device).unsqueeze(0) / 255.0

    def choose_action(self, state=None):
        if (state is None) or (random.random() < self.epsilon):
            return random.randint(0, self.action_dim - 1)

        with torch.no_grad():
            state_t = self.state_to_tensor(state)
            q_values = self.online_net(state_t)
            return int(torch.argmax(q_values, dim=1).item())

    def sync_networks(self):
        self.target_net.load_state_dict(self.online_net.state_dict())
        
    def reduce_epsilon(self): # using linear epsilon decay 
        self.epsilon = max(self.min_epsilon, self.epsilon - self.epsilon_decay)

    def train_network(self, global_step):
        if len(self.experience_buffer) < self.batch_size:
            return

        batch = random.sample(self.experience_buffer, self.batch_size)
        
        # Vectorized batch processing
        states = torch.tensor(np.array([tr.state for tr in batch])).float().to(self.device) / 255.0
        next_states = torch.tensor(np.array([tr.next_state for tr in batch])).float().to(self.device) / 255.0
        actions_b = torch.tensor(np.array([tr.action for tr in batch])).to(self.device)
        rewards = torch.tensor(np.array([tr.reward for tr in batch])).float().to(self.device)
        dones = torch.tensor(np.array([tr.done for tr in batch])).float().to(self.device)

        q_pred = self.online_net(states).gather(1, actions_b.unsqueeze(1)).squeeze()

        # if the episode is NOT done, dones equals 0. (1 - 0) = 1; y_q = rewards + gamma * max_next_q
        # If the episode IS done, dones equals 1. (1 - 1) = 0. Multiplying the future reward by 0; y_q = rewards + 0.
        with torch.no_grad():
            q_next_all = self.target_net(next_states)
            max_next_q = torch.max(q_next_all, dim=1).values
            # y = r + gamma * max_a' Q(s',a')
            y_q = rewards + self.gamma * max_next_q * (1 - dones)

        loss = self.loss_fn(q_pred, y_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.training_error.append(float(loss.item()))
        writer.add_scalar("train/loss", loss.item(), global_step)

    def step(self, state, action, reward, next_state, done, global_step):
        if state is not None: 
            self.memorize(state, action, reward, next_state, done)

        self.step_counter += 1

        if self.step_counter % self.retrain_frequency == 0:
            self.train_network(global_step)

        if self.step_counter % self.target_sync_rate == 0:
            self.sync_networks()


def run_carRacing(episodes: int = 100, max_steps: int = 1000, render_mode: str = "rgb_array", is_training: bool = True):
    base_env = gym.make("CarRacing-v2", render_mode=render_mode) # Car Racing environment
    env = ImagePreprocessingWrapper(base_env)
    agent.env = env

    global_step = 0
    recent_scores = []
    
    # Tracking for matplotlib plots
    reward_history = []
    epsilon_history = []

    for ep in range(episodes):
        state, info = env.reset()
        total_score = 0.0
        steps_taken = 0

        for t in range(max_steps):
            action_idx = agent.choose_action(state)
            actual_action = car_actions[action_idx]
            
            next_state, reward, terminated, truncated, info = env.step(actual_action)
            done = terminated or truncated

            # Clip reward for stability during training
            clipped_reward = np.clip(reward, -1, 1) if is_training else reward

            if is_training:
                agent.step(state, action_idx, clipped_reward, next_state, done, global_step)

            total_score += reward
            state = next_state
            steps_taken += 1
            global_step += 1

            if done:
                break
        
        if is_training:
            agent.reduce_epsilon()

        recent_scores.append(total_score)
        if len(recent_scores) > 10:
            recent_scores.pop(0)
        avg_10 = sum(recent_scores) / len(recent_scores)

        if is_training:
            reward_history.append(total_score)
            epsilon_history.append(agent.epsilon)
            
            writer.add_scalar("episode/return", total_score, ep)
            writer.add_scalar("episode/length", steps_taken, ep)
            writer.add_scalar("episode/epsilon", agent.epsilon, ep)
            writer.add_scalar("episode/return_avg10", avg_10, ep)

        print(f"{' Test' if not is_training else ' Train'} Episode {ep+1}: steps={steps_taken}, total_score={total_score:.1f}, eps={agent.epsilon:.3f}")

    env.close()
    return reward_history, epsilon_history


# Execution Parameters
train_episodes = 100
learning_rate = 0.00025
start_epsilon = 1.0
epsilon_decay = 1.0 / train_episodes 
min_epsilon = 0.1

dummy_env = gym.make("CarRacing-v2", render_mode="rgb_array")
wrapped_dummy = ImagePreprocessingWrapper(dummy_env)

agent = DQNAgent(
    wrapped_dummy, 
    learning_rate, 
    start_epsilon, 
    epsilon_decay, 
    min_epsilon
)

# Train
print("Training")
reward_history, epsilon_history = run_carRacing(episodes=train_episodes, is_training=True)

# Test
print("\nTesting")
agent.epsilon = 0.0 # Turn off random actions
run_carRacing(episodes=5, render_mode="human", is_training=False)

writer.close()

# Plotting Results
plt.figure()
plt.plot(reward_history)
plt.title("Episode Reward")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.savefig("plots/reward_curve.png")

plt.figure()
plt.plot(agent.training_error)
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