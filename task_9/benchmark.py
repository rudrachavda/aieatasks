# One-click solution to train and compare multiple RL algorithms 
# (DQN and Asynchronous Q-Learning) in the CarRacing-v2 environment.

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import gymnasium as gym
import cv2
from collections import deque
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

# Environment Wrapper from Task 8
class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip=4):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        for _ in range(self._skip):
            state, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        return state, total_reward, terminated, truncated, info


# Image Preprocessing from Task 6
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
        grayscale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(grayscale, (84, 84), interpolation=cv2.INTER_AREA)
        return resized


# Transition class from Task 6
class Transition:
    def __init__(self, state, action, reward, next_state, done):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.done = done


# CNN Architecture from Task 6 & 8
class ConvQNetwork(nn.Module):
    def __init__(self, action_dim):
        super().__init__()
        self.net = nn.Sequential(
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
        return self.net(input)


# Actions
car_actions = [
    np.array([-1, 0, 0], dtype=np.float32),   # steer left
    np.array([1, 0, 0], dtype=np.float32),    # steer right
    np.array([0, 1, 0], dtype=np.float32),    # accelerate
    np.array([0, 0, 0.8], dtype=np.float32),  # brake
    np.array([0, 0, 0], dtype=np.float32),    # no action
]


# ALGORITHM 1: DQN (FROM TASK 8)

class DQNAgent:
    def __init__(self, action_dim, lr, gamma=0.99):
        self.gamma = gamma
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = (1.0 - 0.1) / 200 
        
        self.action_dim = action_dim
        self.batch_size = 32
        self.experience_buffer = []
        self.max_memory_size = 20000
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.online_net = ConvQNetwork(action_dim).to(self.device)
        self.target_net = ConvQNetwork(action_dim).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss() 

    def memorize(self, state, action, reward, next_state, done):
        if len(self.experience_buffer) >= self.max_memory_size:
            self.experience_buffer.pop(0)
        self.experience_buffer.append(Transition(state, action, reward, next_state, done))

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        with torch.no_grad():
            state_t = torch.tensor(np.asarray(state, dtype=np.float32), device=self.device).unsqueeze(0) / 255.0
            q_values = self.online_net(state_t)
            return int(torch.argmax(q_values, dim=1).item())

    def train_network(self):
        if len(self.experience_buffer) < self.batch_size:
            return 0.0

        batch = random.sample(self.experience_buffer, self.batch_size)
        states = torch.tensor(np.array([tr.state for tr in batch])).float().to(self.device) / 255.0
        next_states = torch.tensor(np.array([tr.next_state for tr in batch])).float().to(self.device) / 255.0
        actions = torch.tensor(np.array([tr.action for tr in batch])).to(self.device)
        rewards = torch.tensor(np.array([tr.reward for tr in batch])).float().to(self.device)
        dones = torch.tensor(np.array([tr.done for tr in batch])).float().to(self.device)

        q_pred = self.online_net(states).gather(1, actions.unsqueeze(1)).squeeze()

        with torch.no_grad():
            next_actions = self.online_net(next_states).argmax(1)
            tar_action_values = self.target_net(next_states)
            max_next_q = tar_action_values.gather(1, next_actions.unsqueeze(1)).squeeze()
            y_q = rewards + self.gamma * max_next_q * (1 - dones)

        loss = self.loss_fn(q_pred, y_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()


def run_carRacing(episodes=250):
    print("\n--- Starting DQN Benchmark (Task 8 Algorithm) ---")
    env = gym.make("CarRacing-v2", render_mode="rgb_array")
    env = SkipFrame(env, skip=4)
    env = ImagePreprocessingWrapper(env)
    
    agent = DQNAgent(action_dim=len(car_actions), lr=0.0001)
    
    reward_history = []
    loss_history = []
    epsilon_history = []
    
    # Initialize TensorBoard Writer for DQN
    writer = SummaryWriter(log_dir="tensorboard_logs/DQN")
    
    step_counter = 0
    for ep in range(episodes):
        state, _ = env.reset()
        total_score = 0
        ep_loss = 0.0
        
        for t in range(500): 
            action_idx = agent.choose_action(state)
            next_state, reward, terminated, truncated, _ = env.step(car_actions[action_idx])
            done = terminated or truncated
            
            clipped_reward = np.clip(reward, -1, 1)
            agent.memorize(state, action_idx, clipped_reward, next_state, done)
            
            step_counter += 1
            if step_counter % 2 == 0:
                current_loss = agent.train_network()
                if current_loss != 0.0:
                    ep_loss = current_loss 
                    
            if step_counter % 1000 == 0:
                agent.target_net.load_state_dict(agent.online_net.state_dict())
                
            total_score += reward
            state = next_state
            if done: break
            
        agent.epsilon = max(agent.epsilon_min, agent.epsilon - agent.epsilon_decay)
        
        reward_history.append(total_score)
        loss_history.append(ep_loss)
        epsilon_history.append(agent.epsilon)
        
        # Log to TensorBoard
        writer.add_scalar("Benchmark/Episode_Reward", total_score, ep)
        writer.add_scalar("Benchmark/Training_Loss", ep_loss, ep)
        writer.add_scalar("Benchmark/Epsilon", agent.epsilon, ep)
        
        if (ep+1) % 10 == 0:
            print(f"[DQN] Episode {ep+1}/{episodes} | Score: {total_score:.1f} | Loss: {ep_loss:.4f} | Epsilon: {agent.epsilon:.2f}")
            
    env.close()
    writer.close()
    return reward_history, loss_history, epsilon_history


# ALGORITHM 2: ASYNCHRONOUS Q-LEARNING (FROM TASK 7)

class AdamOptimizer(optim.Adam):
    def __init__(self, params, lr=1e-3):
        super().__init__(params, lr=lr)
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                state["step"] = torch.zeros((), dtype=torch.float32)
                state["exp_avg"] = torch.zeros_like(p.data)
                state["exp_avg_sq"] = torch.zeros_like(p.data)
                state["step"].share_memory_()
                state["exp_avg"].share_memory_()
                state["exp_avg_sq"].share_memory_()


class ParallelDrivingAgent:
    def __init__(self, action_dim, lock, online_nn, target_nn, optimizer):
        self.action_dim = action_dim
        self.lock = lock
        self.q_online_net = online_nn
        self.q_target_net = target_nn
        self.optimizer = optimizer
        self.loss_fn = nn.SmoothL1Loss()
        self.gamma = 0.99
        self.device = torch.device("cpu")

    def getAction(self, obs, epsilon):
        if random.random() < epsilon:
            return random.randint(0, self.action_dim - 1)
        with torch.no_grad():
            obs_t = torch.tensor(np.asarray(obs, dtype=np.float32)).unsqueeze(0) / 255.0
            q_values = self.q_online_net(obs_t)
            return int(torch.argmax(q_values, dim=1).item())

    def update(self, obs, action, reward, next_obs, done):
        obs_t = torch.tensor(np.asarray(obs, dtype=np.float32)).unsqueeze(0) / 255.0
        q_pred = self.q_online_net(obs_t)[0, action]
        
        if done:
            y = torch.tensor(float(reward), dtype=torch.float32)
        else:
            with torch.no_grad():
                next_obs_t = torch.tensor(np.asarray(next_obs, dtype=np.float32)).unsqueeze(0) / 255.0
                max_next_q = torch.max(self.q_target_net(next_obs_t), dim=1).values[0]
                y = float(reward) + self.gamma * max_next_q

        loss = self.loss_fn(q_pred, y)
        with self.lock:
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.q_online_net.parameters(), 10.0)
            self.optimizer.step()
            
        return loss.item()


def execute_parallel_worker(wid, online_nn, target_nn, optimizer, lock, global_ep, max_eps, shared_rewards, shared_losses, shared_epsilons):
    env = gym.make("CarRacing-v2", render_mode="rgb_array")
    env = SkipFrame(env, skip=4)
    env = ImagePreprocessingWrapper(env)
    
    agent = ParallelDrivingAgent(len(car_actions), lock, online_nn, target_nn, optimizer)
    epsilon = 1.0
    epsilon_decay = (1.0 - 0.1) / 200
    
    # Initialize TensorBoard Writer for AsyncQ (Only on worker 0 to prevent conflicts)
    writer = SummaryWriter(log_dir="tensorboard_logs/AsyncQ") if wid == 0 else None
    
    while True:
        with global_ep.get_lock():
            if global_ep.value >= max_eps:
                break
            current_ep = global_ep.value
            global_ep.value += 1
            
        obs, _ = env.reset()
        total_reward = 0
        ep_loss = 0.0
        
        for _ in range(500):
            action_idx = agent.getAction(obs, epsilon)
            next_obs, reward, terminated, truncated, _ = env.step(car_actions[action_idx])
            done = terminated or truncated
            
            clipped_reward = np.clip(reward, -1, 1)
            step_loss = agent.update(obs, action_idx, clipped_reward, next_obs, done)
            ep_loss = step_loss
            
            total_reward += reward
            obs = next_obs
            if done: break
            
        epsilon = max(0.1, epsilon - epsilon_decay)
        
        shared_rewards.append((current_ep, total_reward))
        shared_losses.append((current_ep, ep_loss))
        shared_epsilons.append((current_ep, epsilon))
        
        if wid == 0:
            writer.add_scalar("Benchmark/Episode_Reward", total_reward, current_ep)
            writer.add_scalar("Benchmark/Training_Loss", ep_loss, current_ep)
            writer.add_scalar("Benchmark/Epsilon", epsilon, current_ep)
            if current_ep % 10 == 0:
                print(f"[Async Worker] Global Episode {current_ep}/{max_eps} | Score: {total_reward:.1f} | Loss: {ep_loss:.4f} | Epsilon: {epsilon:.2f}")

    env.close()
    if writer:
        writer.close()


def run_async_benchmark(episodes=250):
    print("\n--- Starting Async Q-Learning Benchmark (Task 7 Algorithm) ---")
    online_nn = ConvQNetwork(len(car_actions))
    target_nn = ConvQNetwork(len(car_actions))
    online_nn.share_memory()
    target_nn.share_memory()
    target_nn.load_state_dict(online_nn.state_dict())

    optimizer = AdamOptimizer(online_nn.parameters(), lr=1e-4)
    lock = mp.Lock()
    global_ep = mp.Value("i", 0)
    
    manager = mp.Manager()
    shared_rewards = manager.list()
    shared_losses = manager.list()
    shared_epsilons = manager.list()
    
    procs = []
    for wid in range(8):
        p = mp.Process(target=execute_parallel_worker, args=(wid, online_nn, target_nn, optimizer, lock, global_ep, episodes, shared_rewards, shared_losses, shared_epsilons))
        p.start()
        procs.append(p)

    for p in procs:
        p.join()
        
    sorted_rewards = sorted(list(shared_rewards), key=lambda x: x[0])
    sorted_losses = sorted(list(shared_losses), key=lambda x: x[0])
    sorted_epsilons = sorted(list(shared_epsilons), key=lambda x: x[0])
    
    return [r[1] for r in sorted_rewards], [l[1] for l in sorted_losses], [e[1] for e in sorted_epsilons]


# EXECUTION & PLOTTING

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    
    # 1. Run Benchmarks
    TOTAL_BENCHMARK_EPISODES = 250
    
    dqn_rewards, dqn_losses, dqn_epsilons = run_carRacing(TOTAL_BENCHMARK_EPISODES)
    async_rewards, async_losses, async_epsilons = run_async_benchmark(TOTAL_BENCHMARK_EPISODES)
    
    print("\nBenchmarks Complete! Generating Comparison Graphs...")
    
    # 2. Smooth the data (Moving Average)
    def moving_average(data, window=10):
        if len(data) < window:
            return data
        return np.convolve(data, np.ones(window)/window, mode='valid')
        
    dqn_smooth = moving_average(dqn_rewards)
    async_smooth = moving_average(async_rewards)
    
    os.makedirs("plots", exist_ok=True)
    
    # Plot 1: Raw Episode Reward Comparison
    plt.figure(figsize=(10, 6))
    plt.plot(dqn_rewards, label="DQN (Memory Buffer)", color='blue', alpha=0.6)
    plt.plot(async_rewards, label="Async Q-Learning", color='orange', alpha=0.6)
    plt.title("Algorithm Benchmark: Raw Episode Reward")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join("plots", "task9_benchmark_raw_reward.png"))
    
    # Plot 2: Moving Average Reward Comparison
    plt.figure(figsize=(10, 6))
    plt.plot(dqn_smooth, label="DQN (Smoothed)", color='blue')
    plt.plot(async_smooth, label="Async Q-Learning (Smoothed)", color='orange')
    plt.title("Algorithm Benchmark: Moving Average Reward")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join("plots", "task9_benchmark_moving_avg.png"))

    # Plot 3: Training Loss Comparison
    plt.figure(figsize=(10, 6))
    plt.plot(dqn_losses, label="DQN Loss", color='blue', alpha=0.7)
    plt.plot(async_losses, label="Async Q-Learning Loss", color='orange', alpha=0.7)
    plt.title("Algorithm Benchmark: Training Loss")
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join("plots", "task9_benchmark_loss.png"))

    # Plot 4: Exploration Rate (Epsilon) Comparison
    plt.figure(figsize=(10, 6))
    plt.plot(dqn_epsilons, label="DQN Epsilon", color='blue', linewidth=2)
    plt.plot(async_epsilons, label="Async Q-Learning Epsilon", color='orange', linestyle='dashed', linewidth=2)
    plt.title("Algorithm Benchmark: Exploration Rate Decay")
    plt.xlabel("Episode")
    plt.ylabel("Epsilon")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join("plots", "task9_benchmark_epsilon.png"))
    
    print("All 4 benchmark comparison graphs successfully saved to the /plots directory.")