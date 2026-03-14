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

BASE_DIR = os.path.dirname(__file__)
PLOT_DIR = os.path.join(BASE_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)


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


class AdamOptimizer(optim.Adam):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        super().__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                state["step"] = torch.zeros((), dtype=torch.float32)
                state["exp_avg"] = torch.zeros_like(p.data)
                state["exp_avg_sq"] = torch.zeros_like(p.data)
                state["step"].share_memory_()
                state["exp_avg"].share_memory_()
                state["exp_avg_sq"].share_memory_()

class Transition:
    def __init__(self, obs, action, reward, resultingObs, done):
        self.obs = obs
        self.action = action
        self.reward = reward
        self.resultingObs = resultingObs
        self.done = done


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


class ParallelDrivingAgent:
    def __init__(
        self,
        action_dim: int, 
        agent_id: int,
        thread_lock,
        init_exploration_rate: float,
        exploration_rate_decay: float,
        min_exploration_rate: float,
        primary_network: ConvQNetwork,
        target_network: ConvQNetwork,
        optimizer: AdamOptimizer,
        future_reward_discount_factor: float = 0.95,
        target_update_rate: int = 200,
    ):
        self.agent_id = agent_id
        self.action_dim = action_dim
        self.future_reward_discount_factor = future_reward_discount_factor

        self.exploration_rate = init_exploration_rate
        self.exploration_rate_decay = exploration_rate_decay
        self.min_exploration_rate = min_exploration_rate

        self.target_update_rate = target_update_rate
        self.update_counter = 0

        self.thread_lock = thread_lock

        self.primary_network = primary_network
        self.target_network = target_network
        self.target_network.eval()

        self.device = next(self.primary_network.parameters()).device

        self.optimizer = optimizer
        self.loss_fn = nn.SmoothL1Loss()

        self.training_error = []

        if self.agent_id == 0:
            with self.thread_lock:
                self.sync_networks()

    def obs_to_tensor(self, obs):
        return torch.tensor(np.asarray(obs, dtype=np.float32), device=self.device).unsqueeze(0) / 255.0

    def select_action(self, obs=None):
        if (obs is None) or (random.random() < self.exploration_rate):
            return random.randint(0, self.action_dim - 1)
        with torch.no_grad():
            obs_t = self.obs_to_tensor(obs)
            q_values = self.primary_network(obs_t)
            return int(torch.argmax(q_values, dim=1).item())

    def calculate_online_q(self, obs, action):
        obs_t = self.obs_to_tensor(obs)
        q_values = self.primary_network(obs_t)
        return q_values[0, int(action)]

    def sync_networks(self):
        self.target_network.load_state_dict(self.primary_network.state_dict())

    def calculate_target_q(self, transition):
        if transition.done:
            return torch.tensor(float(transition.reward), dtype=torch.float32, device=self.device)

        with torch.no_grad():
            next_obs_t = self.obs_to_tensor(transition.resultingObs)
            q_next_all = self.target_network(next_obs_t)
            max_next_q = torch.max(q_next_all, dim=1).values[0]
            return float(transition.reward) + self.future_reward_discount_factor * max_next_q

    def compute_loss(self, transition):
        q_pred = self.calculate_online_q(transition.obs, transition.action)
        y = self.calculate_target_q(transition)
        return self.loss_fn(q_pred, y)

    def decay_exploration(self):
        self.exploration_rate = max(
            self.min_exploration_rate,
            self.exploration_rate - self.exploration_rate_decay,
        )

    def optimize_primary_network(self, transition):
        loss = self.compute_loss(transition)

        with self.thread_lock:
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.primary_network.parameters(), max_norm=10.0)
            self.optimizer.step()

        if self.agent_id == 0:
            self.training_error.append(float(loss.item()))
        return float(loss.item())

    def process_step(self, obs, action, reward, resultingObs, done):
        self.update_counter += 1
        loss_val = self.optimize_primary_network(Transition(obs, action, reward, resultingObs, done))

        if (self.update_counter % self.target_update_rate == 0) and (self.agent_id == 0):
            with self.thread_lock:
                self.sync_networks()

        return loss_val


def execute_parallel_worker(
    worker_id: int,
    primary_nn: ConvQNetwork,
    target_nn: ConvQNetwork,
    shared_opt: AdamOptimizer,
    thread_lock,
    global_episode_counter,
    max_total_episodes: int,
    random_seed: int,
    starting_eps: float,
    eps_decay_step: float,
    minimum_eps: float,
    discount_factor: float,
    target_sync_rate: int,
    max_steps_per_episode: int,
    log_directory: str,
    shared_rewards_list, 
    shared_loss_list, 
    shared_epsilon_list
):
    random.seed(random_seed + 1000 * worker_id)
    np.random.seed(random_seed + 1000 * worker_id)
    torch.manual_seed(random_seed + 1000 * worker_id)

    base_env = gym.make("CarRacing-v2", render_mode="rgb_array")
    env = SkipFrame(base_env, skip=4)
    env = ImagePreprocessingWrapper(env)
    
    driving_actions = [
        np.array([-1, 0, 0], dtype=np.float32),
        np.array([1, 0, 0], dtype=np.float32),
        np.array([0, 1, 0], dtype=np.float32),
        np.array([0, 0, 0.8], dtype=np.float32),
        np.array([0, 0, 0], dtype=np.float32),
    ]

    agent = ParallelDrivingAgent(
        action_dim=len(driving_actions),
        agent_id=worker_id,
        thread_lock=thread_lock,
        init_exploration_rate=starting_eps,
        exploration_rate_decay=eps_decay_step,
        min_exploration_rate=minimum_eps,
        primary_network=primary_nn,
        target_network=target_nn,
        optimizer=shared_opt,
        future_reward_discount_factor=discount_factor,
        target_update_rate=target_sync_rate,
    )

    tensorboard_writer = SummaryWriter(log_dir=log_directory) if worker_id == 0 else None

    local_episode = 0
    recent_scores = []

    while True:
        with global_episode_counter.get_lock():
            if global_episode_counter.value >= max_total_episodes:
                break

        obs, _ = env.reset(seed=random_seed + worker_id + local_episode)
        episode_return = 0.0
        steps_survived = 0
        final_loss = None

        for _ in range(max_steps_per_episode):
            action_index = agent.select_action(obs)
            chosen_action = driving_actions[action_index]
            
            next_obs, reward, terminated, truncated, _ = env.step(chosen_action)
            is_done = terminated or truncated

            clipped_reward = np.clip(reward, -1, 1)
            final_loss = agent.process_step(obs, action_index, clipped_reward, next_obs, is_done)

            episode_return += reward
            steps_survived += 1
            obs = next_obs

            if is_done:
                break

        agent.decay_exploration()
        local_episode += 1

        with global_episode_counter.get_lock():
            global_episode_counter.value += 1
            current_global_episode = global_episode_counter.value  

        if worker_id == 0:
            recent_scores.append(episode_return)
            if len(recent_scores) > 20:
                recent_scores.pop(0)
            average_20 = sum(recent_scores) / len(recent_scores)

            shared_rewards_list.append(episode_return)
            shared_epsilon_list.append(agent.exploration_rate)
            if final_loss is not None:
                shared_loss_list.append(final_loss)

            tensorboard_writer.add_scalar("episode/return", episode_return, local_episode)
            tensorboard_writer.add_scalar("episode/return_avg20", average_20, local_episode)
            tensorboard_writer.add_scalar("episode/length", steps_survived, local_episode)
            tensorboard_writer.add_scalar("episode/epsilon", agent.exploration_rate, local_episode)

            if final_loss is not None:
                tensorboard_writer.add_scalar("train/loss", final_loss, current_global_episode)

            with torch.no_grad():
                total_norm_squared = 0.0
                for param in agent.primary_network.parameters():
                    total_norm_squared += param.data.norm(2).item() ** 2
                parameter_l2_norm = total_norm_squared ** 0.5
            tensorboard_writer.add_scalar("model/primary_net_param_l2", parameter_l2_norm, local_episode)

            if local_episode % 10 == 0:
                print(
                    f"[Worker 0] global_ep={current_global_episode} local_ep={local_episode} return={episode_return:.1f} "
                    f"avg20={average_20:.1f} steps={steps_survived} eps={agent.exploration_rate:.3f} "
                    f"loss={(final_loss if final_loss is not None else float('nan')):.4f}"
                )

    env.close()
    if tensorboard_writer is not None:
        tensorboard_writer.flush()
        tensorboard_writer.close()


def evaluate_trained_agent(model, action_dim, num_test_episodes=5):
    print(f"\n--- Starting Testing Phase ({num_test_episodes} Episodes) ---")
    env = gym.make("CarRacing-v2", render_mode="human")
    env = SkipFrame(env, skip=4)
    env = ImagePreprocessingWrapper(env)
    
    computation_device = next(model.parameters()).device
    driving_actions = [
        np.array([-1, 0, 0], dtype=np.float32),
        np.array([1, 0, 0], dtype=np.float32),
        np.array([0, 1, 0], dtype=np.float32),
        np.array([0, 0, 0.8], dtype=np.float32),
        np.array([0, 0, 0], dtype=np.float32),
    ]

    for ep in range(num_test_episodes):
        obs, _ = env.reset()
        total_score = 0.0
        is_done = False
        
        while not is_done:
            obs_tensor = torch.tensor(np.asarray(obs, dtype=np.float32), device=computation_device).unsqueeze(0) / 255.0
            
            with torch.no_grad():
                q_values = model(obs_tensor)
                action_index = int(torch.argmax(q_values, dim=1).item())
                
            chosen_action = driving_actions[action_index]
            obs, reward, terminated, truncated, _ = env.step(chosen_action)
            
            total_score += reward
            is_done = terminated or truncated
            
        print(f"Test Episode {ep+1}: Total Score = {total_score:.1f}")
        
    env.close()


def main():
    mp.set_start_method("spawn", force=True)
    torch.set_num_threads(1)

    total_action_dim = 5

    primary_network = ConvQNetwork(total_action_dim)
    target_network = ConvQNetwork(total_action_dim)

    primary_network.share_memory()
    target_network.share_memory()

    target_network.load_state_dict(primary_network.state_dict())
    target_network.eval()

    shared_optimizer = AdamOptimizer(primary_network.parameters(), lr=1e-4)

    thread_lock = mp.Lock()
    global_episode_counter = mp.Value("i", 0)

    manager = mp.Manager()
    reward_history = manager.list()
    loss_history = manager.list()
    epsilon_history = manager.list()

    num_cpu_workers = 8
    max_total_episodes = 6000 
    max_steps_per_episode = 1000
    random_seed = 42
    discount_factor = 0.99

    starting_eps = 1.0
    minimum_eps = 0.05
    eps_decay_step = (starting_eps - minimum_eps) / 5000.0

    target_sync_rate = 1000

    log_directory = "tensorboard_logs"

    active_processes = []
    print(f"Spawning {num_cpu_workers} Parallel Q-Learning workers")
    for worker_id in range(num_cpu_workers):
        p = mp.Process(
            target=execute_parallel_worker,
            args=(
                worker_id,
                primary_network,
                target_network,
                shared_optimizer,
                thread_lock,
                global_episode_counter,
                max_total_episodes,
                random_seed,
                starting_eps,
                eps_decay_step,
                minimum_eps,
                discount_factor,
                target_sync_rate,
                max_steps_per_episode,
                log_directory,
                reward_history, 
                loss_history, 
                epsilon_history
            ),
        )
        p.start()
        active_processes.append(p)

    for p in active_processes:
        p.join()
        
    print("Parallel Training Complete")

    evaluate_trained_agent(primary_network, total_action_dim, num_test_episodes=5)

    print("\nGenerating Plots")
    
    plt.figure()
    plt.plot(list(reward_history))
    plt.title("Episode Reward (Parallel Worker 0)")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.savefig(os.path.join(PLOT_DIR, "reward_curve_task7.png"))

    plt.figure()
    plt.plot(list(loss_history))
    plt.title("Training Loss (Parallel Worker 0)")
    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    plt.savefig(os.path.join(PLOT_DIR, "loss_curve_task7.png"))

    plt.figure()
    plt.plot(list(epsilon_history))
    plt.title("Exploration Rate (Parallel Worker 0)")
    plt.xlabel("Episode")
    plt.ylabel("Epsilon")
    plt.savefig(os.path.join(PLOT_DIR, "exploration_curve_task7.png"))

    window = 50
    if len(reward_history) >= window:
        moving_avg = np.convolve(list(reward_history), np.ones(window)/window, mode='valid')
        plt.figure()
        plt.plot(moving_avg)
        plt.title("Moving Average Reward (Parallel Worker 0)")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.savefig(os.path.join(PLOT_DIR, "moving_average_reward_task7.png"))

    print("Task 7 complete. Plots successfully saved.")

if __name__ == "__main__":
    main()