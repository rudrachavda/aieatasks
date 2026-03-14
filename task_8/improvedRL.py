# Goals for Fine-Tuning:

# DDQN in DQNAgent.py
# We will be using a double DQN (DDQN) because standard DQN has an issue where it overestimates how "good" actions are.
# DDQN fixes this "overestimation bias" by decoupling the target calculation: we use the Online Network to choose the best next action, 
# but we use the Target Network to evaluate exactly how many points that action is worth.

# Huber Loss (SmoothL1Loss) and Adam Optimizer in DQNAgent.py
# We are swapping out Mean Squared Error (MSE) for Huber Loss. It acts like a parabola (MSE) for small errors, 
# but turns into a straight line for massive errors. This acts as a permanent safety net against exploding gradients.
# 
# We are replacing RMSProp with Adam, which adjusts the learning rate for each individual parameter and generally 
# leads to faster convergence in RL architectures.

# Frame Skipping
# The 2013 DQN research paper states that the agent sees and selects actions on every k^th frame instead of every frame, 
# and its last action is repeated on skipped frames. We are injecting a wrapper to skip 4 frames at a time, 
# allowing the agent to learn momentum and significantly speeding up training time.

# Warmup Phase
# We added a pre-training loop where the agent takes random actions to fill the replay memory, 
# so the neural network has a diverse dataset the second it begins learning.

# I took some inspiration for implementing DDQN and Frame skipping from https://github.com/wiitt/DQN-Car-Racing/blob/main/DQN_model.py.

# Refinements included: Frame Skipping, Double DQN, Huber Loss, Adam Optimizer, and Warmup Phase.

import gymnasium as gym
import numpy as np
import cv2
import matplotlib.pyplot as plt
from collections import deque
from torch.utils.tensorboard import SummaryWriter
import os

# our modular agent
from DQNAgent import DQNAgent

BASE_DIR = os.path.dirname(__file__)
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

# Refinement: Frame Skipping Wrapper
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

# Same Preprocessing function from Task 6
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

# Same carRacing function, but with frame skipping included
def run_carRacing(episodes: int = 100, max_steps: int = 1000, render_mode: str = "rgb_array", is_training: bool = True, is_warmup: bool = False):
    base_env = gym.make("CarRacing-v2", render_mode=render_mode)
    env = SkipFrame(base_env, skip=4)
    env = ImagePreprocessingWrapper(env)
    agent.env = env

    global_step = 0
    recent_scores = []
    
    reward_history = []
    epsilon_history = []

    for ep in range(episodes):
        state, info = env.reset()
        total_score = 0.0
        steps_taken = 0

        for t in range(max_steps):
            if is_warmup:
                action_idx = np.random.randint(0, agent.action_dim)
            else:
                action_idx = agent.choose_action(state)
                
            actual_action = car_actions[action_idx]
            
            next_state, reward, terminated, truncated, info = env.step(actual_action)
            done = terminated or truncated

            # Clip reward for stability during training
            clipped_reward = np.clip(reward, -1, 1) if (is_training or is_warmup) else reward

            if is_training or is_warmup:
                agent.step(state, action_idx, clipped_reward, next_state, done, global_step, is_warmup=is_warmup)

            total_score += reward
            state = next_state
            steps_taken += 1
            global_step += 1

            if done:
                break
        
        if is_training and not is_warmup:
            agent.reduce_epsilon()

        recent_scores.append(total_score)
        if len(recent_scores) > 10:
            recent_scores.pop(0)
        avg_10 = sum(recent_scores) / len(recent_scores)

        if is_training and not is_warmup:
            reward_history.append(total_score)
            epsilon_history.append(agent.epsilon)
            
            writer.add_scalar("episode/return", total_score, ep)
            writer.add_scalar("episode/length", steps_taken, ep)
            writer.add_scalar("episode/epsilon", agent.epsilon, ep)
            writer.add_scalar("episode/return_avg10", avg_10, ep)

        if not is_warmup:
            print(f"{' Test' if not is_training else ' Train'} Episode {ep+1}: steps={steps_taken}, total_score={total_score:.1f}, eps={agent.epsilon:.3f}")
        else:
            if (ep + 1) % 10 == 0:
                print(f" Warmup Episode {ep+1}/{episodes} complete.")

    env.close()
    return reward_history, epsilon_history


# Parameters
train_episodes = 1000
warmup_episodes = 50
learning_rate = 0.0001 # slightly lower learning rate
start_epsilon = 1.0
epsilon_decay = 1.0 / train_episodes 
min_epsilon = 0.1

dummy_env = gym.make("CarRacing-v2", render_mode="rgb_array")
wrapped_dummy = SkipFrame(dummy_env, skip=4)
wrapped_dummy = ImagePreprocessingWrapper(wrapped_dummy)

# agent
agent = DQNAgent(
    env=wrapped_dummy, 
    action_dim=len(car_actions), 
    writer=writer, 
    lr=learning_rate, 
    start_epsilon=start_epsilon, 
    epsilon_decay=epsilon_decay, 
    min_epsilon=min_epsilon
)

# Warmup
print("Warmup")
run_carRacing(episodes=warmup_episodes, is_training=False, is_warmup=True)

# Training
print("Training")
reward_history, epsilon_history = run_carRacing(episodes=train_episodes, is_training=True, is_warmup=False)

# Test
print("\nTesting")
agent.epsilon = 0.0 
run_carRacing(episodes=5, render_mode="human", is_training=False, is_warmup=False)

writer.close()

# Plotting Results
plt.figure()
plt.plot(reward_history)
plt.title("Episode Reward")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.savefig("plots/reward_curve_task8.png")

plt.figure()
plt.plot(agent.training_error)
plt.title("Training Loss")
plt.xlabel("Training Step")
plt.ylabel("Loss")
plt.savefig("plots/loss_curve_task8.png")

plt.figure()
plt.plot(epsilon_history)
plt.title("Exploration Rate")
plt.xlabel("Episode")
plt.ylabel("Epsilon")
plt.savefig("plots/exploration_curve_task8.png")

window = 5
moving_avg = np.convolve(reward_history, np.ones(window)/window, mode='valid')

plt.figure()
plt.plot(moving_avg)
plt.title("Moving Average Reward")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.savefig("plots/moving_average_reward_task8.png")