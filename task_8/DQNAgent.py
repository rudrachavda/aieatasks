# DQN Agent from Task 6 with slight modifications
# Contains the neural network architecture, replay memory, and Double DQN learning logic.

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Same Transition Class from Task 6
class Transition:
    def __init__(self, state, action, reward, next_state, done):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.done = done

# Same CNN from Task 6
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

# Same Agent with minor refinements
class DQNAgent:
    def __init__(
        self, 
        env: gym.Env, 
        
        action_dim: int,       
        writer,
        
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
        
        self.writer = writer
        
        self.retrain_frequency = retrain_frequency
        self.batch_size = batch_size
        self.target_sync_rate = target_sync_rate

        self.experience_buffer = []
        self.max_memory_size = max_memory_size
        self.training_error = []
        self.step_counter = 0

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_dim = action_dim

        self.online_net = ConvQNetwork(self.action_dim).to(self.device)
        self.target_net = ConvQNetwork(self.action_dim).to(self.device)

        self.sync_networks()
        self.target_net.eval()

        # Refinements: Adam Optimizer & Huber Loss
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=self.lr)
        self.loss_fn = nn.SmoothL1Loss() 

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

        # Refinement: Double DQN Target Calculation
        with torch.no_grad():
            next_actions = self.online_net(next_states).argmax(1)
            tar_action_values = self.target_net(next_states)
            max_next_q = tar_action_values.gather(1, next_actions.unsqueeze(1)).squeeze()
            # y = r + gamma * max_a' Q(s',a')
            y_q = rewards + self.gamma * max_next_q * (1 - dones)

        loss = self.loss_fn(q_pred, y_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.training_error.append(float(loss.item()))
        self.writer.add_scalar("train/loss", loss.item(), global_step)

    def step(self, state, action, reward, next_state, done, global_step, is_warmup=False):
        if state is not None: 
            self.memorize(state, action, reward, next_state, done)

        # Skip network updates if we are just warming up the buffer
        if not is_warmup:
            self.step_counter += 1

            if self.step_counter % self.retrain_frequency == 0:
                self.train_network(global_step)

            if self.step_counter % self.target_sync_rate == 0:
                self.sync_networks()