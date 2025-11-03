from typing import Tuple, Any
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from boolrl.replay_buffer import ReplayBuffer

class QNetwork(nn.Module):
    def __init__(self, state_size: int, action_size: int, seed: int, hidden_size: int):
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    def __init__(self, state_size: int, action_size: int, seed: int, hidden_size: int = 64, learning_rate: float = 5e-4, gamma: float = 0.99, tau: float = 1e-3, buffer_size: int = int(1e5), batch_size: int = 64, update_every: int = 4):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.tau = tau
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.update_every = update_every

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.qnet_policy = QNetwork(state_size, action_size, seed, hidden_size).to(self.device)
        self.qnet_target = QNetwork(state_size, action_size, seed, hidden_size).to(self.device)
        self.qnet_target.load_state_dict(self.qnet_policy.state_dict())
        self.qnet_target.eval()

        self.optimizer = optim.Adam(self.qnet_policy.parameters(), lr=self.learning_rate)
        self.memory = ReplayBuffer(buffer_size, batch_size, seed, self.device)

        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.t_step = 0

    def act(self, state: np.ndarray) -> int:
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)

        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.qnet_policy.eval()
        with torch.no_grad():
            qval_tensor = self.qnet_policy(state_tensor)
        self.qnet_policy.train()

        return qval_tensor.argmax(1).item()

    def learn(self, experiences: list) -> None:
        states = torch.from_numpy(np.vstack([e.state for e in experiences])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences]).astype(np.uint8)).float().to(self.device)

        with torch.no_grad():
            q_targets_next = self.qnet_target(next_states).detach().max(1)[0].unsqueeze(1)
            q_targets = rewards + (self.gamma * q_targets_next * (1 - dones))

        q_expected = self.qnet_policy(states).gather(1, actions)

        loss = F.mse_loss(q_expected, q_targets)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.qnet_policy.parameters(), 1.0)
        self.optimizer.step()

        self.soft_update(self.qnet_policy, self.qnet_target, self.tau)

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def soft_update(self, policy_model: nn.Module, target_model: nn.Module, tau: float) -> None:
        for target_param, policy_param in zip(target_model.parameters(), policy_model.parameters()):
            target_param.data.copy_(tau * policy_param.data + (1.0 - tau) * target_param.data)

    def step(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        self.memory.add(state, action, reward, next_state, done)
        self.t_step += 1
        if self.t_step % self.update_every == 0 and len(self.memory) >= self.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences)