from typing import Union
import random
import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from torch_geometric.data.data import Data
from boolrl.gnn_models import GNNQNetwork
from boolrl.replay_buffer import ReplayBuffer

class DQNAgentGNN:
    def __init__(self,
                 gnn_input_size: int,
                 global_feature_size: int,
                 action_size: int,
                 seed: int,
                 hidden_size: int = 64,
                 learning_rate: float = 5e-4,
                 gamma: float = 0.99,
                 tau: float = 1e-3,
                 buffer_size: int = int(1e5),
                 batch_size: int = 64,
                 update_every: int = 4):

        self.action_size = action_size
        self.seed = random.seed(seed)
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.update_every = update_every
        self.learning_rate = learning_rate

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.qnet_policy = GNNQNetwork(gnn_input_size, global_feature_size, hidden_size, action_size).to(self.device)
        self.qnet_target = GNNQNetwork(gnn_input_size, global_feature_size, hidden_size, action_size).to(self.device)
        self.qnet_target.load_state_dict(self.qnet_policy.state_dict())
        self.qnet_target.eval()

        self.optimizer = torch.optim.Adam(self.qnet_policy.parameters(), lr=self.learning_rate)
        self.memory = ReplayBuffer(buffer_size, batch_size, seed, self.device)

        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.t_step = 0

    def act(self, state: Data) -> int:
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)

        state = state.to(self.device)
        self.qnet_policy.eval()
        with torch.no_grad():
            qval_tensor = self.qnet_policy(state)
        self.qnet_policy.train()

        return qval_tensor.argmax(1).item()

    def learn(self, experiences: list) -> None:
        self.optimizer.zero_grad()

        current_states_data = [e.state for e in experiences]
        next_states_data = [e.next_state for e in experiences]

        current_states_loader = DataLoader(current_states_data, batch_size=self.batch_size)
        next_states_loader = DataLoader(next_states_data, batch_size=self.batch_size)

        actions = torch.tensor([e.action for e in experiences], dtype=torch.long).to(self.device)
        rewards = torch.tensor([e.reward for e in experiences], dtype=torch.float).to(self.device)
        dones = torch.tensor([e.done for e in experiences], dtype=torch.float).to(self.device)

        for batch in current_states_loader:
            batch = batch.to(self.device)
            q_expected_all = self.qnet_policy(batch)
            q_expected = q_expected_all.gather(1, actions.unsqueeze(1))

        with torch.no_grad():
            for batch in next_states_loader:
                batch = batch.to(self.device)
                q_targets_next = self.qnet_target(batch).max(1)[0].unsqueeze(1)

            q_targets = rewards.unsqueeze(1) + (self.gamma * q_targets_next * (1 - dones.unsqueeze(1)))

        loss = F.mse_loss(q_expected, q_targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.qnet_policy.parameters(), 1.0)
        self.optimizer.step()

        self.soft_update(self.qnet_policy, self.qnet_target, self.tau)
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def soft_update(self, policy_model: torch.nn.Module, target_model: torch.nn.Module, tau: float) -> None:
        for target_param, policy_param in zip(target_model.parameters(), policy_model.parameters()):
            target_param.data.copy_(tau * policy_param.data + (1.0 - tau) * target_param.data)

    def step(self, state: Data, action: int, reward: float, next_state: Union[Data, None], done: bool) -> None:
        self.memory.add(state, action, reward, next_state, done)
        self.t_step += 1

        if (self.t_step % self.update_every == 0) and (len(self.memory) >= self.batch_size):
            experiences = self.memory.sample()
            self.learn(experiences)
            self.t_step = 0
