from typing import Any, List, Tuple
import random
from collections import deque, namedtuple

class ReplayBuffer:
    def __init__(self, buffer_size: int, batch_size: int, seed: int, device: Any):
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        self.device = device

    def add(self, state: Any, action: int, reward: float, next_state: Any, done: bool) -> None:
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self) -> List:
        return random.sample(self.memory, k=self.batch_size)

    def __len__(self) -> int:
        return len(self.memory)