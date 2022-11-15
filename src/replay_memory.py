import torch as T
from collections import deque
from typing import TypedDict, List, Deque, Literal
import random


class Replay(TypedDict):
    state: T.Tensor
    new_state: T.Tensor
    reward: float


class ReplayMemory:
    def __init__(self, capacity: int):
        self.memory: Deque[Replay] = deque([], maxlen=capacity)

    def store(self, state: T.Tensor, new_state: T.Tensor, reward: float):
        self.memory.append({'state': state, 'new_state': new_state, 'reward': reward})

    def get_batch(self, batch_size: int) -> List[Replay]:
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class StratifiedReplayMemory:
    def __init__(self, reward_capacity: int, no_reward_capacity: int):
        self.reward_memory = deque([], maxlen=reward_capacity)
        self.no_reward_memory = deque([], maxlen=no_reward_capacity)

    def store(self, state: T.Tensor, new_state: T.Tensor, reward: float):
        replay = {'state': state, 'new_state': new_state, 'reward': reward}
        if reward > 0:
            self.reward_memory.append(replay)
        else:
            self.no_reward_memory.append(replay)

    def full(self) -> bool:
        return len(self.reward_memory) == self.reward_memory.maxlen and len(self.no_reward_memory) == self.no_reward_memory.maxlen

    def get_batch(self, reward_batch_size, no_reward_batch_size):
        batch = random.sample(self.reward_memory, reward_batch_size) + random.sample(self.no_reward_memory, no_reward_batch_size)
        random.shuffle(batch)
        return batch


def combine_batch_states(batch: List[Replay], attribute: Literal["state", "new_state"]) -> T.Tensor:
    tensors = T.stack([replay[attribute] for replay in batch])
    tensors = T.unsqueeze(tensors, dim=1)
    return tensors


def get_batch_states(batch: List[Replay]) -> T.Tensor:
    return combine_batch_states(batch, 'state')


def get_batch_new_states(batch: List[Replay]) -> T.Tensor:
    return combine_batch_states(batch, 'new_state')


def get_batch_rewards(batch: List[Replay]) -> T.Tensor:
    return T.tensor([replay['reward'] for replay in batch])