import gym
import numpy as np
import os
import random
import time

import torch
from torch import nn


def main():
    class DQN(nn.Module):
        def __init__(self, observation_size: int, number_of_actions: int, hidden_size: int = 16):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(observation_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, number_of_actions)
            )

        def forward(self, x: torch.Tensor):
            return self.net(x.float())


    Experience = namedtuple(
        'Experience', field_names=['state', 'action', 'reward', 'done', 'new_state'])


    class ReplayBuffer:
        def __init__(self, capacity: int) -> None:
            self.buffer = deque(maxlen=capacity)

        def __len__(self) -> int:
            return len(self.buffer)

        def append(self, experience: Experience) -> None:
            self.buffer.append(experience)

        def sample(self, batch_size: int) -> Tuple:
            indices = np.random.choice(len(self.buffer), batch_size, replace=False)
            states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in indices])
            return (np.array(states), np.array(actions), np.array(rewards, dtype=np.float32),
                    np.array(dones, dtype=np.bool), np.array(next_states))


if __name__ == '__main__':
    main()
