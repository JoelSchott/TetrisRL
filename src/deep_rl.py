import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from typing import Tuple, Union

from tetris import Tetris
import replay_memory
from replay_memory import ReplayMemory, StratifiedReplayMemory, board_to_tensor

ROWS = 20
COLS = 10


class DeepQNetwork(nn.Module):
    def __init__(self):
        super(DeepQNetwork, self).__init__()

        self.criterion = nn.MSELoss()

        self.conv1 = nn.Conv2d(1, 16, 5)
        self.conv2 = nn.Conv2d(16, 32, 4)
        self.conv3 = nn.Conv2d(32, 32, 2)

        self.fc1 = nn.Linear(768, 512)
        self.fc2 = nn.Linear(512, 1)

    def forward(self, states):
        x = F.relu(self.conv1(states))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        # flatten x
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def test(Q_estimation: DeepQNetwork) -> Tuple[Tetris, float]:
    game = Tetris(ROWS, COLS)
    total_reward = 0
    while not game.is_done():
        actions = game.get_actions()
        actions = T.tensor(actions, dtype=T.float).unsqueeze(dim=1)
        action_values = Q_estimation(actions)
        action = T.argmax(action_values)
        reward = game.take_action(action)
        total_reward += reward
    return game, total_reward


def train(n_episodes: int, epsilon_max: float, epsilon_min: float, epsilon_decay: float, reset_steps: int,
          batch_size: Union[int, Tuple[int, int]], gamma: float, memory: Union[ReplayMemory, StratifiedReplayMemory], lr: float = 0.01) -> DeepQNetwork:
    Q_estimation = DeepQNetwork()
    optimizer = optim.RMSprop(Q_estimation.parameters(), lr=lr)
    Q_target = DeepQNetwork()
    Q_target.load_state_dict(Q_estimation.state_dict())
    total_steps = 0
    for i in range(n_episodes):
        total_steps = train_episode(Q_estimation, optimizer, Q_target, memory, epsilon_max, epsilon_min, epsilon_decay,
                                    gamma, reset_steps, batch_size, total_steps)
        if i % 10 == 0:
            game, total_reward = test(Q_estimation)
            print(f'Total reward after training episode {i}: {total_reward}')
    return Q_estimation


def train_episode(Q_estimation: DeepQNetwork, optimizer, Q_target: DeepQNetwork, memory: Union[ReplayMemory, StratifiedReplayMemory], epsilon_max: float,
                  epsilon_min: float, epsilon_decay: float, gamma: float, reset_steps: int, batch_size: Union[int, Tuple[int, int]], total_steps: int) -> int:
    game = Tetris(ROWS, COLS)  # create a new game of tetris
    while not game.is_done():
        epsilon = epsilon_min + (epsilon_max - epsilon_min) * (epsilon_decay ** total_steps)
        state = board_to_tensor(game.get_state())  # get the current state of the game
        actions = game.get_actions()  # all the possible future states of the game
        # epsilon greedy policy
        if random.random() < epsilon:
            # pick a random action
            action = random.randrange(len(actions))
        else:
            # expand dimensions since each state has one channel
            actions = board_to_tensor(actions).unsqueeze(dim=1)
            # calculate the value of each of the future states
            action_values = Q_estimation(actions)
            # pick the action with the greatest value
            action = T.argmax(action_values)
        # get the immediate reward
        reward = game.take_action(action)
        # store the observation in memory
        memory.store(state, board_to_tensor(game.get_state()), reward)
        # continue to the next action if there is not enough states in memory for a batch
        if type(batch_size) == int:
            if memory.can_sample(batch_size):
                replay_batch = memory.get_batch(batch_size)
            else:
                continue
        else:
            if memory.can_sample(batch_size[0], batch_size[1]):
                replay_batch = memory.get_batch(batch_size[0], batch_size[1])
            else:
                continue
        # find the states, rewards, and future states from the memory
        states = replay_memory.get_batch_states(replay_batch)
        new_states = replay_memory.get_batch_new_states(replay_batch)
        rewards = replay_memory.get_batch_rewards(replay_batch)
        # update the estimator network using the batch
        optimizer.zero_grad()
        predictions = Q_estimation(states).squeeze()
        targets = gamma * Q_target(new_states).squeeze() + rewards
        loss = Q_estimation.criterion(predictions, targets)
        loss.backward()
        optimizer.step()
        # if it is time to update the target network, copy over the parameters
        total_steps += 1
        if total_steps % reset_steps == 0:
            Q_target.load_state_dict(Q_estimation.state_dict())
    return total_steps
