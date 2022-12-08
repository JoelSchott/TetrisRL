import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from typing import Tuple, Union, Optional, Callable
import os

from tetris import Tetris
from common import ROWS, COLS
import replay_memory
from replay_memory import ReplayMemory, StratifiedReplayMemory, board_to_tensor
import sample


class ConstantEpsilon:
    def __init__(self, epsilon: float):
        self.epsilon = epsilon

    def __call__(self, steps: int):
        return self.epsilon


class ExponentialDecayEpsilon:
    def __init__(self, epsilon_max: float, epsilon_min: float, epsilon_decay: float):
        self.epsilon_max = epsilon_max
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

    def __call__(self, steps: int):
        return self.epsilon_min + (self.epsilon_max - self.epsilon_min) * (self.epsilon_decay ** steps)


class DeepQNetworkV0(nn.Module):
    def __init__(self):
        super(DeepQNetworkV0, self).__init__()

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


class DeepQNetwork(nn.Module):
    def __init__(self):
        super(DeepQNetwork, self).__init__()

        self.criterion = nn.MSELoss()

        self.conv1 = nn.Conv2d(1, 64, (6, 6))
        self.conv2 = nn.Conv2d(64, 32, (3, 3))
        self.max_pool1 = nn.MaxPool2d(kernel_size=(13, 3))

        self.conv3 = nn.Conv2d(1, 128, (4, 4))
        self.conv4 = nn.Conv2d(128, 32, (3, 3))
        self.max_pool2 = nn.MaxPool2d(kernel_size=(15, 5))

        self.fc1 = nn.Linear(160, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, states):
        # states have shape [num_samples, 1, ROWS, COLS]
        a = F.relu(self.conv1(states))
        a = F.relu(self.conv2(a))
        a = self.max_pool1(a)
        a = T.flatten(a, start_dim=1)

        b = F.relu(self.conv3(states))
        b = F.relu(b)
        b = self.max_pool2(b)
        b = T.flatten(b, start_dim=1)

        x = T.cat((a, b), dim=1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


class TetrisLearner:
    def __init__(self, lr: float, memory: Union[ReplayMemory, StratifiedReplayMemory], batch_size: Union[int, Tuple[int, int]],
                 gamma: float, epsilon_function: Callable, reset_steps: int, estimator: Optional[DeepQNetwork] = None):
        if estimator is None:
            self.Q_estimation = DeepQNetwork()
        else:
            self.Q_estimation = estimator
        self.optimizer = optim.Adam(self.Q_estimation.parameters(), lr=lr)
        self.Q_target = DeepQNetwork()
        self.Q_target.load_state_dict(self.Q_estimation.state_dict())
        self.steps = 0
        self.memory = memory
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon_function = epsilon_function
        self.reset_steps = reset_steps

    def train_model(self, epochs: int, n_samples: int, epsilon: int, n_processors: int, batch_size: int):
        samples = sample.collect_samples_using_model(n_samples, )

    def train(self, n_episodes: int, output_folder: str, model_save_interval: int):
        os.makedirs(output_folder, exist_ok=True)
        history_file = os.path.join(output_folder, 'history.csv')
        with open(history_file, 'w') as f:
            f.write('Experiment,Reward\n')
        for i in range(1, n_episodes + 1):
            self.train_episode()
            test_game, test_reward = self.test()
            with open(history_file, 'a') as f:
                f.write(f'{i},{test_reward}\n')
            if i % model_save_interval == 0:
                self.save_model(os.path.join(output_folder, f'episode_{i}_model.nn'))

    def train_episode(self):
        # create a new game of tetris to train on
        game = Tetris(ROWS, COLS)
        while not game.is_done():
            epsilon = self.epsilon_function(self.steps)
            state = board_to_tensor(game.get_state())  # get the current state of the game
            actions = game.get_actions()  # all the possible future states of the game
            # whether to act randomly or according to policy
            if random.random() < epsilon:
                # pick a random action
                action = random.randrange(len(actions))
            else:
                # expand dimensions since each state has one channel
                actions = board_to_tensor(actions).unsqueeze(dim=1)
                # calculate the value of each of the future states
                action_values = self.Q_estimation(actions)
                # if the states are terminal, set the value of the state to zero
                terminal_actions = T.tensor(game.get_are_terminal())
                action_values[terminal_actions] = 0
                # pick the action with the greatest value
                action = T.argmax(action_values)
            # get the immediate reward
            reward = game.take_action(action)
            # store the observation in memory
            self.memory.store(state, board_to_tensor(game.get_state()), reward, game.is_done())
            # continue to the next action if there is not enough states in memory for a batch
            if not self.memory.can_sample(self.batch_size):
                continue
            replay_batch = self.memory.get_batch(self.batch_size)
            # find the states, rewards, and future states from the memory
            states = replay_memory.get_batch_states(replay_batch)
            new_states = replay_memory.get_batch_new_states(replay_batch)
            rewards = replay_memory.get_batch_rewards(replay_batch)
            are_terminals = replay_memory.get_batch_terminals(replay_batch)
            # update the estimator network using the batch
            self.optimizer.zero_grad()
            predictions = self.Q_estimation(states).squeeze()
            new_state_targets = self.Q_target(new_states).squeeze()
            new_state_targets[are_terminals] = 0
            targets = self.gamma * new_state_targets + rewards
            loss = self.Q_estimation.criterion(predictions, targets)
            loss.backward()
            self.optimizer.step()
            # if it is time to update the target network, copy over the parameters
            self.steps += 1
            if self.steps % self.reset_steps == 0:
                self.Q_target.load_state_dict(self.Q_estimation.state_dict())

    def test(self) -> Tuple[Tetris, float]:
        game = Tetris(ROWS, COLS)
        total_reward = 0
        while not game.is_done():
            actions = game.get_actions()
            actions = T.tensor(actions, dtype=T.float).unsqueeze(dim=1)
            action_values = self.Q_estimation(actions).squeeze()
            action = int(T.argmax(action_values))
            reward = game.take_action(action)
            total_reward += reward
        return game, total_reward

    def save_model(self, file_path: str):
        T.save(self.Q_estimation, file_path)
