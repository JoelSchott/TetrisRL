import multiprocessing
import pickle
import random
from typing import Tuple, List
from multiprocessing import Pool
import torch as T

from tetris import Tetris, Board, over_height
from common import ROWS, COLS

Sample = Tuple[Board, Board, float, bool]


def save_samples(samples: List[Sample], file_path: str):
    with open(file_path, 'wb') as f:
        pickle.dump(samples, f)


def load_samples(file_path: str):
    with open(file_path, 'rb') as f:
        samples = pickle.load(f)
    return samples


def gather_random_samples(n: int, rows: int, cols: int) -> List[Sample]:
    samples = []
    game = Tetris(rows, cols)
    while len(samples) < n:
        state = game.get_state()
        actions = game.get_actions()
        action = random.randrange(len(actions))
        reward = game.take_action(action)
        samples.append((state, game.get_state(), reward, game.is_done()))
        if game.is_done():
            game = Tetris(rows, cols)
    return samples


def _gather_stratified_samples(parameters: Tuple[int, int, int, int, int]) -> List[Sample]:
    num_reward, num_no_reward, rows, cols, max_height = parameters
    reward_samples = []
    no_reward_samples = []
    game = Tetris(rows, cols)
    while len(reward_samples) < num_reward or len(no_reward_samples) < num_no_reward:
        state = game.get_state()
        actions = game.get_actions()
        action = random.randrange(len(actions))
        reward = game.take_action(action)
        sample = (state, game.get_state(), reward, game.is_done())
        if reward > 0 and len(reward_samples) < num_reward:
            reward_samples.append(sample)
        elif reward == 0 and len(no_reward_samples) < num_no_reward:
            no_reward_samples.append(sample)
        over_max_height = False
        for row in range(rows - max_height):
            if sum((game.board[row])) > 0:
                over_max_height = True
                break
        if game.is_done() or over_max_height:
            game = Tetris(rows, cols)
    samples = reward_samples + no_reward_samples
    random.shuffle(samples)
    return samples


def gather_stratified_samples(num_reward: int, num_no_reward: int, rows: int, cols: int, max_height: int,
                              n_processors: int) -> List[Sample]:
    n_processors = min(multiprocessing.cpu_count(), n_processors)
    pool = Pool(n_processors)
    parameters = (int(num_reward / n_processors), int(num_no_reward / n_processors), rows, cols, max_height)
    pooled_samples = pool.map(_gather_stratified_samples, [parameters for _ in range(n_processors)])
    flattened_samples = []
    for samples in pooled_samples:
        flattened_samples.extend(samples)
    return flattened_samples


def _collect_samples_using_model(parameters: Tuple) -> List[Sample]:
    n_samples, model, epsilon, n_samples_with_reward, max_height = parameters
    samples = []
    max_num_no_reward_samples = n_samples - n_samples_with_reward
    game = Tetris(ROWS, COLS)
    while len(samples) < n_samples:
        actions = game.get_actions()
        if random.random() < epsilon:
            action = random.randrange(len(actions))
        else:
            actions = T.tensor(actions, dtype=T.float).unsqueeze(dim=1)
            action_values = model(actions).squeeze()
            action = int(T.argmax(action_values))
        previous_state = game.get_state()
        reward = game.take_action(action)
        sample = (previous_state, game.get_state(), reward, game.is_done())
        if reward == 0:
            if max_num_no_reward_samples > 0:
                samples.append(sample)
                max_num_no_reward_samples -= 1
        else:
            samples.append(sample)
        if game.is_done() or over_height(game.board, max_height):
            game = Tetris(ROWS, COLS)
        print(f'there are {len(samples)} samples')
    random.shuffle(samples)
    return samples


def collect_samples_using_model(n_samples: int, model, epsilon: float, n_samples_with_reward: int, max_height: int,
                                n_processors: int) -> List[Sample]:
    n_processors = min(multiprocessing.cpu_count(), n_processors)
    pool = Pool(n_processors)
    parameters = (int(n_samples / n_processors), model, epsilon, int(n_samples_with_reward / n_processors), max_height)
    pooled_samples = pool.map(_collect_samples_using_model, [parameters for _ in range(n_processors)])
    flattened_samples = []
    for samples in pooled_samples:
        flattened_samples.extend(samples)
    return flattened_samples
