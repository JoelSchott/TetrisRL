import pickle
import random
from typing import Tuple, List

from tetris import Tetris, Board


def save_samples(samples: List[Tuple[Board, Board, float]], file_path: str):
    with open(file_path, 'wb') as f:
        pickle.dump(samples, f)


def load_samples(file_path: str):
    with open(file_path, 'rb') as f:
        samples = pickle.load(f)
    return samples


def gather_random_samples(n: int, rows: int, cols: int, file_path: str):
    samples = []
    game = Tetris(rows, cols)
    while len(samples) < n:
        state = game.get_state()
        actions = game.get_actions()
        action = random.randrange(len(actions))
        reward = game.take_action(action)
        samples.append((state, game.get_state(), reward))
        if game.is_done():
            game = Tetris(rows, cols)
    save_samples(samples, file_path)


def gather_stratified_samples(num_reward: int, num_no_reward: int, rows: int, cols: int, max_height: int, file_path: str):
    reward_samples = []
    no_reward_samples = []
    game = Tetris(rows, cols)
    while len(reward_samples) < num_reward or len(no_reward_samples) < num_no_reward:
        state = game.get_state()
        actions = game.get_actions()
        action = random.randrange(len(actions))
        reward = game.take_action(action)
        sample = (state, game.get_state(), reward)
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
    save_samples(samples, file_path)
