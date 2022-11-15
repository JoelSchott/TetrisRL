from typing import List
import random
import torch as T

import deep_rl
import tetris_display
import sample
from replay_memory import ReplayMemory, StratifiedReplayMemory


def set_seed(seed: int):
    random.seed(seed)
    T.manual_seed(seed)


def test_random(n: int, display: bool = False) -> List[float]:
    total_rewards: List[float] = []
    for _ in range(n):
        Q_estimation = deep_rl.DeepQNetwork()
        game, total_reward = deep_rl.test(Q_estimation)
        total_rewards.append(total_reward)
        if display:
            tetris_display.show(game.archive)
    return total_rewards


def display_samples(sample_path: str):
    samples = sample.load_samples(sample_path)
    for s in samples:
        tetris_display.print_board(s[0])
        tetris_display.print_board(s[1])
        print(s[2])
        print('=' * 40)
    print(f'Found {len(samples)} samples')


def main():
    set_seed(42)
    n_episodes = 100000
    epsilon_max = 1
    epsilon_min = 0.1
    epsilon_decay = 0.9995
    reset_steps = 200
    batch_size = 20
    gamma = 0.99
    memory = ReplayMemory(capacity=1000)
    samples = sample.load_samples('data/standard_random_samples.tet')
    memory.include_samples(samples)
    lr = 0.1
    Q_estimation = deep_rl.train(n_episodes, epsilon_max, epsilon_min, epsilon_decay, reset_steps,
                                 batch_size, gamma, memory, lr)
    T.save(Q_estimation, 'data/11_15_train_from_scratch.nn')
    game, total_reward = deep_rl.test(Q_estimation)
    print(f'Total reward after training: {total_reward}')
    tetris_display.show(game.archive)


if __name__ == '__main__':
    main()
