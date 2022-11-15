from typing import List
import random
import torch
import os

import deep_rl
import tetris_display
import sample


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)


def test_random(n: int, display: bool = False) -> List[float]:
    total_rewards: List[float] = []
    for _ in range(n):
        Q_estimation = deep_rl.DeepQNetwork()
        game, total_reward = deep_rl.test(Q_estimation)
        total_rewards.append(total_reward)
        if display:
            tetris_display.show(game.archive)
    return total_rewards


def main():
    set_seed(42)
    data_sample_path = os.path.join(os.getcwd(), 'data', 'stratified_random_samples_low.tet')
    samples = sample.load_samples(data_sample_path)
    print(f'Found {len(samples)} samples')
    for s in samples:
        tetris_display.print_board(s[0])
        tetris_display.print_board(s[1])
        print(s[2])
        print('=' * 40)
    return
    Q_estimation = deep_rl.train(10000, 1, 0.1, 0.99, 10, 20, 0.95, 1000)
    game, total_reward = deep_rl.test(Q_estimation)
    print(f'Total reward: {total_reward}')
    tetris_display.show(game.archive)


if __name__ == '__main__':
    main()
