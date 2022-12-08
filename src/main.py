from typing import List, Any, Union, Dict
import random
import torch as T
import argparse
import os
import json

from deep_rl import ConstantEpsilon, ExponentialDecayEpsilon, TetrisLearner
import tetris_display
import sample
from replay_memory import ReplayMemory, StratifiedReplayMemory


def set_seed(seed: int):
    random.seed(seed)
    T.manual_seed(seed)


def test_random(n: int, display: bool = False) -> List[float]:
    total_rewards: List[float] = []
    for _ in range(n):
        learner = TetrisLearner(0, ReplayMemory(0), 0, 0, ConstantEpsilon(0), 0)
        game, total_reward = learner.test()
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


def parse_args() -> Dict[str, Any]:
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--seed', help='random seed to use when training and testing', type=int)
    parser.add_argument('-n', '--num-episodes', help='number of episodes to use for training', type=int)
    parser.add_argument('-e', '--epsilon', help='epsilon represents the probability of taking a random action instead' +
                                                ' of an action according to policy, can be a single value or three ' +
                                                'values to represent epsilon max, epsilon min, and epsilon decay for ' +
                                                'an exponential decrease in epsilon across training steps', nargs='+', type=float)
    parser.add_argument('-r', '--reset-steps', help='number of steps between resetting target network to estimator network', type=int)
    parser.add_argument('-b', '--batch-size', help='number of samples to learn from after each action, can be two ' +
                        'values for stratified sampling or one value otherwise', nargs='+', type=int)
    parser.add_argument('-g', '--gamma', help='value decay rate', type=float)
    parser.add_argument('-c', '--capacity', help='capacity of sample memory, can be two values for stratified sampling' +
                        ' or one value otherwise', nargs='+', type=int)
    parser.add_argument('-l', '--learning-rate', help='learning rate of the deep q network', type=float)
    parser.add_argument('-i', '--save-interval', help='number of training episodes between saving model weights', type=int)
    parser.add_argument('-o', '--output', help='path to the output folder')
    parser.add_argument('-p', '--starting-samples', help='path to the file with pre-computed samples that will be used ' +
                        'for initialization', default=None)
    parser.add_argument('-m', '--model', help='path to the file containing the model to use for initialization', default=None)
    pargs = parser.parse_args()

    args = {}
    args['seed'] = pargs.seed
    args['num_episodes'] = pargs.num_episodes
    if len(pargs.epsilon) == 1:
        args['epsilon'] = pargs.epsilon[0]
    elif len(pargs.epsilon) == 3:
        args['epsilon'] = pargs.epsilon
    args['reset_steps'] = pargs.reset_steps
    if len(pargs.batch_size) == 1:
        args['batch_size'] = pargs.batch_size[0]
    elif len(pargs.batch_size) == 2:
        args['batch_size'] = (pargs.batch_size[0], pargs.batch_size[1])
    args['gamma'] = pargs.gamma
    if len(pargs.capacity) == 1:
        args['capacity'] = pargs.capacity[0]
    elif len(pargs.capacity) == 2:
        args['capacity'] = (pargs.capacity[0], pargs.capacity[1])
    args['learning_rate'] = pargs.learning_rate
    args['save_interval'] = pargs.save_interval
    args['output_folder'] = pargs.output
    args['starting_samples'] = pargs.starting_samples
    args['model'] = pargs.model

    return args


def main(args: Dict[str, Any]):
    os.makedirs(args['output_folder'], exist_ok=True)
    with open(os.path.join(args['output_folder'], 'parameters.json'), 'w') as f:
        json.dump(args, f, indent=4)
    set_seed(args['seed'])
    if type(args['epsilon']) == float:
        epsilon = ConstantEpsilon(args['epsilon'])
    else:
        epsilon = ExponentialDecayEpsilon(args['epsilon'][0], args['epsilon'][1], args['epsilon'][2])
    if type(args['capacity']) == int:
        memory = ReplayMemory(args['capacity'])
    else:
        memory = StratifiedReplayMemory(args['capacity'])
    if args['starting_samples'] is not None:
        memory.include_samples(sample.load_samples(args['starting_samples']))
    if args['model'] is not None:
        estimator = T.load(args['model'])
    else:
        estimator = None
    learner = TetrisLearner(
        args['learning_rate'],
        memory,
        args['batch_size'],
        args['gamma'],
        epsilon,
        args['reset_steps'],
        estimator
    )
    learner.train(args['num_episodes'], args['output_folder'], args['save_interval'])
    learner.save_model(os.path.join(args['output_folder'], 'final_model.nn'))
    game, total_reward = learner.test()
    with open(os.path.join(args['output_folder'], 'final_reward.txt'), 'w') as f:
        f.write(f'Final total reward: {total_reward}')


if __name__ == '__main__':
    #main(parse_args())
    import sample
    import deep_rl
    import time
    model = deep_rl.DeepQNetwork()
    start_time = time.time()
    samples = sample.collect_samples_using_model(12000, model, 0.05, 200, 6, 10)
    end_time = time.time()
    print(f'collected 12000 samples in {end_time - start_time} seconds using 10 processes')
