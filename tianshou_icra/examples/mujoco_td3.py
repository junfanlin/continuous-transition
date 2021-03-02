import os
import gym
import torch
import pprint
import argparse
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from tianshou.policy import TD3Policy
from tianshou.trainer import offpolicy_exact_trainer
from tianshou.data import Collector, ReplayBuffer
from tianshou.env import VectorEnv, SubprocVectorEnv
from tianshou.exploration import GaussianNoise
from tianshou.utils.net.continuous import Actor_RLKIT as Actor, Critic_RLKIT as Critic


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='HalfCheetah-v2')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--buffer-size', type=int, default=int(1e6))
    parser.add_argument('--actor-lr', type=float, default=1e-3)
    parser.add_argument('--critic-lr', type=float, default=1e-3)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--tau', type=float, default=0.005)
    parser.add_argument('--exploration-noise', type=float, default=0.1)
    parser.add_argument('--policy-noise', type=float, default=0.2)
    parser.add_argument('--noise-clip', type=float, default=0.5)
    parser.add_argument('--update-actor-freq', type=int, default=2)
    parser.add_argument('--epoch', type=int, default=400)
    parser.add_argument('--step-per-epoch', type=int, default=1000)
    parser.add_argument('--collect-per-step', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--layer-num', type=int, default=2)
    parser.add_argument('--training-num', type=int, default=1)
    parser.add_argument('--test-num', type=int, default=10)
    parser.add_argument('--logdir', type=str, default='log')
    parser.add_argument('--render', type=float, default=0.)
    parser.add_argument('--rew-norm', type=bool, default=False)
    parser.add_argument('--auto_alpha', type=bool, default=True)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--shared', type=bool, default=False)
    args = parser.parse_known_args()[0]
    return args


def test_td3(args=get_args()):
    # initialize environment
    env = gym.make(args.task)
    args.state_shape = env.observation_space.shape or env.observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n
    args.max_action = env.action_space.high[0]
    train_envs = VectorEnv(
        [lambda: gym.make(args.task) for _ in range(args.training_num)])
    test_envs = SubprocVectorEnv(
        [lambda: gym.make(args.task) for _ in range(args.test_num)])
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)
    # model
    actor = Actor(
        args.layer_num, args.state_shape, args.action_shape,
        args.max_action, args.device, hidden_layer_size=args.hidden_size
    ).to(args.device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    critic1 = Critic(
        args.layer_num, args.state_shape, args.action_shape, args.device, hidden_layer_size=args.hidden_size
    ).to(args.device)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=args.critic_lr)
    critic2 = Critic(
        args.layer_num, args.state_shape, args.action_shape, args.device, hidden_layer_size=args.hidden_size
    ).to(args.device)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=args.critic_lr)

    policy = TD3Policy(
        actor, actor_optim, critic1, critic1_optim, critic2, critic2_optim,
        args.tau, args.gamma,
        GaussianNoise(sigma=args.exploration_noise), args.policy_noise,
        args.update_actor_freq, args.noise_clip,
        action_range=[env.action_space.low[0], env.action_space.high[0]],
        reward_normalization=args.rew_norm, ignore_done=False)
    # collector
    if args.training_num == 0:
        max_episode_steps = train_envs._max_episode_steps
    else:
        max_episode_steps = train_envs.envs[0]._max_episode_steps
    train_collector = Collector(
        policy, train_envs, ReplayBuffer(args.buffer_size, max_ep_len=max_episode_steps))
    test_collector = Collector(policy, test_envs, mode='test')
    # log
    log_path = os.path.join(args.logdir, args.task, 'td3', str(args.seed))
    writer = SummaryWriter(log_path)

    def save_fn(policy):
        torch.save(policy.state_dict(), os.path.join(log_path, 'policy.pth'))

    env.spec.reward_threshold = 100000

    def stop_fn(x):
        return x >= env.spec.reward_threshold

    # trainer
    result = offpolicy_exact_trainer(
        policy, train_collector, test_collector, args.epoch,
        args.step_per_epoch, args.collect_per_step, args.test_num,
        args.batch_size, stop_fn=stop_fn, save_fn=save_fn, writer=writer)
    assert stop_fn(result['best_reward'])
    train_collector.close()
    test_collector.close()
    if __name__ == '__main__':
        pprint.pprint(result)
        # Let's watch its performance!
        env = gym.make(args.task)
        collector = Collector(policy, env)
        result = collector.collect(n_episode=1, render=args.render)
        print(f'Final reward: {result["rew"]}, length: {result["len"]}')
        collector.close()


if __name__ == '__main__':
    test_td3()
