import os
import gym
import torch
import pprint
import argparse
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from tianshou.policy import SACMUTRIRB2BPolicy
from tianshou.trainer import offpolicy_exact_trainer
from tianshou.data import Collector, ReplayBufferTriple
from tianshou.env import VectorEnv, SubprocVectorEnv
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import ActorProb_RLKIT as ActorProb, Critic_RLKIT as Critic


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='HalfCheetah-v2')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--buffer-size', type=int, default=int(1e6))
    parser.add_argument('--actor-lr', type=float, default=3e-4)
    parser.add_argument('--alpha-lr', type=float, default=3e-4)
    parser.add_argument('--critic-lr', type=float, default=3e-4)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--tau', type=float, default=0.005)
    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--epoch', type=int, default=1000)
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
    parser.add_argument(
        '--device', type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--shared', type=bool, default=False)
    args = parser.parse_known_args()[0]
    return args

def process_tri(batch, rng, beta=1):
    blend_ratio = rng.beta(beta, beta, *batch.rew.shape) * (1 - batch.done_bk)

    obs = batch.obs + blend_ratio[:, None] * (batch.obs_next - batch.obs)
    obs_next = batch.obs_next + blend_ratio[:, None] * (batch.obs_next_next - batch.obs_next)
    act = batch.act + blend_ratio[:, None] * (batch.act_next - batch.act)
    rew = batch.rew + blend_ratio * (batch.rew_next - batch.rew)
    done = batch.done + blend_ratio * (batch.done_next * 1. - batch.done)

    batch.obs = obs
    batch.obs_next = obs_next
    batch.rew = rew
    batch.act = act
    batch.done = done
    batch.blend = blend_ratio[:, None]

    return batch

def test_sac(args=get_args()):
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
    actor = ActorProb(
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

    disc = Critic(
        args.layer_num, np.prod(args.state_shape)+np.prod(args.action_shape), 0, args.device, hidden_layer_size=args.hidden_size,
        output_dim=np.prod(args.state_shape)+1,
    ).to(args.device)
    disc_optim = torch.optim.Adam(disc.parameters(), lr=args.critic_lr)

    beta = torch.ones(1, requires_grad=True, device=args.device)
    beta_optim = torch.optim.Adam([beta], lr=args.critic_lr)

    if args.auto_alpha:
        target_entropy = -np.prod(env.action_space.shape)
        log_alpha = torch.zeros(1, requires_grad=True, device=args.device)
        alpha_optim = torch.optim.Adam([log_alpha], lr=args.alpha_lr)
        alpha = (target_entropy, log_alpha, alpha_optim)
    else:
        alpha = args.alpha

    rng = np.random.RandomState(seed=args.seed)

    policy = SACMUTRIRB2BPolicy(
        actor, actor_optim, critic1, critic1_optim, critic2, critic2_optim,
        args.tau, args.gamma, alpha,
        [env.action_space.low[0], env.action_space.high[0]], process_tri=(lambda x, beta: process_tri(x, rng=rng, beta=beta)),
        reward_normalization=args.rew_norm, ignore_done=False, norm_diff=False, beta=(beta, beta_optim),
        use_diff=False,
        discriminator=(disc, disc_optim), tor_diff=0.1
    )
    # collector
    if args.training_num == 0:
        max_episode_steps = train_envs._max_episode_steps
    else:
        max_episode_steps = train_envs.envs[0]._max_episode_steps
    train_collector = Collector(
        policy, train_envs, ReplayBufferTriple(args.buffer_size, max_ep_len=max_episode_steps))
    test_collector = Collector(policy, test_envs, mode='test')
    # train_collector.collect(n_step=args.buffer_size)
    # log
    log_path = os.path.join(args.logdir, args.task, 'sac_ct', str(args.seed))
    writer = SummaryWriter(log_path)

    def save_fn(policy, name='policy.pth'):
        torch.save(policy.state_dict(), os.path.join(log_path, name))

    env.spec.reward_threshold = 100000

    def stop_fn(x):
        return x >= env.spec.reward_threshold

    # trainer
    result = offpolicy_exact_trainer(
        policy, train_collector, test_collector, args.epoch,
        args.step_per_epoch, args.collect_per_step, args.test_num,
        args.batch_size, stop_fn=stop_fn, save_fn=save_fn, writer=writer, epochs_to_save=[1, 40, 50, 80, 100, 120, 150, 160, 200])
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
    test_sac()
