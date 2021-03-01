import torch
import numpy as np
from copy import deepcopy
import torch.nn.functional as F
from typing import Dict, Tuple, Union, Optional

from tianshou.policy import DDPGPolicy
from tianshou.policy.dist import DiagGaussian
from tianshou.data import Batch, to_torch_as, ReplayBuffer
from tianshou.exploration import BaseNoise
from tianshou.utils import RunningMeanStd, RewardForwardFilter


class SACMUPolicy(DDPGPolicy):
    """Implementation of Soft Actor-Critic. arXiv:1812.05905

    :param torch.nn.Module actor: the actor network following the rules in
        :class:`~tianshou.policy.BasePolicy`. (s -> logits)
    :param torch.optim.Optimizer actor_optim: the optimizer for actor network.
    :param torch.nn.Module critic1: the first critic network. (s, a -> Q(s,
        a))
    :param torch.optim.Optimizer critic1_optim: the optimizer for the first
        critic network.
    :param torch.nn.Module critic2: the second critic network. (s, a -> Q(s,
        a))
    :param torch.optim.Optimizer critic2_optim: the optimizer for the second
        critic network.
    :param float tau: param for soft update of the target network, defaults to
        0.005.
    :param float gamma: discount factor, in [0, 1], defaults to 0.99.
    :param float exploration_noise: the noise intensity, add to the action,
        defaults to 0.1.
    :param (float, torch.Tensor, torch.optim.Optimizer) or float alpha: entropy
        regularization coefficient, default to 0.2.
        If a tuple (target_entropy, log_alpha, alpha_optim) is provided, then
        alpha is automatatically tuned.
    :param action_range: the action range (minimum, maximum).
    :type action_range: (float, float)
    :param bool reward_normalization: normalize the reward to Normal(0, 1),
        defaults to ``False``.
    :param bool ignore_done: ignore the done flag while training the policy,
        defaults to ``False``.
    :param BaseNoise exploration_noise: add a noise to action for exploration.
        This is useful when solving hard-exploration problem.

    .. seealso::

        Please refer to :class:`~tianshou.policy.BasePolicy` for more detailed
        explanation.
    """

    def __init__(self,
                 actor: torch.nn.Module,
                 actor_optim: torch.optim.Optimizer,
                 critic1: torch.nn.Module,
                 critic1_optim: torch.optim.Optimizer,
                 critic2: torch.nn.Module,
                 critic2_optim: torch.optim.Optimizer,
                 tau: float = 0.005,
                 gamma: float = 0.99,
                 alpha: Tuple[float, torch.Tensor, torch.optim.Optimizer]
                 or float = 0.2,
                 action_range: Optional[Tuple[float, float]] = None,
                 reward_normalization: bool = False,
                 ignore_done: bool = False,
                 estimation_step: int = 1,
                 exploration_noise: Optional[BaseNoise] = None,
                 **kwargs) -> None:
        super().__init__(None, None, None, None, tau, gamma, exploration_noise,
                         action_range, reward_normalization, ignore_done,
                         estimation_step, **kwargs)
        self.actor, self.actor_optim = actor, actor_optim
        self.critic1, self.critic1_old = critic1, deepcopy(critic1)
        self.critic1_old.eval()
        self.critic1_optim = critic1_optim
        self.critic2, self.critic2_old = critic2, deepcopy(critic2)
        self.critic2_old.eval()
        self.critic2_optim = critic2_optim
        self.norm_func = kwargs.get('norm_func', None)
        self.blend_ratio = kwargs.get('blend_ratio', (lambda x : np.zeros((1))))

        self.has_rand = 'rand' in kwargs
        if self.has_rand:
            self.rand_source, self.rand_target, self.rand_optim = kwargs['rand']

        # if self.running_stats:
            self.reward_rms = RunningMeanStd()
            self.obs_rms = RunningMeanStd(shape=self.actor.state_shape)
            self.discounted_reward = RewardForwardFilter(0.99)

        self._automatic_alpha_tuning = not isinstance(alpha, float)
        if self._automatic_alpha_tuning:
            self._target_entropy = alpha[0]
            assert(alpha[1].shape == torch.Size([1])
                   and alpha[1].requires_grad)
            self._log_alpha = alpha[1]
            self._alpha_optim = alpha[2]
            self._alpha = self._log_alpha.exp()
        else:
            self._alpha = alpha

        self.__eps = np.finfo(np.float32).eps.item()

    def train(self, mode=True) -> torch.nn.Module:
        self.training = mode
        self.actor.train(mode)
        self.critic1.train(mode)
        self.critic2.train(mode)
        if self.has_rand:
            self.rand_source.train(mode)
        return self

    def sync_weight(self):
        for o, n in zip(
                self.critic1_old.parameters(), self.critic1.parameters()):
            o.data.copy_(o.data * (1 - self._tau) + n.data * self._tau)
        for o, n in zip(
                self.critic2_old.parameters(), self.critic2.parameters()):
            o.data.copy_(o.data * (1 - self._tau) + n.data * self._tau)

    def forward(self, batch, state=None, input='obs', **kwargs):
        obs = getattr(batch, input)
        logits, h = self.actor(obs, state=state, info=batch.info)
        assert isinstance(logits, tuple)

        mu, sigma = logits

        log_prob = None
        dist = None
        if kwargs.get('deterministic', False):
            act = torch.tanh(mu)
        else:
            dist = torch.distributions.Normal(*logits)
            x = dist.rsample()
            y = torch.tanh(x)
            log_prob = (dist.log_prob(x) - torch.log(
                self._action_scale * (1 - y.pow(2)) + self.__eps)).sum(-1, keepdim=True)  # the smaller the log_prob is, the better, y->0? because it's normalised.
            act = y

        act = act.clamp(self._range[0], self._range[1])

        # if self.has_rand:
        #     obs_normed = np.clip((obs - self.obs_rms.mean) / self.obs_rms.std, -5, 5)
        #     obs_int = (self.rand_source(obs_normed)[0] - self.rand_target(obs_normed)[0]).pow(2).sum(-1, keepdim=True)
        #     return Batch(logits=logits, act=act, state=h, dist=dist, log_prob=log_prob, int_rew=obs_int)
        # else:
        return Batch(logits=logits, act=act, state=h, dist=dist, log_prob=log_prob)

    def learn(self, batch, **kwargs):
        # Critic Loss
        # if self.has_rand:
        #     int_rew = (self.rand_source(batch.obs_next)[0] - self.rand_target(batch.obs_next)[0]).pow(2).sum(-1, keepdim=True).mean()
        #     self.rand_optim.zero_grad()
        #     int_rew.backward()
        #     self.rand_optim.step()

        blend_ratio = self.blend_ratio(batch.rew.shape)[:, None]
        obs_next = batch.obs_next + (batch.obs_next - batch.obs) * blend_ratio
        obs = batch.obs + (batch.obs_next - batch.obs) * blend_ratio
        batch.obs_next = obs_next
        batch.obs = obs

        with torch.no_grad():
            obs_next_result = self(batch, input='obs_next')
            a_ = obs_next_result.act
            dev = a_.device
            batch_act = torch.tensor(batch.act, dtype=torch.float, device=dev)
            target_q = torch.min(
                self.critic1_old(batch.obs_next, a_),
                self.critic2_old(batch.obs_next, a_),
            ) - self._alpha * obs_next_result.log_prob
            rew = torch.tensor(batch.rew,
                               dtype=torch.float, device=dev)[:, None]
            done = torch.tensor(batch.done,
                                dtype=torch.float, device=dev)[:, None]
            target_q = (rew + (1. - done) * self._gamma * target_q)

        # critic 1
        current_q1 = self.critic1(batch.obs, batch_act)
        critic1_loss = F.mse_loss(current_q1, target_q)
        # critic 2
        current_q2 = self.critic2(batch.obs, batch_act)
        critic2_loss = F.mse_loss(current_q2, target_q)

        self.critic1_optim.zero_grad()
        critic1_loss.backward()
        self.critic1_optim.step()

        self.critic2_optim.zero_grad()
        critic2_loss.backward()
        self.critic2_optim.step()

        obs_result = self(batch)
        a = obs_result.act
        current_q1a = self.critic1(batch.obs, a)
        current_q2a = self.critic2(batch.obs, a)
        actor_loss = (self._alpha * obs_result.log_prob - torch.min(  # if current entropy is small, alpha is large, then log_prob is encourge to larger
            current_q1a, current_q2a)).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        if self._automatic_alpha_tuning:
            log_prob = (obs_result.log_prob + self._target_entropy).detach()  # target_entropy - expected_entropy = target_entropy - (-plogp) = target_entropy + plogp
            alpha_loss = -(self._log_alpha * log_prob).mean()  # if current entropy is small, alpha will be large
            self._alpha_optim.zero_grad()
            alpha_loss.backward()
            self._alpha_optim.step()
            self._alpha = self._log_alpha.exp()

        self.sync_weight()

        try:
            alpha = self._alpha.item()
        except Exception as e:
            alpha = self._alpha

        result = {
            'loss/actor': actor_loss.item(),
            'loss/critic1': critic1_loss.item(),
            'loss/critic2': critic2_loss.item(),
            'alpha': alpha,
            'entropy': obs_result.log_prob.mean().item()
        }
        if self._automatic_alpha_tuning:
            result['loss/alpha'] = alpha_loss.item()
        # if self.has_rand:
        #     result['loss/rnd'] = int_rew.item()
        return result

    def process_fn(self, batch: Batch, buffer: ReplayBuffer,
                   indice: np.ndarray) -> Batch:
        if self._rm_done:
            batch.done = batch.done * 0.

        if self._rew_norm:
            if self.norm_func is None:
                bfr = buffer.rew[:min(len(buffer), 1000)]  # avoid large buffer
                mean, std = bfr.mean(), bfr.std()
                if np.isclose(std, 0):
                    mean, std = 0, 1
                batch.rew = (batch.rew - mean) / std
            else:
                batch.rew = self.norm_func(batch.rew)
        return batch

    def update_rand(self, data, batch_size, times=1):
        rns_before = None
        rns_after = None
        for _ in range(times):
            rns_list = []
            for batch in data.split(batch_size):
                # obs_next_normed = batch.obs_next # np.clip((batch.obs_next - self.obs_rms.mean) / self.obs_rms.std, -5, 5)
                obs_next_normed = (batch.obs_next - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + 1)
                # obs_next_normed = obs_next_normed / np.abs(obs_next_normed).sum(-1, keepdims=True)
                obs_next_normed = (obs_next_normed - obs_next_normed.mean(-1, keepdims=True)) / obs_next_normed.std(-1, keepdims=True)

                obs_next_normed = torch.FloatTensor(obs_next_normed, device=self.actor.device)
                pred_target = self.rand_target(obs_next_normed)[0].detach()
                pred_source = self.rand_source(obs_next_normed)[0]
                rew_intrisic = (pred_source - pred_target).pow(2).sum(-1, keepdim=True) / 2

                self.rand_optim.zero_grad()
                rew_intrisic.mean().backward()
                self.rand_optim.step()

                rns_list.append(rew_intrisic.mean().item())

            if rns_before is None:
                rns_before = np.mean(rns_list)

        rns_after = np.mean(rns_list)
        return rns_before, rns_after

    def update_rms(self, data, batch_size, update=True):
        if update:
            # rms_before = np.array(self.obs_rms.mean), np.array(self.obs_rms.std)
            self.obs_rms.update(data.obs_next)
            # rms_after = np.array(self.obs_rms.mean), np.array(self.obs_rms.std)
            # print(rms_before, rms_after, (rms_before[0]-rms_after[0], rms_before[1]-rms_after[1]))

        with torch.no_grad():
            batch_int = []
            for batch in data.split(batch_size, shuffle=False):
                # obs_next_normed = np.clip((batch.obs_next - self.obs_rms.mean) / self.obs_rms.std, -5, 5)
                obs_next_normed = (batch.obs_next - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + 1)
                # obs_next_normed = obs_next_normed / np.abs(obs_next_normed).sum(-1, keepdims=True)
                obs_next_normed = (obs_next_normed - obs_next_normed.mean(-1, keepdims=True)) / obs_next_normed.std(-1, keepdims=True)

                obs_next_normed = torch.FloatTensor(obs_next_normed, device=self.actor.device)
                pred_target = self.rand_target(obs_next_normed)[0].detach()
                pred_source = self.rand_source(obs_next_normed)[0]
                rew_intrisic = (pred_source - pred_target).pow(2).sum(-1, keepdim=True) / 2

                batch_int.append(rew_intrisic.data.numpy())

            batch_int = np.concatenate(batch_int, 0)

            batch_int_per_env = np.array([self.discounted_reward.update(reward_per_step) for reward_per_step in batch_int])
            mean, std, count = np.mean(batch_int_per_env), np.std(batch_int_per_env), len(batch_int_per_env)

        if update:
            self.reward_rms.update_from_moments(mean, std ** 2, count)

        return batch_int