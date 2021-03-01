import torch
import numpy as np
from copy import deepcopy
import torch.nn.functional as F
from typing import Dict, Tuple, Union, Optional

from tianshou.policy import DDPGPolicy
from tianshou.policy.dist import DiagGaussian
from tianshou.data import Batch, to_torch_as, ReplayBuffer
from tianshou.exploration import BaseNoise

import numpy as np
from tianshou.utils import RunningMeanStd, RewardForwardFilter

class SACRAND2Policy(DDPGPolicy):
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
                 rand_source: torch.nn.Module,
                 rand_target: torch.nn.Module,
                 rand_optim: torch.optim.Optimizer,
                 tau: float = 0.005,
                 gamma: float = 0.999,
                 gamma_int: float = 0.99,
                 alpha: Tuple[float, torch.Tensor, torch.optim.Optimizer]
                 or float = 0.2,
                 action_range: Optional[Tuple[float, float]] = None,
                 reward_normalization: bool = False,
                 ignore_done: bool = False,
                 estimation_step: int = 1,
                 extcoef: int = 2,
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
        self._gamma_int = gamma_int
        self.rand_target = rand_target.eval()
        self.rand_source = rand_source
        self.rand_optim = rand_optim
        self.extcoef = extcoef

        self.reward_rms = RunningMeanStd()
        self.obs_rms = RunningMeanStd(shape=self.actor.state_shape)
        self.discounted_reward = RewardForwardFilter(self._gamma_int)

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
                self._action_scale * (1 - y.pow(2)) + self.__eps)).sum(-1, keepdim=True)
            act = y

        act = act.clamp(self._range[0], self._range[1])
        return Batch(logits=logits, act=act, state=h, dist=dist, log_prob=log_prob)

    def learn(self, batch, **kwargs):
        # Critic Loss
        with torch.no_grad():
            obs_next_normed = np.clip((batch.obs_next - self.obs_rms.mean) / self.obs_rms.std, -5, 5)

            obs_next_normed = torch.FloatTensor(obs_next_normed, device=self.actor.device)
            pred_target = self.rand_target(obs_next_normed).detach()
            pred_source = self.rand_source(obs_next_normed)
            rew_intrisic = ((pred_source - pred_target) ** 2).pow(2).sum(-1, keepdim=True) / 2
            rew_intrisic = rew_intrisic / self.reward_rms.std

            obs_next_result = self(batch, input='obs_next')
            a_ = obs_next_result.act
            dev = a_.device
            batch_act = torch.tensor(batch.act, dtype=torch.float, device=dev)
            target_qs1 = self.critic1_old(batch.obs_next, a_)
            target_qs2 = self.critic2_old(batch.obs_next, a_)
            target_q = torch.min(target_qs1[0], target_qs2[0]) - self._alpha * obs_next_result.log_prob
            target_q_int = torch.min(target_qs1[1], target_qs2[1])

            rew = torch.tensor(batch.rew,
                               dtype=torch.float, device=dev)[:, None]
            assert rew.shape == rew_intrisic.shape, (rew.shape, rew_intrisic.shape)

            done = torch.tensor(batch.done,
                                dtype=torch.float, device=dev)[:, None]
            target_q = (rew + (1. - done) * self._gamma * target_q)
            target_q_int = (rew_intrisic + self._gamma_int * target_q_int)

        # critic 1
        current_q1s = self.critic1(batch.obs, batch_act)
        critic1_loss = F.mse_loss(current_q1s[0], target_q)
        critic1_int_loss = F.mse_loss(current_q1s[1], target_q_int)

        # critic 2
        current_q2s = self.critic2(batch.obs, batch_act)
        critic2_loss = F.mse_loss(current_q2s[0], target_q)
        critic2_int_loss = F.mse_loss(current_q2s[1], target_q_int)

        self.critic1_optim.zero_grad()
        (critic1_loss + critic1_int_loss).backward()
        self.critic1_optim.step()

        self.critic2_optim.zero_grad()
        (critic2_loss + critic2_int_loss).backward()
        self.critic2_optim.step()

        obs_result = self(batch)
        a = obs_result.act
        current_q1as = self.critic1(batch.obs, a)
        current_q2as = self.critic2(batch.obs, a)
        actor_loss = (self._alpha * obs_result.log_prob - torch.min(
            current_q1as[0], current_q2as[0]) - (1. / self.extcoef) * torch.min(
            current_q1as[1], current_q2as[1])).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        if self._automatic_alpha_tuning:
            log_prob = (obs_result.log_prob + self._target_entropy).detach()
            alpha_loss = -(self._log_alpha * log_prob).mean()
            self._alpha_optim.zero_grad()
            alpha_loss.backward()
            self._alpha_optim.step()
            self._alpha = self._log_alpha.exp()

        self.sync_weight()

        result = {
            'loss/actor': actor_loss.item(),
            'loss/critic1': critic1_loss.item(),
            'loss/critic2': critic2_loss.item(),
        }
        if self._automatic_alpha_tuning:
            result['loss/alpha'] = alpha_loss.item()
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

    def update_rms(self, data, batch_size):
        self.obs_rms.update(data.obs_next)

        with torch.no_grad():
            batch_int = []
            for batch in data.split(batch_size):
                obs_next_normed = np.clip((batch.obs_next - self.obs_rms.mean) / self.obs_rms.std, -5, 5)

                obs_next_normed = torch.FloatTensor(obs_next_normed, device=self.actor.device)
                pred_target = self.rand_target(obs_next_normed).detach()
                pred_source = self.rand_source(obs_next_normed)
                rew_intrisic = ((pred_source - pred_target) ** 2).pow(2).sum(-1, keepdim=True) / 2

                batch_int.append(rew_intrisic.data.numpy())

            batch_int = np.concatenate(batch_int, 0)

            batch_int_per_env = np.array([self.discounted_reward.update(reward_per_step) for reward_per_step in batch_int])
            mean, std, count = np.mean(batch_int_per_env), np.std(batch_int_per_env), len(batch_int_per_env)

        self.reward_rms.update_from_moments(mean, std ** 2, count)

    def update_rand(self, data, batch_size):
        for batch in data.split(batch_size):
            obs_next_normed = np.clip((batch.obs_next - self.obs_rms.mean) / self.obs_rms.std, -5, 5)

            obs_next_normed = torch.FloatTensor(obs_next_normed, device=self.actor.device)
            pred_target = self.rand_target(obs_next_normed).detach()
            pred_source = self.rand_source(obs_next_normed)
            rew_intrisic = ((pred_source - pred_target) ** 2).pow(2).sum(-1, keepdim=True) / 2

            self.rand_optim.zero_grad()
            rew_intrisic.mean().backward()
            self.rand_optim.step()
