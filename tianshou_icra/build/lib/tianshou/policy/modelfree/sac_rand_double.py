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


class SACRANDDPolicy(DDPGPolicy):
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
                 actor_exp: torch.nn.Module,
                 actor_exp_optim: torch.optim.Optimizer,
                 critic1_exp: torch.nn.Module,
                 critic1_exp_optim: torch.optim.Optimizer,
                 critic2_exp: torch.nn.Module,
                 critic2_exp_optim: torch.optim.Optimizer,
                 rand_source: torch.nn.Module,
                 rand_target: torch.nn.Module,
                 rand_optim: torch.optim.Optimizer,
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

        self.actor_exp, self.actor_exp_optim = actor_exp, actor_exp_optim
        self.critic1_exp, self.critic1_exp_old = critic1_exp, deepcopy(critic1_exp).eval()
        self.critic1_exp_optim = critic1_exp_optim
        self.critic2_exp, self.critic2_exp_old = critic2_exp, deepcopy(critic2_exp).eval()
        self.critic2_exp_optim = critic2_exp_optim

        self.rand_target = rand_target.eval()
        self.rand_source = rand_source
        self.rand_optim = rand_optim

        self.rand_mean = None
        self.rand_std = None

        self._gamma_int = kwargs.get('gamma_int', 0.99)
        self.norm_func = kwargs.get('norm_func', None)
        self.extcoef = kwargs.get('extcoef', 1.)
        self.running_stats = kwargs.get('running_stats', False)
        self.norm_int = kwargs.get('norm_int', True)
        self.clip_int = kwargs.get('clip_int', False)
        self.orinorm = kwargs.get('orinorm', False)
        self.inverse_rew_intrisic = kwargs.get('inverse_rew_intrisic', False)

        if self.running_stats:
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
        self.actor_exp.train(mode)
        self.critic1_exp.train(mode)
        self.critic2_exp.train(mode)
        self.rand_source.train(mode)
        return self

    def sync_weight(self):
        for o, n in zip(
                self.critic1_old.parameters(), self.critic1.parameters()):
            o.data.copy_(o.data * (1 - self._tau) + n.data * self._tau)
        for o, n in zip(
                self.critic2_old.parameters(), self.critic2.parameters()):
            o.data.copy_(o.data * (1 - self._tau) + n.data * self._tau)

        for o, n in zip(
                self.critic1_exp_old.parameters(), self.critic1_exp.parameters()):
            o.data.copy_(o.data * (1 - self._tau) + n.data * self._tau)
        for o, n in zip(
                self.critic2_exp_old.parameters(), self.critic2_exp.parameters()):
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

        if self.running_stats:
            self.obs_rms.update(batch.obs_next)
            obs_next_normed = (batch.obs_next - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + 1)
            obs_next_normed = (obs_next_normed - obs_next_normed.mean(-1, keepdims=True)) / obs_next_normed.std(-1,
                                                                                                                keepdims=True)
        else:
            if self.rand_mean is None:
                self.rand_mean = np.mean(batch.obs_next)
                self.rand_std = np.std(batch.obs_next)
            obs_next_normed = (batch.obs_next - self.rand_mean) / self.rand_std

        if self.inverse_rew_intrisic:
            shuffle_indexes = np.arange(obs_next_normed.shape[0])
            np.random.shuffle(shuffle_indexes)
            mix_factor = np.random.rand(obs_next_normed.shape[0], 1)
            obs_next_normed_mixed = obs_next_normed[shuffle_indexes] * mix_factor + (
                    1 - mix_factor) * obs_next_normed

        obs_next_normed = torch.FloatTensor(obs_next_normed, device=self.actor.device)
        pred_target = self.rand_target(obs_next_normed)[0].detach()
        pred_source = self.rand_source(obs_next_normed)[0]
        rew_intrisic = F.mse_loss(pred_source, pred_target, reduction='none').sum(-1, keepdim=True)

        if self.inverse_rew_intrisic:
            obs_next_normed_mixed = torch.FloatTensor(obs_next_normed_mixed, device=self.actor.device)
            pred_target = self.rand_target(obs_next_normed_mixed)[0].detach()
            pred_source = self.rand_source(obs_next_normed_mixed)[0]
            rew_intrisic_mixed = (F.mse_loss(pred_source, pred_target, reduction='none').sum(-1, keepdim=True) - 20).pow(2).mean()

        self.rand_optim.zero_grad()
        if self.inverse_rew_intrisic:
            (rew_intrisic.mean() + 0.1 * rew_intrisic_mixed).backward()
        else:
            rew_intrisic.mean().backward()
        self.rand_optim.step()

        if self.running_stats and self.norm_int:
            rew_intrisic_np = rew_intrisic.data.numpy()

            if self.orinorm:
                batch_int_per_env = np.array(
                    [self.discounted_reward.update(reward_per_step) for reward_per_step in rew_intrisic_np])
                mean, std, count = np.mean(batch_int_per_env), np.std(batch_int_per_env), len(batch_int_per_env)
                self.reward_rms.update_from_moments(mean, std ** 2, count)
                rew_intrisic = (rew_intrisic - self.reward_rms.mean) / self.reward_rms.std
            else:
                self.reward_rms.update(rew_intrisic_np)
                rew_intrisic = (rew_intrisic - self.reward_rms.mean[0]) / self.reward_rms.std[0]

            if self.clip_int:
                rew_intrisic = F.elu(rew_intrisic)

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
            assert rew.shape == rew_intrisic.shape, (rew.shape, rew_intrisic.shape)
            target_q_exp = torch.min(
                self.critic1_exp_old(batch.obs_next, a_),
                self.critic2_exp_old(batch.obs_next, a_)
            )
            done = torch.tensor(batch.done,
                                dtype=torch.float, device=dev)[:, None]
            target_q = (rew + (1. - done) * self._gamma * target_q)
            target_q_exp = (rew_intrisic + self._gamma_int * target_q_exp)

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

        # critic 1
        current_exp_q1 = self.critic1_exp(batch.obs, batch_act)
        critic1_exp_loss = F.mse_loss(current_exp_q1, target_q_exp)
        # critic 2
        current_exp_q2 = self.critic2_exp(batch.obs, batch_act)
        critic2_exp_loss = F.mse_loss(current_exp_q2, target_q_exp)

        self.critic1_exp_optim.zero_grad()
        critic1_exp_loss.backward()
        self.critic1_exp_optim.step()

        self.critic2_exp_optim.zero_grad()
        critic2_exp_loss.backward()
        self.critic2_exp_optim.step()

        obs_result = self(batch)
        a = obs_result.act
        current_q1a = self.critic1(batch.obs, a)
        current_q2a = self.critic2(batch.obs, a)
        current_q = self._alpha * obs_result.log_prob - torch.min(
            current_q1a, current_q2a)
        current_q1a = self.critic1_exp(batch.obs, a)
        current_q2a = self.critic2_exp(batch.obs, a)
        current_q_exp = - torch.min(current_q1a, current_q2a)

        actor_loss = (current_q + 1. / self.extcoef * current_q_exp).mean()

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
            'loss/critic1_exp': critic1_exp_loss.item(),
            'loss/critic2_exp': critic2_exp_loss.item(),
            'loss/rnd': rew_intrisic.mean().item(),
            'entropy': obs_result.log_prob.mean().item()
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