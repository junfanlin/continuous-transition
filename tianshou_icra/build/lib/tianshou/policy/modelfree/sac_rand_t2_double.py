import torch
import numpy as np
from copy import deepcopy
import torch.nn.functional as F
from typing import Dict, Tuple, Union, Optional

from tianshou.policy import DDPGPolicy
from tianshou.policy.dist import DiagGaussian
from tianshou.data import Batch, to_torch_as, ReplayBuffer
from tianshou.exploration import BaseNoise

from collections import deque
import copy

class SACT2DPolicy(DDPGPolicy):
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
                 w: torch.nn.Module,
                 w_optim: torch.optim.Optimizer,
                 tau: float = 0.005,
                 gamma: float = 0.99,
                 alpha: Tuple[float, torch.Tensor, torch.optim.Optimizer]
                 or float = 0.2,
                 action_range: Optional[Tuple[float, float]] = None,
                 reward_normalization: bool = False,
                 ignore_done: bool = False,
                 estimation_step: int = 1,
                 buf_len: int = 5,
                 buf_interval: int = 1000,
                 use_exp: bool = True,
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

        self.w, self.w_target = w, deepcopy(w).eval()
        self.w_optim = w_optim
        self.use_exp = use_exp
        self.actor_exp_old = deepcopy(actor_exp)
        self.buf_len = buf_len
        self.buf_interval = buf_interval
        self.buf_cnt = 0
        self.buf_actor = deque(maxlen=self.buf_len)

        self.rand_target = rand_target.eval()
        self.rand_source = rand_source
        self.rand_optim = rand_optim

        self.rand_mean = None
        self.rand_std = None

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
        self.w.train(mode)
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

        for o, n in zip(
            self.w_target.parameters(), self.w.parameters()):
            o.data.copy_(o.data * (1 - self._tau) + n.data * self._tau)

    def forward(self, batch, actor=None, state=None, input='obs', **kwargs):
        obs = getattr(batch, input)

        if kwargs.get('deterministic', False):
            logits, h = self.actor(obs, state=state, info=batch.info)
            assert isinstance(logits, tuple)
            mu, sigma = logits
            log_prob = None
            dist = None
            act = torch.tanh(mu)
        else:
            if kwargs.get('explore', True):
                w = kwargs.get('w', self.w.w)
                if w.shape[0] != obs.shape[0]:
                    w = w.repeat(obs.shape[0], 1)
                logits, h = self.actor_exp(obs, w, state=state, info=batch.info)
            else:
                logits, h = self.actor(obs, state=state, info=batch.info)

            assert isinstance(logits, tuple)
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
        batch.obs_next = torch.FloatTensor(batch.obs_next, device=self.actor.device)
        if self.rand_mean is None:
            self.rand_mean = batch.obs_next.mean().detach()#0, keepdim=True).detach()
            self.rand_std = batch.obs_next.std().detach()#0, keepdim=True).detach() / 5 + self.__eps

        norm_obs_next = (batch.obs_next - self.rand_mean) / self.rand_std
        pred_target = self.rand_target(norm_obs_next)[0].detach()
        pred_source = self.rand_source(norm_obs_next)[0]
        rew_intrisic_feature = F.mse_loss(pred_source, pred_target, reduction='none')

        self.rand_optim.zero_grad()
        rew_intrisic_feature.mean().backward()
        self.rand_optim.step()

        with torch.no_grad():
            obs_next_result = self(batch, input='obs_next', explore=False)
            a_ = obs_next_result.act
            dev = a_.device
            batch_act = torch.tensor(batch.act, dtype=torch.float, device=dev)
            target_q = torch.min(
                self.critic1_old(batch.obs_next, a_),
                self.critic2_old(batch.obs_next, a_),
            ) - self._alpha * obs_next_result.log_prob
            rew = torch.tensor((batch.rew - batch.mean) / batch.std,
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

        obs_result = self(batch, explore=False)
        a = obs_result.act
        current_q1a = self.critic1(batch.obs, a)
        current_q2a = self.critic2(batch.obs, a)
        actor_loss = (self._alpha * obs_result.log_prob - torch.min(
            current_q1a, current_q2a)).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()


        w = self.w.sample((batch.obs.shape[0], ))
        rew_intrisic = (rew_intrisic_feature * w).mean(-1, keepdim=True).detach()

        with torch.no_grad():
            obs_next_result = self(batch, w=w, input='obs_next')
            a_ = obs_next_result.act
            dev = a_.device
            batch_act = torch.tensor(batch.act, dtype=torch.float, device=dev)
            target_q = torch.min(
                self.critic1_exp_old(batch.obs_next, w, a_),
                self.critic2_exp_old(batch.obs_next, w, a_),
            ) - self._alpha * obs_next_result.log_prob
            rew = torch.tensor((batch.rew - batch.mean) / batch.std,
                               dtype=torch.float, device=dev)[:, None]

            assert rew.shape == rew_intrisic.shape, (rew.shape, rew_intrisic.shape)
            rew = rew + rew_intrisic
            done = torch.tensor(batch.done,
                                dtype=torch.float, device=dev)[:, None]
            target_q = (rew + (1. - done) * self._gamma * target_q)

        # critic 1
        current_q1 = self.critic1_exp(batch.obs, w, batch_act)
        critic1_loss = F.mse_loss(current_q1, target_q)
        # critic 2
        current_q2 = self.critic2_exp(batch.obs, w, batch_act)
        critic2_loss = F.mse_loss(current_q2, target_q)

        self.critic1_exp_optim.zero_grad()
        critic1_loss.backward()
        self.critic1_exp_optim.step()

        self.critic2_exp_optim.zero_grad()
        critic2_loss.backward()
        self.critic2_exp_optim.step()

        obs_result = self(batch, w=w)
        a = obs_result.act
        current_q1a = self.critic1_exp(batch.obs, w, a)
        current_q2a = self.critic2_exp(batch.obs, w, a)
        actor_loss = (self._alpha * obs_result.log_prob - torch.min(
            current_q1a, current_q2a)).mean()

        self.actor_exp_optim.zero_grad()
        actor_loss.backward()
        self.actor_exp_optim.step()

        if self._automatic_alpha_tuning:
            log_prob = (obs_result.log_prob + self._target_entropy).detach()
            alpha_loss = -(self._log_alpha * log_prob).mean()
            self._alpha_optim.zero_grad()
            alpha_loss.backward()
            self._alpha_optim.step()
            self._alpha = self._log_alpha.exp()

        kl_div = 0
        if len(self.buf_actor) >= 1:
            # is still not for sure whether to use w_target of zero
            logits_old, _ = self.actor_exp_old(batch.obs, self.w.w.repeat(batch.obs.shape[0], 1))
            if self.use_exp:
                logits_ori, _ = self.actor_exp(batch.obs, self.w_target.w.repeat(batch.obs.shape[0], 1).detach())
            else:
                logits_ori, _ = self.actor(batch.obs)

            p = torch.distributions.Normal(*logits_old)
            q = torch.distributions.Normal(*logits_ori)

            kl_div = torch.distributions.kl.kl_divergence(p, q).mean()

            self.w_optim.zero_grad()
            kl_div.backward()
            kl_div = kl_div.item()
            self.w_optim.step()

        if self.buf_cnt % self.buf_interval == 0:
            self.buf_actor.append(copy.deepcopy(self.actor_exp.state_dict()))
            self.actor_exp_old.load_state_dict(self.buf_actor[0])

        self.buf_cnt += 1

        self.sync_weight()

        result = {
            'loss/actor': actor_loss.item(),
            'loss/critic1': critic1_loss.item(),
            'loss/critic2': critic2_loss.item(),
            'loss/kl_div': kl_div,
        }
        if self._automatic_alpha_tuning:
            result['loss/alpha'] = alpha_loss.item()

        return result

    def process_fn(self, batch: Batch, buffer: ReplayBuffer,
                   indice: np.ndarray) -> Batch:
        if self._rm_done:
            batch.done = batch.done * 0.
        if self._rew_norm:
            bfr = buffer.rew[:min(len(buffer), 1000)]  # avoid large buffer
            mean, std = bfr.mean(), bfr.std()
            if np.isclose(std, 0):
                mean, std = 0, 1
        else:
            mean, std = 0, 1
        batch.mean = mean
        batch.std = std
        return batch