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


class SACFECVPolicy(DDPGPolicy):
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
        self.process_tri = kwargs.get('process_tri', (lambda x: x))
        self.disc, self.disc_optim = kwargs.get('discriminator', [None, None])
        self.beta, self.beta_optim = kwargs.get('beta', [None, None])
        self.norm_diff = kwargs.get('norm_diff', False)
        self.tor_diff = kwargs.get('tor_diff', 0.1)
        self.use_diff = kwargs.get('use_diff', True)
        self.use_lbe = kwargs.get('use_lbe', False)
        self.loss_func = kwargs.get('loss_func', F.mse_loss)
        self.norm_loss = kwargs.get('norm_loss', False)

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

        return Batch(logits=logits, act=act, state=h, dist=dist, log_prob=log_prob)

    def learn(self, batch, **kwargs):
        # Critic Loss
        with torch.no_grad():
            obs_next_result = self(batch, input='obs_next')
            a_ = obs_next_result.act
            dev = a_.device
            batch_act = torch.tensor(batch.act, dtype=torch.float, device=dev)
            rew = torch.tensor(batch.rew,
                               dtype=torch.float, device=dev)[:, None]
            done = torch.tensor(batch.done,
                                dtype=torch.float, device=dev)[:, None]

        if self.beta is not None:
            if self.disc is not None:
                with torch.no_grad():
                    obs_bk = np.concatenate(
                        [batch.obs, np.zeros_like(rew), np.ones_like(rew) * 0.5], -1)
                    obs_next_bk = np.concatenate(
                        [batch.obs_next, np.zeros_like(rew), np.ones_like(rew) * 0.5], -1)
                    target_q = torch.min(
                        self.critic1(obs_next_bk, a_),
                        self.critic2(obs_next_bk, a_),
                    ) - self._alpha * obs_next_result.log_prob
                    current_q = torch.min(
                        self.critic1(obs_bk, batch_act),
                        self.critic2(obs_bk, batch_act),
                    )
                    pred_rew = current_q - (1. - done) * self._gamma * target_q
                    scale = (pred_rew * rew).sum() / (pred_rew * pred_rew).sum().detach()  # should we learn this?
                    rew_loss = F.mse_loss(pred_rew * scale, rew)  # usually old_update_dist will be larger no matter what

                if self.disc.avg_loss is None:
                    self.disc.avg_loss = rew_loss.detach()
                else:
                    self.disc.avg_loss = (self.disc.avg_loss * 0.95 + 0.05 * rew_loss).detach()

                # disc_loss = (self.disc.avg_loss - rew_loss) ** 2

                # self.disc_optim.zero_grad()
                # disc_loss.backward()
                # self.disc_optim.step()

                if self.disc.logp is not None:
                    beta_loss = self.disc.logp * (rew_loss - self.disc.avg_loss.detach()) # if larger than average, punish the logp
                    self.beta_optim.zero_grad()
                    beta_loss.backward()
                    self.beta_optim.step()
                else:
                    beta_loss = torch.zeros(1)

                dist = torch.distributions.Normal(self.beta, torch.ones_like(self.beta) * 0.1)
                sample = dist.rsample()
                beta = torch.tanh(sample)
                self.disc.logp = dist.log_prob(sample) - torch.log(2 * (1 - beta.pow(2)) + self.__eps)
                beta = ((beta + 1) / 2).clamp(1e-4, 1).detach()
            else:
                beta = self.beta

            batch = self.process_tri(batch, beta=beta.squeeze().data.numpy())
        else:
            beta = None
            batch = self.process_tri(batch, beta=beta)

        targ_blend = torch.FloatTensor(batch.blend)

        ratio = 1

        if self.norm_diff:
            ratio = ratio / ratio.sum() * ratio.shape[0]
            ratio_log = ratio.mean().item()

        if not self.use_diff:
            ratio = 1
            ratio_log = 1

        if self.norm_loss:
            ratio = 0.5 + (targ_blend - 0.5).abs()
            ratio = ratio / ratio.sum() * ratio.shape[0]
            ratio_log = ratio.mean().item()

        with torch.no_grad():
            target_q = torch.min(
                self.critic1_old(batch.obs_next, a_),
                self.critic2_old(batch.obs_next, a_),
            ) - self._alpha * obs_next_result.log_prob
            target_q = (rew + (1. - done) * self._gamma * target_q)

            if self.use_lbe:
                obs_next_ori_result = self(batch, input='obs_next_ori_bk')
                a_ = obs_next_ori_result.act
                target_q_ori = torch.min(
                    self.critic1_old(batch.obs_next_ori_bk, a_),
                    self.critic2_old(batch.obs_next_ori_bk, a_),
                ) - self._alpha * obs_next_ori_result.log_prob
                done_ori = torch.tensor(batch.done_ori_bk, dtype=torch.float, device=dev)[:, None]
                target_q_ori = (1. - done_ori) * self._gamma * target_q_ori

                obs_next_next_ori_result = self(batch, input='obs_next_next_ori_bk')
                a_ = obs_next_next_ori_result.act
                target_q_next_ori = torch.min(
                    self.critic1_old(batch.obs_next_next_ori_bk, a_),
                    self.critic2_old(batch.obs_next_next_ori_bk, a_),
                ) - self._alpha * obs_next_next_ori_result.log_prob
                done_next_ori = torch.tensor(batch.done_next_ori_bk, dtype=torch.float, device=dev)[:, None]
                target_q_next_ori = (1. - done_next_ori) * self._gamma * target_q_next_ori

                target_q_ori = rew + target_q_ori + targ_blend * (target_q_next_ori - target_q_ori)
                target_q = torch.min(target_q, target_q_ori)

        current_q1 = self.critic1(batch.obs, batch_act)
        critic1_loss = (self.loss_func(current_q1, target_q, reduction='none') * ratio).mean()
        current_q2 = self.critic2(batch.obs, batch_act)
        critic2_loss = (self.loss_func(current_q2, target_q, reduction='none') * ratio).mean()

        self.critic1_optim.zero_grad()
        critic1_loss.backward()
        self.critic1_optim.step()

        self.critic2_optim.zero_grad()
        critic2_loss.backward()
        self.critic2_optim.step()

        obs_result = self(batch, input='obs_bk')
        a = obs_result.act
        current_q1a = self.critic1(batch.obs_bk, a)
        current_q2a = self.critic2(batch.obs_bk, a)
        actor_loss = ((self._alpha * obs_result.log_prob - torch.min(  # if current entropy is small, alpha is large, then log_prob is encourge to larger
            current_q1a, current_q2a)) * ratio).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        if self._automatic_alpha_tuning:
            log_prob = (obs_result.log_prob + self._target_entropy).detach()  # target_entropy - expected_entropy = target_entropy - (-plogp) = target_entropy + plogp
            alpha_loss = -((self._log_alpha * log_prob) * ratio).mean()  # if current entropy is small, alpha will be large
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
            'blend': ratio_log,
            'alpha': alpha,
            'entropy': obs_result.log_prob.mean().item()
        }
        if beta is not None:
            result['beta'] = beta.item()

        if self.disc is not None:
            result['disc/beta_mu'] = self.beta.item()
            # result['disc/disc_loss'] = disc_loss.item()
            result['disc/rew_loss'] = rew_loss.item()
            result['disc/avg_loss'] = self.disc.avg_loss.item()
            result['disc/scale'] = scale.item()
            result['disc/beta_loss'] = beta_loss.item()
            result['disc/avg_l_rew'] = self.disc.avg_loss.item() > rew_loss.item()

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
