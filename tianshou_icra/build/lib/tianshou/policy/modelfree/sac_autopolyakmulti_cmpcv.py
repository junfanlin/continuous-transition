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


class SACAPMCCVPolicy(DDPGPolicy):
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
        self.critic1_old#.eval()
        self.critic1_optim = critic1_optim
        self.critic2, self.critic2_old = critic2, deepcopy(critic2)
        self.critic2_old#.eval()
        self.critic2_optim = critic2_optim
        self.norm_func = kwargs.get('norm_func', None)
        self.polyak, self.polyak_optim = kwargs.get('polyak', (None, None))
        self.use_autopolyak = kwargs.get('use_autopolyak', False)

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

    def sync_weight(self, polyaks):
        for p, o, n in zip(
                polyaks, self.critic1_old.parameters(), self.critic1.parameters()):
            o.data.copy_(o.data * (1 - p) + n.data * p)
        for p, o, n in zip(
                polyaks, self.critic2_old.parameters(), self.critic2.parameters()):
            o.data.copy_(o.data * (1 - p) + n.data * p)

    def forward(self, batch, state=None, input='obs', **kwargs):
        obs = getattr(batch, input)
        logits, h = self.actor(obs, state=state, info=batch.info)
        assert isinstance(logits, tuple)

        mu, sigma = logits

        log_prob = None
        dist = None

        if kwargs.get('deterministic', False):
            act = torch.tanh(mu) * self._action_scale
        else:
            dist = torch.distributions.Normal(*logits)
            x = dist.rsample()
            y = torch.tanh(x)
            log_prob = (dist.log_prob(x) - torch.log(
                self._action_scale * (1 - y.pow(2)) + self.__eps)).sum(-1, keepdim=True)  # the smaller the log_prob is, the better, y->0? because it's normalised.
            act = y * self._action_scale

        act = act.clamp(self._range[0], self._range[1])

        return Batch(logits=logits, act=act, state=h, dist=dist, log_prob=log_prob)

    def learn(self, batch, **kwargs):
        # Critic Loss
        with torch.no_grad():
            obs_next_result = self(batch, input='obs_next', deterministic=True)
            a_ = obs_next_result.act
            dev = a_.device
            batch_act = torch.tensor(batch.act, dtype=torch.float, device=dev)
            rew = torch.tensor(batch.rew,
                               dtype=torch.float, device=dev)[:, None]
            done = torch.tensor(batch.done,
                                dtype=torch.float, device=dev)[:, None]

        critic1_old_param = []
        critic2_old_param = []
        if self.use_autopolyak:
            polyaks = self.polyak.polyaks
        else:
            polyaks = [self._tau] * self.polyak.size
        for polyak, o, n in zip(
                polyaks, self.critic1_old.parameters(), self.critic1.parameters()):
            critic1_old_param.append(o * (1 - polyak) + n * polyak)
        for polyak, o, n in zip(
                polyaks, self.critic2_old.parameters(), self.critic2.parameters()):
            critic2_old_param.append(o * (1 - polyak) + n * polyak)

        old_update_target_q = torch.min(
            self.critic1_old(batch.obs_next, a_, vars=critic1_old_param),
            self.critic2_old(batch.obs_next, a_, vars=critic2_old_param)
        )
        old_update_q = torch.min(
            self.critic1_old(batch.obs, batch_act, vars=critic1_old_param),
            self.critic2_old(batch.obs, batch_act, vars=critic2_old_param),
        )
        old_update_rew = old_update_q - (1. - done) * self._gamma * old_update_target_q
        with torch.no_grad():
            # old_update_dist = F.mse_loss(old_update_rew, rew)
            old_update_scale = (old_update_rew * rew).sum() / (old_update_rew * old_update_rew).sum().detach()
        old_update_dist_scaled = F.mse_loss(old_update_rew * old_update_scale, rew)

        polyloss = old_update_dist_scaled

        self.polyak_optim.zero_grad()
        polyloss.backward()
        self.polyak_optim.step()

        if self.use_autopolyak:
            self.sync_weight(self.polyak.polyaks.detach())
        else:
            self.sync_weight([self._tau] * self.polyak.size)


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


        ###################################################################
        # with torch.no_grad():
        #     critic1_old_param = []
        #     critic2_old_param = []
        #     polyaks = self.polyak.polyaks
        #     for p, o, n in zip(
        #             polyaks, self.critic1_old.parameters(), self.critic1.parameters()):
        #         critic1_old_param.append(o.detach() * (1 - p) + n.detach() * p)
        #     for p, o, n in zip(
        #             polyaks, self.critic2_old.parameters(), self.critic2.parameters()):
        #         critic2_old_param.append(o.detach() * (1 - p) + n.detach() * p)
        #     obs_result_tmp = self(batch)
        #     a = obs_result_tmp.act
        #
        #     current_q1 = self.critic1_old(batch.obs, a.detach(), vars=critic1_old_param)
        #     current_q2 = self.critic2_old(batch.obs, a.detach(), vars=critic2_old_param)
        #     current_qa = torch.min(current_q1, current_q2)
        #
        #     current_q1 = self.critic1_old(batch.obs, batch_act, vars=critic1_old_param)
        #     current_q2 = self.critic2_old(batch.obs, batch_act, vars=critic2_old_param)
        #     current_qb = torch.min(current_q1, current_q2)
        #
        #     opt_rand = (current_qa > current_qb).float().detach()  # those qualified
        #
        #     with torch.no_grad():
        #         obs_result_tmp = self(batch, 'obs_next')
        #         a_next = obs_result_tmp.act
        #     current_next_q1 = self.critic1_old(batch.obs_next, a_next.detach(), vars=critic1_old_param)
        #     current_next_q2 = self.critic2_old(batch.obs_next, a_next.detach(), vars=critic2_old_param)
        #     current_next_q = torch.min(current_next_q1,
        #                                current_next_q2) - self._alpha.detach() * obs_result_tmp.log_prob
        #
        #     pred_rew = current_qb - (1. - done) * self._gamma * current_next_q
        #
        #     rew_loss = (F.mse_loss(pred_rew, rew, reduction='none') * opt_rand).sum() / opt_rand.sum()
        #
        #     current_q1 = self.critic1(batch.obs, batch_act)
        #     current_q2 = self.critic2(batch.obs, batch_act)
        #     current_qb = torch.min(current_q1, current_q2)
        #
        #     current_next_q1 = self.critic1(batch.obs_next, a_next.detach())
        #     current_next_q2 = self.critic2(batch.obs_next, a_next.detach())
        #     current_next_q = torch.min(current_next_q1,
        #                                current_next_q2) - self._alpha.detach() * obs_result_tmp.log_prob
        #
        #     pred_rew = current_qb - (1. - done) * self._gamma * current_next_q
        #
        #     cur_rew_loss = (F.mse_loss(pred_rew, rew, reduction='none') * opt_rand).sum() / opt_rand.sum()
        #
        # polyak_loss = (self.polyak.polyaks.log() * (cur_rew_loss - rew_loss)).mean()
        #
        # self.polyak_optim.zero_grad()
        # polyak_loss.backward()
        # self.polyak_optim.step()

        ##################################################################

        # if self.use_autopolyak:
        #     self.sync_weight(self.polyak.polyaks.detach())
        # else:
        #     self.sync_weight([self._tau] * self.polyak.size)

        try:
            alpha = self._alpha.item()
        except Exception as e:
            alpha = self._alpha

        result = {
            'loss/actor': actor_loss.item(),
            'loss/critic1': critic1_loss.item(),
            'loss/critic2': critic2_loss.item(),
            'loss/polyak': polyloss.item(),
            'alpha': alpha,
            'entropy': obs_result.log_prob.mean().item(),
            # 'opt_rand': opt_rand.mean().item(),
            # 'cur_dis': (cur_rew_loss > rew_loss).float().item(),
        }
        for i in range(self.polyak.size):
            result['polyak/' + str(i)] = self.polyak.polyaks[i].item()

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




# with torch.no_grad():
        #     obs_result = self(batch)
        #     a = obs_result.act
        #     current_q1 = self.critic1_old(batch.obs, a)
        #     current_q2 = self.critic2_old(batch.obs, a)
        #     # current_q1 = self.critic1(batch.obs, batch_act)
        #     # current_q2 = self.critic2(batch.obs, batch_act)
        #     current_q = torch.min(current_q1, current_q2)# - self._alpha * obs_result.log_prob
        #
        #     obs_next_result = self(batch, 'obs_next')
        #     a_next = obs_next_result.act
        #     current_next_q1 = self.critic1_old(batch.obs_next, a_next)
        #     current_next_q2 = self.critic2_old(batch.obs_next, a_next)
        #     current_next_q = torch.min(current_next_q1, current_next_q2) - self._alpha * obs_next_result.log_prob
        #
        #     pred_rew = current_q - self.dist(batch.obs, batch_act, a_next) - (1. - done) * self._gamma * current_next_q
        #
        # # pred_rew = pred_rew / self.dist.scale # scale should be positive
        # rew_loss = F.mse_loss(rew * self.dist.scale, pred_rew)
        #
        # self.dist_optim.zero_grad()
        # rew_loss.backward()
        # self.dist_optim.step()