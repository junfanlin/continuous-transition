import torch
import numpy as np
from copy import deepcopy
import torch.nn.functional as F
from typing import Dict, Tuple, Union, Optional, List

from tianshou.policy import DDPGPolicy
from tianshou.policy.dist import DiagGaussian
from tianshou.data import Batch, to_torch_as, ReplayBuffer
from tianshou.exploration import BaseNoise


class SACENSPolicy(DDPGPolicy):
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
                 l_actor: List[torch.nn.Module],
                 l_actor_optim: List[torch.optim.Optimizer],
                 l_critic1: List[torch.nn.Module],
                 l_critic1_optim: List[torch.optim.Optimizer],
                 l_critic2: List[torch.nn.Module],
                 l_critic2_optim: List[torch.optim.Optimizer],
                 tau: float = 0.005,
                 gamma: float = 0.99,
                 l_alpha: List[Tuple[float, torch.Tensor, torch.optim.Optimizer]]
                 or List[float] = [0.2],
                 action_range: Optional[Tuple[float, float]] = None,
                 reward_normalization: bool = False,
                 ignore_done: bool = False,
                 estimation_step: int = 1,
                 exploration_noise: Optional[BaseNoise] = None,
                 **kwargs) -> None:
        super().__init__(None, None, None, None, tau, gamma, exploration_noise,
                         action_range, reward_normalization, ignore_done,
                         estimation_step, **kwargs)

        self.ens_num = kwargs.get('ens_num', None)
        self._lambda = kwargs.get('lamba', None)
        assert self.ens_num is not None, "SAC Ensembles requires ens_num"
        assert self._lambda is not None, "SAC Ensembles requires lamba"

        self.l_actor, self.l_actor_optim = l_actor, l_actor_optim
        self.l_critic1, self.l_critic1_old = l_critic1, [deepcopy(critic).eval() for critic in l_critic1]
        self.l_critic1_optim = l_critic1_optim
        self.l_critic2, self.l_critic2_old = l_critic2, [deepcopy(critic).eval() for critic in l_critic2]
        self.l_critic2_optim = l_critic2_optim

        self.norm_func = kwargs.get('norm_func', None)
        self._automatic_alpha_tuning = not isinstance(l_alpha[0], float)
        if self._automatic_alpha_tuning:
            self._target_entropy = l_alpha[0][0]
            assert(l_alpha[0][1].shape == torch.Size([1])
                   and l_alpha[0][1].requires_grad)
            self._l_log_alpha = [alpha[1] for alpha in l_alpha]
            self._l_alpha_optim = [alpha[2] for alpha in l_alpha]
            self._l_alpha = [log_alpha.exp() for log_alpha in self._l_log_alpha]
        else:
            self._l_alpha = l_alpha

        self.__eps = np.finfo(np.float32).eps.item()

    def train(self, mode=True) -> torch.nn.Module:
        self.training = mode
        [actor.train(mode) for actor in self.l_actor]
        [critic.train(mode) for critic in self.l_critic1]
        [critic.train(mode) for critic in self.l_critic2]
        return self

    def sync_weight(self) -> None:
        for critic_old, critic in zip(self.l_critic1_old, self.l_critic1):
            for o, n in zip(
                    critic_old.parameters(), critic.parameters()):
                o.data.copy_(o.data * (1 - self._tau) + n.data * self._tau)

        for critic_old, critic in zip(self.l_critic2_old, self.l_critic2):
            for o, n in zip(
                    critic_old.parameters(), critic.parameters()):
                o.data.copy_(o.data * (1 - self._tau) + n.data * self._tau)

    def forward(self, batch: Batch,
                state: Optional[Union[dict, Batch, np.ndarray]] = None,
                input: str = 'obs',
                explorating: bool = True,
                **kwargs) -> Batch:
        obs = getattr(batch, input)
        l_batch = []

        for i in range(self.ens_num):
            logits, h = self.l_actor[i](obs, state=state, info=batch.info)
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

            l_batch.append(Batch(
                logits=logits, act=act, state=h, dist=dist, log_prob=log_prob))

            if explorating and not kwargs.get('deterministic', False):
                l_act_q = []
                for j in range(self.ens_num):
                    val = self.l_critic1[j](obs, act).squeeze(-1)
                    l_act_q.append(val)

                    val = self.l_critic2[j](obs, act).squeeze(-1)
                    l_act_q.append(val)

                l_act_q = torch.cat(l_act_q, -1)
                act_mean = l_act_q.mean(-1)
                act_std = l_act_q.std(-1)
                act_val = act_mean + self._lambda * act_std

                if i == 0:
                    max_act_val = act_val
                    batch_ret = l_batch[-1]
                elif act_val > max_act_val:
                    max_act_val = act_val
                    batch_ret = l_batch[-1]

        if kwargs.get('deterministic', False):
            act_ret = torch.cat([b.act[:, None] for b in l_batch], -2).mean(-2)
            return Batch(act=act_ret)
        elif explorating:
            return batch_ret
        else:
            return l_batch

    def learn(self, batch: Batch, **kwargs) -> Dict[str, float]:
        l_actor_loss = []
        l_critic1_loss = []
        l_critic2_loss = []
        if self._automatic_alpha_tuning:
            l_alpha_loss = []

        device = self.l_critic1[0].device
        batch.act = torch.FloatTensor(batch.act, device=device)
        batch.mask = torch.FloatTensor(batch.mask, device=device)
        with torch.no_grad():
            l_obs_next_result = self(batch, input='obs_next', explorating=False)

        for i in range(self.ens_num):
            with torch.no_grad():
                b = l_obs_next_result[i]
                a_ = l_obs_next_result[i].act
                target_q = torch.min(
                    self.l_critic1_old[i](batch.obs_next, a_),
                    self.l_critic2_old[i](batch.obs_next, a_),
                ) - self._l_alpha[i] * b.log_prob
                rew = to_torch_as(batch.rew, a_)[:, None]
                done = to_torch_as(batch.done, a_)[:, None]
                target_q = (rew + (1. - done) * self._gamma * target_q)

            if batch.mask[:, i].sum() > 0:
                # critic 1
                current_q1 = self.l_critic1[i](batch.obs, batch.act)
                critic1_loss = (F.mse_loss(current_q1, target_q, reduction='none') * batch.mask[:, [i]]).sum() / batch.mask[:, [i]].sum()
                # critic 2
                current_q2 = self.l_critic2[i](batch.obs, batch.act)
                critic2_loss = (F.mse_loss(current_q2, target_q, reduction='none') * batch.mask[:, [i]]).sum() / batch.mask[:, [i]].sum()

                self.l_critic1_optim[i].zero_grad()
                critic1_loss.backward()
                self.l_critic1_optim[i].step()

                self.l_critic2_optim[i].zero_grad()
                critic2_loss.backward()
                self.l_critic2_optim[i].step()
                # actor

                logits, h = self.l_actor[i](batch.obs)
                assert isinstance(logits, tuple)
                dist = torch.distributions.Normal(*logits)
                x = dist.rsample()
                y = torch.tanh(x)
                log_prob = (dist.log_prob(x) - torch.log(
                    self._action_scale * (1 - y.pow(2)) + self.__eps)).sum(-1, keepdim=True)
                a = y.clamp(self._range[0], self._range[1])

                current_q1a = self.l_critic1[i](batch.obs, a)
                current_q2a = self.l_critic2[i](batch.obs, a)
                actor_loss = ((self._l_alpha[i] * log_prob - torch.min(current_q1a, current_q2a))
                              * batch.mask[:, [i]]).sum() / batch.mask[:, [i]].sum()

                self.l_actor_optim[i].zero_grad()
                actor_loss.backward()
                self.l_actor_optim[i].step()

                if self._automatic_alpha_tuning:
                    log_prob = (log_prob + self._target_entropy).detach()
                    alpha_loss = -(self._l_log_alpha[i] * log_prob).mean()
                    self._l_alpha_optim[i].zero_grad()
                    alpha_loss.backward()
                    self._l_alpha_optim[i].step()
                    self._l_alpha[i] = self._l_log_alpha[i].exp()
                    l_alpha_loss.append(alpha_loss.item())

                l_actor_loss.append(actor_loss.item())
                l_critic1_loss.append(critic1_loss.item())
                l_critic2_loss.append(critic2_loss.item())

        self.sync_weight()

        result = {
            'loss/actor': np.mean(l_actor_loss),
            'loss/critic1': np.mean(l_critic1_loss),
            'loss/critic2': np.mean(l_critic2_loss)
        }
        if self._automatic_alpha_tuning:
            result['loss/alpha'] = np.mean(l_alpha_loss)
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
