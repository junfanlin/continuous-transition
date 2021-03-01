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


def simulate_beta(alpha, size=1, rng=np.random):
    """
    from binary to uniform
    alpha (0, 1]
    """
    return (rng.randn(size) * alpha / 2) % 1




class SACFEMBPolicy(DDPGPolicy):
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
        self.process_tri = kwargs.get('process_tri', (lambda x: x))
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
        if self.beta is None:
            # beta_a, beta_b = torch.ones(1), torch.ones(1)
            beta = torch.ones(1) * 1e-5
        else:
            beta = torch.exp(self.beta)

        beta_dist = torch.distributions.beta.Beta(beta, 1)
        blend_ratio = beta_dist.rsample((256,))

        # batch = self.process_tri(batch, blend_ratio)
        # batch_val = batch[256:]
        batch_val = batch[:256]
        batch = batch[:256]

        ############## PROCESS DATA###################
        obs = torch.FloatTensor(batch.obs)
        act = torch.FloatTensor(batch.act)
        obs_next = torch.FloatTensor(batch.obs_next)
        rew = torch.FloatTensor(batch.rew)[:, None]
        done = torch.FloatTensor(batch.done)[:, None]
        done_bk = torch.FloatTensor(batch.done_bk)[:, None]

        act_next = torch.FloatTensor(batch.act_next)
        obs_next_next = torch.FloatTensor(batch.obs_next_next)
        rew_next = torch.FloatTensor(batch.rew_next)[:, None]
        done_next = torch.FloatTensor(batch.done_next)[:, None]

        blend_ratio = (1 - done_bk) * blend_ratio

        batch.obs_blend = obs + blend_ratio * (obs_next - obs)
        batch.obs_next_blend = obs_next + blend_ratio * (obs_next_next - obs_next)
        batch.act_blend = act + blend_ratio * (act_next - act)
        batch.rew_blend = rew + blend_ratio * (rew_next - rew)
        batch.done_blend = done + blend_ratio * (done_next - done)
        ##############################################

        ratio = 1

        if not self.use_diff:
            ratio = 1
            ratio_log = 1

        # with torch.no_grad():
        obs_next_result = self(batch, input='obs_next_blend')
        a_ = obs_next_result.act
        dev = a_.device
        # batch_act = torch.tensor(batch.act, dtype=torch.float, device=dev)
        batch_act = batch.act_blend
        target_q = torch.min(
            self.critic1_old(batch.obs_next_blend, a_),
            self.critic2_old(batch.obs_next_blend, a_),
        ) - self._alpha * obs_next_result.log_prob
        rew = batch.rew_blend # torch.tensor(batch.rew, dtype=torch.float, device=dev)[:, None]
        done = batch.done_blend # torch.tensor(batch.done, dtype=torch.float, device=dev)[:, None]
        target_q = (rew + (1. - done) * self._gamma * target_q)

        fast_weight_c1 = [param for param in self.critic1.parameters()]
        fast_weight_c2 = [param for param in self.critic2.parameters()]

        # critic 1
        current_q1 = self.critic1(batch.obs, batch_act, vars=fast_weight_c1)
        critic1_loss = ((current_q1 - target_q).square() * ratio).mean()
        # critic 2
        current_q2 = self.critic2(batch.obs, batch_act, vars=fast_weight_c2)
        critic2_loss = ((current_q2 - target_q).square() * ratio).mean()

        grad = torch.autograd.grad(critic1_loss, fast_weight_c1, create_graph=True, retain_graph=True)
        # self.clip_grad_norm(grad, config.gradient_clip)
        fast_weight_c1 = self.critic1_optim.step(fast_weight_c1, grad)
        # self.critic1_optim.zero_grad()
        # critic1_loss.backward()
        # self.critic1_optim.step()

        grad = torch.autograd.grad(critic2_loss, fast_weight_c2, create_graph=True, retain_graph=True)
        # self.clip_grad_norm(grad, config.gradient_clip)
        fast_weight_c2 = self.critic2_optim.step(fast_weight_c2, grad)
        # self.critic2_optim.zero_grad()
        # critic2_loss.backward()
        # self.critic2_optim.step()

        if self.beta is not None:
            obs_result_tmp = self(batch_val, input='obs_next')
            a = obs_result_tmp.act.detach()
            done = torch.FloatTensor(batch_val.done)[:, None]
            rew = torch.FloatTensor(batch_val.rew)[:, None]
            current_q = torch.min(self.critic1(batch_val.obs, batch_val.act, vars=fast_weight_c1),
                                  self.critic2(batch_val.obs, batch_val.act, vars=fast_weight_c2))
            current_targ_q = torch.min(self.critic1_old(batch_val.obs_next, a),
                                       self.critic2_old(batch_val.obs_next, a)) - self._alpha * obs_result_tmp.log_prob
            current_rew = current_q - (1 - done) * self._gamma * current_targ_q
            pred_loss = F.mse_loss(current_rew, rew)

            self.beta_optim.zero_grad()
            pred_loss.backward(retain_graph=True)
            self.beta_optim.step()

            beta = torch.exp(self.beta)

        for o, n in zip(self.critic1.parameters(), fast_weight_c1):
            o.data.copy_(n.data)

        for o, n in zip(self.critic2.parameters(), fast_weight_c2):
            o.data.copy_(n.data)

        obs_result = self(batch, input='obs_blend') # obs_bk
        a = obs_result.act
        current_q1a = self.critic1(batch.obs_blend, a)
        current_q2a = self.critic2(batch.obs_blend, a)
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
            'entropy': obs_result.log_prob.mean().item(),
        }
        if self.beta is not None:
            result['loss/val'] = pred_loss.item(),

        result['beta'] = beta.item()

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
