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

from scipy import stats

class SACMUTRIHSPolicy(DDPGPolicy):
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

        self.class_num = kwargs.get('class_num', 4)
        self.goal = torch.zeros(self.class_num)
        self.mapper = np.random.rand(np.prod(self.actor.action_shape) + np.prod(self.actor.state_shape), self.class_num)

        self.norm_func = kwargs.get('norm_func', None)
        self.process_tri = kwargs.get('process_tri', (lambda x: x))

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

        deterministic = kwargs.get('deterministic', False)

        goal = kwargs.get('goal', None)
        if goal is None:
            goal = self.goal
            # if deterministic: # what goal should we choose when exploration / exploitation ?
            #     goal = torch.zeros(obs.shape[0], 1)
            # else:
            #     goal = self.goal

        logits, h = self.actor(obs, state=state, info=batch.info, goal=goal)
        assert isinstance(logits, tuple)

        mu, sigma = logits

        log_prob = None
        dist = None
        if deterministic:
            act = torch.tanh(mu)
        else:
            dist = torch.distributions.Normal(*logits)
            x = dist.rsample()
            y = torch.tanh(x)
            log_prob = (dist.log_prob(x) - torch.log(
                self._action_scale * (1 - y.pow(2)) + self.__eps)).sum(-1, keepdim=True)  # the smaller the log_prob is, the better, y->0? because it's normalised.
            act = y

        act = act.clamp(self._range[0], self._range[1])
        act_determ = torch.tanh(mu).clamp(self._range[0], self._range[1])

        return Batch(logits=logits, act=act, state=h, dist=dist, log_prob=log_prob, act_determ=act_determ)

    def learn(self, batch, **kwargs):
        # Critic Loss
        batch = self.process_tri(batch)

        with torch.no_grad():
            goal = self.goal # should we sample a better goal?
            obs_next_result = self(batch, input='obs_next', goal=goal)
            a_ = obs_next_result.act
            dev = a_.device
            batch_act = torch.tensor(batch.act, dtype=torch.float, device=dev)
            target_q = torch.min(
                self.critic1_old(batch.obs_next, a_, goal=goal),
                self.critic2_old(batch.obs_next, a_, goal=goal),
            ) - self._alpha * obs_next_result.log_prob
            rew = torch.tensor(batch.rew,
                               dtype=torch.float, device=dev)[:, None]
            done = torch.tensor(batch.done,
                                dtype=torch.float, device=dev)[:, None]
            target_q = (rew + (1. - done) * self._gamma * target_q)


            ######################### context ############################
            current_q1 = self.critic1_old(batch.obs, batch_act, goal=goal)
            current_q2 = self.critic2_old(batch.obs, batch_act, goal=goal)
            current_qb = torch.min(current_q1, current_q2)
            current_next_q1 = self.critic1_old(batch.obs_next, obs_next_result.act_determ, goal=goal)
            current_next_q2 = self.critic2_old(batch.obs_next, obs_next_result.act_determ, goal=goal)
            current_next_q = torch.min(current_next_q1, current_next_q2)
            target_q_goal = (1. - done) * self._gamma * current_next_q
            pred_rew = current_qb - target_q_goal
            old_rew_loss = F.mse_loss(pred_rew, rew, reduction='none')

            logits = np.dot(np.concatenate([batch.obs, batch.act], -1), self.mapper)
            prob = torch.softmax(torch.FloatTensor(logits), -1)
            goal_prob = prob * old_rew_loss
            goal = goal_prob.mean(0)

            kendal, _ = stats.kendalltau(pred_rew.numpy(), batch.rew[:, None])
            # relabled transition, obs, act, obs_next, done, pred_rew, goal

            self.goal = goal # ?
            # goal = self.goal
            ##############################################################

        # critic 1
        # current_q1 = self.critic1(batch.obs, batch_act, goal=goal)
        # critic1_loss_goal = F.mse_loss(current_q1, current_qb)
        current_q1 = self.critic1(batch.obs, batch_act, goal=goal)#torch.zeros(self.class_num))
        critic1_loss = F.mse_loss(current_q1, target_q)
        # critic 2
        # current_q2 = self.critic2(batch.obs, batch_act, goal=goal)
        # critic2_loss_goal = F.mse_loss(current_q2, current_qb)
        current_q2 = self.critic2(batch.obs, batch_act, goal=goal)#torch.zeros(self.class_num))
        critic2_loss = F.mse_loss(current_q2, target_q)

        self.critic1_optim.zero_grad()
        # (critic1_loss + critic1_loss_goal).backward()
        critic1_loss.backward()
        self.critic1_optim.step()

        self.critic2_optim.zero_grad()
        # (critic2_loss + critic2_loss_goal).backward()
        critic2_loss.backward()
        self.critic2_optim.step()

        obs_result = self(batch, goal=goal)
        a = obs_result.act
        current_q1a = self.critic1(batch.obs, a, goal=goal)
        current_q2a = self.critic2(batch.obs, a, goal=goal)
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
            'kendal': kendal,
            'entropy': obs_result.log_prob.mean().item()
        }
        for i in range(self.actor.goalsize):
            result['goal/'+str(i)] = goal[i]


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
