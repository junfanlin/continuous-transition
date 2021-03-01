import torch
import numpy as np
from copy import deepcopy
import torch.nn.functional as F
from typing import Dict, Tuple, Union, Optional

from tianshou.policy import DDPGPolicy
from tianshou.policy.dist import DiagGaussian
from tianshou.data import Batch, to_torch_as, ReplayBuffer
from tianshou.exploration import BaseNoise


def sigmoid(x):
    return 1./(np.exp(-x) + 1.)

class SACNGUPolicy(DDPGPolicy):
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
                 embnet: torch.nn.Module,
                 embnet_optim: torch.optim.Optimizer,
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
                 n_policies: int = 32,
                 beta: float = 0.3,
                 gamma_min: float = 0.99,
                 gamma_max: float = 0.997,
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

        self.embnet, self.embnet_optim = embnet, embnet_optim

        self.rand_target = rand_target.eval()
        self.rand_source = rand_source
        self.rand_optim = rand_optim

        self.rand_mean = None
        self.rand_std = None

        self.rew_intrisic_mean = None
        self.rew_intrisic_std = None

        self.n_policies = n_policies
        self.betas = sigmoid(10 * (2 * np.arange(n_policies) - n_policies + 2)/(n_policies - 2)) * beta
        self.betas[0] = 0
        self.betas[-1] = beta
        self.gammas = np.array([(1 - np.exp(((n_policies - 1 - i) * np.log(1 - gamma_max) + i * np.log(1 - gamma_min))
                                  / (n_policies - 1))) for i in range(n_policies)])

        self.init_goal()

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
        self.embnet.train(mode)
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
        if kwargs.get('deterministic', False):
            goal = np.zeros((1, self.n_policies))
            goal[0, 0] = 1
            # obs = np.concatenate([obs, goal], -1)
            logits, h = self.actor(obs, goal, state=state, info=batch.info)
            assert isinstance(logits, tuple)
            mu, sigma = logits
            log_prob = None
            dist = None
            act = torch.tanh(mu)
        else:
            goal = kwargs.get('goal', self.goal)
            # obs = np.concatenate([obs, goal], -1)
            logits, h = self.actor(obs, goal, state=state, info=batch.info)
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
        batch.obs_next = torch.FloatTensor(batch.obs_next, device=self.actor.device)
        if self.rand_mean is None:
            self.rand_mean = batch.obs_next.mean().detach()  # 0, keepdim=True).detach()
            self.rand_std = batch.obs_next.std().detach()  # 0, keepdim=True).detach() / 5 + self.__eps

        norm_obs_next = (batch.obs_next - self.rand_mean) / self.rand_std

        pred_target = self.rand_target(norm_obs_next)[0].detach()
        pred_source = self.rand_source(norm_obs_next)[0]
        rew_intrisic = F.mse_loss(pred_source, pred_target, reduction='none').mean(-1, keepdim=True)
        if self.rew_intrisic_mean is None:
            self.rew_intrisic_mean = rew_intrisic.mean().item()
            self.rew_intrisic_std = rew_intrisic.std().item()
        else:
            self.rew_intrisic_mean = self.rew_intrisic_mean * 0.9 + rew_intrisic.mean().item() * 0.1
            self.rew_intrisic_std = self.rew_intrisic_std * 0.9 + rew_intrisic.std().item() * 0.1

        alpha = (1 + (rew_intrisic - self.rew_intrisic_mean) / (self.rew_intrisic_std + 1e-6)).clamp(0, 5)

        self.rand_optim.zero_grad()
        rew_intrisic.mean().backward()
        self.rand_optim.step()


        goal = np.eye(self.n_policies)[batch.goal]
        # Critic Loss
        with torch.no_grad():
            obs_next_result = self(batch, input='obs_next', goal=goal)
            a_ = obs_next_result.act
            dev = a_.device
            batch_act = torch.tensor(batch.act, dtype=torch.float, device=dev)
            target_q = torch.min(
                self.critic1_old(batch.obs_next, goal, a_),
                self.critic2_old(batch.obs_next, goal, a_),
            ) - self._alpha * obs_next_result.log_prob

            beta = torch.tensor(self.betas[batch.goal],
                                dtype=torch.float, device=dev)[:, None]
            gamma = torch.tensor(self.gammas[batch.goal],
                                 dtype=torch.float, device=dev)[:, None]
            rew_ext = torch.tensor(batch.rew,
                               dtype=torch.float, device=dev)[:, None]
            rew_int = torch.tensor(batch.erew,
                               dtype=torch.float, device=dev)[:, None] * alpha
            rew = rew_ext + beta * rew_int
            done = torch.tensor(batch.done,
                                dtype=torch.float, device=dev)[:, None]
            target_q = (rew + (1. - done) * gamma * target_q)

        logits_emb = self.embnet.inverse_actor(batch.obs, batch.obs_next)
        dist = torch.distributions.Normal(*logits_emb)
        emb_loss = -dist.log_prob(batch_act).mean()
        self.embnet_optim.zero_grad()
        emb_loss.backward()
        self.embnet_optim.step()

        # critic 1
        current_q1 = self.critic1(batch.obs, goal, batch_act)
        critic1_loss = F.mse_loss(current_q1, target_q)
        # critic 2
        current_q2 = self.critic2(batch.obs, goal, batch_act)
        critic2_loss = F.mse_loss(current_q2, target_q)

        self.critic1_optim.zero_grad()
        critic1_loss.backward()
        self.critic1_optim.step()

        self.critic2_optim.zero_grad()
        critic2_loss.backward()
        self.critic2_optim.step()

        obs_result = self(batch, goal=goal)
        a = obs_result.act
        current_q1a = self.critic1(batch.obs, goal, a)
        current_q2a = self.critic2(batch.obs, goal, a)
        actor_loss = (self._alpha * obs_result.log_prob - torch.min(
            current_q1a, current_q2a)).mean()

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
            bfr = buffer.rew[:min(len(buffer), 1000)]  # avoid large buffer
            mean, std = bfr.mean(), bfr.std()
            if np.isclose(std, 0):
                mean, std = 0, 1
        else:
            mean, std = 0, 1
        batch.rew = (batch.rew - mean) / std
        return batch

    def init_goal(self):
        self._current_goal_id = np.random.randint(self.n_policies)
        self.goal = np.zeros((1, self.n_policies))
        self.goal[0, self._current_goal_id] = 1


    def assign_ngu(self, buffer, length, batch_size):
        # cal episodic reward
        indexes = []
        cur_ind = buffer._index - 1
        for i in range(length):
            indexes = [cur_ind] + indexes
            cur_ind -= 1
            if cur_ind < 0:
                cur_ind = buffer._size - 1
        indexes = np.array(indexes)

        embs = None
        for i in range(0, length, batch_size):
            indexes_ = indexes[i:np.minimum(i+batch_size, length)]
            batch = buffer[indexes_]
            with torch.no_grad():
                emb = self.embnet(batch.obs).data.numpy()
                if embs is None:
                    embs = emb
                else:
                    embs = np.concatenate([embs, emb], 0)

        buffer.erew[indexes[0]] = 0
        buffer[indexes[0]].episodic_reward = 0
        for i in range(1, length):
            distes = ((embs[:i] - embs[[i]]) ** 2).sum(-1)
            distes = distes[distes.argsort()[:10]] # top 10 smallest
            distes = distes / distes.mean()
            distes = np.maximum(distes - 0.008, 0)
            kv = 1e-4 / (distes + 1e-4)
            s = np.sqrt(kv.sum()) + 1e-3
            if s > 8:
                episodic_reward = 0
            else:
                episodic_reward = 1 / s
            buffer.erew[indexes[i]] = episodic_reward

        # assign goal
        for i in range(length):
            buffer.goal[indexes[i]] = self._current_goal_id
