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


class SACIDNPolicy(DDPGPolicy):
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
        self.embedder, self.embedder_optim = kwargs['embed']

        self.norm_func = kwargs.get('norm_func', None)
        self.action_num = kwargs.get('action_num', 10)


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
        self.embedder.train(mode)
        return self

    def sync_weight(self):
        for o, n in zip(
                self.critic1_old.parameters(), self.critic1.parameters()):
            o.data.copy_(o.data * (1 - self._tau) + n.data * self._tau)
        for o, n in zip(
                self.critic2_old.parameters(), self.critic2.parameters()):
            o.data.copy_(o.data * (1 - self._tau) + n.data * self._tau)

    def _ind_bias(self, obs, acts, hist):
        # only 1 currently
        max_len = np.minimum(50, len(hist[0]))

        # notice: since we take from the end, the similarity is also descend
        # obs_hist = np.concatenate([[hist_[-1-i].obs for i in range(max_len)] for hist_ in hist], 0)
        # act_hist = np.concatenate([[hist_[-1-i].act for i in range(max_len)] for hist_ in hist], 0)
        obs_next_hist = np.concatenate([[hist_[-1-i].obs_next for i in range(max_len)] for hist_ in hist], 0)

        batch_size, state_size = obs.shape
        act_size = acts.shape[-1]
        obses = np.tile(obs, (self.action_num, 1, 1))

        with torch.no_grad():
            emb_sas = self.embedder.forward_sa(obses.reshape(-1, state_size), acts.reshape(-1, act_size)).\
                view(self.action_num, batch_size, -1)

            # emb_sa_hist = self.embedder.forward_sa(obs_hist, act_hist)
            emb_s_next_hist = self.embedder.forward_s(obs_next_hist)

            # sim = torch.nn.functional.cosine_similarity(emb_sas - emb_sa_hist[None], emb_s_next_hist[None], -1, 1e-6)

            sim = torch.nn.functional.cosine_similarity(emb_sas, emb_s_next_hist[None], -1, 1e-6)
            # sim_base = torch.nn.functional.cosine_similarity(emb_sa_hist, emb_s_next_hist, -1, 1e-6)
            # sim = sim - sim_base[None]
            # sim = -torch.nn.functional.relu(-sim)

        return sim.min(-1, keepdim=True)[0].argmin(0) # choose the action minimize the similarity
        # return torch.zeros(obs.shape[0]).long() # sim.max(-1, keepdim=True)[0].argmin(0) # choose the action minimize the similarity

    def forward(self, batch, state=None, input='obs', **kwargs):
        obs = getattr(batch, input)
        logits, h = self.actor(obs, state=state, info=batch.info)
        assert isinstance(logits, tuple)

        mu, sigma = logits

        log_prob = None
        dist = None

        hist = kwargs.get("hist", None)

        if kwargs.get('deterministic', False):
            act = (torch.tanh(mu) * self._action_scale).clamp(self._range[0], self._range[1])
        else:
            dist = torch.distributions.Normal(*logits)
            # if len(batch) > 1:
            #     print("wtf")
            if hist is not None and len(hist[0]) > 0:
                xs = dist.rsample([self.action_num]) #.permute((1, 0, 2))
                ys = torch.tanh(xs)
                acts = (ys * self._action_scale).clamp(self._range[0], self._range[1])

                index = self._ind_bias(obs, acts, hist)
                x = xs[index, np.arange(xs.shape[1])]
            else:
                x = dist.rsample()
            y = torch.tanh(x)
            act = (y * self._action_scale).clamp(self._range[0], self._range[1])

            log_prob = (dist.log_prob(x) - torch.log(
                self._action_scale * (1 - y.pow(2)) + self.__eps)).sum(-1, keepdim=True) # the smaller the log_prob is, the better, y->0? because it's normalised.

        return Batch(logits=logits, act=act, state=h, dist=dist, log_prob=log_prob)

    def learn(self, batch, **kwargs):
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

        rand_obs = np.array(batch.obs)
        np.random.shuffle(rand_obs)
        emb_next = self.embedder.forward_s(batch.obs_next)
        emb_rand = self.embedder.forward_s(rand_obs)
        emb_sa = self.embedder.forward_sa(batch.obs, batch_act)

        pos_sim = (1 - torch.nn.functional.cosine_similarity(emb_sa, emb_next)).mean()
        neg_sim = torch.nn.functional.relu(torch.nn.functional.cosine_similarity(emb_sa, emb_rand)).mean()

        self.embedder_optim.zero_grad()
        (pos_sim + neg_sim).backward()
        self.embedder_optim.step()


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
            'loss/pos_sim': pos_sim.item(),
            'loss/neg_sim': neg_sim.item(),
            'alpha': alpha,
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
