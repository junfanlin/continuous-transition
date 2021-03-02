import torch
import numpy as np
from copy import deepcopy
import torch.nn.functional as F
from typing import Dict, Tuple, Optional

from tianshou.policy import DDPGPolicy
from tianshou.data import Batch, ReplayBuffer
from tianshou.exploration import BaseNoise, GaussianNoise

from tianshou.policy import BasePolicy
from tianshou.exploration import BaseNoise, GaussianNoise
from tianshou.data import Batch, ReplayBuffer, to_torch_as
from typing import Dict, Tuple, Union, Optional

class TD3MUTRIRB2BPolicy(DDPGPolicy):
    """Implementation of Twin Delayed Deep Deterministic Policy Gradient,
    arXiv:1802.09477

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
    :param float exploration_noise: the exploration noise, add to the action,
        defaults to ``GaussianNoise(sigma=0.1)``
    :param float policy_noise: the noise used in updating policy network,
        default to 0.2.
    :param int update_actor_freq: the update frequency of actor network,
        default to 2.
    :param float noise_clip: the clipping range used in updating policy
        network, default to 0.5.
    :param action_range: the action range (minimum, maximum).
    :type action_range: (float, float)
    :param bool reward_normalization: normalize the reward to Normal(0, 1),
        defaults to ``False``.
    :param bool ignore_done: ignore the done flag while training the policy,
        defaults to ``False``.

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
                 exploration_noise: Optional[BaseNoise]
                 = GaussianNoise(sigma=0.1),
                 policy_noise: float = 0.2,
                 update_actor_freq: int = 2,
                 noise_clip: float = 0.5,
                 action_range: Optional[Tuple[float, float]] = None,
                 reward_normalization: bool = False,
                 ignore_done: bool = False,
                 estimation_step: int = 1,
                 **kwargs) -> None:
        super().__init__(actor, actor_optim, None, None, tau, gamma,
                         exploration_noise, action_range, reward_normalization,
                         ignore_done, estimation_step, **kwargs)
        self.critic1, self.critic1_old = critic1, deepcopy(critic1)
        self.critic1_old.eval()
        self.critic1_optim = critic1_optim
        self.critic2, self.critic2_old = critic2, deepcopy(critic2)
        self.critic2_old.eval()
        self.critic2_optim = critic2_optim

        self.norm_func = kwargs.get('norm_func', None)
        self.process_tri = kwargs.get('process_tri', (lambda x, beta: x))
        self.disc, self.disc_optim = kwargs.get('discriminator', [None, None])
        self.beta, self.beta_optim = kwargs.get('beta', [None, None])
        self.norm_diff = kwargs.get('norm_diff', False)
        self.tor_diff = kwargs.get('tor_diff', 0.1)
        self.use_diff = kwargs.get('use_diff', True)

        self._policy_noise = policy_noise
        self._freq = update_actor_freq
        self._noise_clip = noise_clip
        self._cnt = 0
        self._last = 0

    def train(self, mode=True) -> torch.nn.Module:
        self.training = mode
        self.actor.train(mode)
        self.critic1.train(mode)
        self.critic2.train(mode)
        return self

    def forward(self, batch: Batch,
                state: Optional[Union[dict, Batch, np.ndarray]] = None,
                model: str = 'actor',
                input: str = 'obs',
                explorating: bool = True,
                **kwargs) -> Batch:
        """Compute action over the given batch data.

        :return: A :class:`~tianshou.data.Batch` which has 2 keys:

            * ``act`` the action.
            * ``state`` the hidden state.

        .. seealso::

            Please refer to :meth:`~tianshou.policy.BasePolicy.forward` for
            more detailed explanation.
        """
        model = getattr(self, model)
        obs = getattr(batch, input)
        actions, h = model(obs, state=state, info=batch.info)
        actions = actions + self._action_bias
        if not kwargs.get('deterministic', False):
            actions += to_torch_as(self._noise(actions.shape), actions)
        actions = actions.clamp(self._range[0], self._range[1])
        return Batch(act=actions, state=h)


    def sync_weight(self) -> None:
        for o, n in zip(self.actor_old.parameters(), self.actor.parameters()):
            o.data.copy_(o.data * (1 - self._tau) + n.data * self._tau)
        for o, n in zip(
                self.critic1_old.parameters(), self.critic1.parameters()):
            o.data.copy_(o.data * (1 - self._tau) + n.data * self._tau)
        for o, n in zip(
                self.critic2_old.parameters(), self.critic2.parameters()):
            o.data.copy_(o.data * (1 - self._tau) + n.data * self._tau)


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

    def learn(self, batch: Batch, **kwargs) -> Dict[str, float]:

        if self.disc is not None:
            trans_pred = self.disc(np.concatenate([batch.obs, batch.act], -1))
            trans_pred_next = self.disc(np.concatenate([batch.obs_next, batch.act_next], -1))
            trans_targ = torch.FloatTensor(np.concatenate([batch.obs_next, batch.rew[:, None]], -1))
            trans_targ_next = torch.FloatTensor(np.concatenate([batch.obs_next_next, batch.rew_next[:, None]], -1))
            loss_disc = F.mse_loss(trans_pred, trans_targ) + F.mse_loss(trans_pred_next, trans_targ_next)

            self.disc_optim.zero_grad()
            loss_disc.backward()
            self.disc_optim.step()

            with torch.no_grad():
                trans_pred = self.disc(np.concatenate([batch.obs, batch.act], -1))
                trans_pred_next = self.disc(np.concatenate([batch.obs_next, batch.act_next], -1))

        if self.beta is not None:
            beta = self.beta
        else:
            beta = torch.ones(1)

        batch = self.process_tri(batch, beta=beta.squeeze().data.numpy())

        if self.disc is not None:
            targ_blend = torch.FloatTensor(batch.blend)
            trans_mix = self.disc(np.concatenate([batch.obs, batch.act], -1))
            trans_diff = (trans_mix - (trans_pred + targ_blend * (trans_pred_next - trans_pred))
                          ).pow(2).mean(-1, keepdim=True).detach()
        if self.beta is not None:
            beta_loss = (beta.log() * (trans_diff - self.tor_diff)).mean()

            self.beta_optim.zero_grad()
            beta_loss.backward()
            self.beta_optim.step()

            self.beta.data.clamp_(1e-6, 1)

        ratio = 1

        if self.norm_diff:
            ratio = ratio / ratio.sum() * ratio.shape[0]
            ratio_log = ratio.mean().item(),

        if not self.use_diff:
            ratio = 1
            ratio_log = 1

        with torch.no_grad():
            a_ = self(batch, model='actor_old', input='obs_next', deterministic=True).act
            dev = a_.device
            noise = torch.randn(size=a_.shape, device=dev) * self._policy_noise
            if self._noise_clip > 0:
                noise = noise.clamp(-self._noise_clip, self._noise_clip)
            a_ += noise
            a_ = a_.clamp(self._range[0], self._range[1])
            target_q = torch.min(
                self.critic1_old(batch.obs_next, a_),
                self.critic2_old(batch.obs_next, a_))
            rew = torch.tensor(batch.rew,
                               dtype=torch.float, device=dev)[:, None]
            done = torch.tensor(batch.done,
                                dtype=torch.float, device=dev)[:, None]
            target_q = (rew + (1. - done) * self._gamma * target_q)

        # critic 1
        current_q1 = self.critic1(batch.obs, batch.act)
        critic1_loss = F.mse_loss(current_q1, target_q)
        self.critic1_optim.zero_grad()
        critic1_loss.backward()
        self.critic1_optim.step()
        # critic 2
        current_q2 = self.critic2(batch.obs, batch.act)
        critic2_loss = F.mse_loss(current_q2, target_q)
        self.critic2_optim.zero_grad()
        critic2_loss.backward()
        self.critic2_optim.step()

        if self._cnt % self._freq == 0:
            actor_loss = -self.critic1(
                batch.obs, self(batch, deterministic=True).act).mean()
            self.actor_optim.zero_grad()
            actor_loss.backward()
            self._last = actor_loss.item()
            self.actor_optim.step()
            self.sync_weight()

        self._cnt += 1
        result = {
            'loss/actor': self._last,
            'loss/critic1': critic1_loss.item(),
            'loss/critic2': critic2_loss.item(),
        }
        if self.disc is not None:
            result['blend'] = ratio_log
            result['loss/trans_diff'] = trans_diff.mean().item()

        if self.beta is not None:
            result['beta'] = beta.mean().item()
            result['loss/beta'] = beta_loss.item()

        return result
