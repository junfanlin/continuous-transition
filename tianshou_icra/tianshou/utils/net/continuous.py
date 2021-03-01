import torch
import numpy as np
from torch import nn

from tianshou.data import to_torch


class Actor(nn.Module):
    """For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.
    """

    def __init__(self, preprocess_net, action_shape,
                 max_action, device='cpu', hidden_layer_size=128):
        super().__init__()
        self.preprocess = preprocess_net
        self.last = nn.Linear(hidden_layer_size, np.prod(action_shape))
        self._max = max_action

    def forward(self, s, state=None, info={}):
        """s -> logits -> action"""
        logits, h = self.preprocess(s, state)
        logits = self._max * torch.tanh(self.last(logits))
        return logits, h


class Critic(nn.Module):
    """For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.
    """

    def __init__(self, preprocess_net, device='cpu', hidden_layer_size=128):
        super().__init__()
        self.device = device
        self.preprocess = preprocess_net
        self.last = nn.Linear(hidden_layer_size, 1)

    def forward(self, s, a=None, **kwargs):
        """(s, a) -> logits -> Q(s, a)"""
        s = to_torch(s, device=self.device, dtype=torch.float32)
        s = s.flatten(1)
        if a is not None:
            a = to_torch(a, device=self.device, dtype=torch.float32)
            a = a.flatten(1)
            s = torch.cat([s, a], dim=1)
        logits, h = self.preprocess(s)
        logits = self.last(logits)
        return logits


class Critic_RLKIT(nn.Module):
    def __init__(self, layer_num, state_shape, action_shape=0, device='cpu', hidden_layer_size=128, output_dim=1):
        super().__init__()
        self.device = device
        self.model = [
            nn.Linear(np.prod(state_shape) + np.prod(action_shape), hidden_layer_size),
            nn.ReLU(inplace=True)]
        for i in range(layer_num):
            self.model += [nn.Linear(hidden_layer_size, hidden_layer_size), nn.ReLU(inplace=True)]
        self.model += [nn.Linear(hidden_layer_size, output_dim)]
        self.model = nn.Sequential(*self.model)

    def forward(self, s, a=None, **kwargs):
        s = to_torch(s, device=self.device, dtype=torch.float)
        batch = s.shape[0]
        s = s.view(batch, -1)
        if a is not None:
            if not isinstance(a, torch.Tensor):
                a = torch.tensor(a, device=self.device, dtype=torch.float)
            a = a.view(batch, -1)
            s = torch.cat([s, a], dim=1)
        logits = self.model(s)
        return logits


class Critic_RLKIT_HS(nn.Module):
    def __init__(self, layer_num, state_shape, action_shape=0, device='cpu', hidden_layer_size=128, output_dim=1,
                 goalsize=1):
        super().__init__()
        self.device = device
        self.goalsize = goalsize
        self.model = [
            nn.Linear(np.prod(state_shape) + np.prod(action_shape) + goalsize, hidden_layer_size),
            nn.ReLU(inplace=True)]
        for i in range(layer_num):
            self.model += [nn.Linear(hidden_layer_size, hidden_layer_size), nn.ReLU(inplace=True)]
        self.model += [nn.Linear(hidden_layer_size, output_dim)]
        self.model = nn.Sequential(*self.model)

    def forward(self, s, a=None, **kwargs):
        s = to_torch(s, device=self.device, dtype=torch.float)
        batch = s.shape[0]
        s = s.view(batch, -1)

        ######## goal ##########
        if self.goalsize != 0:
            goal = kwargs['goal']
            goal = torch.ones(batch, self.goalsize) * goal[None]
            s = torch.cat([s, goal], -1)

        if a is not None:
            if not isinstance(a, torch.Tensor):
                a = torch.tensor(a, device=self.device, dtype=torch.float)
            a = a.view(batch, -1)
            s = torch.cat([s, a], dim=1)
        logits = self.model(s)
        return logits


import torch.nn.functional as F


class LinearHO(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LinearHO, self).__init__()

        # this dict contains all tensors needed to be optimized
        self.vars = nn.ParameterList()


        sqrtk = np.sqrt(1/in_channels)
        # [ch_out, ch_in]
        w = nn.Parameter(torch.ones(out_channels, in_channels))
        torch.nn.init.uniform_(w, a=-sqrtk, b=sqrtk)
        self.vars.append(w)
        b = nn.Parameter(torch.zeros(out_channels))
        torch.nn.init.uniform_(b, a=-sqrtk, b=sqrtk)
        self.vars.append(b)

    def forward(self, x, vars=None):
        if vars is None:
            vars = self.vars

        return F.linear(x, vars[0], vars[1])

    def zero_grad(self, vars=None):
        """
        :param vars:
        :return:
        """
        with torch.no_grad():
            if vars is None:
                for p in self.vars:
                    if p.grad is not None:
                        p.grad.zero_()
            else:
                for p in vars:
                    if p.grad is not None:
                        p.grad.zero_()

    def parameters(self):
        """
        override this function since initial parameters will return with a generator.
        :return:
        """
        return self.vars


class MLPHO(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim, n_hidden_layer=0, nl=(lambda: nn.ReLU(True))):
        super().__init__()
        self.n_hidden_layer = n_hidden_layer
        self.add_module('linear_0', LinearHO(input_dim, hidden_dim))
        self.add_module('nl_0', nl())
        for i in range(n_hidden_layer):
            self.add_module('linear_'+str(i+1), LinearHO(hidden_dim, hidden_dim))
            self.add_module('nl_'+str(i+1), nl())
        self.add_module('linear_'+str(n_hidden_layer+1), LinearHO(hidden_dim, out_dim))

    def forward(self, x, vars=None):
        for i in range(self.n_hidden_layer+2):
            x = self.__getattr__('linear_'+str(i))(x, None if vars is None else [vars[i*2], vars[i*2+1]])
            if i+1 < self.n_hidden_layer+1:
                x = self.__getattr__('nl_'+str(i))(x)
        return x



class Critic_RLKIT_AP(nn.Module):
    def __init__(self, layer_num, state_shape, action_shape=0, device='cpu', hidden_layer_size=128, output_dim=1):
        super().__init__()
        self.device = device
        self.model = MLPHO(np.prod(state_shape) + np.prod(action_shape), hidden_layer_size, output_dim, layer_num)

    def forward(self, s, a=None, **kwargs):
        if not isinstance(s, torch.Tensor):
            s = to_torch(s, device=self.device, dtype=torch.float)
        batch = s.shape[0]
        s = s.view(batch, -1)
        if a is not None:
            if not isinstance(a, torch.Tensor):
                a = torch.tensor(a, device=self.device, dtype=torch.float)
            a = a.view(batch, -1)
            s = torch.cat([s, a], dim=1)
        logits = self.model(s, vars=kwargs.get('vars', None))
        return logits


class Critic_RLKIT_APFE(nn.Module):
    def __init__(self, layer_num, state_shape, action_shape=0, device='cpu', hidden_layer_size=128, output_dim=1, ef=2):
        super().__init__()
        self.device = device
        self.state_dim = np.prod(state_shape)
        # self.state_shape = np.prod(state_shape)
        self.model = MLPHO(np.prod(state_shape) + np.prod(action_shape) + ef, hidden_layer_size, output_dim, layer_num)

    def forward(self, s, a=None, **kwargs):
        if not isinstance(s, torch.Tensor):
            s = to_torch(s, device=self.device, dtype=torch.float)
        batch = s.shape[0]
        s = s.view(batch, -1)
        if a is not None:
            if not isinstance(a, torch.Tensor):
                a = torch.tensor(a, device=self.device, dtype=torch.float)
            a = a.view(batch, -1)
            s = torch.cat([s, a], dim=1)
        logits = self.model(s, vars=kwargs.get('vars', None))
        return logits


class Critic_RLKIT_FE(nn.Module):
    def __init__(self, layer_num, state_shape, action_shape=0, device='cpu', hidden_layer_size=128, output_dim=1, ef=2):
        super().__init__()
        self.device = device
        self.state_dim = np.prod(state_shape)
        # self.state_shape = np.prod(state_shape)
        self.model = [
            nn.Linear(np.prod(state_shape) + ef + np.prod(action_shape), hidden_layer_size),
            nn.ReLU(inplace=True)]
        for i in range(layer_num):
            self.model += [nn.Linear(hidden_layer_size, hidden_layer_size), nn.ReLU(inplace=True)]
        self.model += [nn.Linear(hidden_layer_size, output_dim)]
        self.model = nn.Sequential(*self.model)

    def forward(self, s, a=None, **kwargs):
        if not isinstance(s, torch.Tensor):
            s = to_torch(s, device=self.device, dtype=torch.float)
        batch = s.shape[0]
        s = s.view(batch, -1)
        if a is not None:
            if not isinstance(a, torch.Tensor):
                a = torch.tensor(a, device=self.device, dtype=torch.float)
            a = a.view(batch, -1)
            s = torch.cat([s, a], dim=1)
        logits = self.model(s)
        return logits


class Critic_RLKIT_NS(nn.Module):
    def __init__(self, layer_num, state_shape, action_shape=0, device='cpu', hidden_layer_size=128, output_dim=1):
        super().__init__()
        self.device = device
        self.model = [
            nn.Linear(np.prod(state_shape) + np.prod(action_shape), hidden_layer_size),
            nn.ReLU(inplace=True)]
        for i in range(layer_num):
            self.model += [nn.Linear(hidden_layer_size, hidden_layer_size), nn.ReLU(inplace=True)]
        self.model += [nn.Linear(hidden_layer_size, output_dim)]
        self.model = nn.Sequential(*self.model)

        self.embedder_model = torch.nn.LSTM(np.prod(action_shape), np.prod(action_shape), 1)

    def embedder(self, acts):
        h0 = torch.zeros(1, acts[0].shape[0], acts[0].shape[1])
        c0 = torch.zeros(1, acts[0].shape[0], acts[0].shape[1])

        act_list = []
        for act in acts:

            if not isinstance(act, torch.Tensor):
                act = torch.tensor(act, device=self.device, dtype=torch.float)
                act = act.view(act.shape[0], -1)

            act = act[None]
            act, (h0, c0) = self.embedder_model(act, (h0, c0))
            act = act[0]
            act_list.append(act)

        return act_list

    def forward(self, s, a, second=None, **kwargs):
        s = to_torch(s, device=self.device, dtype=torch.float)
        batch = s.shape[0]
        s = s.view(batch, -1)
        a = self.embedder(a)
        if second is not None:
            a = a[0] * (1 - second) + a[1] * second
        else:
            a = a[0]
        s = torch.cat([s, a], dim=1)
        logits = self.model(s)
        return logits


# class Critic_RLKIT_UNIFY(nn.Module):
#     def __init__(self, layer_num, state_shape, action_shape=0, device='cpu', hidden_layer_size=128, output_dim=1, emb_size=32):
#         super().__init__()
#         self.device = device
#         model_emb = [
#             nn.Linear(np.prod(state_shape) + np.prod(action_shape), hidden_layer_size),
#             nn.ReLU(inplace=True)]
#         for i in range(layer_num):
#             model_emb += [nn.Linear(hidden_layer_size, hidden_layer_size), nn.ReLU(inplace=True)]
#         self.model_emb = nn.Sequential(*model_emb)
#
#         self.cos = nn.Linear(hidden_layer_size, emb_size)
#         self.model = nn.Linear(hidden_layer_size, output_dim)
#         self.model_lcb = nn.Linear(hidden_layer_size, output_dim)
#
#     def forward(self, s, a, a_opt, **kwargs):
#         s = to_torch(s, device=self.device, dtype=torch.float)
#         batch = s.shape[0]
#         s = s.view(batch, -1)
#
#         if not isinstance(a_opt, torch.Tensor):
#             a_opt = torch.tensor(a_opt, device=self.device, dtype=torch.float)
#         a_opt = a_opt.view(batch, -1)
#         s_a_opt = torch.cat([s, a_opt], dim=1)
#         emb_s_a_opt = self.model_emb(s_a_opt)
#
#         if not isinstance(a, torch.Tensor):
#             a = torch.tensor(a, device=self.device, dtype=torch.float)
#         a = a.view(batch, -1)
#         s_a = torch.cat([s, a], dim=1)
#         emb_s_a = self.model_emb(s_a)
#
#         met_opt = self.cos(emb_s_a_opt)
#         met = self.cos(emb_s_a)
#         logits_opt = self.model(emb_s_a_opt)
#         logits_opt_lcb = self.model_lcb(emb_s_a_opt).square()
#
#         similarity = torch.nn.functional.cosine_similarity(met, met_opt, -1, 1e-6) # in [-1, 1]
#         similarity = (similarity + 1) / 2
#
#         logits = logits_opt - (1-similarity) * logits_opt_lcb
#         return logits


class Critic_RLKIT_UNIFY(nn.Module):
    def __init__(self, layer_num, state_shape, action_shape=0, device='cpu', hidden_layer_size=128, output_dim=1,
                 emb_size=32):
        super().__init__()
        self.device = device
        self.model_emb = [
            nn.Linear(np.prod(state_shape)+np.prod(action_shape), hidden_layer_size),
            nn.ReLU(inplace=True)]
        for i in range(layer_num):
            self.model_emb += [nn.Linear(hidden_layer_size, hidden_layer_size), nn.ReLU(inplace=True)]
        self.model_emb += [nn.Linear(hidden_layer_size, emb_size)]
        self.model_emb = nn.Sequential(*self.model_emb)

        self.model_body = [
            nn.Linear(np.prod(state_shape), hidden_layer_size),
            nn.ReLU(inplace=True)]
        for i in range(layer_num):
            self.model_body += [nn.Linear(hidden_layer_size, hidden_layer_size), nn.ReLU(inplace=True)]
        self.model_body = nn.Sequential(*self.model_body)
        self.model = nn.Linear(hidden_layer_size, output_dim)
        self.model_lcb = nn.Sequential(nn.Linear(hidden_layer_size, output_dim), nn.Softplus())

    def forward(self, s, a, a_opt, **kwargs):
        s = to_torch(s, device=self.device, dtype=torch.float)
        batch = s.shape[0]
        s = s.view(batch, -1)

        model_int = self.model_body(s)
        maxq = self.model(model_int)
        lcb = self.model_lcb(model_int)

        if not isinstance(a_opt, torch.Tensor):
            a_opt = torch.tensor(a_opt, device=self.device, dtype=torch.float)
        a_opt = a_opt.view(batch, -1)
        s_a_opt = torch.cat([s, a_opt], dim=1)
        emb_s_a_opt = self.model_emb(s_a_opt)

        if not isinstance(a, torch.Tensor):
            a = torch.tensor(a, device=self.device, dtype=torch.float)
        a = a.view(batch, -1)
        s_a = torch.cat([s, a], dim=1)
        emb_s_a = self.model_emb(s_a)

        similarity = torch.nn.functional.cosine_similarity(emb_s_a_opt, emb_s_a, -1, 1e-6)  # in [-1, 1]
        similarity = (similarity + 1) / 2

        logits = maxq - (1 - similarity) * lcb
        return logits


# class Critic_RLKIT_UNIFY(nn.Module):
#     def __init__(self, layer_num, state_shape, action_shape=0, device='cpu', hidden_layer_size=128, output_dim=1,
#                  emb_size=32):
#         super().__init__()
#         self.device = device
#         self.model_emb = [
#             nn.Linear(np.prod(state_shape)+np.prod(action_shape), hidden_layer_size),
#             nn.ReLU(inplace=True)]
#         for i in range(layer_num):
#             self.model_emb += [nn.Linear(hidden_layer_size, hidden_layer_size), nn.ReLU(inplace=True)]
#         self.model_emb += [nn.Linear(hidden_layer_size, emb_size)]
#         self.model_emb = nn.Sequential(*self.model_emb)
#
#         self.model_body = [
#             nn.Linear(np.prod(state_shape), hidden_layer_size),
#             nn.ReLU(inplace=True)]
#         for i in range(layer_num):
#             self.model_body += [nn.Linear(hidden_layer_size, hidden_layer_size), nn.ReLU(inplace=True)]
#         self.model_body = nn.Sequential(*self.model_body)
#         self.model = nn.Linear(hidden_layer_size, output_dim)
#         self.model_lcb = nn.Sequential(nn.Linear(hidden_layer_size, output_dim), nn.Softplus())
#
#     def forward(self, s, a, a_opt, **kwargs):
#         s = to_torch(s, device=self.device, dtype=torch.float)
#         batch = s.shape[0]
#         s = s.view(batch, -1)
#
#         model_int = self.model_body(s)
#         maxq = self.model(model_int)
#         lcb = self.model_lcb(model_int)
#
#         if not isinstance(a_opt, torch.Tensor):
#             a_opt = torch.tensor(a_opt, device=self.device, dtype=torch.float)
#         a_opt = a_opt.view(batch, -1)
#         s_a_opt = torch.cat([s, a_opt], dim=1)
#         emb_s_a_opt = self.model_emb(s_a_opt)
#
#         if not isinstance(a, torch.Tensor):
#             a = torch.tensor(a, device=self.device, dtype=torch.float)
#         a = a.view(batch, -1)
#         s_a = torch.cat([s, a], dim=1)
#         emb_s_a = self.model_emb(s_a)
#
#         similarity = torch.nn.functional.cosine_similarity(emb_s_a_opt, emb_s_a, -1, 1e-6)  # in [-1, 1]
#         similarity = (similarity + 1) / 2
#
#         logits = maxq - (1 - similarity) * lcb
#         return logits



class Dynamics(nn.Module):
    def __init__(self, layer_num, state_shape, action_shape=0, device='cpu', hidden_layer_size=128):
        super().__init__()
        self.device = device
        self.model = [
            nn.Linear(np.prod(state_shape) + np.prod(action_shape), hidden_layer_size),
            nn.ReLU(inplace=True)]
        for i in range(layer_num):
            self.model += [nn.Linear(hidden_layer_size, hidden_layer_size), nn.ReLU(inplace=True)]
        self.model += [nn.Linear(hidden_layer_size, np.prod(state_shape))]
        self.model = nn.Sequential(*self.model)

    def forward(self, s, a=None, **kwargs):
        s = to_torch(s, device=self.device, dtype=torch.float)
        batch = s.shape[0]
        s = s.view(batch, -1)
        if a is not None:
            if not isinstance(a, torch.Tensor):
                a = torch.tensor(a, device=self.device, dtype=torch.float)
            a = a.view(batch, -1)
            s = torch.cat([s, a], dim=1)
        logits = self.model(s)
        return logits


class Critic_RLKIT_HEADS(nn.Module):
    def __init__(self, layer_num, state_shape, action_shape=0, device='cpu', hidden_layer_size=128, head_num=1):
        super().__init__()
        self.device = device

        self.head_num = head_num
        self.models = []
        for nid in range(head_num):
            model = [
                nn.Linear(np.prod(state_shape) + np.prod(action_shape), hidden_layer_size),
                nn.ReLU(inplace=True)]
            for i in range(layer_num):
                model += [nn.Linear(hidden_layer_size, hidden_layer_size), nn.ReLU(inplace=True)]
            model += [nn.Linear(hidden_layer_size, 1)]
            model = nn.Sequential(*model)
            self.add_module('model_'+str(nid), model)
            self.models.append(model)


    def forward(self, s, a=None, **kwargs):
        s = to_torch(s, device=self.device, dtype=torch.float)
        batch = s.shape[0]
        s = s.view(batch, -1)
        if a is not None:
            if not isinstance(a, torch.Tensor):
                a = torch.tensor(a, device=self.device, dtype=torch.float)
            a = a.view(batch, -1)
            s = torch.cat([s, a], dim=1)
        rets = []
        for model in self.models:
            logits = model(s)
            rets.append(logits)
        return rets


class Critic_RLKIT_GOAL(nn.Module):
    def __init__(self, layer_num, goal_size, state_shape, action_shape=0, device='cpu', hidden_layer_size=128):
        super().__init__()
        self.device = device
        self.model = [
            nn.Linear(np.prod(state_shape) + np.prod(action_shape) + goal_size, hidden_layer_size),
            nn.ReLU(inplace=True)]
        for i in range(layer_num):
            self.model += [nn.Linear(hidden_layer_size, hidden_layer_size), nn.ReLU(inplace=True)]
        self.model += [nn.Linear(hidden_layer_size, 1)]
        self.model = nn.Sequential(*self.model)

    def forward(self, s, w, a=None, **kwargs):
        if not isinstance(s, torch.Tensor):
            s = torch.tensor(s, device=self.device, dtype=torch.float)

        if not isinstance(w, torch.Tensor):
            w = torch.tensor(w, device=self.device, dtype=torch.float)

        batch = s.shape[0]
        s = s.view(batch, -1)
        w = w.view(batch, -1)
        if a is not None:
            if not isinstance(a, torch.Tensor):
                a = torch.tensor(a, device=self.device, dtype=torch.float)
            a = a.view(batch, -1)
            s = torch.cat([s, a], dim=1)
        s_w = torch.cat([s, w], dim=1)
        logits = self.model(s_w)
        return logits


# class ActorProb(nn.Module):
#     """For advanced usage (how to customize the network), please refer to
#     :ref:`build_the_network`.
#     """
#
#     def __init__(self, preprocess_net, action_shape, max_action,
#                  device='cpu', unbounded=False, hidden_layer_size=128):
#         super().__init__()
#         self.preprocess = preprocess_net
#         self.device = device
#         self.mu = nn.Linear(hidden_layer_size, np.prod(action_shape))
#         self.sigma = nn.Parameter(torch.zeros(np.prod(action_shape), 1))
#         self._max = max_action
#         self._unbounded = unbounded
#
#     def forward(self, s, state=None, **kwargs):
#         """s -> logits -> (mu, sigma)"""
#         logits, h = self.preprocess(s, state)
#         mu = self.mu(logits)
#         if not self._unbounded:
#             mu = self._max * torch.tanh(mu)
#         shape = [1] * len(mu.shape)
#         shape[1] = -1
#         sigma = (self.sigma.view(shape) + torch.zeros_like(mu)).exp()
#         return (mu, sigma), None


import torch.nn.init as init

class RNDModel(nn.Module):
    def __init__(self, layer_num, state_shape, device='cpu', hidden_layer_size=128):
        super().__init__()
        self.device = device
        self.model = [
            nn.Linear(np.prod(state_shape), hidden_layer_size),
            nn.ReLU(inplace=True)]
        for i in range(layer_num):
            self.model += [nn.Linear(hidden_layer_size, hidden_layer_size), nn.ReLU(inplace=True)]
        self.model += [nn.Linear(hidden_layer_size, hidden_layer_size)]
        self.model = nn.Sequential(*self.model)

        for p in self.modules():
            if isinstance(p, nn.Conv2d):
                init.orthogonal_(p.weight, np.sqrt(2))
                p.bias.data.zero_()

            if isinstance(p, nn.Linear):
                init.orthogonal_(p.weight, np.sqrt(2))
                p.bias.data.zero_()

    def forward(self, s, **kwargs):
        if not isinstance(s, torch.Tensor):
            s = torch.tensor(s, device=self.device, dtype=torch.float)

        batch = s.shape[0]
        s = s.view(batch, -1)
        logits = self.model(s)
        return logits


class RNDNet(nn.Module):
    def __init__(self, layer_num, state_shape, device='cpu', hidden_layer_size=128, output_dim=128):
        super().__init__()
        self.device = device
        self.model = [
            nn.Linear(np.prod(state_shape), hidden_layer_size),
            nn.ReLU(inplace=True)]
        for i in range(layer_num):
            self.model += [nn.Linear(hidden_layer_size, hidden_layer_size), nn.ReLU(inplace=True)]
        self.model += [nn.Linear(hidden_layer_size, output_dim)]
        self.model = nn.Sequential(*self.model)

        for p in self.modules():
            if isinstance(p, nn.Conv2d):
                init.orthogonal_(p.weight, np.sqrt(2))
                p.bias.data.zero_()

            if isinstance(p, nn.Linear):
                init.orthogonal_(p.weight, np.sqrt(2))
                p.bias.data.zero_()

    def forward(self, s, **kwargs):
        if not isinstance(s, torch.Tensor):
            s = torch.tensor(s, device=self.device, dtype=torch.float)

        batch = s.shape[0]
        s = s.view(batch, -1)
        logits = self.model(s)
        return logits, None


class ICMNet(nn.Module):
    def __init__(self, layer_num, state_shape, act_shape, device='cpu', hidden_layer_size=128, output_dim=128):
        super().__init__()
        self.device = device
        self.model = [
            nn.Linear(np.prod(state_shape)+np.prod(act_shape), hidden_layer_size),
            nn.ReLU(inplace=True)]
        for i in range(layer_num):
            self.model += [nn.Linear(hidden_layer_size, hidden_layer_size), nn.ReLU(inplace=True)]
        self.model += [nn.Linear(hidden_layer_size, output_dim)]
        self.model = nn.Sequential(*self.model)

        for p in self.modules():
            if isinstance(p, nn.Conv2d):
                init.orthogonal_(p.weight, np.sqrt(2))
                p.bias.data.zero_()

            if isinstance(p, nn.Linear):
                init.orthogonal_(p.weight, np.sqrt(2))
                p.bias.data.zero_()

    def forward(self, s, a, **kwargs):
        if not isinstance(s, torch.Tensor):
            s = torch.tensor(s, device=self.device, dtype=torch.float)

        if not isinstance(a, torch.Tensor):
            a = torch.tensor(a, device=self.device, dtype=torch.float)

        batch = s.shape[0]
        s = s.view(batch, -1)
        s = torch.cat([s, a], -1)
        logits = self.model(s)
        return logits, None


class RNDNetEns(nn.Module):
    def __init__(self, layer_num, state_shape, device='cpu', hidden_layer_size=128, num_ens=1, sigmoid=False):
        super().__init__()
        self.device = device
        self.num_ens = num_ens

        for idx in range(self.num_ens):
            model = [
                nn.Linear(np.prod(state_shape), hidden_layer_size),
                nn.ReLU(inplace=True)]
            for i in range(layer_num):
                model += [nn.Linear(hidden_layer_size, hidden_layer_size), nn.ReLU(inplace=True)]
            model += [nn.Linear(hidden_layer_size, hidden_layer_size)]
            if sigmoid:
                model += [nn.Sigmoid()]
            model = nn.Sequential(*model)

            self.add_module('model'+str(idx), model)


        for p in self.modules():
            if isinstance(p, nn.Conv2d):
                init.orthogonal_(p.weight, np.sqrt(2))
                p.bias.data.zero_()

            if isinstance(p, nn.Linear):
                init.orthogonal_(p.weight, np.sqrt(2))
                p.bias.data.zero_()

    def forward(self, s, **kwargs):
        if not isinstance(s, torch.Tensor):
            s = torch.tensor(s, device=self.device, dtype=torch.float)

        batch = s.shape[0]
        s = s.view(batch, -1)
        out_list = []
        for idx in range(self.num_ens):
            out_list.append(self.__getattr__('model'+str(idx))(s).unsqueeze(-1))
        out_list = torch.cat(out_list, -1)
        return out_list, None


class RNDNetClass(nn.Module):
    def __init__(self, layer_num, state_shape, device='cpu', hidden_layer_size=128):
        super().__init__()
        self.device = device
        self.model = [
            nn.Linear(np.prod(state_shape), hidden_layer_size),
            nn.ReLU(inplace=True)]
        for i in range(layer_num):
            self.model += [nn.Linear(hidden_layer_size, hidden_layer_size), nn.ReLU(inplace=True)]
        self.model += [nn.Linear(hidden_layer_size, hidden_layer_size), nn.Sigmoid()]
        self.model = nn.Sequential(*self.model)

        for p in self.modules():
            if isinstance(p, nn.Conv2d):
                init.orthogonal_(p.weight, np.sqrt(2))
                p.bias.data.zero_()

            if isinstance(p, nn.Linear):
                init.orthogonal_(p.weight, np.sqrt(2))
                p.bias.data.zero_()

    def forward(self, s, **kwargs):
        if not isinstance(s, torch.Tensor):
            s = torch.tensor(s, device=self.device, dtype=torch.float)

        batch = s.shape[0]
        s = s.view(batch, -1)
        logits = self.model(s)
        return logits, None


class ActorProb_RLKIT(nn.Module):
    def __init__(self, layer_num, state_shape, action_shape,
                 max_action=1, device='cpu', hidden_layer_size=128, lsgm_range=[-20, 2]):
        super().__init__()
        self.device = device
        self.action_shape = action_shape
        self.state_shape = state_shape
        self.model = [
            nn.Linear(np.prod(state_shape), hidden_layer_size),
            nn.ReLU(inplace=True)]
        for i in range(layer_num):
            self.model += [nn.Linear(hidden_layer_size, hidden_layer_size), nn.ReLU(inplace=True)]
        self.model = nn.Sequential(*self.model)
        self.mu = nn.Linear(hidden_layer_size, np.prod(action_shape))
        self.sigma = nn.Linear(hidden_layer_size, np.prod(action_shape))
        self._lsgm_range = lsgm_range
        self._max = max_action

    def forward(self, s, **kwargs):
        if not isinstance(s, torch.Tensor):
            s = torch.tensor(s, device=self.device, dtype=torch.float)
        batch = s.shape[0]
        s = s.view(batch, -1)
        logits = self.model(s)

        mu = self.mu(logits)
        log_sigma = self.sigma(logits)
        log_sigma = torch.clamp(log_sigma, self._lsgm_range[0], self._lsgm_range[1])
        sigma = torch.exp(log_sigma)

        return (mu, sigma), None


class ActorProb_RLKIT_HS(nn.Module):
    def __init__(self, layer_num, state_shape, action_shape,
                 max_action=1, device='cpu', hidden_layer_size=128, lsgm_range=[-20, 2], goalsize=1):
        super().__init__()
        self.device = device
        self.action_shape = action_shape
        self.state_shape = state_shape
        self.goalsize=goalsize
        self.model = [
            nn.Linear(np.prod(state_shape) + goalsize, hidden_layer_size),
            nn.ReLU(inplace=True)]
        for i in range(layer_num):
            self.model += [nn.Linear(hidden_layer_size, hidden_layer_size), nn.ReLU(inplace=True)]
        self.model = nn.Sequential(*self.model)
        self.mu = nn.Linear(hidden_layer_size, np.prod(action_shape))
        self.sigma = nn.Linear(hidden_layer_size, np.prod(action_shape))
        self._lsgm_range = lsgm_range
        self._max = max_action

    def forward(self, s, **kwargs):
        if not isinstance(s, torch.Tensor):
            s = torch.tensor(s, device=self.device, dtype=torch.float)
        batch = s.shape[0]
        s = s.view(batch, -1)

        ######## goal ##########
        if self.goalsize != 0:
            goal = kwargs['goal']
            goal = torch.ones(batch, self.goalsize) * goal[None]
            s = torch.cat([s, goal], -1)

        logits = self.model(s)

        mu = self.mu(logits)
        log_sigma = self.sigma(logits)
        log_sigma = torch.clamp(log_sigma, self._lsgm_range[0], self._lsgm_range[1])
        sigma = torch.exp(log_sigma)

        return (mu, sigma), None


class ActorProb_RLKIT_FITREW(nn.Module):
    def __init__(self, layer_num, state_shape, action_shape,
                 max_action=1, device='cpu', hidden_layer_size=128, lsgm_range=[-20, 2]):
        super().__init__()
        self.device = device
        self.action_shape = action_shape
        self.state_shape = state_shape
        self.model = [
            nn.Linear(np.prod(state_shape), hidden_layer_size),
            nn.ReLU(inplace=True)]
        for i in range(layer_num):
            self.model += [nn.Linear(hidden_layer_size, hidden_layer_size), nn.ReLU(inplace=True)]
        self.model = nn.Sequential(*self.model)
        self.mu = nn.Linear(hidden_layer_size, np.prod(action_shape))
        self.sigma = nn.Linear(hidden_layer_size, np.prod(action_shape))
        self.a_weight = nn.Linear(hidden_layer_size, np.prod(action_shape))
        self._lsgm_range = lsgm_range
        self._max = max_action

    def forward(self, s, **kwargs):
        if not isinstance(s, torch.Tensor):
            s = torch.tensor(s, device=self.device, dtype=torch.float)
        batch = s.shape[0]
        s = s.view(batch, -1)
        logits = self.model(s)

        mu = self.mu(logits)
        log_sigma = self.sigma(logits)
        log_sigma = torch.clamp(log_sigma, self._lsgm_range[0], self._lsgm_range[1])
        sigma = torch.exp(log_sigma)
        a_weight = self.a_weight(logits)
        a_weight = torch.clamp(a_weight, self._lsgm_range[0], self._lsgm_range[1])
        a_weight = a_weight.exp()

        return (mu, sigma), None, a_weight


class ActorProb_RLKIT_FE(nn.Module):
    def __init__(self, layer_num, state_shape, action_shape,
                 max_action=1, device='cpu', hidden_layer_size=128, lsgm_range=[-20, 2], fe=0):
        super().__init__()
        self.device = device
        self.action_shape = action_shape
        self.state_shape = state_shape
        self.fe = fe
        self.model = [
            nn.Linear(np.prod(state_shape)+fe, hidden_layer_size),
            nn.ReLU(inplace=True)]
        for i in range(layer_num):
            self.model += [nn.Linear(hidden_layer_size, hidden_layer_size), nn.ReLU(inplace=True)]
        self.model = nn.Sequential(*self.model)
        self.mu = nn.Linear(hidden_layer_size, np.prod(action_shape))
        self.sigma = nn.Linear(hidden_layer_size, np.prod(action_shape))
        self._lsgm_range = lsgm_range
        self._max = max_action

    def forward(self, s, **kwargs):
        if not isinstance(s, torch.Tensor):
            s = torch.tensor(s, device=self.device, dtype=torch.float)
        batch = s.shape[0]
        s = s.view(batch, -1)
        if s.shape[-1] < np.prod(self.state_shape) + self.fe:
            s = torch.cat([s, torch.zeros(batch, 1), torch.ones(batch, 1) * 0.5], -1)
        else:
            s = s[:, :np.prod(self.state_shape)+self.fe]
        logits = self.model(s)

        mu = self.mu(logits)
        log_sigma = self.sigma(logits)
        log_sigma = torch.clamp(log_sigma, self._lsgm_range[0], self._lsgm_range[1])
        sigma = torch.exp(log_sigma)

        return (mu, sigma), None


class Actor_RLKIT(nn.Module):
    def __init__(self, layer_num, state_shape, action_shape,
                 max_action=1, device='cpu', hidden_layer_size=128):
        super().__init__()
        self.device = device
        self.action_shape = action_shape
        self.state_shape = state_shape
        self.model = [
            nn.Linear(np.prod(state_shape), hidden_layer_size),
            nn.ReLU(inplace=True)]
        for i in range(layer_num):
            self.model += [nn.Linear(hidden_layer_size, hidden_layer_size), nn.ReLU(inplace=True)]
        self.model = nn.Sequential(*self.model)
        self.mu = nn.Linear(hidden_layer_size, np.prod(action_shape))

    def forward(self, s, **kwargs):
        if not isinstance(s, torch.Tensor):
            s = torch.tensor(s, device=self.device, dtype=torch.float)
        batch = s.shape[0]
        s = s.view(batch, -1)
        logits = self.model(s)

        mu = self.mu(logits)

        return torch.tanh(mu), None


class EmbNet(nn.Module):
    def __init__(self, layer_num, goal_size, state_shape, action_shape, embsize=32,
                 max_action=1, device='cpu', hidden_layer_size=128, lsgm_range=[-20, 2]):
        super().__init__()
        self.device = device
        self.action_shape = action_shape
        self.model = [
            nn.Linear(np.prod(state_shape), hidden_layer_size),
            nn.ReLU(inplace=True)]
        for i in range(layer_num-1):
            self.model += [nn.Linear(hidden_layer_size, hidden_layer_size), nn.ReLU(inplace=True)]
        self.model += [nn.Linear(hidden_layer_size, embsize), nn.ReLU(inplace=True)]
        self.model = nn.Sequential(*self.model)

        self.inverse_l1 = nn.Linear(embsize * 2, hidden_layer_size)
        self.mu = nn.Linear(hidden_layer_size, np.prod(action_shape))
        self.sigma = nn.Linear(hidden_layer_size, np.prod(action_shape))
        self._lsgm_range = lsgm_range
        self._max = max_action

    def forward(self, s, **kwargs):
        if not isinstance(s, torch.Tensor):
            s = torch.tensor(s, device=self.device, dtype=torch.float)
        batch = s.shape[0]
        s = s.view(batch, -1)
        logits = self.model(s)
        return logits

    def inverse_actor(self, s1, s2):
        if not isinstance(s1, torch.Tensor):
            s1 = torch.tensor(s1, device=self.device, dtype=torch.float)
        batch = s1.shape[0]
        s1 = s1.view(batch, -1)
        s1 = self(s1)

        if not isinstance(s2, torch.Tensor):
            s2 = torch.tensor(s2, device=self.device, dtype=torch.float)
        batch = s2.shape[0]
        s2 = s2.view(batch, -1)
        s2 = self(s2)

        logits = torch.nn.functional.relu(self.inverse_l1(torch.cat([s1, s2], -1)))
        mu = self.mu(logits)
        log_sigma = self.sigma(logits)
        log_sigma = torch.clamp(log_sigma, self._lsgm_range[0], self._lsgm_range[1])
        sigma = torch.exp(log_sigma)

        return (mu, sigma)



class ActorProb_RLKIT_GOAL(nn.Module):
    def __init__(self, layer_num, goal_size, state_shape, action_shape,
                 max_action=1, device='cpu', hidden_layer_size=128, lsgm_range=[-20, 2]):
        super().__init__()
        self.device = device
        self.action_shape = action_shape
        self.model = [
            nn.Linear(np.prod(state_shape) + goal_size, hidden_layer_size),
            nn.ReLU(inplace=True)]
        for i in range(layer_num):
            self.model += [nn.Linear(hidden_layer_size, hidden_layer_size), nn.ReLU(inplace=True)]
        self.model = nn.Sequential(*self.model)
        self.mu = nn.Linear(hidden_layer_size, np.prod(action_shape))
        self.sigma = nn.Linear(hidden_layer_size, np.prod(action_shape))
        self._lsgm_range = lsgm_range
        self._max = max_action

    def forward(self, s, w, **kwargs):
        if not isinstance(s, torch.Tensor):
            s = torch.tensor(s, device=self.device, dtype=torch.float)

        if not isinstance(w, torch.Tensor):
            w = torch.tensor(w, device=self.device, dtype=torch.float)

        batch = s.shape[0]
        s = s.view(batch, -1)
        w = w.view(batch, -1)
        s_w = torch.cat([s, w], -1)
        logits = self.model(s_w)

        mu = self.mu(logits)
        log_sigma = self.sigma(logits)
        log_sigma = torch.clamp(log_sigma, self._lsgm_range[0], self._lsgm_range[1])
        sigma = torch.exp(log_sigma)

        return (mu, sigma), None


class ActorProb(nn.Module):
    """For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.
    """

    def __init__(self, preprocess_net, action_shape, max_action,
                 device='cpu', unbounded=False, hidden_layer_size=128, lsgm_range=[-20, 2]):
        super().__init__()
        self.preprocess = preprocess_net
        self.device = device
        self.mu = nn.Linear(hidden_layer_size, np.prod(action_shape))
        self.sigma = nn.Linear(hidden_layer_size, np.prod(action_shape))
        self._max = max_action
        self._unbounded = unbounded
        self._lsgm_range = lsgm_range

    def forward(self, s, state=None, **kwargs):
        """s -> logits -> (mu, sigma)"""
        logits, h = self.preprocess(s, state)
        mu = self.mu(logits)
        if not self._unbounded:
            mu = self._max * torch.tanh(mu)
        log_sigma = self.sigma(logits)
        log_sigma = torch.clamp(log_sigma, self._lsgm_range[0], self._lsgm_range[1])
        sigma = torch.exp(log_sigma)
        return (mu, sigma), None



class RecurrentActorProb(nn.Module):
    """For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.
    """

    def __init__(self, layer_num, state_shape, action_shape,
                 max_action, device='cpu', hidden_layer_size=128):
        super().__init__()
        self.device = device
        self.nn = nn.LSTM(input_size=np.prod(state_shape),
                          hidden_size=hidden_layer_size,
                          num_layers=layer_num, batch_first=True)
        self.mu = nn.Linear(hidden_layer_size, np.prod(action_shape))
        self.sigma = nn.Parameter(torch.zeros(np.prod(action_shape), 1))

    def forward(self, s, **kwargs):
        """Almost the same as :class:`~tianshou.utils.net.common.Recurrent`."""
        s = to_torch(s, device=self.device, dtype=torch.float32)
        # s [bsz, len, dim] (training) or [bsz, dim] (evaluation)
        # In short, the tensor's shape in training phase is longer than which
        # in evaluation phase.
        if len(s.shape) == 2:
            s = s.unsqueeze(-2)
        logits, _ = self.nn(s)
        logits = logits[:, -1]
        mu = self.mu(logits)
        shape = [1] * len(mu.shape)
        shape[1] = -1
        sigma = (self.sigma.view(shape) + torch.zeros_like(mu)).exp()
        return (mu, sigma), None


class RecurrentCritic(nn.Module):
    """For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.
    """

    def __init__(self, layer_num, state_shape,
                 action_shape=0, device='cpu', hidden_layer_size=128):
        super().__init__()
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.device = device
        self.nn = nn.LSTM(input_size=np.prod(state_shape),
                          hidden_size=hidden_layer_size,
                          num_layers=layer_num, batch_first=True)
        self.fc2 = nn.Linear(hidden_layer_size + np.prod(action_shape), 1)

    def forward(self, s, a=None):
        """Almost the same as :class:`~tianshou.utils.net.common.Recurrent`."""
        s = to_torch(s, device=self.device, dtype=torch.float32)
        # s [bsz, len, dim] (training) or [bsz, dim] (evaluation)
        # In short, the tensor's shape in training phase is longer than which
        # in evaluation phase.
        assert len(s.shape) == 3
        self.nn.flatten_parameters()
        s, (h, c) = self.nn(s)
        s = s[:, -1]
        if a is not None:
            if not isinstance(a, torch.Tensor):
                a = torch.tensor(a, device=self.device, dtype=torch.float32)
            s = torch.cat([s, a], dim=1)
        s = self.fc2(s)
        return s
