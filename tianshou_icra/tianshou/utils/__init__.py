from tianshou.utils.config import tqdm_config
from tianshou.utils.moving_average import MovAvg

import numpy as np

class RunningMeanStd(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        self.inp_shape = shape
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

        self.min = None#np.zeros(shape, 'float64')
        self.max = None#np.zeros(shape, 'float64')

    def update(self, x):
        if self.min is None:
            self.min = np.min(x, axis=0)
            self.max = np.max(x, axis=0)
            same_pos = self.min == self.max

            self.min = self.min - same_pos * 1.
            self.max = self.max + same_pos * 1.

        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def reset(self):
        self.count = 1e-4
        self.mean = np.zeros(self.inp_shape, 'float64')
        self.var = np.ones(self.inp_shape, 'float64')

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * (self.count)
        m_b = batch_var * (batch_count)
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / (self.count + batch_count)
        new_var = M2 / (self.count + batch_count)

        new_count = batch_count + self.count

        np.where(np.isclose(new_var, 0), np.ones(10), new_var)

        self.mean = new_mean
        self.var = new_var
        self.count = new_count

    @property
    def std(self):
        return np.sqrt(self.var)

    def toclass(self, input):
        v_min = (self.max - input) / (self.max - self.min)
        # v_max = (input - self.min) / (self.max - self.min)
        v_max = 1 - v_min
        out = np.concatenate([v_min, v_max], -1)
        return out


class RewardForwardFilter(object):
    def __init__(self, gamma):
        self.rewems = None
        self.gamma = gamma

    def update(self, rews):
        if self.rewems is None:
            self.rewems = rews
        else:
            self.rewems = self.rewems * self.gamma + rews
        return self.rewems


__all__ = [
    'MovAvg',
    'tqdm_config',
    'RunningMeanStd',
    'RewardForwardFilter'
]
