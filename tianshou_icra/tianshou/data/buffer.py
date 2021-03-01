import numpy as np
from typing import Any, Tuple, Union, Optional

from tianshou.data.batch import Batch, _create_value


class ReplayBuffer:
    """:class:`~tianshou.data.ReplayBuffer` stores data generated from
    interaction between the policy and environment. The current implementation
    of Tianshou typically use 7 reserved keys in :class:`~tianshou.data.Batch`:

    * ``obs`` the observation of step :math:`t` ;
    * ``act`` the action of step :math:`t` ;
    * ``rew`` the reward of step :math:`t` ;
    * ``done`` the done flag of step :math:`t` ;
    * ``obs_next`` the observation of step :math:`t+1` ;
    * ``info`` the info of step :math:`t` (in ``gym.Env``, the ``env.step()`` \
    function returns 4 arguments, and the last one is ``info``);
    * ``policy`` the data computed by policy in step :math:`t`;

    The following code snippet illustrates its usage:
    ::

        >>> import numpy as np
        >>> from tianshou.data import ReplayBuffer
        >>> buf = ReplayBuffer(size=20)
        >>> for i in range(3):
        ...     buf.add(obs=i, act=i, rew=i, done=i, obs_next=i + 1, info={})
        >>> buf.obs
        # since we set size = 20, len(buf.obs) == 20.
        array([0., 1., 2., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
               0., 0., 0., 0.])
        >>> # but there are only three valid items, so len(buf) == 3.
        >>> len(buf)
        3
        >>> buf2 = ReplayBuffer(size=10)
        >>> for i in range(15):
        ...     buf2.add(obs=i, act=i, rew=i, done=i, obs_next=i + 1, info={})
        >>> len(buf2)
        10
        >>> buf2.obs
        # since its size = 10, it only stores the last 10 steps' result.
        array([10., 11., 12., 13., 14.,  5.,  6.,  7.,  8.,  9.])

        >>> # move buf2's result into buf (meanwhile keep it chronologically)
        >>> buf.update(buf2)
        array([ 0.,  1.,  2.,  5.,  6.,  7.,  8.,  9., 10., 11., 12., 13., 14.,
                0.,  0.,  0.,  0.,  0.,  0.,  0.])

        >>> # get a random sample from buffer
        >>> # the batch_data is equal to buf[incide].
        >>> batch_data, indice = buf.sample(batch_size=4)
        >>> batch_data.obs == buf[indice].obs
        array([ True,  True,  True,  True])

    :class:`~tianshou.data.ReplayBuffer` also supports frame_stack sampling
    (typically for RNN usage, see issue#19), ignoring storing the next
    observation (save memory in atari tasks), and multi-modal observation (see
    issue#38):
    ::

        >>> buf = ReplayBuffer(size=9, stack_num=4, ignore_obs_next=True)
        >>> for i in range(16):
        ...     done = i % 5 == 0
        ...     buf.add(obs={'id': i}, act=i, rew=i, done=done,
        ...             obs_next={'id': i + 1})
        >>> print(buf)  # you can see obs_next is not saved in buf
        ReplayBuffer(
            act: array([ 9., 10., 11., 12., 13., 14., 15.,  7.,  8.]),
            done: array([0., 1., 0., 0., 0., 0., 1., 0., 0.]),
            info: Batch(),
            obs: Batch(
                     id: array([ 9., 10., 11., 12., 13., 14., 15.,  7.,  8.]),
                 ),
            policy: Batch(),
            rew: array([ 9., 10., 11., 12., 13., 14., 15.,  7.,  8.]),
        )
        >>> index = np.arange(len(buf))
        >>> print(buf.get(index, 'obs').id)
        [[ 7.  7.  8.  9.]
         [ 7.  8.  9. 10.]
         [11. 11. 11. 11.]
         [11. 11. 11. 12.]
         [11. 11. 12. 13.]
         [11. 12. 13. 14.]
         [12. 13. 14. 15.]
         [ 7.  7.  7.  7.]
         [ 7.  7.  7.  8.]]
        >>> # here is another way to get the stacked data
        >>> # (stack only for obs and obs_next)
        >>> abs(buf.get(index, 'obs')['id'] - buf[index].obs.id).sum().sum()
        0.0
        >>> # we can get obs_next through __getitem__, even if it doesn't exist
        >>> print(buf[:].obs_next.id)
        [[ 7.  8.  9. 10.]
         [ 7.  8.  9. 10.]
         [11. 11. 11. 12.]
         [11. 11. 12. 13.]
         [11. 12. 13. 14.]
         [12. 13. 14. 15.]
         [12. 13. 14. 15.]
         [ 7.  7.  7.  8.]
         [ 7.  7.  8.  9.]]

    :param int size: the size of replay buffer.
    :param int stack_num: the frame-stack sampling argument, should be greater
        than 1, defaults to 0 (no stacking).
    :param bool ignore_obs_next: whether to store obs_next, defaults to
        ``False``.
    :param bool sample_avail: the parameter indicating sampling only available
        index when using frame-stack sampling method, defaults to ``False``.
        This feature is not supported in Prioritized Replay Buffer currently.
    """

    def __init__(self, size: int, stack_num: Optional[int] = 0,
                 ignore_obs_next: bool = False,
                 sample_avail: bool = False, **kwargs) -> None:
        super().__init__()
        self._maxsize = size
        self._stack = stack_num
        assert stack_num != 1, 'stack_num should greater than 1'
        self._avail = sample_avail and stack_num > 1
        self._avail_index = []
        self._save_s_ = not ignore_obs_next
        self._index = 0
        self._size = 0
        self._meta = Batch()
        self._max_ep_len = kwargs.get('max_ep_len', None) # used to identify whether the end is terminated by time limit.
        self._ens_num = kwargs.get('ens_num', None) # if not none, add mask
        self._ngu = kwargs.get('ngu', False)
        self._rand2 = kwargs.get('rand2', False)
        self.non_episodic = kwargs.get('non_episodic', False)
        self.reset()

    def __len__(self) -> int:
        """Return len(self)."""
        return self._size

    def __repr__(self) -> str:
        """Return str(self)."""
        return self.__class__.__name__ + self._meta.__repr__()[5:]

    def __getattr__(self, key: str) -> Union['Batch', Any]:
        """Return self.key"""
        if key not in self._meta.__dict__:
            return
        return self._meta.__dict__[key]

    def _add_to_buffer(self, name: str, inst: Any) -> None:
        try:
            value = self._meta.__dict__[name]
        except KeyError:
            self._meta.__dict__[name] = _create_value(inst, self._maxsize)
            value = self._meta.__dict__[name]
        if isinstance(inst, np.ndarray) and value.shape[1:] != inst.shape:
            raise ValueError(
                "Cannot add data to a buffer with different shape, key: "
                f"{name}, expect shape: {value.shape[1:]}, "
                f"given shape: {inst.shape}.")
        try:
            value[self._index] = inst
        except KeyError:
            for key in set(inst.keys()).difference(value.__dict__.keys()):
                value.__dict__[key] = _create_value(inst[key], self._maxsize)
            value[self._index] = inst

    def _get_stack_num(self):
        return self._stack

    def _set_stack_num(self, num):
        self._stack = num

    def update(self, buffer: 'ReplayBuffer') -> None:
        """Move the data from the given buffer to self."""
        if len(buffer) == 0:
            return
        i = begin = buffer._index % len(buffer)
        origin = buffer._get_stack_num()
        buffer._set_stack_num(0)

        _len = 0
        while True:
            _len += 1

            if self._max_ep_len is not None and _len >= self._max_ep_len:
                buffer.done[i] *= 0

            if ((i + 1) % len(buffer) == begin):
                buffer.done_bk[i] = 1

            if self.non_episodic and ((i + 1) % len(buffer) == begin) and buffer.done[i] != 0:
                buffer.obs_next[i] = buffer.obs[begin]

            if not self._save_s_:
                buffer.obs_next[i] = None
            self.add(**buffer[i])
            i = (i + 1) % len(buffer)
            if i == begin:
                break
        buffer._set_stack_num(origin)

    def add(self,
            obs: Union[dict, Batch, np.ndarray],
            act: Union[np.ndarray, float],
            rew: Union[int, float],
            done: bool,
            obs_next: Optional[Union[dict, Batch, np.ndarray]] = None,
            info: dict = {},
            policy: Optional[Union[dict, Batch]] = {},
            **kwargs) -> None:
        """Add a batch of data into replay buffer."""
        assert isinstance(info, (dict, Batch)), \
            'You should return a dict in the last argument of env.step().'
        self._add_to_buffer('obs', obs)
        self._add_to_buffer('act', act)
        self._add_to_buffer('rew', rew)
        self._add_to_buffer('done', done)
        self._add_to_buffer('done_bk', kwargs.get('done_bk', done))
        if self._save_s_:
            if obs_next is None:
                obs_next = Batch()
            self._add_to_buffer('obs_next', obs_next)
        self._add_to_buffer('info', info)
        self._add_to_buffer('policy', policy)
        if self._ens_num is not None:
            mask = np.random.randint(2, size=self._ens_num)
            self._add_to_buffer('mask', mask)
        if self._ngu:
            self._add_to_buffer('erew', 0)
            self._add_to_buffer('goal', 0)
        if self._rand2:
            self._add_to_buffer('irew', 0)
        if 'log_prob' in kwargs:
            self._add_to_buffer('log_prob', kwargs.get('log_prob', 0))
        if 'alpha' in kwargs:
            self._add_to_buffer('alpha', kwargs.get('alpha', 0))
        if 'int_rew' in kwargs:
            self._add_to_buffer('int_rew', kwargs.get('int_rew', 0))

        # maintain available index for frame-stack sampling
        if self._avail:
            # update current frame
            avail = sum(self.done[i] for i in range(
                self._index - self._stack + 1, self._index)) == 0
            if self._size < self._stack - 1:
                avail = False
            if avail and self._index not in self._avail_index:
                self._avail_index.append(self._index)
            elif not avail and self._index in self._avail_index:
                self._avail_index.remove(self._index)
            # remove the later available frame because of broken storage
            t = (self._index + self._stack - 1) % self._maxsize
            if t in self._avail_index:
                self._avail_index.remove(t)

        if self._maxsize > 0:
            self._size = min(self._size + 1, self._maxsize)
            self._index = (self._index + 1) % self._maxsize
        else:
            self._size = self._index = self._index + 1

    def reset(self) -> None:
        """Clear all the data in replay buffer."""
        self._index = 0
        self._size = 0
        self._avail_index = []

    def sample(self, batch_size: int) -> Tuple[Batch, np.ndarray]:
        """Get a random sample from buffer with size equal to batch_size. \
        Return all the data in the buffer if batch_size is ``0``.

        :return: Sample data and its corresponding index inside the buffer.
        """
        if batch_size > 0:
            _all = self._avail_index if self._avail else self._size
            indice = np.random.choice(_all, batch_size)
        else:
            if self._avail:
                indice = np.array(self._avail_index)
            else:
                indice = np.concatenate([
                    np.arange(self._index, self._size),
                    np.arange(0, self._index),
                ])
        assert len(indice) > 0, 'No available indice can be sampled.'
        return self[indice], indice

    def get(self, indice: Union[slice, int, np.integer, np.ndarray], key: str,
            stack_num: Optional[int] = None) -> Union[Batch, np.ndarray]:
        """Return the stacked result, e.g. [s_{t-3}, s_{t-2}, s_{t-1}, s_t],
        where s is self.key, t is indice. The stack_num (here equals to 4) is
        given from buffer initialization procedure.
        """
        if stack_num is None:
            stack_num = self._stack
        if isinstance(indice, slice):
            indice = np.arange(
                0 if indice.start is None
                else self._size - indice.start if indice.start < 0
                else indice.start,
                self._size if indice.stop is None
                else self._size - indice.stop if indice.stop < 0
                else indice.stop,
                1 if indice.step is None else indice.step)
        else:
            indice = np.array(indice, copy=True)
        # set last frame done to True
        last_index = (self._index - 1 + self._size) % self._size
        last_done, self.done[last_index] = self.done[last_index], True
        if key == 'obs_next' and (not self._save_s_ or self.obs_next is None):
            indice += 1 - self.done[indice].astype(np.int)
            indice[indice == self._size] = 0
            key = 'obs'
        val = self._meta.__dict__[key]
        try:
            if stack_num > 0:
                stack = []
                for _ in range(stack_num):
                    stack = [val[indice]] + stack
                    pre_indice = np.asarray(indice - 1)
                    pre_indice[pre_indice == -1] = self._size - 1
                    indice = np.asarray(
                        pre_indice + self.done[pre_indice].astype(np.int))
                    indice[indice == self._size] = 0
                if isinstance(val, Batch):
                    stack = Batch.stack(stack, axis=indice.ndim)
                else:
                    stack = np.stack(stack, axis=indice.ndim)
            else:
                stack = val[indice]
        except IndexError as e:
            stack = Batch()
            if not isinstance(val, Batch) or len(val.__dict__) > 0:
                raise e
        self.done[last_index] = last_done
        return stack

    def __getitem__(self, index: Union[
            slice, int, np.integer, np.ndarray]) -> Batch:
        """Return a data batch: self[index]. If stack_num is set to be > 0,
        return the stacked obs and obs_next with shape [batch, len, ...].
        """
        ret =  Batch(
            obs=self.get(index, 'obs'),
            act=self.act[index],
            rew=self.rew[index],
            done=self.done[index],
            done_bk=self.done_bk[index],
            obs_next=self.get(index, 'obs_next'),
            info=self.get(index, 'info'),
            policy=self.get(index, 'policy')
        )

        if self.log_prob is not None:
            ret.log_prob = self.log_prob[index]
        if self.alpha is not None:
            ret.alpha = self.alpha[index]
        # if hasattr(self, 'int_rew'):
        #     ret.int_rew = self.int_rew[index]

        if self._ens_num is not None:
            ret.mask=self.mask[index]
        if self._ngu:
            ret.erew=self.erew[index]
            ret.goal=self.goal[index]
        if self._rand2:
            ret.irew=self.irew[index]
        return ret


# class ReplayBufferTriple(ReplayBuffer):
#     def __init__(self, size, **kwargs) -> None:
#         super().__init__(size, **kwargs)
#
#
#     def __getitem__(self, index: Union[
#             slice, int, np.integer, np.ndarray]) -> Batch:
#         """Return a data batch: self[index]. If stack_num is set to be > 0,
#         return the stacked obs and obs_next with shape [batch, len, ...].
#         """
#         ret =  Batch(
#             obs=self.get(index, 'obs'),
#             act=self.act[index],
#             rew=self.rew[index],
#             done=self.done[index],
#             done_bk=self.done_bk[index],
#             obs_next=self.get(index, 'obs_next'),
#             act_next=self.act[(index+1) % self._size],
#             rew_next=self.rew[(index+1) % self._size],
#             obs_next_next=self.get((index+1) % self._size, 'obs_next'),
#             done_next=self.done[(index + 1) % self._size],
#             info=self.get(index, 'info'),
#             policy=self.get(index, 'policy')
#         )
#
#         if self.log_prob is not None:
#             ret.log_prob = self.log_prob[index]
#         if self.alpha is not None:
#             ret.alpha = self.alpha[index]
#         # if hasattr(self, 'int_rew'):
#         #     ret.int_rew = self.int_rew[index]
#
#         if self._ens_num is not None:
#             ret.mask=self.mask[index]
#         if self._ngu:
#             ret.erew=self.erew[index]
#             ret.goal=self.goal[index]
#         if self._rand2:
#             ret.irew=self.irew[index]
#         return ret


class ReplayBufferTriple(ReplayBuffer):
    def __init__(self, size, **kwargs) -> None:
        super().__init__(size, **kwargs)
        self.num = kwargs.get('num', 2)


    def __getitem__(self, index: Union[
            slice, int, np.integer, np.ndarray]) -> Batch:
        """Return a data batch: self[index]. If stack_num is set to be > 0,
        return the stacked obs and obs_next with shape [batch, len, ...].
        """
        kwargs = dict()
        for i in range(1, self.num):
            kwargs[''.join(['act']+['_next'] * i)] = self.act[(index + i) % self._size]
            kwargs[''.join(['rew']+['_next'] * i)] = self.rew[(index + i) % self._size]
            kwargs[''.join(['obs_next']+['_next'] * i)] = self.get((index + i) % self._size, 'obs_next')
            kwargs[''.join(['done']+['_next'] * i)] = self.done[(index + i) % self._size]
            kwargs[''.join(['done_bk']+['_next'] * (i - 1))] = self.done_bk[(index + i - 1) % self._size]

        ret =  Batch(
            obs=self.get(index, 'obs'),
            obs_next=self.get(index, 'obs_next'),
            act=self.act[index],
            rew=self.rew[index],
            done=self.done[index],
            info=self.get(index, 'info'),
            policy=self.get(index, 'policy'),
            **kwargs
        )

        if self.log_prob is not None:
            ret.log_prob = self.log_prob[index]
        if self.alpha is not None:
            ret.alpha = self.alpha[index]
        # if hasattr(self, 'int_rew'):
        #     ret.int_rew = self.int_rew[index]

        if self._ens_num is not None:
            ret.mask=self.mask[index]
        if self._ngu:
            ret.erew=self.erew[index]
            ret.goal=self.goal[index]
        if self._rand2:
            ret.irew=self.irew[index]
        return ret


class ReplayBufferProtect(ReplayBuffer):
    def __init__(self, size, **kwargs) -> None:
        super().__init__(size, **kwargs)
        self.protect_num = kwargs.get('protect_num', 1)
        self.with_pt = kwargs.get('with_pt', True)


    def __getitem__(self, index: Union[
            slice, int, np.integer, np.ndarray]) -> Batch:
        """Return a data batch: self[index]. If stack_num is set to be > 0,
        return the stacked obs and obs_next with shape [batch, len, ...].
        """

        # protect_range = [(self._index - self.protect_num) % self._size, self._index] # [a, b)

        qualified = ((self._index - index) % self._size) > self.protect_num
        index_jump = (index + self.protect_num) % self._size
        index = (index * qualified + (1-qualified) * index_jump).astype(np.int)

        protect_index = np.random.randint(0, self.protect_num, size=index.shape)
        protect_index = (self._index - protect_index - 1) % self._size

        if self.with_pt:
            index = np.concatenate([index, protect_index], 0)

        ret =  Batch(
            obs=self.get(index, 'obs'),
            obs_next=self.get(index, 'obs_next'),
            act=self.act[index],
            rew=self.rew[index],
            done=self.done[index],
            info=self.get(index, 'info'),
            policy=self.get(index, 'policy'),
        )

        if self.log_prob is not None:
            ret.log_prob = self.log_prob[index]
        if self.alpha is not None:
            ret.alpha = self.alpha[index]
        # if hasattr(self, 'int_rew'):
        #     ret.int_rew = self.int_rew[index]

        if self._ens_num is not None:
            ret.mask=self.mask[index]
        if self._ngu:
            ret.erew=self.erew[index]
            ret.goal=self.goal[index]
        if self._rand2:
            ret.irew=self.irew[index]
        return ret


class ReplayBufferTriProtect(ReplayBuffer):
    def __init__(self, size, **kwargs) -> None:
        super().__init__(size, **kwargs)
        self.num = kwargs.get('num', 2)
        self.protect_num = kwargs.get('protect_num', 1)
        self.with_pt = kwargs.get('with_pt', True)


    def __getitem__(self, index: Union[
            slice, int, np.integer, np.ndarray]) -> Batch:
        """Return a data batch: self[index]. If stack_num is set to be > 0,
        return the stacked obs and obs_next with shape [batch, len, ...].
        """

        # protect_range = [(self._index - self.protect_num) % self._size, self._index] # [a, b)

        if self.protect_num > 0:
            qualified = ((self._index - index) % self._size) > self.protect_num
            index_jump = (index + self.protect_num) % self._size
            index = (index * qualified + (1-qualified) * index_jump).astype(np.int)

            protect_index = np.random.randint(0, self.protect_num, size=index.shape)
            protect_index = (self._index - protect_index - 1) % self._size

            if self.with_pt:
                index = np.concatenate([index, protect_index], 0)


        kwargs = dict()
        for i in range(1, self.num):
            kwargs[''.join(['act'] + ['_next'] * i)] = self.act[(index + i) % self._size]
            kwargs[''.join(['rew'] + ['_next'] * i)] = self.rew[(index + i) % self._size]
            kwargs[''.join(['obs_next'] + ['_next'] * i)] = self.get((index + i) % self._size, 'obs_next')
            kwargs[''.join(['done'] + ['_next'] * i)] = self.done[(index + i) % self._size]
            kwargs[''.join(['done_bk'] + ['_next'] * (i - 1))] = self.done_bk[(index + i - 1) % self._size]

        ret =  Batch(
            obs=self.get(index, 'obs'),
            obs_next=self.get(index, 'obs_next'),
            act=self.act[index],
            rew=self.rew[index],
            done=self.done[index],
            info=self.get(index, 'info'),
            policy=self.get(index, 'policy'),
            **kwargs
        )

        if self.log_prob is not None:
            ret.log_prob = self.log_prob[index]
        if self.alpha is not None:
            ret.alpha = self.alpha[index]
        # if hasattr(self, 'int_rew'):
        #     ret.int_rew = self.int_rew[index]

        if self._ens_num is not None:
            ret.mask=self.mask[index]
        if self._ngu:
            ret.erew=self.erew[index]
            ret.goal=self.goal[index]
        if self._rand2:
            ret.irew=self.irew[index]
        return ret



class ListReplayBuffer(ReplayBuffer):
    """The function of :class:`~tianshou.data.ListReplayBuffer` is almost the
    same as :class:`~tianshou.data.ReplayBuffer`. The only difference is that
    :class:`~tianshou.data.ListReplayBuffer` is based on ``list``. Therefore,
    it does not support advanced indexing, which means you cannot sample a
    batch of data out of it. It is typically used for storing data.

    .. seealso::

        Please refer to :class:`~tianshou.data.ReplayBuffer` for more
        detailed explanation.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(size=0, ignore_obs_next=False, **kwargs)

    def sample(self, batch_size: int) -> Tuple[Batch, np.ndarray]:
        raise NotImplementedError("ListReplayBuffer cannot be sampled!")

    def _add_to_buffer(
            self, name: str,
            inst: Union[dict, Batch, np.ndarray, float, int, bool]) -> None:
        if inst is None:
            return
        if self._meta.__dict__.get(name, None) is None:
            self._meta.__dict__[name] = []
        self._meta.__dict__[name].append(inst)

    def reset(self) -> None:
        self._index = self._size = 0
        for k in list(self._meta.__dict__.keys()):
            if isinstance(self._meta.__dict__[k], list):
                self._meta.__dict__[k] = []


class PrioritizedReplayBuffer(ReplayBuffer):
    """Prioritized replay buffer implementation.

    :param float alpha: the prioritization exponent.
    :param float beta: the importance sample soft coefficient.
    :param str mode: defaults to ``weight``.
    :param bool replace: whether to sample with replacement

    .. seealso::

        Please refer to :class:`~tianshou.data.ReplayBuffer` for more
        detailed explanation.
    """

    def __init__(self, size: int, alpha: float, beta: float,
                 mode: str = 'weight',
                 replace: bool = False, **kwargs) -> None:
        if mode != 'weight':
            raise NotImplementedError
        super().__init__(size, **kwargs)
        self._alpha = alpha
        self._beta = beta
        self._weight_sum = 0.0
        self._amortization_freq = 50
        self._replace = replace
        self._meta.weight = np.zeros(size, dtype=np.float64)
        self.max_priority = 1

    def add(self,
            obs: Union[dict, np.ndarray],
            act: Union[np.ndarray, float],
            rew: Union[int, float],
            done: bool,
            obs_next: Optional[Union[dict, np.ndarray]] = None,
            info: dict = {},
            policy: Optional[Union[dict, Batch]] = {},
            weight: Optional[float] = None,
            **kwargs) -> None:
        """Add a batch of data into replay buffer."""
        # we have to sacrifice some convenience for speed
        if weight is None:
            weight = self.max_priority
        else:
            weight = np.abs(weight) ** self._alpha
        self._weight_sum += weight - \
            self._meta.weight[self._index]
        self._add_to_buffer('weight', weight)
        super().add(obs, act, rew, done, obs_next, info, policy)

    @property
    def replace(self):
        return self._replace

    @replace.setter
    def replace(self, v: bool):
        self._replace = v

    def sample(self, batch_size: int) -> Tuple[Batch, np.ndarray]:
        """Get a random sample from buffer with priority probability. \
        Return all the data in the buffer if batch_size is ``0``.

        :return: Sample data and its corresponding index inside the buffer.
        """
        assert self._size > 0, 'cannot sample a buffer with size == 0 !'
        p = None
        if batch_size > 0 and (self._replace or batch_size <= self._size):
            # sampling weight
            p = (self.weight / self.weight.sum())[:self._size]
            indice = np.random.choice(
                self._size, batch_size, p=p,
                replace=self._replace)
            p = p[indice]  # weight of each sample
        elif batch_size == 0:
            p = np.full(shape=self._size, fill_value=1.0 / self._size)
            indice = np.concatenate([
                np.arange(self._index, self._size),
                np.arange(0, self._index),
            ])
        else:
            raise ValueError(
                f"batch_size should be less than {len(self)}, \
                    or set replace=True")
        batch = self[indice]
        batch["impt_weight"] = (self._size * p) ** (-self._beta)
        return batch, indice

    def update_weight(self, indice: Union[slice, np.ndarray],
                      new_weight: np.ndarray) -> None:
        """Update priority weight by indice in this buffer.

        :param np.ndarray indice: indice you want to update weight
        :param np.ndarray new_weight: new priority weight you want to update
        """
        if self._replace:
            if isinstance(indice, slice):
                # convert slice to ndarray
                indice = np.arange(indice.stop)[indice]
            # remove the same values in indice
            indice, unique_indice = np.unique(
                indice, return_index=True)
            new_weight = new_weight[unique_indice]
        new_weight = np.power(np.abs(new_weight), self._alpha)
        self.max_priority = np.maximum(np.max(new_weight), self.max_priority)
        self.weight[indice] = new_weight

    def __getitem__(self, index: Union[
            slice, int, np.integer, np.ndarray]) -> Batch:
        return Batch(
            obs=self.get(index, 'obs'),
            act=self.act[index],
            rew=self.rew[index],
            done=self.done[index],
            obs_next=self.get(index, 'obs_next'),
            info=self.get(index, 'info'),
            weight=self.weight[index],
            policy=self.get(index, 'policy'),
        )



from tianshou.utils.sum_tree import SumTree


class PrioritizedReplayBuffer_RLKit(ReplayBuffer):
    """Prioritized replay buffer implementation.

    :param float alpha: the prioritization exponent.
    :param float beta: the importance sample soft coefficient.
    :param str mode: defaults to ``weight``.
    :param bool replace: whether to sample with replacement

    .. seealso::

        Please refer to :class:`~tianshou.data.ReplayBuffer` for more
        detailed explanation.
    """

    def __init__(self, size: int, alpha: float, beta: float,
                 mode: str = 'weight',
                 replace: bool = False, **kwargs) -> None:
        if mode != 'weight':
            raise NotImplementedError
        super().__init__(size, **kwargs)
        self._alpha = alpha
        self._beta = beta
        self._amortization_freq = 50
        self._replace = replace
        self.tree = SumTree(size)
        self.max_priority = 1

    def add(self,
            obs: Union[dict, np.ndarray],
            act: Union[np.ndarray, float],
            rew: Union[int, float],
            done: bool,
            obs_next: Optional[Union[dict, np.ndarray]] = None,
            info: dict = {},
            policy: Optional[Union[dict, Batch]] = {},
            **kwargs) -> None:
        """Add a batch of data into replay buffer."""
        # we have to sacrifice some convenience for speed
        super().add(obs, act, rew, done, obs_next, info, policy)
        self.tree.add(self.max_priority)

    def sample(self, batch_size: int) -> Tuple[Batch, np.ndarray]:
        """Get a random sample from buffer with priority probability. \
        Return all the data in the buffer if batch_size is ``0``.

        :return: Sample data and its corresponding index inside the buffer.
        """
        assert self._size > 0, 'cannot sample a buffer with size == 0 !'

        if batch_size > 0:
            segment = self.tree.total() / batch_size

            # sampled_data = []
            indice_data = []
            indice_p = []
            p = []
            for i in range(batch_size):
                a = segment * i
                b = segment * (i + 1)
                s = np.random.uniform(a, b)
                (idx, pri, data_index) = self.tree.get(s)
                if data_index >= self._index and self._size != self._maxsize:
                    continue
                    # raise AssertionError#, (data_index, self._index, self._size, self._maxsize)
                indice_p.append(idx)
                indice_data.append(data_index)
                p.append(pri)

            while len(p) < batch_size:
                # This should rarely happen
                idx = np.random.randint(len(p))
                p.append(p[idx])
                indice_p.append(indice_p[idx])
                indice_data.append(indice_data[idx])
        else:
            raise ValueError(
                f"batch_size should be less than {len(self)}, \
                    or set replace=True")
        p = np.array(p)
        indice_p = np.array(indice_p)
        indice_data = np.array(indice_data)

        batch = self[indice_data]
        batch["impt_weight"] = (self._size * p) ** (-self._beta)
        batch["impt_weight"] = batch['impt_weight'] / (np.max(batch['impt_weight']))
        return batch, indice_p

    def update_weight(self, indice: Union[slice, np.ndarray],
                      new_weight: np.ndarray) -> None:
        """Update priority weight by indice in this buffer.

        :param np.ndarray indice: indice you want to update weight
        :param np.ndarray new_weight: new priority weight you want to update
        """
        new_weight = np.power(np.abs(new_weight), self._alpha)
        self.max_prioirty = np.maximum(np.max(new_weight), self.max_priority)
        for idx, priority in zip(indice, new_weight):
            self.tree.update(idx, priority)



# class ReplayBuffer_debug(ReplayBuffer):
#     def __init__(self, size: int, stack_num: Optional[int] = 0,
#                  ignore_obs_next: bool = False,
#                  sample_avail: bool = False, **kwargs) -> None:
#         super().__init__(size, stack_num, ignore_obs_next, sample_avail, **kwargs)
#
#     def add(self,
#             obs: Union[dict, Batch, np.ndarray],
#             act: Union[np.ndarray, float],
#             rew: Union[int, float],
#             done: bool,
#             obs_next: Optional[Union[dict, Batch, np.ndarray]] = None,
#             info: dict = {},
#             policy: Optional[Union[dict, Batch]] = {},
#             **kwargs) -> None:
#         """Add a batch of data into replay buffer."""
#         assert isinstance(info, (dict, Batch)), \
#             'You should return a dict in the last argument of env.step().'
#         self._add_to_buffer('obs', obs)
#         self._add_to_buffer('act', act)
#         self._add_to_buffer('rew', rew)
#         self._add_to_buffer('done', done)
#         if self._save_s_:
#             if obs_next is None:
#                 obs_next = Batch()
#             self._add_to_buffer('obs_next', obs_next)
#         self._add_to_buffer('info', info)
#         self._add_to_buffer('policy', policy)
#         self._add_to_buffer('logp', kwargs.get('logp', 0))
#         if self._ens_num is not None:
#             mask = np.random.randint(2, size=self._ens_num)
#             self._add_to_buffer('mask', mask)
#         if self._ngu:
#             self._add_to_buffer('erew', 0)
#             self._add_to_buffer('goal', 0)
#         if self._rand2:
#             self._add_to_buffer('irew', 0)
#
#         # maintain available index for frame-stack sampling
#         if self._avail:
#             # update current frame
#             avail = sum(self.done[i] for i in range(
#                 self._index - self._stack + 1, self._index)) == 0
#             if self._size < self._stack - 1:
#                 avail = False
#             if avail and self._index not in self._avail_index:
#                 self._avail_index.append(self._index)
#             elif not avail and self._index in self._avail_index:
#                 self._avail_index.remove(self._index)
#             # remove the later available frame because of broken storage
#             t = (self._index + self._stack - 1) % self._maxsize
#             if t in self._avail_index:
#                 self._avail_index.remove(t)
#
#         if self._maxsize > 0:
#             self._size = min(self._size + 1, self._maxsize)
#             self._index = (self._index + 1) % self._maxsize
#         else:
#             self._size = self._index = self._index + 1
#
#     def __getitem__(self, index: Union[
#             slice, int, np.integer, np.ndarray]) -> Batch:
#         """Return a data batch: self[index]. If stack_num is set to be > 0,
#         return the stacked obs and obs_next with shape [batch, len, ...].
#         """
#         if self._ens_num is not None:
#             return Batch(
#                 obs=self.get(index, 'obs'),
#                 act=self.act[index],
#                 rew=self.rew[index],
#                 done=self.done[index],
#                 obs_next=self.get(index, 'obs_next'),
#                 info=self.get(index, 'info'),
#                 policy=self.get(index, 'policy'),
#                 mask=self.mask[index],
#                 logp=self.logp[index]
#             )
#         elif self._ngu:
#             return Batch(
#                 obs=self.get(index, 'obs'),
#                 act=self.act[index],
#                 rew=self.rew[index],
#                 done=self.done[index],
#                 obs_next=self.get(index, 'obs_next'),
#                 info=self.get(index, 'info'),
#                 policy=self.get(index, 'policy'),
#                 erew=self.erew[index],
#                 goal=self.goal[index],
#                 logp = self.logp[index]
#             )
#         elif self._rand2:
#             return Batch(
#                 obs=self.get(index, 'obs'),
#                 act=self.act[index],
#                 rew=self.rew[index],
#                 done=self.done[index],
#                 obs_next=self.get(index, 'obs_next'),
#                 info=self.get(index, 'info'),
#                 policy=self.get(index, 'policy'),
#                 irew=self.irew[index],
#                 logp = self.logp[index]
#             )
#         else:
#             return Batch(
#                 obs=self.get(index, 'obs'),
#                 act=self.act[index],
#                 rew=self.rew[index],
#                 done=self.done[index],
#                 obs_next=self.get(index, 'obs_next'),
#                 info=self.get(index, 'info'),
#                 policy=self.get(index, 'policy'),
#                 logp = self.logp[index]
#             )

# from tianshou.utils.sum_tree import SumTree
#
# class PrioritizedReplay(ReplayBuffer):
#     TransitionCLS = PrioritizedTransition
#
#     def __init__(self, memory_size, batch_size, n_step=1, discount=1, history_length=1, keys=None):
#         super(PrioritizedReplay, self).__init__(memory_size, batch_size, n_step, discount, history_length, keys)
#         self.tree = SumTree(memory_size)
#         self.max_priority = 1
#
#     def feed(self, data):
#         super().feed(data)
#         self.tree.add(self.max_priority, None)
#
#     def sample(self, batch_size=None):
#         if batch_size is None:
#             batch_size = self.batch_size
#
#         segment = self.tree.total() / batch_size
#
#         sampled_data = []
#         for i in range(batch_size):
#             a = segment * i
#             b = segment * (i + 1)
#             s = random.uniform(a, b)
#             (idx, p, data_index) = self.tree.get(s)
#             transition = super().construct_transition(data_index)
#             if transition is None:
#                 continue
#             sampled_data.append(PrioritizedTransition(
#                 *transition,
#                 sampling_prob=p / self.tree.total(),
#                 idx=idx,
#             ))
#         while len(sampled_data) < batch_size:
#             # This should rarely happen
#             sampled_data.append(random.choice(sampled_data))
#
#         sampled_data = zip(*sampled_data)
#         sampled_data = list(map(lambda x: np.asarray(x), sampled_data))
#         sampled_data = PrioritizedTransition(*sampled_data)
#         return sampled_data
#
#     def update_priorities(self, info):
#         for idx, priority in info:
#             self.max_priority = max(self.max_priority, priority)
#             self.tree.update(idx, priority)