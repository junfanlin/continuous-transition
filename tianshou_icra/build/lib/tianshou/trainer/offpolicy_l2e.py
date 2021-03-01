import time
import tqdm
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, List, Union, Callable, Optional

from tianshou.data import Collector
from tianshou.policy import BasePolicy
from tianshou.utils import tqdm_config, MovAvg
from tianshou.trainer import test_episode, gather_info

import copy
import numpy as np

def compute_value_last(collector, n_step):
    it = collector.buffer._index - 1
    last_frame = collector.buffer[it]
    v_boot = collector.policy.get_value(last_frame.obs_next[None, :]).item()

    if collector.policy._rew_norm:
        bfr = collector.buffer.rew[:min(len(collector.buffer), 1000)]  # avoid large buffer
        mean, std = bfr.mean(), bfr.std()
        if np.isclose(std, 0):
            mean, std = 0, 1
    else:
        mean, std = 0, 1

    for i in range(n_step):
        last_frame = collector.buffer[it]
        v_boot = (last_frame.rew - mean) / std + collector.policy._gamma * (1 - last_frame.done) * v_boot
        it -= 1
        if it < 0:
            it = collector.buffer._size - 1
    return v_boot



def offpolicy_l2e_trainer(
        policy: BasePolicy,
        train_collector: Collector,
        explore_collector: Collector,
        test_collector: Collector,
        max_epoch: int,
        step_per_epoch: int,
        collect_per_step: int,
        episode_per_test: Union[int, List[int]],
        batch_size: int,
        update_per_step: int = 1,
        train_fn: Optional[Callable[[int], None]] = None,
        test_fn: Optional[Callable[[int], None]] = None,
        stop_fn: Optional[Callable[[float], bool]] = None,
        save_fn: Optional[Callable[[BasePolicy], None]] = None,
        log_fn: Optional[Callable[[dict], None]] = None,
        writer: Optional[SummaryWriter] = None,
        log_interval: int = 1,
        verbose: bool = True,
        test_in_train: bool = True,
        l2e_times: int = 1,
) -> Dict[str, Union[float, str]]:
    """A wrapper for off-policy trainer procedure.

    :param policy: an instance of the :class:`~tianshou.policy.BasePolicy`
        class.
    :param train_collector: the collector used for training.
    :type train_collector: :class:`~tianshou.data.Collector`
    :param test_collector: the collector used for testing.
    :type test_collector: :class:`~tianshou.data.Collector`
    :param int max_epoch: the maximum of epochs for training. The training
        process might be finished before reaching the ``max_epoch``.
    :param int step_per_epoch: the number of step for updating policy network
        in one epoch.
    :param int collect_per_step: the number of frames the collector would
        collect before the network update. In other words, collect some frames
        and do some policy network update.
    :param episode_per_test: the number of episodes for one policy evaluation.
    :param int batch_size: the batch size of sample data, which is going to
        feed in the policy network.
    :param int update_per_step: the number of times the policy network would
        be updated after frames be collected. In other words, collect some
        frames and do some policy network update.
    :param function train_fn: a function receives the current number of epoch
        index and performs some operations at the beginning of training in this
        epoch.
    :param function test_fn: a function receives the current number of epoch
        index and performs some operations at the beginning of testing in this
        epoch.
    :param function save_fn: a function for saving policy when the undiscounted
        average mean reward in evaluation phase gets better.
    :param function stop_fn: a function receives the average undiscounted
        returns of the testing result, return a boolean which indicates whether
        reaching the goal.
    :param function log_fn: a function receives env info for logging.
    :param torch.utils.tensorboard.SummaryWriter writer: a TensorBoard
        SummaryWriter.
    :param int log_interval: the log interval of the writer.
    :param bool verbose: whether to print the information.
    :param bool test_in_train: whether to test in the training phase.

    :return: See :func:`~tianshou.trainer.gather_info`.
    """
    global_step = 0
    best_epoch, best_reward = -1, -1
    stat = {}
    start_time = time.time()
    test_in_train = test_in_train and train_collector.policy == policy
    policy_prime = copy.deepcopy(policy)
    for epoch in range(1, 1 + max_epoch):
        # train
        policy.train()
        if train_fn:
            train_fn(epoch)
        with tqdm.tqdm(total=step_per_epoch, desc=f'Epoch #{epoch}',
                       **tqdm_config) as t:
            while t.n < t.total:  # this is shown in the progress bar, represents the update times
                result_e = explore_collector.collect(n_step=collect_per_step, log_fn=log_fn)
                result_before = train_collector.collect(n_step=collect_per_step, log_fn=log_fn)

                value_before = compute_value_last(train_collector, result_before['n/st'])

                for par_pri, par in zip(policy_prime.parameters(), policy.parameters()):
                    par_pri.data.copy_(par.data)

                for opt_pri, opt in zip(policy_prime.get_optimizers(), policy.get_optimizers()):
                    opt_pri.load_state_dict(opt.state_dict())

                for i in range(result_e['n/st'] // collect_per_step * l2e_times):
                    policy.learn(explore_collector.sample(batch_size))

                result_after = train_collector.collect(n_step=collect_per_step, log_fn=log_fn)

                value_after = compute_value_last(train_collector, result_after['n/st'])

                for par, par_pri in zip(policy.parameters(), policy_prime.parameters()):
                    par.data.copy_(par_pri.data)

                for opt, opt_pri in zip(policy.get_optimizers(), policy_prime.get_optimizers()):
                    opt.load_state_dict(opt_pri.state_dict())

                result_value = value_after - value_before

                policy.learn_e(explore_collector.sample(0), result_value, batch_size)

                train_collector.buffer.update(explore_collector.buffer)
                explore_collector.reset_buffer()

                data = {}
                if test_in_train and stop_fn and stop_fn(result_after['rew']):
                    test_result = test_episode(
                        policy, test_collector, test_fn,
                        epoch, episode_per_test)
                    if stop_fn and stop_fn(test_result['rew']):
                        if save_fn:
                            save_fn(policy)
                        for k in result.keys():
                            data[k] = f'{result[k]:.2f}'
                        t.set_postfix(**data)
                        return gather_info(
                            start_time, train_collector, test_collector,
                            test_result['rew'])
                    else:
                        policy.train()
                        if train_fn:
                            train_fn(epoch)

                for i in range(update_per_step * min(
                        result_after['n/st'] + result_e['n/st'] + result_before['n/st'] // collect_per_step,
                        t.total - t.n)):
                    global_step += 1
                    losses = policy.learn(train_collector.sample(batch_size))
                    for k in result_after.keys():
                        data[k] = f'{result_after[k]:.2f}'
                        if writer and global_step % log_interval == 0:
                            writer.add_scalar(
                                k, result_after[k], global_step=global_step)
                    for k in losses.keys():
                        if stat.get(k) is None:
                            stat[k] = MovAvg()
                        stat[k].add(losses[k])
                        data[k] = f'{stat[k].get():.6f}'
                        if writer and global_step % log_interval == 0:
                            writer.add_scalar(
                                k, stat[k].get(), global_step=global_step)
                    t.update(1)
                    t.set_postfix(**data)
            if t.n <= t.total:
                t.update()
        # test
        result = test_episode(
            policy, test_collector, test_fn, epoch, episode_per_test)
        if best_epoch == -1 or best_reward < result['rew']:
            best_reward = result['rew']
            best_epoch = epoch
            if save_fn:
                save_fn(policy)
        if verbose:
            print(f'Epoch #{epoch}: test_reward: {result["rew"]:.6f}, '
                  f'best_reward: {best_reward:.6f} in #{best_epoch}')
        if stop_fn and stop_fn(best_reward):
            break
    return gather_info(
        start_time, train_collector, test_collector, best_reward)
