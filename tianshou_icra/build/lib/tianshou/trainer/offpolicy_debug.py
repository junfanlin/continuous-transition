import time
import tqdm
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, List, Union, Callable, Optional

from tianshou.data import Collector, Batch
from tianshou.policy import BasePolicy
from tianshou.utils import tqdm_config, MovAvg
from tianshou.trainer import test_episode, gather_info

import numpy as np

def offpolicy_debug_trainer(
        policy: BasePolicy,
        train_collector: Collector,
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
        test_data: Optional[Batch] = None,
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
    for epoch in range(1, 1 + max_epoch):
        # train
        policy.train()
        if train_fn:
            train_fn(epoch)
        with tqdm.tqdm(total=step_per_epoch, desc=f'Epoch #{epoch}',
                       **tqdm_config) as t:
            while t.n < t.total:  # this is shown in the progress bar, represents the update times
                result = train_collector.collect(n_step=collect_per_step,
                                                 log_fn=log_fn)

                n_bar = 10
                l_logp = []
                # l_int_rew = []
                indexes = []
                index = train_collector.buffer._index - 1
                for ii in range(result['n/st']):
                    l_logp.append(train_collector.buffer[index].log_prob)
                    # l_int_rew.append(train_collector.buffer[index].int_rew)
                    indexes.append(index)
                    index -= 1
                    if index < 0:
                        index = train_collector.buffer._size - 1
                indexes = np.array(indexes)[::-1]

                l_int_rew, l_alpha = policy.update_rms(train_collector.buffer[indexes], batch_size)
                rns_before, rns_after = policy.update_rand(train_collector.buffer[np.array(indexes)], batch_size,
                                                           times=1)

                l_int_rew = list(l_int_rew.reshape(-1))
                l_alpha = list(l_alpha.reshape(-1))
                if test_data is not None:
                    l_int_rew2, l_alpha2 = policy.update_rms(test_data, batch_size, update=False)
                    l_int_rew2 = list(l_int_rew2.reshape(-1))
                    l_alpha2 = list(l_alpha2.reshape(-1))

                while True:
                    if len(l_logp) % n_bar != 0:
                        l_logp.append(l_logp[-1])
                        l_int_rew.append(l_int_rew[-1])
                        l_alpha.append(l_alpha[-1])
                    else:
                        break

                while True:
                    if len(l_int_rew2) % n_bar != 0:
                        l_int_rew2.append(l_int_rew2[-1])
                        l_alpha2.append(l_alpha2[-1])
                    else:
                        break

                l_logp = np.array(l_logp).reshape(n_bar, -1).mean(-1)
                l_alpha = np.array(l_alpha).reshape(n_bar, -1).mean(-1)
                l_int_rew = np.array(l_int_rew).reshape(n_bar, -1).mean(-1)
                if test_data is not None:
                    l_int_rew2 = np.array(l_int_rew2).reshape(n_bar, -1).mean(-1)
                    l_alpha2 = np.array(l_alpha2).reshape(n_bar, -1).mean(-1)

                for ii in range(n_bar):
                    writer.add_scalar('logp/p' + str(ii), l_logp[ii], global_step=global_step)
                    writer.add_scalar('alpha/a'+str(ii), l_alpha[ii], global_step=global_step)
                    writer.add_scalar('int/i' + str(ii), l_int_rew[ii], global_step=global_step)
                    if test_data is not None:
                        writer.add_scalar('int2/i' + str(ii), l_int_rew2[ii], global_step=global_step)
                        writer.add_scalar('alpha2/a'+str(ii), l_alpha2[ii], global_step=global_step)

                data = {}
                if test_in_train and stop_fn and stop_fn(result['rew']):
                    test_result = test_episode(
                        policy, test_collector, test_fn,
                        epoch, episode_per_test)

                    for k in test_result.keys():
                        if writer:
                            writer.add_scalar('test/' + str(k), test_result[k], global_step=global_step)

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
                        result['n/st'] // collect_per_step, t.total - t.n)):
                    global_step += 1
                    losses = policy.learn(train_collector.sample(batch_size))
                    for k in result.keys():
                        data[k] = f'{result[k]:.2f}'
                        if writer and global_step % log_interval == 0:
                            writer.add_scalar(
                                k, result[k], global_step=global_step)
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

                rns_before, rns_after = policy.update_rand(train_collector.buffer[np.array(indexes)], batch_size, times=10)

                print(rns_before, rns_after, rns_before - rns_after)

            if t.n <= t.total:
                t.update()
        # test
        result = test_episode(
            policy, test_collector, test_fn, epoch, episode_per_test)

        for k in result.keys():
            if writer:
                writer.add_scalar('test/'+str(k), result[k], global_step=global_step)

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
