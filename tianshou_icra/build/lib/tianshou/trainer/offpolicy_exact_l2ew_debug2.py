import time
import tqdm
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, List, Union, Callable, Optional

from tianshou.data import Collector, Batch
from tianshou.policy import BasePolicy
from tianshou.utils import tqdm_config, MovAvg
from tianshou.trainer import test_episode, gather_info

import numpy as np
import copy

def offpolicy_exact_l2ew_debug2_trainer(
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
        test_data: Optional[Batch] = None,
        n_policy: int = 1,
        n_sample: int = 1,
        update_times: int = 1000,
        int_driven: bool = True
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
    should_update_times = 0
    for epoch in range(1, 1 + max_epoch):
        # train
        policy.train()
        if train_fn:
            train_fn(epoch)
        with tqdm.tqdm(total=step_per_epoch, desc=f'Epoch #{epoch}',
                       **tqdm_config) as t:
            while t.n < t.total:  # this is shown in the progress bar, represents the update times
                data = {}
                if should_update_times <= 0:
                    if epoch == 1:
                        result = train_collector.collect(n_step=collect_per_step, log_fn=log_fn)
                        for k in result.keys():
                            data[k] = f'{result[k]:.2f}'
                            if writer and global_step % log_interval == 0:
                                writer.add_scalar(
                                    k, result[k], global_step=global_step)
                        collect_data_num = result['n/st']
                    else:
                        # result = train_collector.collect(n_step=collect_per_step, log_fn=log_fn)
                        # save current parameters
                        for par_pri, par in zip(policy_prime.parameters(), policy.parameters()):
                            par_pri.data.copy_(par.data)

                        for opt_pri, opt in zip(policy_prime.get_optimizers(), policy.get_optimizers()):
                            opt_pri.load_state_dict(opt.state_dict())

                        collect_data_num = 0
                        ws = []
                        for i in range(n_policy):
                            # recover from the original
                            for par, par_pri in zip(policy.parameters(), policy_prime.parameters()):
                                par.data.copy_(par_pri.data)

                            for opt, opt_pri in zip(policy.get_optimizers(), policy_prime.get_optimizers()):
                                opt.load_state_dict(opt_pri.state_dict())

                            w = policy.sample_w()
                            policy.set_w(w)

                            i_values = []
                            e_values = []
                            for _ in range(update_times):
                                policy.learn(train_collector.sample(batch_size), update_rand=False)

                            for j in range(n_sample):
                                result = explore_collector.collect(n_step=collect_per_step, log_fn=log_fn)
                                for k in result.keys():
                                    data[k] = f'{result[k]:.2f}'
                                    if writer and global_step % log_interval == 0:
                                        writer.add_scalar(
                                            k, result[k], global_step=global_step + collect_data_num)
                                collect_data_num += result['n/st']
                                indexes = []
                                index = explore_collector.buffer._index - 1
                                for _ in range(result['n/st']):
                                    indexes.append(index)
                                    index -= 1
                                    if index < 0:
                                        index = explore_collector.buffer._size - 1
                                indexes = np.array(indexes)[::-1]
                                ivalue = policy.check_intrisic_value(explore_collector.buffer[indexes])
                                evalue = result['rew']
                                i_values.append(ivalue)
                                e_values.append(evalue)
                                print(i, j, w.exp().mean().item(), result['n/st'], ivalue, evalue)

                            if int_driven:
                                ws.append([copy.deepcopy(w), i_values])
                            else:
                                ws.append([copy.deepcopy(w), e_values])

                        # recover from the original
                        for par, par_pri in zip(policy.parameters(), policy_prime.parameters()):
                            par.data.copy_(par_pri.data)

                        for opt, opt_pri in zip(policy.get_optimizers(), policy_prime.get_optimizers()):
                            opt.load_state_dict(opt_pri.state_dict())

                        train_collector.buffer.update(explore_collector.buffer)
                        explore_collector.reset_buffer()
                        policy.update_w(ws)

                        n_bar = 10
                        if test_data is not None:
                            l_int_rew2, l_alpha2 = policy.update_rms(test_data, batch_size, update=False)
                            l_int_rew2 = list(l_int_rew2.reshape(-1))
                            l_alpha2 = list(l_alpha2.reshape(-1))

                            while True:
                                if len(l_int_rew2) % n_bar != 0:
                                    l_int_rew2.append(l_int_rew2[-1])
                                else:
                                    break

                            while True:
                                if len(l_alpha2) % n_bar != 0:
                                    l_alpha2.append(l_alpha2[-1])
                                else:
                                    break

                            l_int_rew2 = np.array(l_int_rew2).reshape(n_bar, -1).mean(-1)
                            l_alpha2 = np.array(l_alpha2).reshape(n_bar, -1).mean(-1)

                            for ii in range(n_bar):
                                if test_data is not None:
                                    writer.add_scalar('int2/i' + str(ii), l_int_rew2[ii], global_step=global_step)
                                    writer.add_scalar('alpha2/a'+str(ii), l_alpha2[ii], global_step=global_step)

                    should_update_times = collect_data_num // collect_per_step

                update_times_ = 0
                data = {}
                for i in range(update_per_step * min(
                        should_update_times, t.total - t.n)):
                    global_step += 1
                    update_times_ += 1
                    losses = policy.learn(train_collector.sample(batch_size))

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

                should_update_times -= update_times_

            if t.n <= t.total:
                t.update()
        # test
        result_test = test_episode(
            policy, test_collector, test_fn, epoch, episode_per_test)

        for k in result_test.keys():
            if writer:
                writer.add_scalar('test/'+str(k), result_test[k], global_step=global_step)

        if best_epoch == -1 or best_reward < result_test['rew']:
            best_reward = result_test['rew']
            best_epoch = epoch
            if save_fn:
                save_fn(policy)
        if verbose:
            print(f'Epoch #{epoch}: test_reward: {result_test["rew"]:.6f}, '
                  f'best_reward: {best_reward:.6f} in #{best_epoch}')
        if stop_fn and stop_fn(best_reward):
            break

    return gather_info(
        start_time, train_collector, test_collector, best_reward)
