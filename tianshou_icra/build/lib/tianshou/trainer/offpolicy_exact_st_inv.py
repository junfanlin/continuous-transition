import time
import tqdm
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, List, Union, Callable, Optional

from tianshou.data import Collector
from tianshou.policy import BasePolicy
from tianshou.utils import tqdm_config, MovAvg
from tianshou.trainer import test_episode, gather_info

from collections import deque
import numpy as np
import torch
import copy

def offpolicy_exact_st_inv_trainer(
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
        self_taught_duration: int = 1,
        n_meta_update: int = 1,
        n_loop: int = 1,
        n_rnd_update: int = 1
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
    should_update_times = 0

    for epoch in range(1, 1 + max_epoch):
        # train
        policy.train()
        if train_fn:
            train_fn(epoch)
        with tqdm.tqdm(total=step_per_epoch, desc=f'Epoch #{epoch}',
                       **tqdm_config) as t:
            while t.n < t.total:  # this is shown in the progress bar, represents the update times
                if should_update_times <= 0:
                    if epoch == 1:
                        result = train_collector.collect(n_step=collect_per_step,
                                                         log_fn=log_fn)
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

                        should_update_times = result['n/st'] // collect_per_step
                        for k in result.keys():
                            data[k] = f'{result[k]:.2f}'
                            if writer and global_step % log_interval == 0:
                                writer.add_scalar(
                                    k, result[k], global_step=global_step)
                    else:
                        data = {}

                        w1, w2 = policy.sample_w()

                        # save the original
                        mem_param = []
                        for p in policy.parameters():
                            mem_param.append(p.data.clone())
                        mem_dict = copy.deepcopy([opt.state_dict() for opt in policy.get_optimizers()])
                        log_alpha = copy.deepcopy(policy._log_alpha.data)


                        # start left evaluation
                        policy.set_w(w1)
                        for i in range(n_meta_update):
                            losses = policy.learn(train_collector.sample(batch_size), update_rand=False)
                        explore_collector.reset_buffer()
                        result = explore_collector.collect(n_step=collect_per_step,
                                                         log_fn=log_fn)

                        # recover from the original
                        for par, par_pri in zip(policy.parameters(), mem_param):
                            par.data.copy_(copy.deepcopy(par_pri))
                        for opt, opt_pri in zip(policy.get_optimizers(), mem_dict):
                            opt.load_state_dict(copy.deepcopy(opt_pri))
                        policy._log_alpha.data.copy_(log_alpha)
                        # evaluate use the original
                        p1 = policy.gather_evaluation(explore_collector.sample(0), batch_size)
                        should_update_times = result['n/st'] // collect_per_step


                        # start right evaluation
                        policy.set_w(w2)
                        for i in range(n_meta_update):
                            losses = policy.learn(train_collector.sample(batch_size), update_rand=False)
                        train_collector.buffer.update(explore_collector.buffer)
                        explore_collector.reset_buffer()
                        result2 = explore_collector.collect(n_step=collect_per_step,
                                                           log_fn=log_fn)

                        # recover from the original
                        for par, par_pri in zip(policy.parameters(), mem_param):
                            par.data.copy_(copy.deepcopy(par_pri))
                        for opt, opt_pri in zip(policy.get_optimizers(), mem_dict):
                            opt.load_state_dict(copy.deepcopy(opt_pri))
                        policy._log_alpha.data.copy_(log_alpha)
                        # evaluate use the original
                        p2 = policy.gather_evaluation(explore_collector.sample(0), batch_size)
                        should_update_times += result2['n/st'] // collect_per_step
                        train_collector.buffer.update(explore_collector.buffer)
                        explore_collector.reset_buffer()

                        # udpate w
                        loss_w, grad_w, w = policy.update_w([w1, p1], [w2, p2])


                        for k in result.keys():
                            data[k] = f'{(result[k]+result2[k])/ 2:.2f}'
                            if writer:
                                writer.add_scalar(
                                    k, (result[k]+result2[k])/2, global_step=global_step)

                        if writer:
                            writer.add_scalar('w/loss', loss_w, global_step=global_step)
                            writer.add_scalar('w/grad', grad_w, global_step=global_step)
                            writer.add_scalar('w/w', w, global_step=global_step)

                        for _ in range(n_rnd_update):
                            policy.update_rand(train_collector.sample(batch_size))

                update_times = 0
                for i in range(update_per_step * min(
                        should_update_times, t.total - t.n)):
                    global_step += 1
                    update_times += 1
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

                should_update_times -= update_times

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
