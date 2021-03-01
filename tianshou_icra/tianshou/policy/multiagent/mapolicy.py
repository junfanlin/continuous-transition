import numpy as np
from typing import Union, Optional, Dict, List

from tianshou.policy import BasePolicy
from tianshou.data import Batch, ReplayBuffer


class MultiAgentPolicyManager(BasePolicy):
    """This multi-agent policy manager accepts a list of
    :class:`~tianshou.policy.BasePolicy`. It dispatches the batch data to each
    of these policies when the "forward" is called. The same as "process_fn"
    and "learn": it splits the data and feeds them to each policy. A figure in
    :ref:`marl_example` can help you better understand this procedure.
    """

    def __init__(self, policies: List[BasePolicy]):
        super().__init__()
        self.policies = policies
        for i, policy in enumerate(policies):
            # agent_id 0 is reserved for the environment proxy
            # (this MultiAgentPolicyManager)
            policy.set_agent_id(i + 1)

    def replace_policy(self, policy, agent_id):
        """Replace the "agent_id"th policy in this manager."""
        self.policies[agent_id - 1] = policy
        policy.set_agent_id(agent_id)

    def process_fn(self, batch: Batch, buffer: ReplayBuffer,
                   indice: np.ndarray) -> Batch:
        """Save original multi-dimensional rew in "save_rew", set rew to the
        reward of each agent during their ``process_fn``, and restore the
        original reward afterwards.
        """
        results = {}
        # reward can be empty Batch (after initial reset) or nparray.
        has_rew = isinstance(buffer.rew, np.ndarray)
        if has_rew:  # save the original reward in save_rew
            save_rew, buffer.rew = buffer.rew, Batch()
        for policy in self.policies:
            agent_index = np.nonzero(batch.obs.agent_id == policy.agent_id)[0]
            if len(agent_index) == 0:
                results[f'agent_{policy.agent_id}'] = Batch()
                continue
            tmp_batch, tmp_indice = batch[agent_index], indice[agent_index]
            if has_rew:
                tmp_batch.rew = tmp_batch.rew[:, policy.agent_id - 1]
                buffer.rew = save_rew[:, policy.agent_id - 1]
            results[f'agent_{policy.agent_id}'] = \
                policy.process_fn(tmp_batch, buffer, tmp_indice)
        if has_rew:  # restore from save_rew
            buffer.rew = save_rew
        return Batch(results)

    def forward(self, batch: Batch,
                state: Optional[Union[dict, Batch]] = None,
                **kwargs) -> Batch:
        """:param state: if None, it means all agents have no state. If not
            None, it should contain keys of "agent_1", "agent_2", ...

        :return: a Batch with the following contents:

        ::

            {
                "act": actions corresponding to the input
                "state":{
                    "agent_1": output state of agent_1's policy for the state
                    "agent_2": xxx
                    ...
                    "agent_n": xxx}
                "out":{
                    "agent_1": output of agent_1's policy for the input
                    "agent_2": xxx
                    ...
                    "agent_n": xxx}
            }
        """
        results = []
        for policy in self.policies:
            # This part of code is difficult to understand.
            # Let's follow an example with two agents
            # batch.obs.agent_id is [1, 2, 1, 2, 1, 2] (with batch_size == 6)
            # each agent plays for three transitions
            # agent_index for agent 1 is [0, 2, 4]
            # agent_index for agent 2 is [1, 3, 5]
            # we separate the transition of each agent according to agent_id
            agent_index = np.nonzero(batch.obs.agent_id == policy.agent_id)[0]
            if len(agent_index) == 0:
                # (has_data, agent_index, out, act, state)
                results.append((False, None, Batch(), None, Batch()))
                continue
            tmp_batch = batch[agent_index]
            if isinstance(tmp_batch.rew, np.ndarray):
                # reward can be empty Batch (after initial reset) or nparray.
                tmp_batch.rew = tmp_batch.rew[:, policy.agent_id - 1]
            out = policy(batch=tmp_batch, state=None if state is None
                         else state["agent_" + str(policy.agent_id)],
                         **kwargs)
            act = out.act
            each_state = out.state \
                if (hasattr(out, 'state') and out.state is not None) \
                else Batch()
            results.append((True, agent_index, out, act, each_state))
        holder = Batch.cat([{'act': act} for
                            (has_data, agent_index, out, act, each_state)
                            in results if has_data])
        state_dict, out_dict = {}, {}
        for policy, (has_data, agent_index, out, act, state) in \
                zip(self.policies, results):
            if has_data:
                holder.act[agent_index] = act
            state_dict["agent_" + str(policy.agent_id)] = state
            out_dict["agent_" + str(policy.agent_id)] = out
        holder["out"] = out_dict
        holder["state"] = state_dict
        return holder

    def learn(self, batch: Batch, **kwargs
              ) -> Dict[str, Union[float, List[float]]]:
        """:return: a dict with the following contents:

        ::

            {
                "agent_1/item1": item 1 of agent_1's policy.learn output
                "agent_1/item2": item 2 of agent_1's policy.learn output
                "agent_2/xxx": xxx
                ...
                "agent_n/xxx": xxx
            }
        """
        results = {}
        for policy in self.policies:
            data = batch[f'agent_{policy.agent_id}']
            if not data.is_empty():
                out = policy.learn(batch=data, **kwargs)
                for k, v in out.items():
                    results["agent_" + str(policy.agent_id) + '/' + k] = v
        return results
