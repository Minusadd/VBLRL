import numpy as np

from rlkit.samplers.util import rollout
from rlkit.torch.sac.policies import MakeDeterministic


class InPlacePathSampler(object):

    def __init__(self, env, policy, max_path_length):
        self.env = env
        self.policy = policy

        self.max_path_length = max_path_length

    def start_worker(self):
        pass

    def shutdown_worker(self):
        pass

    def obtain_samples(self, env_idx, deterministic=False, max_samples=np.inf, max_trajs=1, accum_context=True,planning=True,backward=False,prt=False,singlemodel=False,resample=1):

        assert max_samples < np.inf or max_trajs < np.inf, "either max_samples or max_trajs must be finite"
        policy = MakeDeterministic(self.policy) if deterministic else self.policy
        paths = []
        n_steps_total = 0
        n_trajs = 0
        n_success_num = 0
        success = -1
        while n_steps_total < max_samples and n_trajs < max_trajs:
            path = rollout(
                self.env[env_idx], policy, env_idx, planning=planning, backward=backward, prt=prt,singlemodel=singlemodel,max_path_length=self.max_path_length, accum_context=accum_context)

            paths.append(path)
            n_steps_total += len(path['observations'])
            n_success_num += path['success']
            n_trajs += 1

        success=n_success_num/n_trajs
        return paths, n_steps_total,dict(n_success_num=n_success_num, n_trajs=n_trajs, success=success)

