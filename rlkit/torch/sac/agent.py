import numpy as np
import time
import torch
from torch import nn as nn
import torch.nn.functional as F
from rlkit.torch.core import np_ify
import rlkit.torch.pytorch_util as ptu
from optimizers import CEMOptimizer

def detect_done(ob, env_name, check_done):
    if env_name == 'gym_fhopper':
        height, ang = ob[:, 0], ob[:, 1]
        done = np.logical_or(height <= 0.7, abs(ang) >= 0.2)

    elif env_name == 'gym_fwalker2d':
        height, ang = ob[:, 0], ob[:, 1]
        done = np.logical_or(
            height > 2.0,
            np.logical_or(height < 0.8, np.logical_or(abs(ang) > 1.0, abs(ang) < -1.0))
        )

    elif env_name == 'gym_fant':
        height = ob[:, 0]
        done = np.logical_or(height > 1.0, height < 0.2)

    elif env_name in ['gym_fant2', 'gym_fant5', 'gym_fant10',
                      'gym_fant20', 'gym_fant30']:
        height = ob[:, 0]
        done = np.logical_or(height > 1.0, height < 0.2)
    else:
        done = np.zeros([ob.shape[0]])

    if not check_done:
        done[:] = False

    return done

def _product_of_gaussians(mus, sigmas_squared):
    '''
    compute mu, sigma of product of gaussians
    '''
    sigmas_squared = torch.clamp(sigmas_squared, min=1e-7)
    sigma_squared = 1. / torch.sum(torch.reciprocal(sigmas_squared), dim=0)
    mu = sigma_squared * torch.sum(mus / sigmas_squared, dim=0)
    return mu, sigma_squared


def _mean_of_gaussians(mus, sigmas_squared):
    '''
    compute mu, sigma of mean of gaussians
    '''
    mu = torch.mean(mus, dim=0)
    sigma_squared = torch.mean(sigmas_squared, dim=0)
    return mu, sigma_squared


def _natural_to_canonical(n1, n2):
    ''' convert from natural to canonical gaussian parameters '''
    mu = -0.5 * n1 / n2
    sigma_squared = -0.5 * 1 / n2
    return mu, sigma_squared


def _canonical_to_natural(mu, sigma_squared):
    ''' convert from canonical to natural gaussian parameters '''
    n1 = mu / sigma_squared
    n2 = -0.5 * 1 / sigma_squared
    return n1, n2


class Agent(nn.Module):#context encoder -> action output (during training and sampling)

    def __init__(self,
                 forward_dyna,
                 dyna,
                 action_dim,
                 per=1,
                 plan_hor=20,
                 npart=50,
                 popsize=500,
                 env_name=None,
                 **kwargs
    ):
        super().__init__()
        self.forw_dyna_set = forward_dyna
        self.dyna = dyna
        self.dU = action_dim
        self.per = per
        self.plan_hor = plan_hor
        print("planhor:", plan_hor)
        self.npart = npart
        print("popsize:", popsize)
        self.prt = True
        self.env_name = env_name
        if env_name is None:
            self.check_done=False
        else:
            self.check_done=True
        self.optimizer = CEMOptimizer(
            sol_dim=self.plan_hor * self.dU,
            lower_bound=np.tile(-np.ones(self.dU), [self.plan_hor]),
            upper_bound=np.tile(np.ones(self.dU), [self.plan_hor]),
            cost_function=self._compile_cost,
            popsize=popsize,
            num_elites=50,
            max_iters=5,
            alpha=0.1
        )

       


        self.ac_buf = np.array([]).reshape(0, self.dU)
        self.prev_sol = np.tile(np.zeros(self.dU), [self.plan_hor])
        self.init_var = np.tile(np.ones(self.dU) * 1.5, [self.plan_hor])


    def get_action(self, obs, env_idx, get_pred_cost=False, planning=True, backward=False, prt=False, singlemodel=False):


        self.current_id = env_idx
        if not planning:
            return np.random.uniform(-1, 1, self.dU)

        if self.ac_buf.shape[0] > 0:

            action, self.ac_buf = self.ac_buf[0], self.ac_buf[1:]

            return action

        self.sy_cur_obs = obs

        soln = self.optimizer.obtain_solution(self.prev_sol, self.init_var, backward, prt, singlemodel)

        self.prev_sol = np.concatenate([np.copy(soln)[self.per * self.dU:], np.zeros(self.per * self.dU)])
        self.ac_buf = soln[:self.per * self.dU].reshape(-1, self.dU)

        return self.get_action(obs, env_idx)

    @torch.no_grad()
    def _compile_cost(self, ac_seqs, backward=False, prt=False, singlemodel=False, know_rew=False):

        nopt = ac_seqs.shape[0]

        ac_seqs = torch.from_numpy(ac_seqs).float().to(ptu.device)

        ac_seqs = ac_seqs.view(-1, self.plan_hor, self.dU)


        transposed = ac_seqs.transpose(0, 1)


        expanded = transposed[:, :, None]


        tiled = expanded.expand(-1, -1, self.npart, -1)


        ac_seqs = tiled.contiguous().view(self.plan_hor, -1, self.dU)


        # Expand current observation
        cur_obs = torch.from_numpy(self.sy_cur_obs).float().to(ptu.device)
        cur_obs = cur_obs[None]

        cur_obs = cur_obs.expand(nopt * self.npart, -1)

        rews = torch.zeros(nopt, self.npart, device=ptu.device)
        done = np.zeros([nopt * self.npart])
        for t in range(self.plan_hor):
            cur_acs = ac_seqs[t]
            proc_obs = cur_obs
            acs = cur_acs

            if singlemodel:
                rew, rew_mean, rew_var, next_obs, no_mean, no_var = self.dyna.infer(proc_obs, acs)
            else:
                rew, rew_mean, rew_var, next_obs, no_mean, no_var = self.forw_dyna_set[self.current_id].infer(proc_obs, acs)



            rew = rew.cpu().numpy()*(1-np.expand_dims(done,1))
            
            rew = torch.from_numpy(rew).float().to(ptu.device)

            rew = rew.view(-1, self.npart)
            if backward and not singlemodel:
                rew1, rew_mean1, rew_var1, next_obs1, no_mean1, no_var1 = self.dyna.infer(proc_obs, acs)
                rew1 = rew1.view(-1, self.npart)
                #mean
                rew_mean = rew_mean.view(-1, self.npart)
                no_mean = no_mean.view(nopt, self.npart, -1)
                no_mean = no_mean.transpose(1, 2)
                no_mean = no_mean.contiguous().view(-1, self.npart)
                #var
                rew_var = rew_var.view(-1, self.npart)
                no_var = no_var.view(nopt, self.npart, -1)
                no_var = no_var.transpose(1, 2)
                no_var = no_var.contiguous().view(-1, self.npart)
                #mean
                var_r_mean = torch.var(rew_mean, dim=1)
                var_no_mean = torch.var(no_mean, dim=1)
                #var
                var_r_var = torch.var(rew_var, dim=1)
                var_no_var = torch.var(no_var, dim=1)
                #mean
                rew_mean1 = rew_mean1.view(-1, self.npart)
                no_mean1 = no_mean1.view(nopt, self.npart, -1)
                no_mean1 = no_mean1.transpose(1, 2)
                no_mean1 = no_mean1.contiguous().view(-1, self.npart)
                #var
                rew_var1 = rew_var1.view(-1, self.npart)
                no_var1 = no_var1.view(nopt, self.npart, -1)
                no_var1 = no_var1.transpose(1, 2)
                no_var1 = no_var1.contiguous().view(-1, self.npart)
                #mean
                var_r_mean1 = torch.var(rew_mean1, dim=1)
                var_no_mean1 = torch.var(no_mean1, dim=1)
                #var
                var_r_var1 = torch.var(rew_var1, dim=1)
                var_no_var1 = torch.var(no_var1, dim=1)
                #mean&var

                rew[var_r_mean>var_r_mean1] = rew1[var_r_mean>var_r_mean1]
                rew[var_r_var > var_r_var1] = rew1[var_r_var > var_r_var1]
                    #rew = rew1
                next_obs = next_obs.view(nopt, self.npart, -1)
                next_obs = next_obs.transpose(1, 2)
                next_obs = next_obs.contiguous().view(-1, self.npart)
                next_obs1 = next_obs1.view(nopt, self.npart, -1)
                next_obs1 = next_obs1.transpose(1, 2)
                next_obs1 = next_obs1.contiguous().view(-1, self.npart)
                next_obs[var_no_mean > var_no_mean1] = next_obs1[var_no_mean > var_no_mean1]
                next_obs[var_no_var > var_no_var1] = next_obs1[var_no_var > var_no_var1]
                next_obs = next_obs1
                next_obs = next_obs.view(nopt, -1, self.npart)
                next_obs = next_obs.transpose(1, 2)
                next_obs = next_obs.contiguous().view(nopt*self.npart, -1)

            next_obs = proc_obs + next_obs

            this_done = detect_done(next_obs.cpu().numpy(), env_name=self.env_name, check_done=self.check_done)

            done = np.logical_or(this_done, done)




            rews += rew
            cur_obs = next_obs

        # Replace nan with high cost
        rews[rews != rews] = -1e6


        return rews.mean(dim=1).detach().cpu().numpy()

    def _expand_to_ts_format(self, mat):
        dim = mat.shape[-1]

        # Before, [10, 5] in case of proc_obs
        reshaped = mat.view(-1, self.npart, 1, dim)
        # After, [2, 5, 1, 5]

        transposed = reshaped.transpose(0, 1)
        # After, [5, 2, 1, 5]

        reshaped = transposed.contiguous().view(-1, dim)
        # After. [5, 2, 5]

        return reshaped

    def _flatten_to_matrix(self, ts_fmt_arr):
        dim = ts_fmt_arr.shape[-1]

        reshaped = ts_fmt_arr.view(self.npart, -1, 1, dim)

        transposed = reshaped.transpose(0, 1)

        reshaped = transposed.contiguous().view(-1, dim)

        return reshaped



    def forward(self, obs):

        return obs

    def log_diagnostics(self, eval_statistics):
        '''
        adds logging data about encodings to eval_statistics
        '''
        pass


    @property
    def networks(self):
        return self.forw_dyna_set + [self.dyna]
