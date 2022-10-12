from collections import OrderedDict
import numpy as np
import copy
import time
import torch
import torch.optim as optim
from torch import nn as nn
from rlkit.core import logger
import rlkit.torch.pytorch_util as ptu
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.core.rl_algorithm import MetaRLAlgorithm
from rl_alg import BNNdynamics
from torch.optim import Adam

class BayesianLifelongRL(MetaRLAlgorithm):
    def __init__(
            self,
            env,
            nets,
            mw=False,
            policy_lr=1e-3,
            qf_lr=1e-3,
            vf_lr=1e-3,
            context_lr=1e-3,
            kl_lambda=1.,
            encoder_tau=0.005,
            policy_mean_reg_weight=1e-3,
            policy_std_reg_weight=1e-3,
            policy_pre_activation_weight=0.,
            optimizer_class=optim.Adam,
            recurrent=False,
            use_information_bottleneck=True,
            sparse_rewards=False,

            soft_target_tau=1e-2,
            plotter=None,
            render_eval_paths=False,
            **kwargs
    ):
        super().__init__(
            env=env,
            agent=nets[0],
            mw=mw,
            **kwargs
        )

        self.soft_target_tau = soft_target_tau
        self.policy_mean_reg_weight = policy_mean_reg_weight
        self.policy_std_reg_weight = policy_std_reg_weight
        self.policy_pre_activation_weight = policy_pre_activation_weight
        self.plotter = plotter
        self.render_eval_paths = render_eval_paths

        self.recurrent = recurrent
        self.qf_criterion = nn.MSELoss()
        self.vf_criterion = nn.MSELoss()
        self.vib_criterion = nn.MSELoss()
        self.l2_reg_criterion = nn.MSELoss()
        self.repre_criterion = nn.MSELoss()
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.kl_lambda = kl_lambda
        self.encoder_tau = encoder_tau
        self.use_information_bottleneck = use_information_bottleneck
        self.sparse_rewards = sparse_rewards

        self.forw_dyna_set = nets[1]


    ###### Torch stuff #####
    @property
    def networks(self):
        return self.agent.networks + [self.agent]
# agent.networks: [self.context_encoder, self.policy]
    def training_mode(self, mode):
        for net in self.networks:
            net.train(mode)

    def to(self, device=None):
        if device == None:
            device = ptu.device
        for net in self.networks:
            net.to(device)

    ##### Data handling #####
    def sample_data(self, indices, task=False):
        ''' sample data from replay buffers to construct a training meta-batch '''
        # collect data from multiple tasks for the meta-batch
        obs, actions, rewards, next_obs, terms = [], [], [], [], []
        for idx in indices:
            if task:
                batch = ptu.np_to_pytorch_batch(self.replay_buffer.random_batch(idx, batch_size=self.batch_size_task))
            else:
                batch = ptu.np_to_pytorch_batch(self.replay_buffer.random_batch(idx, batch_size=self.batch_size))
            o = batch['observations'][None, ...]
            a = batch['actions'][None, ...]
            r = batch['rewards'][None, ...]
            no = batch['next_observations'][None, ...]
            t = batch['terminals'][None, ...]
            obs.append(o)
            actions.append(a)
            rewards.append(r)
            next_obs.append(no)
            terms.append(t)
        obs = torch.cat(obs, dim=0)
        actions = torch.cat(actions, dim=0)
        rewards = torch.cat(rewards, dim=0)
        next_obs = torch.cat(next_obs, dim=0)
        terms = torch.cat(terms, dim=0)
        return [obs, actions, rewards, next_obs, terms]


    ##### Training #####
    def pretrain(self, env_id, backward=False):
        if backward:
            env_idx = env_id - len(self.agent.forw_dyna_set)
        else:
            env_idx = env_id

        self.agent.dyna._dynamics_model.set_params(self.agent.dyna._params_mu, self.agent.dyna._params_rho)
        self.agent.forw_dyna_set[env_idx]._params_mu.data.copy_(self.agent.dyna._params_mu.data)
        self.agent.forw_dyna_set[env_idx]._params_rho.data.copy_(self.agent.dyna._params_rho.data)
        self.agent.forw_dyna_set[env_idx]._dynamics_model.set_params(self.agent.forw_dyna_set[env_idx]._params_mu, self.agent.forw_dyna_set[env_idx]._params_rho)

    def _do_training(self, env_idx, backward=False, single_model=False, update_post=False, test=False):

        if self.new_task:
            self.agent.forw_dyna_set[env_idx].save_old_para()

        if self.task_step // self.global_update_interval != self.update_global or test:
            logger.push_prefix('Iteration #%d | ' % self.task_step)

            for j in range(self.num_updates_task):
                self._take_step_task(env_idx, single_model, backward, update_post)



            if not backward and not test:
                for i in range(self.num_updates_global):
                    self._take_step(env_idx, backward)
            else:
                self._take_step(env_idx, backward)

        else:
            for j in range(self.num_updates_task):
                self._take_step_task(env_idx, single_model, backward, update_post)

    def _min_q(self, obs, actions, task_z):
        q1 = self.qf1(obs, actions, task_z.detach())
        q2 = self.qf2(obs, actions, task_z.detach())
        min_q = torch.min(q1, q2)
        return min_q

    def _update_target_network(self):
        ptu.soft_update_from_to(self.vf, self.target_vf, self.soft_target_tau)

    def _take_step_task(self, env_idx, single_model=False, backward=False, update_post=False):

        if single_model or backward or self.replay_buffer.task_buffers[env_idx]._size < 200:
            self.eval_statistics = OrderedDict()
            self.eval_statistics['Pred_obs_loss_task'] = 0
            self.eval_statistics['Pred_rew_loss_task'] = 0
            self.eval_statistics['ELBO_task'] = 0
            self.eval_statistics['loglikelihood_task'] = 0
            self.eval_statistics['KL_loss_task'] = 0
            return
        if update_post:
            if env_idx > 10:
                update_post = True
            else:
                update_post = False
        obs, actions, rewards, next_obs, terms = self.sample_data([env_idx], task=True)
        t, b, nok = obs.size()

        obs = obs.view(t * b, -1)
        actions = actions.view(t * b, -1)
        next_obs = next_obs.view(t * b, -1)
        rewards_flat = rewards.view(t * b, -1)
        rewards_flat = rewards_flat * self.reward_scale
        next_obs = next_obs - obs
        elbo, pred, kl, obs_loss, r_loss = self.agent.forw_dyna_set[env_idx].update_posterior(obs, actions, next_obs, rewards_flat, update_post=update_post, weight_kl=0.0001)
        self.update_step += 1
        self.eval_statistics = OrderedDict()
        self.eval_statistics['ELBO_task'] = np.mean(elbo)
        self.eval_statistics['loglikelihood_task'] = np.mean(ptu.get_numpy(pred))
        self.eval_statistics['KL_loss_task'] = np.mean(ptu.get_numpy(kl))
        self.eval_statistics['Pred_obs_loss_task'] = np.mean(ptu.get_numpy(obs_loss))
        self.eval_statistics['Pred_rew_loss_task'] = np.mean(ptu.get_numpy(r_loss))
    def _take_step(self, env_id, backward=False):
        if backward:
            self.eval_statistics['Pred_obs_loss'] = 0
            self.eval_statistics['Pred_rew_loss'] = 0
            self.eval_statistics['ELBO'] = 0
            self.eval_statistics['loglikelihood'] = 0
            self.eval_statistics['KL_loss'] = 0
            return
        indices = np.random.choice(env_id + 1, np.min([env_id + 1, self.meta_batch]), replace=False)
        num_tasks = len(indices)
        obs, actions, rewards, next_obs, terms = self.sample_data(indices)
        t, b, _ = obs.size()
        obs = obs.view(t * b, -1)
        actions = actions.view(t * b, -1)
        next_obs = next_obs.view(t * b, -1)
        rewards_flat = rewards.view(t * b, -1)
        rewards_flat = rewards_flat * self.reward_scale
        next_obs = next_obs - obs
        elbo, pred, kl, obs_loss, r_loss = self.agent.dyna.update_posterior(obs, actions, next_obs, rewards_flat, update_post=False, weight_kl=0.0001)
        self.global_update_step += 1


        self.eval_statistics['Pred_obs_loss'] = np.mean(ptu.get_numpy(obs_loss))
        self.eval_statistics['Pred_rew_loss'] = np.mean(ptu.get_numpy(r_loss))
        self.eval_statistics['ELBO'] = np.mean(elbo)
        self.eval_statistics['loglikelihood'] = np.mean(ptu.get_numpy(pred))
        self.eval_statistics['KL_loss'] = np.mean(ptu.get_numpy(kl))
    def get_epoch_snapshot(self, epoch):
        # NOTE: overriding parent method which also optionally saves the env
        snapshot = OrderedDict(
            forwardpred=self.agent.dyna.state_dict(),
        )
        for i in range(len(self.agent.forw_dyna_set)):
            snapshot['fowardpred' + str(i)] = self.agent.forw_dyna_set[i].state_dict(),

        return snapshot
