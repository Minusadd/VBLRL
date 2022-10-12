from collections import deque
import numpy as np
import torch
from torch import nn
from torch.optim import Adam
import time
from network import BNN


class BNNdynamics(nn.Module):
    _ATTRIBUTES_TO_SAVE = [
        '_D_KL_smooth_length', '_prev_D_KL_medians',
        '_eta', '_lamb',
        '_dynamics_model',
        '_params_mu', '_params_rho',
        '_optim',
    ]
    def __init__(self, observation_size, action_size, device='cpu', reward_size=1, eta=0.1, lamb=0.01, update_iterations=500, learning_rate=0.0005,
            hidden_layers=2, hidden_layer_size=128, D_KL_smooth_length=10, max_logvar=2, min_logvar=-9., deterministic=False, weight_out=0.1):
        super().__init__()

        self._update_iterations = update_iterations
        self._eta = eta
        self._lamb = lamb
        self._device = device
        self._D_KL_smooth_length = D_KL_smooth_length
        self._prev_D_KL_medians = deque(maxlen=D_KL_smooth_length)
        self.lr = learning_rate
        self._dynamics_model = BNN(observation_size, action_size, reward_size, hidden_layers, hidden_layer_size, max_logvar, min_logvar, deterministic,weight_out, device)
        init_params_mu, init_params_rho = self._dynamics_model.get_parameters()
        self._params_mu = nn.Parameter(init_params_mu.to(device))
        self._params_rho = nn.Parameter(init_params_rho.to(device))
        init_params_mu_old, init_params_rho_old = self._dynamics_model.get_parameters_old()
        self._params_mu_old = nn.Parameter(init_params_mu_old.to(device))
        self._params_rho_old = nn.Parameter(init_params_rho_old.to(device))
        self._dynamics_model.set_params(self._params_mu, self._params_rho)
        self._optim = Adam([self._params_mu, self._params_rho], lr=learning_rate)

        self.init_var = torch.ones_like(self._params_rho).to(device) * 0.5**2





    # def memorize_episodic_info_gains(self, info_gains: np.array):
    #
    #     self._prev_D_KL_medians.append(np.median(info_gains))

    def infer(self, s, a, share_para=False):
        """
        Params
        ---
        batch_s: (batch, observation_size)
        batch_a: (batch, action_size)
        batch_s_next: (batch, observation_size)
        Return
        ---
        loss: (float)
        """
        #t1 = time.time()
        batch_s = s.float().to(self._device)
        batch_a = a.float().to(self._device)

        # self._dynamics_model.set_params(self._params_mu, self._params_rho)
        #td1 = time.time()
        r_mean, r_var, no_mean, no_var = self._dynamics_model.infer(torch.cat([batch_s, batch_a], dim=1), share_paremeters_among_samples=share_para)
        #td2 = time.time()
        #print("td1:", td2-td1)
        r_pred = r_mean + torch.randn_like(r_mean, device=self._device) * torch.sqrt(torch.exp(r_var))
        no_pred = no_mean + torch.randn_like(no_mean, device=self._device) * torch.sqrt(torch.exp(no_var))

        return r_pred, r_mean, r_var, no_pred, no_mean, no_var

    def save_old_para(self):
        self._dynamics_model.save_old_parameters()

    def _calc_div_kl_old(self):
        #TODO 0005 prior should be the previous posterior instead of the initial prior
        """Calculate D_{KL} [ q(\theta | \phi) || p(\theta) ]
        = \frac{1}{2} \sum^d_i [ \log(var^{init}_i) - \log(var_i) + \frac{var_i}{var^{init}_i} + \frac{(\mu_i - \mu^{init}_i)^2}{var^{init}_i} ] - \frac{d}{2}
        """
        var = (1 + self._params_rho.exp()).log().pow(2)
        var_old = (1 + self._params_rho_old.exp()).log().pow(2)
        return .5 * (var_old.log() - var.log() + var / var_old + (self._params_mu-self._params_mu_old).pow(2) / var_old ).sum() - .5 * len(self._params_mu)


    def _calc_div_kl(self):
        #TODO 0005 prior should be the previous posterior instead of the initial prior
        """Calculate D_{KL} [ q(\theta | \phi) || p(\theta) ]
        = \frac{1}{2} \sum^d_i [ \log(var^{init}_i) - \log(var_i) + \frac{var_i}{var^{init}_i} + \frac{(\mu_i - \mu^{init}_i)^2}{var^{init}_i} ] - \frac{d}{2}
        """
        var = (1 + self._params_rho.exp()).log().pow(2)
        #init_var = torch.ones_like(self._params_rho) * 0.5**2

        return .5 * ( self.init_var.log() - var.log() + var / self.init_var + (self._params_mu).pow(2) / self.init_var ).sum() - .5 * len(self._params_mu)

    def update_posterior(self, batch_s, batch_a, batch_s_next, batch_r, update_post=False, weight_kl=0.0001):
        """
        Params
        ---
        batch_s: (batch, observation_size)
        batch_a: (batch, action_size)
        batch_s_next: (batch, observation_size)
        Return
        ---
        loss: (float)
        """
        #tt1 = time.time()
        batch_s = batch_s.float().to(self._device)
        batch_a = batch_a.float().to(self._device)
        
        batch_s_next = batch_s_next.float().to(self._device)
        batch_r = batch_r.float().to(self._device)
        #self._dynamics_model.set_params(self._params_mu, self._params_rho)
        
        log_likelihood, obs_loss, r_loss = self._dynamics_model.log_likelihood(torch.cat([batch_s, batch_a], dim=1), batch_r, batch_s_next)
        #tt2 = time.time()
        #print("tt1:", tt2-tt1)
        if update_post:
            div_kl = self._calc_div_kl_old()
        else:
            div_kl = self._calc_div_kl()

        elbo = log_likelihood - weight_kl * div_kl
        #assert not torch.isnan(elbo).any() and not torch.isinf(elbo).any(), elbo.item()
        #tt22 = time.time()
        #print("tt22:", tt22-tt2)
        self._optim.zero_grad()
        (-elbo).backward()
        self._optim.step()

        # Check parameters
        # assert not torch.isnan(self._params_mu).any() and not torch.isinf(self._params_mu).any(), self._params_mu
        # assert not torch.isnan(self._params_rho).any() and not torch.isinf(self._params_rho).any(), self._params_rho

        #tt3 = time.time()
        #print("tt33:", tt3-tt22)
        # update self._params
        self._dynamics_model.set_params(self._params_mu, self._params_rho)
        #tt4 = time.time()
        #print("tt3:",tt4-tt3)
        return elbo.item(), log_likelihood, div_kl, obs_loss, r_loss

    def state_dict(self):
        return {
            k: getattr(self, k).state_dict() if hasattr(getattr(self, k), 'state_dict') else getattr(self, k)
            for k in self._ATTRIBUTES_TO_SAVE
        }

    def load_state_dict(self, state_dict):
        for k, v in state_dict.items():
            if hasattr(getattr(self, k), 'load_state_dict'):
                getattr(self, k).load_state_dict(v)
            else:
                setattr(self, k, v)
