"""
Launcher for experiments with PEARL

"""
import os
import pathlib
import numpy as np
import click
import json
import torch

from rlkit.envs import ENVS
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.torch.sac.policies import TanhGaussianPolicy
from rlkit.torch.networks import FlattenMlp, MlpEncoder, RecurrentEncoder
from rlkit.torch.sac.sac import BayesianLifelongRL
from rlkit.torch.sac.agent import Agent
from rlkit.launchers.launcher_util import setup_logger
import rlkit.torch.pytorch_util as ptu
from configs.default import default_config
from rl_alg import BNNdynamics
from gym_env import GymEnv
import pickle
import gym

def experiment(variant):



    SEED = variant['algo_params']['seed']
    job_name_mtl = 'results/walker_mtl_gravity_exp'
    torch.set_num_threads(5)
    ptu.set_gpu_mode(variant['util_params']['use_gpu'], variant['util_params']['gpu_id'])

    num_tasks = 20


    f = open(job_name_mtl + '/env_factors.pickle', 'rb')
    gravity_factors = pickle.load(f)

    f.close()
    f = open(job_name_mtl + '/env_ids.pickle', 'rb')
    env_ids = pickle.load(f)
    f.close()
    e_unshuffled = {}
    for task_id in range(num_tasks):
        gravity_factor = gravity_factors[task_id]
        env_id = env_ids[task_id]
        gym.envs.register(
            id=env_id,
            entry_point='gym_extensions.continuous.mujoco.modified_walker2d:Walker2dGravityEnv',
            max_episode_steps=variant['algo_params']['max_path_length'],
            reward_threshold=3800.0,
            kwargs=dict(gravity=-gravity_factor* 9.81)
        )
        e_unshuffled[task_id] = GymEnv(env_id)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    env = e_unshuffled
    obs_dim = env[0].spec.observation_dim
    action_dim = env[0].spec.action_dim
    print("a:",action_dim)

    #TODO 0002 Net_size and hyperparameters options
    dyna = BNNdynamics(obs_dim, action_dim, device=ptu.device, learning_rate=0.0006, weight_out=0.1)
    forward_dyna_set = []
    num_nets = int(num_tasks)
    for i in range(num_nets):

        forward_dyna = BNNdynamics(obs_dim, action_dim, device=ptu.device, learning_rate=0.0006, deterministic=variant['deterministic'],weight_out=0.1)
        forward_dyna_set.append(forward_dyna)
    agent = Agent(
        forward_dyna=forward_dyna_set,
        dyna=dyna,
        action_dim=action_dim,
        **variant['algo_params']
    )

    algorithm = BayesianLifelongRL(
        env=env,
        nets=[agent, forward_dyna_set],
        **variant['algo_params']
    )



    if ptu.gpu_enabled():
        algorithm.to()

    # debugging triggers a lot of printing and logs to a debug directory
    DEBUG = variant['util_params']['debug']
    os.environ['DEBUG'] = str(int(DEBUG))

    # create logging directory
    # TODO support Docker
    exp_id = 'debug' if DEBUG else None
    experiment_log_dir = setup_logger(variant['env_name'], variant=variant, exp_id=exp_id, base_log_dir=variant['util_params']['base_log_dir'])

    # optionally save eval trajectories as pkl files
    if variant['algo_params']['dump_eval_paths']:
        pickle_dir = experiment_log_dir + '/eval_trajectories'
        pathlib.Path(pickle_dir).mkdir(parents=True, exist_ok=True)

    # run the algorithm
    algorithm.train()

def deep_update_dict(fr, to):
    ''' update dict of dicts with new values '''
    # assume dicts have same keys
    for k, v in fr.items():
        if type(v) is dict:
            deep_update_dict(v, to[k])
        else:
            to[k] = v
    return to

@click.command()
@click.argument('config', default=None)
@click.option('--gpu', default=0)
@click.option('--docker', is_flag=True, default=False)
@click.option('--debug', is_flag=True, default=False)
def main(config, gpu, docker, debug):

    variant = default_config
    if config:
        with open(os.path.join(config)) as f:
            exp_params = json.load(f)
        variant = deep_update_dict(exp_params, variant)
    variant['util_params']['gpu_id'] = gpu

    experiment(variant)

if __name__ == "__main__":
    main()

