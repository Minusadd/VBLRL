"""
Launcher for experiments with PEARL

"""
import os
import pathlib
import numpy as np
import click
import json
import torch
import time
#from rlkit.envs import ENVS
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
    torch.set_num_threads(SEED)
    ptu.set_gpu_mode(variant['util_params']['use_gpu'], variant['util_params']['gpu_id'])

    num_tasks = 10


    e_unshuffled = []

    import metaworld
    import random

    ml50 = metaworld.MT50()  # Construct the benchmark, sampling tasks

    testing_envs = []
    for name, env_cls in ml50.train_classes.items():
        env = env_cls()
        task = [task for task in ml50.train_tasks if task.env_name == name][0]
        env.set_task(task)
        e_unshuffled.append(env)



    np.random.seed(SEED)
    torch.manual_seed(SEED)
    env = e_unshuffled
    #breakpoint()
    obs_dim = int(np.prod(env[0].observation_space.shape))#env[0].spec.observation_dim
    action_dim = int(np.prod(env[0].action_space.shape)) #env[0].spec.action_dim
    print("a:",action_dim)

    #TODO 0002 Net_size and hyperparameters options


    dyna = BNNdynamics(obs_dim, action_dim, device = ptu.device,learning_rate=0.001,weight_out=0.1)
    forward_dyna_set = []
    for i in range(len(env)):
        forward_dyna = BNNdynamics(obs_dim, action_dim, device=ptu.device, learning_rate=0.0005, deterministic=variant['deterministic'],weight_out=0.1)
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
        mw=True,
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

