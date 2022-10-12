import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
import os.path as osp
from gym_extensions.continuous.mujoco.wall_envs import WallEnvFactory
from gym_extensions.continuous.mujoco.gravity_envs import GravityEnvFactory
from gym.envs.mujoco.walker2d import Walker2dEnv
from gym_extensions.continuous.mujoco.perturbed_bodypart_env import ModifiedSizeEnvFactory

import os
import gym

Walker2dWallEnv = lambda *args, **kwargs : WallEnvFactory(ModifiedWalker2dEnv)(model_path=os.path.dirname(gym.envs.mujoco.__file__) + "/assets/walker2d.xml", ori_ind=-1, *args, **kwargs)

Walker2dGravityEnv = lambda *args, **kwargs : GravityEnvFactory(ModifiedWalker2dEnv)(model_path=os.path.dirname(gym.envs.mujoco.__file__) + "/assets/walker2d.xml", *args, **kwargs)

Walker2dModifiedBodyPartSizeEnv = lambda *args, **kwargs : ModifiedSizeEnvFactory(ModifiedWalker2dEnv)(model_path=os.path.dirname(gym.envs.mujoco.__file__) + "/assets/walker2d.xml", *args, **kwargs)


class ModifiedWalker2dEnv(Walker2dEnv, utils.EzPickle):
    """
    Simply allows changing of XML file, probably not necessary if we pull request the xml name as a kwarg in openai gym
    """
    def __init__(self, **kwargs):
        mujoco_env.MujocoEnv.__init__(self, kwargs["model_path"], 4)
        utils.EzPickle.__init__(self)

    def step(self, a):
        posbefore = self.sim.data.qpos[0]
        obbefore = self._get_obs()
        self.do_simulation(a, self.frame_skip)
        posafter, height, ang = self.sim.data.qpos[0:3]
        #print("dt:", self.dt)
        #alive_bonus = 1.0
        ob = self._get_obs()
        alive_bonus = 1.0
        reward = 1*(posafter - posbefore) / self.dt

        reward -= 1e-3 * np.square(a).sum()
        done = not (height > 0.8 and height < 2.0 and
                    ang > -1.0 and ang < 1.0)
        reward += float(not done)
        if done:
            reward -= 0
        #done = False
        ob[-9:] = ob[-9:] / 5
        return ob, reward, done, {}

    def get_image(self, width=640, height=640, camera_name='track'):
        return self.sim.render(
            width=width,
            height=height,
            camera_name=camera_name,
        )

class Walker2dWithSensorEnv(Walker2dEnv, utils.EzPickle):
    """
    Adds empty sensor readouts, this is to be used when transfering to WallEnvs where we get sensor readouts with distances to the wall
    """

    def __init__(self, n_bins=10, **kwargs):
        self.n_bins = n_bins
        mujoco_env.MujocoEnv.__init__(self, kwargs["model_path"], 4)
        utils.EzPickle.__init__(self)


    def _get_obs(self):
        obs = np.concatenate([
            Walker2dEnv._get_obs(self),
            np.zeros(self.n_bins)
            # goal_readings
        ])
        return obs
