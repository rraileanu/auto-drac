import gym
import torch
from gym.spaces.box import Box

from baselines.common.vec_env import VecEnvWrapper


class TransposeObs(gym.ObservationWrapper):
    def __init__(self, env=None):
        """
        Transpose observation space (base class)
        """
        super(TransposeObs, self).__init__(env)


class TransposeImageProcgen(TransposeObs):
    def __init__(self, env=None, op=[0, 3, 2, 1]):
        """
        Transpose observation space for images
        """
        super(TransposeImageProcgen, self).__init__(env)
        self.op = op
        obs_shape = self.observation_space.shape
        self.observation_space = Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0], [
                obs_shape[2], obs_shape[1], obs_shape[0]
            ],
            dtype=self.observation_space.dtype)

    def observation(self, ob):
        if ob.shape[0] == 1:
            ob = ob[0]
        return ob.transpose(self.op[0], self.op[1], self.op[2], self.op[3])


class VecPyTorchProcgen(VecEnvWrapper):
    def __init__(self, venv, device):
        """
        Environment wrapper that returns tensors (for obs and reward)
        """
        super(VecPyTorchProcgen, self).__init__(venv)
        self.device = device

        self.observation_space = Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0], 
            [3, 64, 64],
            dtype=self.observation_space.dtype)

    def reset(self):
        obs = self.venv.reset()
        if obs.shape[1] != 3:
            obs = obs.transpose(0, 3, 1, 2)
        obs = torch.from_numpy(obs).float().to(self.device) / 255.
        return obs

    def step_async(self, actions):
        if isinstance(actions, torch.LongTensor) or len(actions.shape) > 1:
            # Squeeze the dimension for discrete actions
            actions = actions.squeeze(1)
        actions = actions.cpu().numpy()
        self.venv.step_async(actions)

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        if obs.shape[1] != 3:
            obs = obs.transpose(0, 3, 1, 2)
        obs = torch.from_numpy(obs).float().to(self.device) / 255.
        reward = torch.from_numpy(reward).unsqueeze(dim=1).float()
        return obs, reward, done, info
        
