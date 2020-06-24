import numpy as np
import torch

from ucb_rl2_meta import utils

from procgen import ProcgenEnv
from baselines.common.vec_env import (
    VecExtractDictObs,
    VecMonitor,
    VecNormalize
)
from ucb_rl2_meta.envs import VecPyTorchProcgen, TransposeImageProcgen


def evaluate(args, actor_critic, device, num_processes=1, aug_id=None):
    actor_critic.eval()
    
    # Sample Levels From the Full Distribution 
    venv = ProcgenEnv(num_envs=num_processes, env_name=args.env_name, \
        num_levels=0, start_level=0, \
        distribution_mode=args.distribution_mode)
    venv = VecExtractDictObs(venv, "rgb")
    venv = VecMonitor(venv=venv, filename=None, keep_buf=100)
    venv = VecNormalize(venv=venv, ob=False)
    eval_envs = VecPyTorchProcgen(venv, device)

    eval_episode_rewards = []

    obs = eval_envs.reset()
    eval_recurrent_hidden_states = torch.zeros(
        num_processes, actor_critic.recurrent_hidden_state_size, device=device)
    eval_masks = torch.ones(num_processes, 1, device=device)

    while len(eval_episode_rewards) < 10:
        with torch.no_grad():
            if aug_id:
                obs = aug_id(obs)
            _, action, _, eval_recurrent_hidden_states = actor_critic.act(
                obs,
                eval_recurrent_hidden_states,
                eval_masks,
                deterministic=False)

        obs, _, done, infos = eval_envs.step(action)
         
        eval_masks = torch.tensor(
            [[0.0] if done_ else [1.0] for done_ in done],
            dtype=torch.float32,
            device=device)

        for info in infos:
            if 'episode' in info.keys():
                eval_episode_rewards.append(info['episode']['r'])

    eval_envs.close()

    print("Last {} test episodes: mean/median reward {:.1f}/{:.1f}\n"\
        .format(len(eval_episode_rewards), \
        np.mean(eval_episode_rewards), np.median(eval_episode_rewards)))

    return eval_episode_rewards
