import argparse

import torch

parser = argparse.ArgumentParser(description='RL')

# PPO Arguments. 
parser.add_argument(
    '--lr', 
    type=float, 
    default=5e-4, 
    help='learning rate')
parser.add_argument(
    '--eps',
    type=float,
    default=1e-5,
    help='RMSprop optimizer epsilon')
parser.add_argument(
    '--alpha',
    type=float,
    default=0.99,
    help='RMSprop optimizer apha')
parser.add_argument(
    '--gamma',
    type=float,
    default=0.999,
    help='discount factor for rewards')
parser.add_argument(
    '--gae_lambda',
    type=float,
    default=0.95,
    help='gae lambda parameter')
parser.add_argument(
    '--entropy_coef',
    type=float,
    default=0.01,
    help='entropy term coefficient')
parser.add_argument(
    '--value_loss_coef',
    type=float,
    default=0.5,
    help='value loss coefficient (default: 0.5)')
parser.add_argument(
    '--max_grad_norm',
    type=float,
    default=0.5,
    help='max norm of gradients)')
parser.add_argument(
    '--seed', 
    type=int, 
    default=1, 
    help='random seed')
parser.add_argument(
    '--num_processes',
    type=int,
    default=64,
    help='how many training CPU processes to use')
parser.add_argument(
    '--num_steps',
    type=int,
    default=256,
    help='number of forward steps in A2C')
parser.add_argument(
    '--ppo_epoch',
    type=int,
    default=3,
    help='number of ppo epochs')
parser.add_argument(
    '--num_mini_batch',
    type=int,
    default=8,
    help='number of batches for ppo')
parser.add_argument(
    '--clip_param',
    type=float,
    default=0.2,
    help='ppo clip parameter')
parser.add_argument(
    '--log_interval',
    type=int,
    default=10,
    help='log interval, one log per n updates')
parser.add_argument(
    '--save_interval',
    type=int,
    default=100,
    help='save interval, one save per n update')
parser.add_argument(
    '--num_env_steps',
    type=int,
    default=25e6,
    help='number of environment steps to train')
parser.add_argument(
    '--env_name',
    type=str,
    default='bigfish',
    help='environment to train on')
parser.add_argument(
    '--run_name',
    default='drac',
    help='name for the run - prefix to log files')
parser.add_argument(
    '--log_dir',
    default='/tmp/drac/',
    help='directory to save agent logs')
parser.add_argument(
    '--no_cuda',
    action='store_true',
    default=False,
    help='disables CUDA training')
parser.add_argument(
    '--hidden_size',
    type=int,
    default=256,
    help='state embedding dimension')

# Procgen Arguments.
parser.add_argument(
    '--distribution_mode',
    default='easy',
    help='distribution of envs for procgen')
parser.add_argument(
    '--num_levels',
    type=int,
    default=200,
    help='number of Procgen levels to use for training')
parser.add_argument(
    '--start_level',
    type=int,
    default=0,
    help='start level id for sampling Procgen levels')

# DrAC Arguments.
parser.add_argument(
    '--aug_type',
    type=str,
    default='crop',
    help='augmentation type')
parser.add_argument(
    '--aug_coef', 
    type=float, 
    default=0.1, 
    help='coefficient on the augmented loss')
parser.add_argument(
    '--aug_extra_shape', 
    type=int, 
    default=0, 
    help='increase image size by')
parser.add_argument(
    '--image_pad', 
    type=int, 
    default=12, 
    help='increase image size by')

# UCB-DrAC Arguments.
parser.add_argument(
    '--use_ucb',
    action='store_true',
    default=False,
    help='use UCB to select an augmentation')
parser.add_argument(
    '--ucb_window_length', 
    type=int, 
    default=10, 
    help='length of sliding window for UCB (i.e. number of UCB actions)')
parser.add_argument(
    '--ucb_exploration_coef', 
    type=float, 
    default=0.1, 
    help='exploration coefficient for UCB')

# RL^2 Arguments.
parser.add_argument(
    '--use_rl2',
    action='store_true',
    default=False,
    help='use RL^2 for selecting an augmentation')
parser.add_argument(
    '--rl2_hidden_size',
    type=int,
    default=32,
    help='hidden size in RL^2 policy for selecting an augmentation')
parser.add_argument(
    '--rl2_lr', 
    type=float, 
    default=5e-4, 
    help='learning rate for RL^2')
parser.add_argument(
    '--rl2_eps',
    type=float,
    default=1e-5,
    help='RMSprop optimizer epsilon for RL^2')
parser.add_argument(
    '--rl2_entropy_coef',
    type=float,
    default=0.001,
    help='entropy term coefficient for RL^2')

# Meta-DrAC Arguments. 
parser.add_argument(
    '--use_meta_learning',
    action='store_true',
    default=False,
    help='use meta learning for finding the best augmentation')
parser.add_argument(
    '--meta_batch_size', 
    type=int,
    default=64, 
    help='batch size for the meta-learner.')
parser.add_argument(
    '--meta_num_train_steps', 
    type=int,
    default=1, 
    help='number of training steps in one meta-update')
parser.add_argument(
    '--meta_num_test_steps', 
    type=int, 
    default=1, 
    help='number of test steps in one meta-update')
parser.add_argument(
    '--meta_grad_clip', 
    type=float, 
    default=100, 
    help='clip value used for the meta-gradients')
parser.add_argument(
    '--split_ratio', 
    type=float,
    default=0.10, 
    help='fraction of the PPO buffer used for test by the meta-learner.')
