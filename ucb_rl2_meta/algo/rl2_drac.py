import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random 
import numpy as np
import sys 
from collections import deque

class RL2DrAC():
    """
    RL^2 Data-regularized Actor-Critic (RL2-DrAC) object
    """
    def __init__(self,
                 actor_critic,
                 rl2_learner,
                 clip_param,
                 ppo_epoch,
                 num_mini_batch,
                 value_loss_coef,
                 entropy_coef,
                 rl2_entropy_coef,
                 lr=None,
                 rl2_lr=None,
                 eps=None,
                 rl2_eps=None,
                 max_grad_norm=None,
                 aug_list=None,
                 aug_id=None,
                 aug_coef=0.1,
                 num_aug_types=8, 
                 recurrent_hidden_size=32, 
                 num_actions=15, 
                 device='cuda'): 
                 
        self.actor_critic = actor_critic
        self.optimizer = optim.Adam(actor_critic.parameters(), lr=lr, eps=eps)

        self.rl2_learner = rl2_learner
        self.rl2_optimizer = optim.Adam(rl2_learner.parameters(), lr=rl2_lr, eps=rl2_eps)

        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.rl2_entropy_coef = rl2_entropy_coef

        self.max_grad_norm = max_grad_norm

        self.aug_list = aug_list
        self.aug_id = aug_id
        self.aug_coef = aug_coef

        self.num_aug_types = num_aug_types
        self.num_action_selected = [0.] * self.num_aug_types 

        self.device = device
        self.num_actions = num_actions

        self.rl2_masks =  torch.ones(1, 1).to(device)
        self.rl2_recurrent_hidden_states = torch.zeros(1, recurrent_hidden_size).to(device)
        self.rl2_obs = torch.zeros((1, num_actions + 1)).to(device)

        self.step = 0
        
    def convert_to_onehot(self, action_value):
        self.action_onehot = torch.zeros(1, self.num_actions).to(self.device)
        self.action_onehot[0][action_value] = 1
        return self.action_onehot 

    def update(self, rollouts):
        if self.step > 0 :
            # Update RL^2 Learner 
            rl2_advantages = rollouts.returns.mean().reshape(1, 1) - self.rl2_value
            rl2_value_loss = rl2_advantages.pow(2).mean()
            rl2_action_loss = -(rl2_advantages.detach() * self.rl2_action_log_prob.mean())
            rl2_loss = rl2_value_loss * self.value_loss_coef + rl2_action_loss - \
                self.rl2_dist_entropy * self.rl2_entropy_coef

            self.rl2_optimizer.zero_grad()
            rl2_loss.backward()
            nn.utils.clip_grad_norm_(self.rl2_learner.parameters(), self.max_grad_norm)
            self.rl2_optimizer.step()

        # Select Augmentation Type using RL^2 Leaner
        self.rl2_value, self.rl2_action, self.rl2_action_log_prob, rl2_recurrent_hidden_states \
            = self.rl2_learner.act(self.rl2_obs, self.rl2_recurrent_hidden_states, self.rl2_masks)
        # Get Entropy of RL^2 Learner for its loss
        _, _, self.rl2_dist_entropy, _ = self.rl2_learner.evaluate_actions(
            self.rl2_obs, self.rl2_recurrent_hidden_states, self.rl2_masks, self.rl2_action
        )
        
        # Update the Next Inputs of the RL^2 Learner (reward, action, recurrent state)
        prev_reward = rollouts.returns.mean().reshape(1, 1)
        rl2_action_onehot = self.convert_to_onehot(self.rl2_action.item())
        self.rl2_obs = torch.cat((prev_reward, rl2_action_onehot), dim=1)
        self.rl2_recurrent_hidden_states = rl2_recurrent_hidden_states.detach()

        # Get the Augmentation Type
        self.current_aug_func = self.aug_list[self.rl2_action]
        self.num_action_selected[self.rl2_action.item()] += 1

        # DrAC Update
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-5)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0

        for e in range(self.ppo_epoch):
            data_generator = rollouts.recurrent_generator(
                advantages, self.num_mini_batch)
            
            for sample in data_generator:
                obs_batch, recurrent_hidden_states_batch, actions_batch, \
                value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, \
                        adv_targ = sample

                values, action_log_probs, dist_entropy, _ = self.actor_critic.evaluate_actions(
                    obs_batch, recurrent_hidden_states_batch, masks_batch,
                    actions_batch)
                    
                ratio = torch.exp(action_log_probs -
                                  old_action_log_probs_batch)
                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                                    1.0 + self.clip_param) * adv_targ
                action_loss = -torch.min(surr1, surr2).mean()

                value_pred_clipped = value_preds_batch + \
                    (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
                value_losses = (values - return_batch).pow(2)
                value_losses_clipped = (
                    value_pred_clipped - return_batch).pow(2)
                value_loss = 0.5 * torch.max(value_losses,
                                                value_losses_clipped).mean()
                
                obs_batch_aug = self.current_aug_func.do_augmentation(obs_batch)
                obs_batch_id = self.aug_id(obs_batch)
                
                _, new_actions_batch, _, _ = self.actor_critic.act(\
                    obs_batch_id, recurrent_hidden_states_batch, masks_batch)
                values_aug, action_log_probs_aug, dist_entropy_aug, _ = \
                    self.actor_critic.evaluate_actions(obs_batch_aug, \
                    recurrent_hidden_states_batch, masks_batch, new_actions_batch)
                # Compute Augmented Loss
                action_loss_aug = - action_log_probs_aug.mean()
                value_loss_aug = .5 * (torch.detach(values) - values_aug).pow(2).mean()

                # Update actor-critic using both PPO and Augmented Loss
                self.optimizer.zero_grad()
                aug_loss = value_loss_aug + action_loss_aug
                (value_loss * self.value_loss_coef + action_loss -
                    dist_entropy * self.entropy_coef + 
                    aug_loss * self.aug_coef).backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                        self.max_grad_norm)
                self.optimizer.step()  

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()

                self.current_aug_func.change_randomization_params_all()

        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates
        self.step += 1
     
        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch