import io
import os.path
import pathlib
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Type, Union, Iterable

import gym
import numpy as np
import torch as th
from stable_baselines3 import SAC
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.save_util import load_from_zip_file, recursive_setattr
from stable_baselines3.common.type_aliases import GymEnv, Schedule
from stable_baselines3.common.utils import polyak_update, check_for_correct_spaces
from stable_baselines3.sac.policies import SACPolicy
from torch.nn import functional as F

from mfnlc.config import default_device
from mfnlc.learn.utils import list_dict_to_dict_list
from mfnlc.learn.subgoal import LaplacePolicy, GaussianPolicy, EnsembleCritic


class SafetyRis(SAC):
    def __init__(
        self,
        actor: GaussianPolicy,
        critic: EnsembleCritic,
        subgoal_net: LaplacePolicy,
        state_dim: int,
        action_dim: int,
        policy: Union[str, Type[SACPolicy]],
        env: Union[GymEnv, str],
        h_lr: float = 1e-4, 
        q_lr: float = 1e-3,
        pi_lr: float = 1e-4, 
        epsilon: float = 1e-16,
        learning_rate: Union[float, Schedule] = 3e-4,
        buffer_size: int = 1_000_000,  # 1e6
        learning_starts: int = 100,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = 1,
        gradient_steps: int = 1,
        action_noise: Optional[ActionNoise] = None,
        replay_buffer_class: Optional[ReplayBuffer] = None,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        ent_coef: Union[str, float] = "auto",
        target_update_interval: int = 1,
        target_entropy: Union[str, float] = "auto",
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        use_sde_at_warmup: bool = False,
        alpha: float = 0.1,
        Lambda: float = 0.1, 
        n_ensemble: int = 10, 
        clip_v_function: float = -150,
        tensorboard_log: Optional[str] = None,
        create_eval_env: bool = False,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ):

        super(SafetyRis, self).__init__(
            policy,
            env,
            learning_rate,
            buffer_size,
            learning_starts,
            batch_size,
            tau,
            gamma,
            train_freq,
            gradient_steps,
            action_noise,
            replay_buffer_class,
            replay_buffer_kwargs,
            optimize_memory_usage,
            ent_coef,
            target_update_interval,
            target_entropy,
            use_sde,
            sde_sample_freq,
            use_sde_at_warmup,
            tensorboard_log,
            create_eval_env,
            policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=False,
        )


        self.state_dim = state_dim
        self.action_dim = action_dim

        # subgoal
        self.subgoal_net = subgoal_net
        self.subgoal_optimizer = th.optim.Adam(self.subgoal_net.parameters(), lr=h_lr)
        self.alpha = alpha
        self.Lambda = Lambda
        self.n_ensemble = n_ensemble
        self.clip_v_function = clip_v_function
        self.epsilon = epsilon
        # actor
        self.new_actor = actor
        self.actor_optimizer = th.optim.Adam(self.new_actor.parameters(), lr=pi_lr)
        # critic
        self.new_critic = critic
        self.critic_optimizer = th.optim.Adam(self.new_critic.parameters(), lr=q_lr)

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super(SafetyRis, self)._setup_model()
        self._setup_alias()

    def _setup_alias(self) -> None:
        del self.actor
        del self.critic
        del self.critic_target
        del self.policy.actor
        del self.policy.critic
        del self.policy.critic_target
        self.actor = self.new_actor
        self.actor_target = deepcopy(self.actor)
        self.critic = self.new_critic
        self.critic_target = deepcopy(self.critic)
        self.policy.actor = self.actor
        self.policy.critic = self.critic
    
    # RIS requires training each time when env.step()
    def _on_step(self):
        if self.num_timesteps > 0 and self.num_timesteps > self.learning_starts:
            gradient_steps = self.gradient_steps
            # Special case when the user passes `gradient_steps=0`
            if gradient_steps > 0:
                self.train(batch_size=self.batch_size, gradient_steps=gradient_steps)
        
    def train_highlevel_policy(self, state, goal, subgoal, subgoal_net_losses=[], advs=[]):
		# Compute subgoal distribution 
        batch_size = state.shape[0] # 2048
        subgoal_distribution = self.subgoal_net(state, goal)
        with th.no_grad():
            # Compute target value
            new_subgoal = subgoal_distribution.loc # 2048 x 20
            policy_v_1 = self.value(state, new_subgoal) # 2048 x 1
            policy_v_2 = self.value(new_subgoal, goal) # 2048 x 1
            policy_v = th.cat([policy_v_1, policy_v_2], -1).clamp(min=self.clip_v_function, max=0.0).abs().max(-1)[0]

            # Compute subgoal distance loss
            v_1 = self.value(state, subgoal)
            v_2 = self.value(subgoal, goal)
            v = th.cat([v_1, v_2], -1).clamp(min=self.clip_v_function, max=0.0).abs().max(-1)[0]
            adv = - (v - policy_v)
            weight = F.softmax(adv/self.Lambda, dim=0)

        log_prob = subgoal_distribution.log_prob(subgoal).sum(-1)
        subgoal_loss = - (log_prob * weight).mean()
        subgoal_net_losses.append(subgoal_loss.item())
        advs.append(adv.mean().item())

        # Update network
        self.subgoal_optimizer.zero_grad()
        subgoal_loss.backward()
        self.subgoal_optimizer.step()
        #with th.no_grad():
        #    subgoal_grad_norm = (
        #    sum(p.grad.data.norm(2).item() ** 2 for p in self.subgoal_net.parameters() if p.grad is not None) ** 0.5
        #    )
		
        """
		# Log variables
        if self.logger is not None:
            self.logger.store(
                #subgoal_grad_norm = subgoal_grad_norm,
                subgoal_weight = weight.mean().item(),
                subgoal_weight_max = weight.max().item(),
                subgoal_weight_min = weight.min().item(),
                log_prob_target_subgoal = log_prob.mean().item(),
                adv = adv.mean().item(),
                ratio_adv = adv.ge(0.0).float().mean().item(),
                subgoal_loss = subgoal_loss.item(),
                high_policy_v = policy_v.mean().item(),
                high_v = v.mean().item(),
                v1_v2_diff = policy_v_1.mean().item() - policy_v_2.mean().item(),
            )
        """

    def sample_action_and_KL(self, state, goal):
        batch_size = state.size(0)
        # Sample action, subgoals and KL-divergence
        action_dist = self.actor(state, goal)
        action = action_dist.rsample()

        with th.no_grad():
            subgoal = self.sample_subgoal(state, goal)

        prior_action_dist = self.actor_target(state.unsqueeze(1).expand(batch_size, subgoal.size(1), self.state_dim), subgoal)
        prior_prob = prior_action_dist.log_prob(action.unsqueeze(1).expand(batch_size, subgoal.size(1), self.action_dim)).sum(-1, keepdim=True).exp()
        prior_log_prob = th.log(prior_prob.mean(1) + self.epsilon)
        D_KL = action_dist.log_prob(action).sum(-1, keepdim=True) - prior_log_prob

        action = th.tanh(action)
        return action, D_KL
    
    def value(self, state, goal):
        _, _, action = self.actor.sample(state, goal)
        V = self.critic(state, action, goal).min(-1, keepdim=True)[0]
        return V
    
    def sample_subgoal(self, state, goal):
        subgoal_distribution = self.subgoal_net(state, goal)
        subgoal = subgoal_distribution.rsample((self.n_ensemble,))
        subgoal = th.transpose(subgoal, 0, 1) # 2048x10x20
        return subgoal

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)

        actor_losses, critic_losses, subgoal_net_losses, advs = [], [], [], []

        for gradient_step in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            # sample more states for subgoals
            subgoals_replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            state = replay_data.observations["observation"]
            goal = replay_data.observations["desired_goal"]
            action = replay_data.actions
            reward = replay_data.rewards
            next_state = replay_data.next_observations["observation"]
            done = replay_data.dones
            subgoal = subgoals_replay_data.observations["observation"]

            """ Critic """
            # Compute target Q
            with th.no_grad():
                next_action, log_prob, _ = self.actor.sample(next_state, goal)
                target_Q = self.critic_target(next_state, next_action, goal)
                target_Q = th.min(target_Q, -1, keepdim=True)[0]
                target_Q = reward + (1.0-done) * self.gamma*target_Q

            # Compute critic loss
            Q = self.critic(state, action, goal)
            critic_loss = 0.5 * (Q - target_Q).pow(2).sum(-1).mean()
            critic_losses.append(critic_loss.item())

            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            #if self.max_grad_norm > 0:
            #    th.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.max_grad_norm)
            self.critic_optimizer.step()
                
            # Optimize the subgoal policy
            self.train_highlevel_policy(state, goal, subgoal, subgoal_net_losses, advs) # test
            
            """ Actor """
            action, D_KL = self.sample_action_and_KL(state, goal)
            # Compute actor loss
            Q = self.critic(state, action, goal)
            Q = th.min(Q, -1, keepdim=True)[0]
            actor_loss = (self.alpha*D_KL - Q).mean()
            actor_losses.append(actor_loss.item())
            # Optimize the actor 
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            #if self.actor_max_grad_norm > 0:
            #    th.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.actor_max_grad_norm)
            self.actor_optimizer.step()

            # Update target networks
            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.actor.parameters(), self.actor_target.parameters(), self.tau) # test
        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        self.logger.record("train/subgoal_net_loss", np.mean(subgoal_net_losses))
        self.logger.record("train/adv", np.mean(advs))

    def _excluded_save_params(self) -> List[str]:
        return super(SafetyRis, self)._excluded_save_params() + ["actor", "critic", "critic_target"]

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy", "actor.optimizer", "critic.optimizer"]
        if self.ent_coef_optimizer is not None:
            saved_pytorch_variables = ["log_ent_coef"]
            state_dicts.append("ent_coef_optimizer")
        else:
            saved_pytorch_variables = ["ent_coef_tensor"]
        return state_dicts, saved_pytorch_variables
