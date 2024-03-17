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
from stable_baselines3.common.type_aliases import TrainFreq, TrainFrequencyUnit
from torch.nn import functional as F

from mfnlc.config import default_device
from mfnlc.learn.utils import list_dict_to_dict_list
from mfnlc.learn.subgoal import LaplacePolicy, GaussianPolicy, EnsembleCritic, CustomActorCriticPolicy
from mfnlc.learn.HER import HERReplayBuffer, PathBuilder


class SafetyRis(SAC):
    def __init__(
        self,
        policy: CustomActorCriticPolicy,
        subgoal_net: LaplacePolicy,
        state_dim: int,
        action_dim: int,
        policy_to_delete: Union[str, Type[SACPolicy]],
        env: Union[GymEnv, str],
        h_lr: float = 1e-4, 
        q_lr: float = 1e-3,
        pi_lr: float = 1e-4, 
        epsilon: float = 1e-16,
        critic_max_grad_norm: float = None,
        actor_max_grad_norm: float = None,
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
        fraction_goals_are_rollout_goals: float = 0.2,
        fraction_resampled_goals_are_env_goals: float = 0.0,
        fraction_resampled_goals_are_replay_buffer_goals: float = 0.5,
        tensorboard_log: Optional[str] = None,
        create_eval_env: bool = False,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ):

        super(SafetyRis, self).__init__(
            policy_to_delete,
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

        # policy
        self.pi_lr = pi_lr
        self.q_lr = q_lr
        self.new_policy = policy
        self.critic_max_grad_norm = critic_max_grad_norm
        self.actor_max_grad_norm = actor_max_grad_norm

        # subgoal
        self.subgoal_net = subgoal_net
        self.subgoal_optimizer = th.optim.Adam(self.subgoal_net.parameters(), lr=h_lr)
        self.alpha = alpha
        self.Lambda = Lambda
        self.n_ensemble = n_ensemble
        self.clip_v_function = clip_v_function
        self.epsilon = epsilon

        # HER
        vectorized = False
        self.path_builder = PathBuilder()
        self.custom_replay_buffer = HERReplayBuffer(
            max_size=500_000,
            env=env,
            fraction_goals_are_rollout_goals = fraction_goals_are_rollout_goals,
            fraction_resampled_goals_are_env_goals = fraction_resampled_goals_are_env_goals,
            fraction_resampled_goals_are_replay_buffer_goals = fraction_resampled_goals_are_replay_buffer_goals,
            ob_keys_to_save     =["collision", "clearance_is_enough"],
            desired_goal_keys   =["desired_goal"],
            observation_key     = 'observation',
            desired_goal_key    = 'desired_goal',
            achieved_goal_key   = 'achieved_goal',
            vectorized          = vectorized 
        )

        # safety
        self.safety = False

        # sac
        self.sac = False
        self.sac_alpha = 0.2

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super(SafetyRis, self)._setup_model()
        self._setup_alias()

    def _setup_alias(self) -> None:
        # setup new actor critic
        del self.actor
        del self.critic
        del self.critic_target
        del self.policy.actor
        del self.policy.critic
        del self.policy.critic_target
        del self.policy
        self.policy = self.new_policy
        self.actor = self.policy.actor
        self.critic = self.policy.critic
        self.actor_target = deepcopy(self.actor)
        self.critic_target = deepcopy(self.critic)
        self.actor_optimizer = th.optim.Adam(self.actor.parameters(), lr=self.pi_lr)
        self.critic_optimizer = th.optim.Adam(self.critic.parameters(), lr=self.q_lr)

    
    def sample_and_preprocess_batch(self, replay_buffer, env, batch_size=256, device=th.device("cuda")):
        # Extract 
        batch = replay_buffer.random_batch(batch_size)
        state_batch         = batch["observations"]
        action_batch        = batch["actions"]
        next_state_batch    = batch["next_observations"]
        goal_batch          = batch["resampled_goals"]
        reward_batch        = batch["rewards"]
        done_batch          = batch["terminals"]
        clearance_is_enough_batch = batch["clearance_is_enough"]
        collision_batch     = batch["collision"]       
        
        # vel_pos = int(env.envs[0].env.obstacle_in_obs) * 2
        # i_v = vel_pos + 1
        # e_v = i_v + 2
        shift_v = int(next_state_batch.shape[1] / env.envs[0].frame_stack * (env.envs[0].frame_stack - 1))
        # Compute sparse rewards: -1 for all actions until the goal is reached
        reward_batch = np.sqrt(np.power(np.array(next_state_batch - goal_batch)[:, shift_v:shift_v+2], 2).sum(-1, keepdims=True)) # distance: next_state to goal
        done_batch   = 1.0 * (reward_batch <= env.envs[0].env.env.goal_size)# terminal condition
        # done_batch   = 1.0 * (reward_batch <= env.envs[0].env.env.goal_size) + \
        #     1.0 * (np.sqrt(np.power(np.array(next_state_batch)[:, -e_v:-i_v], 2).sum(-1, keepdims=True)) > 0.1)
        done_batch = 1.0 * collision_batch + (1.0 - 1.0 * collision_batch) * (done_batch)
        # done_batch = 1.0 * collision_batch + (1.0 - 1.0 * collision_batch) * (done_batch // (1.0 + 1.0))
        reward_batch = (- np.ones_like(done_batch) * (-env.envs[0].env.time_step_reward)) * (1.0 - collision_batch) \
                        + (env.envs[0].env.collision_penalty) * collision_batch

        cost_batch = (- np.ones_like(done_batch) * 0)
        """
        if env.static_env:
            velocity_array = np.abs(next_state_batch[:, 3:4])
            if args.use_lower_velocity_bound:
                min_ris_velocity = 0.3
                # cost_collision = fabs(env.collision_reward)
                # adding threshold to lower velocity bound
                velocity_limit_exceeded = velocity_array >= min_ris_velocity
                updated_velocity_array = velocity_array * velocity_limit_exceeded
                cost_batch = (np.ones_like(done_batch) * updated_velocity_array) * (1.0 - clearance_is_enough_batch)
                # cost_batch = (1.0 - collision_batch) * cost_batch + cost_collision * collision_batch
            else:
                cost_batch = (np.ones_like(done_batch) * velocity_array) * (1.0 - clearance_is_enough_batch)

        else:
            cost_batch = (- np.ones_like(done_batch) * 0)
        """
        # Scaling
        # if args.scaling > 0.0:
        #     reward_batch = reward_batch * args.scaling
        # check if (collision == 1) then (done == 1)
        #if env.static_env and not env.teleport_back_on_collision:
        #    assert ( (1.0 - 1.0 * collision_batch) + (1.0 * collision_batch) * (1.0 * done_batch) ).all()

        # Convert to Pytorch
        state_batch         = th.FloatTensor(state_batch).to(device)
        action_batch        = th.FloatTensor(action_batch).to(device)
        reward_batch        = th.FloatTensor(reward_batch).to(device)
        cost_batch        = th.FloatTensor(cost_batch).to(device)
        next_state_batch    = th.FloatTensor(next_state_batch).to(device)
        done_batch          = th.FloatTensor(done_batch).to(device)
        goal_batch          = th.FloatTensor(goal_batch).to(device)

        return state_batch, action_batch, reward_batch, cost_batch, next_state_batch, done_batch, goal_batch

    
    # RIS requires training each time when env.step()
    def _on_step(self):
        if self.num_timesteps > 0 and self.num_timesteps > self.learning_starts:
            gradient_steps = self.gradient_steps
            # Special case when the user passes `gradient_steps=0`
            if gradient_steps > 0:
                self.train(batch_size=self.batch_size, gradient_steps=gradient_steps)

    def collect_rollouts(
        self,
        env,
        callback,
        train_freq,
        replay_buffer: ReplayBuffer,
        action_noise: Optional[ActionNoise] = None,
        learning_starts: int = 0,
        log_interval: Optional[int] = None,
    ):
        # collect_rollouts should collect one full episode to self.path_builder
        # TODO vec env resets env if it is terminated, so i have to accout last obs
        assert train_freq.unit == TrainFrequencyUnit.EPISODE and \
               train_freq.frequency == 1 
        self.path_builder = PathBuilder()
        rollout = super().collect_rollouts(
                self.env,
                train_freq=self.train_freq,
                action_noise=self.action_noise,
                callback=callback,
                learning_starts=self.learning_starts,
                replay_buffer=self.replay_buffer,
                log_interval=log_interval,
            )
        self.custom_replay_buffer.add_path(self.path_builder.get_all_stacked())  
        return rollout

    def _store_transition(
        self,
        replay_buffer: ReplayBuffer,
        buffer_action: np.ndarray,
        new_obs: Union[np.ndarray, Dict[str, np.ndarray]],
        reward: np.ndarray,
        dones: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        """
        Store transition in the replay buffer.
        We store the normalized action and the unnormalized observation.
        It also handles terminal observations (because VecEnv resets automatically).

        :param replay_buffer: Replay buffer object where to store the transition.
        :param buffer_action: normalized action
        :param new_obs: next observation in the current episode
            or first observation of the episode (when dones is True)
        :param reward: reward for the current transition
        :param dones: Termination signal
        :param infos: List of additional information about the transition.
            It may contain the terminal observations and information about timeout.
        """
        # Store only the unnormalized version
        if self._vec_normalize_env is not None:
            new_obs_ = self._vec_normalize_env.get_original_obs()
            reward_ = self._vec_normalize_env.get_original_reward()
        else:
            # Avoid changing the original ones
            self._last_original_obs, new_obs_, reward_ = self._last_obs, new_obs, reward

        # Avoid modification by reference
        next_obs = deepcopy(new_obs_)
        # As the VecEnv resets automatically, new_obs is already the
        # first observation of the next episode
        for i, done in enumerate(dones):
            if done and infos[i].get("terminal_observation") is not None:
                if isinstance(next_obs, dict):
                    next_obs_ = infos[i]["terminal_observation"]
                    # VecNormalize normalizes the terminal observation
                    if self._vec_normalize_env is not None:
                        next_obs_ = self._vec_normalize_env.unnormalize_obs(next_obs_)
                    # Replace next obs for the correct envs
                    for key in next_obs.keys():
                        next_obs[key][i] = next_obs_[key]
                else:
                    next_obs[i] = infos[i]["terminal_observation"]
                    # VecNormalize normalizes the terminal observation
                    if self._vec_normalize_env is not None:
                        next_obs[i] = self._vec_normalize_env.unnormalize_obs(next_obs[i, :])

        assert self.n_envs == 1
        self.path_builder.add_all(
            observations=self._last_original_obs,
            actions=buffer_action,
            rewards=reward_,
            next_observations=next_obs,
            terminals=[1.0*dones[0]]
        )
        
        self._last_obs = new_obs
        # Save the unnormalized observation
        if self._vec_normalize_env is not None:
            self._last_original_obs = new_obs_
        
    def train_highlevel_policy(self, state, goal, subgoal, debug_info={}):
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
        debug_info["subgoal_net_losses"].append(subgoal_loss.item())
        debug_info["advs"].append(adv.mean().item())
        debug_info["target_subgoal_V"].append(v.mean().item())
        debug_info["subgoal_V"].append(policy_v.mean().item())

        # Update network
        self.subgoal_optimizer.zero_grad()
        subgoal_loss.backward()
        self.subgoal_optimizer.step()

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

        actor_losses, critic_losses = [], []
        debug_info = {}
        debug_info["subgoal_net_losses"] = []
        debug_info["advs"] = []
        debug_info["Q"] = []
        debug_info["target_subgoal_V"] = []
        debug_info["subgoal_V"] = []
        debug_info["target_Q"] = []

        for gradient_step in range(gradient_steps):
            state, action, reward, cost, next_state, done, goal = self.sample_and_preprocess_batch(
                self.custom_replay_buffer, 
                env=self.env,
                batch_size=batch_size,
                device=self.device
            )
            # Sample subgoal candidates uniformly in the replay buffer
            subgoal = th.FloatTensor(self.custom_replay_buffer.random_state_batch(batch_size)).to(self.device)
            

            """ Critic """
            # Compute target Q
            with th.no_grad():
                next_action, next_log_prob, _ = self.actor.sample(next_state, goal)
                target_Q = self.critic_target(next_state, next_action, goal)
                if self.sac:
                    target_Q -= self.sac_alpha * next_log_prob
                target_Q = th.min(target_Q, -1, keepdim=True)[0]
                target_Q = reward + (1.0-done) * self.gamma*target_Q

            # Compute critic loss
            Q = self.critic(state, action, goal)
            critic_loss = 0.5 * (Q - target_Q).pow(2).sum(-1).mean()
            critic_losses.append(critic_loss.item())
            debug_info["Q"].append(Q.mean().item())
            debug_info["target_Q"].append(target_Q.mean().item())

            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            if not(self.critic_max_grad_norm is None):
                if self.critic_max_grad_norm > 0:
                    th.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.critic_max_grad_norm)
            self.critic_optimizer.step()
                
            # Optimize the subgoal policy
            if not self.sac:
                self.train_highlevel_policy(state, goal, subgoal, debug_info) # test
            
            """ Actor """
            if self.sac:
                action, log_prob, _ = self.actor.sample(state, goal)
            else:
                action, D_KL = self.sample_action_and_KL(state, goal)                
            # Compute actor loss
            Q = self.critic(state, action, goal)
            Q = th.min(Q, -1, keepdim=True)[0]
            
            if self.sac:
                actor_loss = (self.sac_alpha * log_prob - Q).mean()
            else:
                actor_loss = (self.alpha*D_KL - Q).mean()
            actor_losses.append(actor_loss.item())
            # Optimize the actor 
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            if not(self.actor_max_grad_norm is None):
                if self.actor_max_grad_norm > 0:
                    th.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.actor_max_grad_norm)
            self.actor_optimizer.step()

            # Update target networks
            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.actor.parameters(), self.actor_target.parameters(), self.tau) # test
        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))        
        self.logger.record("train/Q", np.mean(debug_info["Q"])) 
        self.logger.record("train/target_Q", np.mean(debug_info["target_Q"]))
        if not self.sac:      
            self.logger.record("train/subgoal_net_loss", np.mean(debug_info["subgoal_net_losses"]))
            self.logger.record("train/adv", np.mean(debug_info["advs"]))
            self.logger.record("train/D_KL", D_KL.mean().item())
            self.logger.record("train/target_subgoal_V", np.mean(debug_info["target_subgoal_V"]))
            self.logger.record("train/subgoal_V", np.mean(debug_info["subgoal_V"]))

    def save(self, folder, save_optims=False):
        th.save(self.actor.state_dict(),		 folder + "actor.pth")
        th.save(self.critic.state_dict(),		folder + "critic.pth")
        if self.safety:
            th.save(self.critic_cost.state_dict(),		folder + "critic_cost.pth")
        th.save(self.subgoal_net.state_dict(),   folder + "subgoal_net.pth")
        if save_optims:
            th.save(self.actor_optimizer.state_dict(), 	folder + "actor_opti.pth")
            th.save(self.critic_optimizer.state_dict(), 	folder + "critic_opti.pth")
            th.save(self.subgoal_optimizer.state_dict(), folder + "subgoal_opti.pth")
    
    def load(self, folder, old_version=False, best=True):
        if old_version:
            run_name = ""	
        else:
            run_name = "best_" if best else "last_"
        print(f"load run_name: {run_name}")
        self.actor.load_state_dict(th.load(folder+run_name+"actor.pth", map_location=self.device))
        self.critic.load_state_dict(th.load(folder+run_name+"critic.pth", map_location=self.device))
        if self.safety:
            self.critic_cost.load_state_dict(th.load(folder+run_name+"critic_cost.pth", map_location=self.device))
        self.subgoal_net.load_state_dict(th.load(folder+run_name+"subgoal_net.pth", map_location=self.device))

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
