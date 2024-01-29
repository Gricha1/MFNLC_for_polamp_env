import os
from typing import Union, Type, Tuple, Optional, Dict, Any

import numpy as np
import gym
import torch as th
from stable_baselines3 import TD3, HerReplayBuffer, SAC
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.type_aliases import Schedule, MaybeCallback, GymEnv
from stable_baselines3.td3.policies import TD3Policy
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import Video
import wandb
from wandb.integration.sb3 import WandbCallback

from mfnlc.config import get_path, default_device
from mfnlc.envs import get_env, get_sub_proc_env
from mfnlc.exps.train.utils import copy_current_model_to_log_dir
from mfnlc.learn.safety_ris import SafetyRis
from mfnlc.learn.subgoal import LaplacePolicy, EnsembleCritic, GaussianPolicy, CustomActorCriticPolicy


def train(env_name,
          total_timesteps: int,
          policy_to_delete: Union[str, Type[TD3Policy]] = "MlpPolicy",
          learning_rate: Union[float, Schedule] = 1e-3,
          buffer_size: int = 1_000_000,  # 1e6
          learning_starts: int = 100,
          batch_size: int = 100,
          tau: float = 0.005,
          gamma: float = 0.99,
          train_freq: Union[int, Tuple[int, str]] = (1, "episode"),
          gradient_steps: int = 40,
          action_noise: Optional[ActionNoise] = None,
          goal_selection_strategy = "future", # Available strategies (cf paper): future, final, episode
          replay_buffer_class: Optional[ReplayBuffer] = None,
          replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
          optimize_memory_usage: bool = False,
          ent_coef = "auto",
          target_update_interval = 1,
          target_entropy = "auto",
          use_sde = False,
          sde_sample_freq = -1,
          use_sde_at_warmup = False,
          new_policy_kwargs: Optional[Dict[str, Any]] = None, # RIS
          h_lr = 1e-3, # RIS
          q_lr = 1e-3, # RIS
          pi_lr = 1e-4, # RIS
          epsilon: float = 1e-16, # RIS
          alpha = 0.1, # RIS
          Lambda = 0.1, # RIS
          n_ensemble = 10, # RIS
          clip_v_function = -150, # RIS,
          critic_max_grad_norm: float = None, # RIS
          actor_max_grad_norm: float = None, # RIS
          create_eval_env: bool = False,
          policy_to_delete_kwargs: Optional[Dict[str, Any]] = None,
          verbose: int = 1,
          seed: Optional[int] = None,
          callback: MaybeCallback = None,
          log_interval: int = 4,
          eval_env: Optional[GymEnv] = None,
          eval_freq: int = -1,
          n_eval_episodes: int = 5,
          tb_log_name: str = "SAFETY_RIS",
          eval_log_path: Optional[str] = None,
          reset_num_timesteps: bool = True,
          n_envs: int = 1,
          validate_freq: int = 5000,
          use_wandb = True
          ):
    algo = "ris"
    if use_wandb:
        run = wandb.init(
            project="train_safety_ris_safety_gym",
            sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        )

    if n_envs == 1:
        env = get_env(env_name)
    elif n_envs > 1:
        env = get_sub_proc_env(env_name, n_envs)
    else:
        raise ValueError(f"n_envs should be greater than 0, but it is {n_envs}")

    robot_name = env_name.split("-")[0]

    tensorboard_log = get_path(robot_name, algo, "log")

    # add custom video callback
    class VideoRecorderCallback(WandbCallback):
        def __init__(self, 
                    eval_env: gym.Env, 
                    render_freq: int, 
                    n_eval_episodes: int = 1, 
                    deterministic: bool = True,
                    verbose: int = 0,
                    model_save_path: Optional[str] = None,
                    model_save_freq: int = 0,
                    gradient_save_freq: int = 0):
            super().__init__(gradient_save_freq=gradient_save_freq, # error if > 0 
                             model_save_path=model_save_path,
                             verbose=verbose)
            self._eval_env = eval_env
            self._render_freq = render_freq
            self._n_eval_episodes = n_eval_episodes
            self._deterministic = deterministic
            self._is_success_buffer = []

        def _on_step(self) -> bool:
            if self.n_calls % self._render_freq == 0:
                robot_screens = []
                positions_screens = []
                self._is_success_buffer = []
                self.collisions = []
                dubug_info = {"acc_reward" : 0, "t": 0, "acc_cost" : 0}

                def grab_screens(_locals: Dict[str, Any], _globals: Dict[str, Any]) -> None:
                    # predict subgoal and set to env
                    assert len(_locals["env"].envs) == 1
                    if _locals['done'] and _locals['info']["collision"]:
                        self.collisions.append(1.0)
                    elif _locals['done']:
                        self.collisions.append(0.0)
                    dubug_info["a0"] = _locals["actions"][0]
                    dubug_info["a1"] = _locals["actions"][1]
                    dubug_info["acc_reward"] += _locals["reward"]
                    dubug_info["acc_cost"] += _locals["info"]["cost"]
                    dubug_info["t"] += 1
                    with th.no_grad():
                        state = _locals["observations"]["observation"]
                        goal = _locals["observations"]["desired_goal"]
                        to_torch_state = th.FloatTensor(state).to(default_device).unsqueeze(0)
                        to_torch_goal = th.FloatTensor(goal).to(default_device).unsqueeze(0)
                        subgoal_distribution = self.model.subgoal_net(to_torch_state, to_torch_goal)
                        subgoal = subgoal_distribution.loc
                    _locals["env"].envs[0].set_subgoal_pos(subgoal)
                    # dubug subgoal
                    if _locals["episode_counts"][_locals["i"]] == 0 and dubug_info["t"] == 1:
                        print("state:", state)
                        print("subgoal:", subgoal[0])
                        print("goal:", goal)
                    # get video
                    if _locals["episode_counts"][_locals["i"]] == 0:
                        screen = self._eval_env.custom_render(positions_render=False)
                        robot_screens.append(screen.transpose(2, 0, 1))
                        screen = self._eval_env.custom_render(positions_render=True, dubug_info=dubug_info)
                        positions_screens.append(screen.transpose(2, 0, 1))
                    # get success rate
                    if _locals["done"]:
                        maybe_is_success = _locals["info"].get("goal_is_arrived")
                        if maybe_is_success is not None:
                            self._is_success_buffer.append(maybe_is_success)

                episode_rewards, episode_lengths = evaluate_policy(
                    self.model,
                    self._eval_env,
                    callback=grab_screens,
                    return_episode_rewards=True,
                    n_eval_episodes=self._n_eval_episodes,
                    deterministic=self._deterministic,
                )
                self.logger.record(
                    "trajectory/video",
                    Video(th.ByteTensor([robot_screens]), fps=40),
                    exclude=("stdout", "log", "json", "csv"),
                )
                self.logger.record(
                    "trajectory/pos_video",
                    Video(th.ByteTensor([positions_screens]), fps=40),
                    exclude=("stdout", "log", "json", "csv"),
                )
                del robot_screens
                del positions_screens

                mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
                min_reward, max_reward = np.min(episode_rewards), np.max(episode_rewards)
                mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
                collision_rate = np.mean(self.collisions)
                # Add to current Logger
                self.logger.record("eval/reward", float(mean_reward))
                self.logger.record("eval/ep_length", mean_ep_length)
                self.logger.record("eval/reward_min", min_reward)
                self.logger.record("eval/reward_max", max_reward)
                self.logger.record("eval/collision_rate", collision_rate)
                if len(self._is_success_buffer) > 0:
                    success_rate = np.mean(self._is_success_buffer)
                    self.logger.record("eval/custom_success_rate", success_rate)
                else:
                    success_rate = np.mean(self._is_success_buffer)
                    self.logger.record("eval/custom_success_rate", 0)
            
                return True
            else:
                return super()._on_step()
        
    if n_envs == 1:
        callback_eval_env = get_env(env_name)
    else:
        assert 1 == 0

    # test eval env
    obs = callback_eval_env.reset()
    print("obs type:", type(obs))
    print("obs:", callback_eval_env.observation_space)
    print("image shape:", callback_eval_env.custom_render().shape)

    env_obs_dim = env.observation_space["observation"].shape[0]
    env_goal_dim = env.observation_space["desired_goal"].shape[0]
    action_dim = env.action_space.shape[0]
    assert env_obs_dim == env_goal_dim
    run_id = 0
    video_recorder = None
    if use_wandb:
        run_id = run.id
        video_recorder = VideoRecorderCallback(callback_eval_env, 
                                            n_eval_episodes=10, 
                                            render_freq=validate_freq,
                                            gradient_save_freq=0, # error if > 0 
                                            model_save_path=f"models/{run_id}",
                                            verbose=2)

    print("Ending")
    state_dim = env_obs_dim
    goal_dim = env_goal_dim
    actor = GaussianPolicy(state_dim, action_dim, 
                           hidden_dims=new_policy_kwargs["net_arch"]).to(default_device)
    critic = EnsembleCritic(state_dim, action_dim, 
                            hidden_dims=new_policy_kwargs["net_arch"],
                            n_Q=2).to(default_device)
    subgoal_net = LaplacePolicy(state_dim=state_dim, 
                                goal_dim=state_dim, 
                                hidden_dims=new_policy_kwargs["net_arch"]).to(default_device)
    policy = CustomActorCriticPolicy(default_device)
    policy.actor = actor
    policy.critic = critic

    model = SafetyRis(
        policy,
        subgoal_net,
        state_dim,
        action_dim,
        "MultiInputPolicy",  
        env, 
        h_lr, 
        q_lr,
        pi_lr,
        epsilon,
        critic_max_grad_norm,
        actor_max_grad_norm,
        learning_rate, buffer_size, learning_starts, batch_size, tau, gamma,
        train_freq, gradient_steps, action_noise, 
        HerReplayBuffer, #replay_buffer_class
        dict(
            n_sampled_goal=4,
            goal_selection_strategy=goal_selection_strategy,
            online_sampling=True,
        ), # replay_buffer_kwargs
        optimize_memory_usage, ent_coef, target_update_interval, target_entropy, 
        use_sde, sde_sample_freq, use_sde_at_warmup, 
        alpha, Lambda, n_ensemble, clip_v_function,
        tensorboard_log, create_eval_env, policy_to_delete_kwargs, verbose, seed, default_device)

    model.learn(total_timesteps, video_recorder, log_interval, eval_env, eval_freq,
                n_eval_episodes, tb_log_name, eval_log_path, reset_num_timesteps)

    model_path = get_path(robot_name, algo, "model")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)

    copy_current_model_to_log_dir(robot_name, algo)
