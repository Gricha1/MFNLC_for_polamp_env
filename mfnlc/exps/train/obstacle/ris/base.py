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

from mfnlc.config import get_path, default_device
from mfnlc.envs import get_env, get_sub_proc_env
from mfnlc.exps.train.utils import copy_current_model_to_log_dir
from mfnlc.learn.safety_ris import SafetyRis
from mfnlc.learn.subgoal import LaplacePolicy, EnsembleCritic, GaussianPolicy


def train(env_name,
          total_timesteps: int,
          policy: Union[str, Type[TD3Policy]] = "MlpPolicy",
          learning_rate: Union[float, Schedule] = 1e-3,
          buffer_size: int = 1_000_000,  # 1e6
          learning_starts: int = 100,
          batch_size: int = 100,
          tau: float = 0.005,
          gamma: float = 0.99,
          train_freq: Union[int, Tuple[int, str]] = (1, "episode"),
          gradient_steps: int = 40,
          action_noise: Optional[ActionNoise] = None,
          # HER hyperparams
          # Available strategies (cf paper): future, final, episode
          goal_selection_strategy = "future", # equivalent to GoalSelectionStrategy.FUTURE
          replay_buffer_class: Optional[ReplayBuffer] = None,
          replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
          optimize_memory_usage: bool = False,
          ent_coef = "auto",
          target_update_interval = 1,
          target_entropy = "auto",
          use_sde = False,
          sde_sample_freq = -1,
          use_sde_at_warmup = False,
          #policy_delay: int = 2, # TD3
          #target_policy_noise: float = 0.2, # TD3
          #target_noise_clip: float = 0.5, # TD3
          subgoal_policy_kwargs: Optional[Dict[str, Any]] = None, # RIS
          h_lr = 1e-3, # RIS
          q_lr = 1e-3, # RIS
          pi_lr = 1e-4, # RIS
          epsilon: float = 1e-16, # RIS
          alpha = 0.1, # RIS
          Lambda = 0.1, # RIS
          n_ensemble = 10, # RIS
          clip_v_function = -150, # RIS,
          create_eval_env: bool = False,
          policy_kwargs: Optional[Dict[str, Any]] = None,
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
          ):
    algo = "ris"

    if n_envs == 1:
        env = get_env(env_name)
    elif n_envs > 1:
        env = get_sub_proc_env(env_name, n_envs)
    else:
        raise ValueError(f"n_envs should be greater than 0, but it is {n_envs}")

    robot_name = env_name.split("-")[0]

    tensorboard_log = get_path(robot_name, algo, "log")

    # add custom video callback
    class VideoRecorderCallback(BaseCallback):
        def __init__(self, eval_env: gym.Env, render_freq: int, n_eval_episodes: int = 1, deterministic: bool = True):
            super().__init__()
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

                def grab_screens(_locals: Dict[str, Any], _globals: Dict[str, Any]) -> None:
                    # predict subgoal and set to env
                    assert len(_locals["env"].envs) == 1
                    with th.no_grad():
                        state = _locals["observations"]["observation"]
                        goal = _locals["observations"]["desired_goal"]
                        to_torch_state = th.FloatTensor(state).to(default_device).unsqueeze(0)
                        to_torch_goal = th.FloatTensor(goal).to(default_device).unsqueeze(0)
                        subgoal_distribution = self.model.subgoal_net(to_torch_state, to_torch_goal)
                        subgoal = subgoal_distribution.loc
                    _locals["env"].envs[0].set_subgoal_pos(subgoal)
                    # get video
                    if _locals["episode_counts"][_locals["i"]] == 0:
                        #screen = self._eval_env.custom_render(positions_render=False)
                        #robot_screens.append(screen.transpose(2, 0, 1))
                        screen = self._eval_env.custom_render(positions_render=True)
                        positions_screens.append(screen.transpose(2, 0, 1))
                    # get success rate
                    info = _locals["info"]
                    if _locals["done"]:
                        maybe_is_success = info.get("is_success")
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
                #self.logger.record(
                #    "trajectory/video",
                #    Video(th.ByteTensor([robot_screens]), fps=40),
                #    exclude=("stdout", "log", "json", "csv"),
                #)
                self.logger.record(
                    "trajectory/pos_video",
                    Video(th.ByteTensor([positions_screens]), fps=40),
                    exclude=("stdout", "log", "json", "csv"),
                )
                del robot_screens
                #del positions_screens

                mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
                mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
                # Add to current Logger
                self.logger.record("eval/reward", float(mean_reward))
                self.logger.record("eval/ep_length", mean_ep_length)
                if len(self._is_success_buffer) > 0:
                    success_rate = np.mean(self._is_success_buffer)
                    self.logger.record("eval/success_rate", success_rate)
                else:
                    success_rate = np.mean(self._is_success_buffer)
                    self.logger.record("eval/success_rate", 0)
            return True
        
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
    video_recorder = VideoRecorderCallback(callback_eval_env, n_eval_episodes=10, render_freq=500)

    state_dim = env_obs_dim
    goal_dim = env_goal_dim
    actor = GaussianPolicy(state_dim, action_dim, hidden_dims=subgoal_policy_kwargs["net_arch"]).to(default_device)
    critic = EnsembleCritic(state_dim, action_dim, hidden_dims=subgoal_policy_kwargs["net_arch"],
                            n_Q=2).to(default_device)
    subgoal_net = LaplacePolicy(state_dim=state_dim, 
                                goal_dim=state_dim, 
                                hidden_dims=subgoal_policy_kwargs["net_arch"]).to(default_device)

    model = SafetyRis(
        actor,
        critic,
        subgoal_net,
        state_dim,
        action_dim,
        "MultiInputPolicy",  
        env, 
        h_lr, 
        q_lr,
        pi_lr,
        epsilon,
        learning_rate, buffer_size, learning_starts, batch_size, tau, gamma,
        train_freq, gradient_steps, action_noise, 
        HerReplayBuffer, #replay_buffer_class
        dict(
            n_sampled_goal=4,
            goal_selection_strategy=goal_selection_strategy,
        ), # replay_buffer_kwargs
        optimize_memory_usage, ent_coef, target_update_interval, target_entropy, 
        use_sde, sde_sample_freq, use_sde_at_warmup, 
        alpha, Lambda, n_ensemble, clip_v_function,
        tensorboard_log, create_eval_env, policy_kwargs, verbose, seed, default_device)

    """
    model = TD3(
        "MultiInputPolicy", 
        #"MlpPolicy", 
        env, learning_rate, buffer_size, learning_starts, batch_size, tau, gamma,
        train_freq, gradient_steps, action_noise, 
        HerReplayBuffer, #replay_buffer_class
        dict(
            n_sampled_goal=4,
            goal_selection_strategy=goal_selection_strategy,
        ), # replay_buffer_kwargs
        #replay_buffer_class,
        #replay_buffer_kwargs,
        optimize_memory_usage, policy_delay, target_policy_noise, target_noise_clip,
        tensorboard_log, create_eval_env, policy_kwargs, verbose, seed, default_device)
    """
        
    model.learn(total_timesteps, video_recorder, log_interval, eval_env, eval_freq,
                n_eval_episodes, tb_log_name, eval_log_path, reset_num_timesteps)

    model_path = get_path(robot_name, algo, "model")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)

    copy_current_model_to_log_dir(robot_name, algo)
