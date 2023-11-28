import os
from typing import Any, Dict, Optional, Tuple, Type, Union

import gym
import numpy as np
import torch as th
from matplotlib import pyplot as plt
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.type_aliases import (GymEnv, MaybeCallback,
                                                   Schedule)
from stable_baselines3.td3.policies import TD3Policy
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import Video

from mfnlc.config import env_config, get_path, default_device
from mfnlc.envs import get_env, get_sub_proc_env
from mfnlc.evaluation.simulation import inspect_training_simu
from mfnlc.exps.train.utils import copy_current_model_to_log_dir
from mfnlc.learn.lyapunov_td3 import LyapunovTD3
from mfnlc.learn.tclf import TwinControlLyapunovFunction, InputAmplifierBase


def train(env_name,
          lf_structure,
          lqf_structure,
          tclf_ub: float = 5,
          tclf_q_sigma: float = None,
          tclf_input_amplifier: InputAmplifierBase = None,
          tclf_lie_derivative_upper: float = 0.2,
          total_timesteps: int = 100_00,
          lqf_loss_cnst: float = 1.0,
          policy: Union[str, Type[TD3Policy]] = "MlpPolicy",
          learning_rate: Union[float, Schedule] = 1e-3,
          buffer_size: int = 1_000_000,  # 1e6
          learning_starts: int = 100,
          batch_size: int = 100,
          tau: float = 0.005,
          gamma: float = 0.99,
          train_freq: Union[int, Tuple[int, str]] = (1, "episode"),
          gradient_steps: int = -1,
          action_noise: Optional[ActionNoise] = None,
          replay_buffer_class: Optional[ReplayBuffer] = None,
          replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
          optimize_memory_usage: bool = False,
          policy_delay: int = 2,
          target_policy_noise: float = 0.2,
          target_noise_clip: float = 0.5,
          create_eval_env: bool = False,
          policy_kwargs: Optional[Dict[str, Any]] = None,
          verbose: int = 1,
          seed: Optional[int] = None,
          callback: MaybeCallback = None,
          log_interval: int = 4,
          eval_env: Optional[GymEnv] = None,
          eval_freq: int = -1,
          n_eval_episodes: int = 5,
          tb_log_name: str = "TD3",
          eval_log_path: Optional[str] = None,
          reset_num_timesteps: bool = True,
          n_envs: int = 1,
          ):
    algo = "lyapunov_td3"

    if n_envs == 1:
        env = get_env(env_name)
    elif n_envs > 1:
        env = get_sub_proc_env(env_name, n_envs)
    else:
        raise ValueError(f"n_envs should be greater than 0, but it is {n_envs}")

    robot_name = env_name.split("-")[0]
    tclf = TwinControlLyapunovFunction(lf_structure,
                                       lqf_structure,
                                       tclf_ub,
                                       env_config[robot_name]["sink"],
                                       tclf_input_amplifier,
                                       tclf_lie_derivative_upper,
                                       default_device)

    tensorboard_log = get_path(robot_name, algo, "log")

    # add custom video callback
    class VideoRecorderCallback(BaseCallback):
        def __init__(self, eval_env: gym.Env, render_freq: int, n_eval_episodes: int = 1, deterministic: bool = True):
            """
            Records a video of an agent's trajectory traversing ``eval_env`` and logs it to TensorBoard

            :param eval_env: A gym environment from which the trajectory is recorded
            :param render_freq: Render the agent's trajectory every eval_freq call of the callback.
            :param n_eval_episodes: Number of episodes to render
            :param deterministic: Whether to use deterministic or stochastic policy
            """
            super().__init__()
            self._eval_env = eval_env
            self._render_freq = render_freq
            self._n_eval_episodes = n_eval_episodes
            self._deterministic = deterministic

        def _on_step(self) -> bool:
            if self.n_calls % self._render_freq == 0:
                screens = []

                def grab_screens(_locals: Dict[str, Any], _globals: Dict[str, Any]) -> None:
                    """
                    Renders the environment in its current state, recording the screen in the captured `screens` list

                    :param _locals: A dictionary containing all local variables of the callback's scope
                    :param _globals: A dictionary containing all global variables of the callback's scope
                    """
                    #screen = self._eval_env.render(mode="rgb_array")
                    #screen = self._eval_env.render()
                    screen = self._eval_env.custom_render()
                    # PyTorch uses CxHxW vs HxWxC gym (and tensorflow) image convention
                    screens.append(screen.transpose(2, 0, 1))

                evaluate_policy(
                    self.model,
                    self._eval_env,
                    callback=grab_screens,
                    n_eval_episodes=self._n_eval_episodes,
                    deterministic=self._deterministic,
                )
                self.logger.record(
                    "trajectory/video",
                    Video(th.ByteTensor([screens]), fps=40),
                    exclude=("stdout", "log", "json", "csv"),
                )
                del screens
            return True
        
    if n_envs == 1:
        callback_eval_env = get_env(env_name)
    else:
        assert 1 == 0

    # test
    callback_eval_env.reset()
    print(callback_eval_env.custom_render().shape)
    video_recorder = VideoRecorderCallback(callback_eval_env, render_freq=50_000)
    
    model = LyapunovTD3(
        tclf, policy, env, lqf_loss_cnst, tclf_q_sigma, learning_rate, buffer_size, learning_starts, batch_size, tau,
        gamma, train_freq, gradient_steps, action_noise, replay_buffer_class, replay_buffer_kwargs,
        optimize_memory_usage, policy_delay, target_policy_noise, target_noise_clip,
        tensorboard_log, create_eval_env, policy_kwargs, verbose, seed, default_device)

    model.learn(total_timesteps, video_recorder, log_interval, eval_env, eval_freq,
                n_eval_episodes, tb_log_name, eval_log_path, reset_num_timesteps)

    model_path = get_path(robot_name, algo, "model")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)

    copy_current_model_to_log_dir(robot_name, algo)


def evaluate_lyapunov_of(robot_name):
    simu_info = inspect_training_simu(env_name=f"{robot_name}-no-obst",
                                      algo="lyapunov_td3",
                                      n_rollout=20,
                                      render=False)
    trajectories = simu_info["obs"]

    tclf = th.load(get_path(robot_name, "lyapunov_td3", "tclf"))

    for traj in trajectories:
        traj_tensor = th.tensor(np.array(traj), device=default_device, dtype=th.float32)
        with th.no_grad():
            values = tclf.forward_lf(traj_tensor)
        values = values.cpu().numpy()
        plt.plot(np.arange(len(values)), values)
    plt.show()

    transitions = []
    for traj in trajectories:
        transitions.append(np.array([traj[:-1], traj[1:]]).transpose([1, 0, 2]))
    transitions = np.concatenate(transitions)
    transitions_tensor = th.tensor(transitions, device=default_device, dtype=th.float32)

    with th.no_grad():
        values = tclf.forward_lf(transitions_tensor)
        lyapunov_acc = sum((values[:, 1, 0] - values[:, 0, 0]) < 0) / len(values)
        print(f"{robot_name} - {lyapunov_acc} of transitions' lie derivative is smaller than 0")
