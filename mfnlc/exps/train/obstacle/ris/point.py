import numpy as np
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise

from mfnlc.evaluation.simulation import inspect_training_simu
from mfnlc.exps.train.obstacle.ris.base import train


def learn():
    action_noise = OrnsteinUhlenbeckActionNoise(np.array([0, 0]), np.array([0.1, 0.1]))
    train(env_name="GCPoint",
          total_timesteps=1600_000,
          learning_starts=10_000,
          action_noise=None,
          subgoal_policy_kwargs={"net_arch": [100, 100]},
          policy_kwargs={"net_arch": [100, 100]},
          train_freq=(1, "episode"), #train_freq=(200, "step"),
          gradient_steps=1, #gradient_steps=100,
          h_lr=1e-4, # RIS
          q_lr=1e-3, # RIS
          pi_lr=1e-4, # RIS
          epsilon=1e-16, # RIS
          alpha=0.1, # RIS
          Lambda=0.1, # RIS
          n_ensemble=10, # RIS
          clip_v_function=-150, # RIS,
          n_envs=1,
          batch_size=2048,
          log_interval=4)


def evaluate_controller():
    inspect_training_simu(env_name="Point",
                          algo="e2e",
                          n_rollout=20,
                          render=True)


if __name__ == '__main__':
    learn()
    # evaluate_controller()
