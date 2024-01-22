import numpy as np
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise

from mfnlc.evaluation.simulation import inspect_training_simu
from mfnlc.exps.train.obstacle.ris.base import train


def learn():
    action_noise = OrnsteinUhlenbeckActionNoise(np.array([0, 0]), np.array([0.1, 0.1]))
    train(env_name="GCPoint",
          total_timesteps=1600_000,
          action_noise=action_noise,
          subgoal_policy_kwargs={"net_arch": [100, 100]},
          policy_kwargs={"net_arch": [100, 100]},
          train_freq=(200, "step"),
          #train_freq=(1, "step"),
          #gradient_steps=100,
          n_envs=1,
          batch_size=10000,
          log_interval=4)


def evaluate_controller():
    inspect_training_simu(env_name="Point",
                          algo="e2e",
                          n_rollout=20,
                          render=True)


if __name__ == '__main__':
    learn()
    # evaluate_controller()
