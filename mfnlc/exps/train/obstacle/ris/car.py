import numpy as np
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise

from mfnlc.evaluation.simulation import inspect_training_simu
from mfnlc.exps.train.obstacle.ris.base import train


def learn():
    train(env_name="GCCar",
          total_timesteps=2_000_000,
          learning_starts=10_000,
          action_noise=None,
          new_policy_kwargs={"net_arch": [256, 256]},
          policy_to_delete_kwargs={"net_arch": [100, 100]},
          train_freq=(1, "episode"), #train_freq=(200, "step"),
          gradient_steps=1, #gradient_steps=100,
          h_lr=1e-4, # RIS
          q_lr=1e-3, # RIS
          pi_lr=1e-4, # RIS
          epsilon=1e-16, # RIS
          alpha=0.05, # RIS
          Lambda=0.05, # RIS
          n_ensemble=20, # RIS
          clip_v_function=-150, # RIS,
          critic_max_grad_norm=6.0, # RIS
          actor_max_grad_norm=2.0, # RIS
          log_interval=4,
          n_envs=1,
          batch_size=4096,
          validate_freq=10_000,
          use_wandb=True)


def evaluate_controller():
    inspect_training_simu(env_name="Car",
                          algo="e2e",
                          n_rollout=20,
                          render=True)


if __name__ == '__main__':
    learn()
    #evaluate_controller()
