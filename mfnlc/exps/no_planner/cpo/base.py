import sys
import importlib.util

SPEC_OS = importlib.util.find_spec('mfnlc')
os1 = importlib.util.module_from_spec(SPEC_OS)
SPEC_OS.loader.exec_module(os1)
sys.modules['shrl'] = os1

import os

import pandas as pd
import numpy as np

from torch.utils.tensorboard import SummaryWriter
from mfnlc.config import get_path
from mfnlc.envs import get_env
from mfnlc.envs.difficulty import choose_level
from mfnlc.evaluation.model import load_model
from mfnlc.evaluation.simulation import simu

ALGO = "cpo"


def evaluate(env_name,
             n_rollout: int = 1,
             n_steps: int = 1000,
             level: int = 1,
             arrive_radius: float = 0.3,
             render: bool = False,
             pretrained: bool = False):
    model = load_model(env_name, algo=ALGO, pretrained=pretrained)
    env = get_env(env_name)
    env.update_env_config({
        "obstacle_in_obs": 8
    })
    choose_level(env, level)

    if render:
        tensorboard_log = get_path(robot_name=env.robot_name,
                              algo=ALGO, task="video") + \
                              f" pretrained: {pretrained}"
        log_idx = 0
        while os.path.exists(tensorboard_log + f"_{log_idx}"):
            log_idx += 1
        tensorboard_log += f"_{log_idx}"
        writer = SummaryWriter(tensorboard_log)
        print("log dir:", tensorboard_log)

    i = 0
    visual_episodes = [i for i in range(n_rollout) if (i + 1) % 5 == 0]
    #visual_episodes = [0]
    print("visual_episodes:", visual_episodes)
    running_data = {
        "total_step": [],
        "goal_met": [],
        "collision": [],
        "reward_sum": [],
        "cost_sum": []
    }
    while i < n_rollout:
        print(f"********** task {i}:")
        env.reset()
        res = simu(env=env,
                   model=model,
                   n_steps=n_steps,
                   path=None,
                   arrive_radius=arrive_radius,
                   render=(render and (i in visual_episodes)))
        for k in res:
            if k == "screens":
                if render and (i in visual_episodes):
                    print("add video to tensorboard")
                    writer.add_video('eval_trajectory', res[k], global_step=i, fps=30)
                continue
            print(f"{k}:", res[k])
            running_data[k].append(res[k])
        del res
        i += 1
        print("**********")
        print()

    stat = pd.DataFrame(running_data)
    print("result data:", running_data)
    for k in running_data:
        print(f"mean {k}", np.mean(running_data[k]))

    os.makedirs(tensorboard_log, exist_ok=True)
    stat.to_csv(tensorboard_log + f"/{level}.csv")
    print("results are saved to:", tensorboard_log + f"/{level}.csv")

    print(stat)
    writer.close()
    env.close()
    return stat
