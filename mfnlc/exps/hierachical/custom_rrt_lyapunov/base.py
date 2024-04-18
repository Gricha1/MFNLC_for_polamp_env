import sys
import importlib.util

SPEC_OS = importlib.util.find_spec('mfnlc')
os1 = importlib.util.module_from_spec(SPEC_OS)
SPEC_OS.loader.exec_module(os1)
sys.modules['shrl'] = os1

import os
import random
from typing import Dict

import numpy as np
import pandas as pd
from gym.wrappers.monitor import Monitor as VideoMonitor
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3.common.logger import Video
#from PIL import Image
from stable_baselines3.common.logger import Image
import torch as th

from mfnlc.config import get_path
from mfnlc.envs import get_env
from mfnlc.envs.base import ObstacleMaskWrapper
from mfnlc.envs.difficulty import choose_level
from mfnlc.evaluation.model import load_model
from mfnlc.evaluation.simulation import simu
from mfnlc.learn.lyapunov_td3 import LyapunovTD3
from mfnlc.monitor.monitor import Monitor, LyapunovValueTable
from mfnlc.plan import Planner
from mfnlc.plan.common.plot import plot_path_2d

ALGO = "rrt_lyapunov"


def evaluate(env_name,
             n_rollout: int = 1,
             n_steps: int = 1000,
             level: int = 1,
             planning_algo: str = "rrt*",
             planner_max_iter: int = 200,
             planning_algo_kwargs: Dict = {},  # noqa
             arrive_radius: float = 0.1,
             monitor_max_step_size: float = 0.2,
             monitor_search_step_size: float = 0.01,
             render: bool = False,
             check_plan: bool = False,
             video: bool = False,
             render_config: Dict = {},  # noqa
             seed: int = None, 
             pretrained: bool = False,
             ):
    model: LyapunovTD3 = load_model(env_name, algo=ALGO, pretrained=pretrained)
    env = ObstacleMaskWrapper(get_env(env_name))

    env.seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    choose_level(env, level) # doesnt work fix position

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

    if video:
        video_path = get_path(robot_name=env.robot_name,
                              algo=ALGO, task="video") + f"{planning_algo}-level-{level}"
        env = VideoMonitor(env, video_path, force=True)
    planner = Planner(env, planning_algo)
    lv_table = LyapunovValueTable.load(get_path(env.robot_name, ALGO, "lv_table", pretrained=pretrained))
    monitor = Monitor(lv_table, max_step_size=monitor_max_step_size, search_step_size=monitor_search_step_size)
    monitor = None

    i = 0
    visual_episodes = [i for i in range(n_rollout) if (i + 1) % 5 == 0]
    print("visual_episodes:", visual_episodes)
    
    running_data = {
        "total_step": [],
        "goal_met": [],
        "collision": [],
        "reward_sum": [],
        "cost_sum": []
    }
    replan_idxs = []

    re_plan = False
    while i < n_rollout:
        print(f"********** task {i}:")
        if not re_plan:
            # reset video monitor
            env.reset()
        else:
            env.unwrapped.reset()
        path = planner.plan(planner_max_iter, **planning_algo_kwargs)
        if check_plan:
            image_plan = plot_path_2d(planner.algo.search_space, path, planner.algo.tree)
        if len(path) == 0:
            re_plan = True
            replan_idxs.append(i)
            continue
        re_plan = False

        res = simu(env=env,
                   model=model,
                   n_steps=n_steps,
                   path=path,
                   arrive_radius=arrive_radius,
                   monitor=monitor,
                   render=(render and (i in visual_episodes)),
                   render_config=render_config,
                   planner=planner)
        for k in res:
            if render and k == "screens" and (i in visual_episodes):
                print("add video to tensorboard")
                writer.add_video('eval_trajectory', res[k], global_step=i, fps=30)
                continue
            print(f"{k}:", res[k])
            running_data[k].append(res[k])
        i += 1
        print("**********")
        print()

    stat = pd.DataFrame(running_data)
    print("result data:", running_data)
    print("replan_idxs:", replan_idxs)
    for k in running_data:
        print(f"mean {k}", np.mean(running_data[k]))
    
    stat.to_csv(tensorboard_log + f"/{level}.csv")
    print("results are saved to:", tensorboard_log + f"/{level}.csv")

    for key_ in running_data:
        writer.add_scalar(f'testing/mean_{key_}', np.mean(running_data[key_]), 0)

    writer.close()
    env.close()
    return stat, env.env.traj


def build_lyapunov_table(env_name: str,
                         obs_lb: np.ndarray,
                         obs_ub: np.ndarray,
                         n_levels: int = 10,
                         pgd_max_iter: int = 100,
                         pgd_lr: float = 1e-3,
                         n_range_est_sample: int = 10,
                         n_radius_est_sample: int = 10,
                         bound_cnst: float = 100,
                         pretrained: bool = False):
    model: LyapunovTD3 = load_model(env_name, algo=ALGO, pretrained=pretrained)
    lv_table = LyapunovValueTable(model.tclf,
                                  obs_lb,
                                  obs_ub,
                                  n_levels=n_levels,
                                  pgd_max_iter=pgd_max_iter,
                                  pgd_lr=pgd_lr,
                                  n_range_est_sample=n_range_est_sample,
                                  n_radius_est_sample=n_radius_est_sample,
                                  bound_cnst=bound_cnst)
    lv_table.build()
    print(lv_table.lyapunov_values)
    print(lv_table.lyapunov_radius)
    robot_name = env_name.split("-")[0]
    lv_table.save(get_path(robot_name, algo=ALGO, task="lv_table", pretrained=pretrained))
