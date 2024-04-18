from typing import Dict

import numpy as np
import torch as th

from mfnlc.config import env_config
from mfnlc.envs import get_env
from mfnlc.evaluation.model import load_model
from mfnlc.monitor.monitor import Monitor
from mfnlc.plan.common.path import Path
from mfnlc.plan.common.plot import plot_path_2d
from mfnlc.plan import Planner


def inspect_training_simu(env_name: str,
                          algo: str,
                          n_rollout: int,
                          render: bool = False):
    env = get_env(env_name)
    robot_name = env_name.split("-")[0]

    model = load_model(env_name, algo)

    simu_data = {
        "rewards": [],
        "obs": [],
        "actions": [],
        "infos": []
    }

    for ep in range(n_rollout):
        reward_list = []
        obs_list = []
        action_list = []
        info_list = []

        obs = env.reset()
        obs_list.append(obs)
        for i in range(env_config[robot_name]["max_step"]):
            action = model.predict(obs)[0]
            obs, reward, done, info = env.step(action)

            action_list.append(action)
            obs_list.append(obs)
            reward_list.append(reward)
            info_list.append(info)

            if render:
                env.render()

            if done:
                break

        simu_data["obs"].append(obs_list)
        simu_data["actions"].append(action_list)
        simu_data["rewards"].append(reward_list)
        simu_data["infos"].append(info_list)

    return simu_data


def simu(env,
         model,
         n_steps: int,
         path: Path = None,
         arrive_radius: float = 0.0,
         monitor: Monitor = None,
         render: bool = False,
         render_config: Dict = {},  # noqa
         planner: Planner = None
         ):
    obs = env.get_obs()

    if monitor is not None:
        monitor.reset()

    subgoal_index = 0
    if path is not None:
        env.set_subgoal(path[subgoal_index])

    env.set_render_config(render_config)
    if render:
        screens = []

    total_step = 0
    goal_met = False
    collision = False
    reward_sum = 0.0
    cost_sum = 0.0

    for i in range(n_steps):
        action = model.predict(obs)[0]
        obs, reward, done, info = env.step(action)
        assert not info.get("collision") or (info.get("collision") and done)
        total_step += 1
        reward_sum += reward
        cost_sum = info["episode_cost"]

        if path is not None:
            if np.linalg.norm(env.robot_pos - path[subgoal_index]) < arrive_radius:
                subgoal_index += 1
                subgoal_index = min(len(path) - 1, subgoal_index)
                subgoal = path[subgoal_index]
                env.set_subgoal(subgoal, store=True)
            else:
                subgoal = path[subgoal_index]

            if monitor is not None:
                subgoal, lyapunov_r = monitor.select_subgoal(env, subgoal)
            env.set_subgoal(subgoal, store=False)
            #env.set_roa(subgoal, lyapunov_r)  # noqa

        step_info = {"t": i, 
                     "acc_reward": reward_sum, 
                     "acc_cost": cost_sum, 
                    }
        if render:
            if (planner is None): # cpo
                screen = plot_path_2d(None, path, None, 
                                      plot_current_pose=True, 
                                      curret_pose=env.robot_pos, 
                                      goal_pose=env.env.goal_pos[:2],
                                      obst_poses=env.env.hazards_pos, 
                                      step_info=step_info) 
            else: # lyapunov
                screen = plot_path_2d(planner.algo.search_space, path, planner.algo.tree, 
                                    plot_current_pose=True, curret_pose=env.robot_pos, 
                                    step_info=step_info)            
            screens.append(screen)

        if done:
            goal_met = info.get("goal_met", False)
            collision = info.get("collision", False)
            break

    if not render:
        return {"total_step": total_step,
                "collision": collision,
                "goal_met": goal_met,
                "reward_sum": reward_sum,
                "cost_sum": cost_sum}
    else:
        screens = np.transpose(np.array(screens), axes=[0, 3, 1, 2])
        return {"total_step": total_step,
                "collision": collision,
                "goal_met": goal_met,
                "reward_sum": reward_sum,
                "cost_sum": cost_sum,
                "screens" : th.ByteTensor([screens])}
