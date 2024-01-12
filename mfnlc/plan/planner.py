from typing import Dict

import numpy as np
#from safety_gym.envs.engine import Engine

from mfnlc.config import env_config
#from mfnlc.envs import Continuous2DNav
#from mfnlc.envs.base import SafetyGymBase
from mfnlc.plan.common.geometry import Circle
from mfnlc.plan.common.geometry import Polygon
from mfnlc.plan.common.path import Path
from mfnlc.plan.common.space import SearchSpace
from mfnlc.plan.rrt import RRT
from mfnlc.plan.rrt_star import RRTStar


class Planner:
    def __init__(self,
                 env,
                 algo_name: str = "rrt*"):
        self.env = env
        self.algo_name = algo_name
        self.algo = None
        self.support_margin = 0.0

    def plan(self,
             max_iteration: int,
             support_margin: int = 0.0,
             heuristic=None,
             n_sample=1,
             ) -> Path:
        if self.algo is None:
            self.algo = self._build_planning_algorithm(support_margin)
        else:
            self.support_margin = support_margin
            search_space = SearchSpace.build_from_env(self.env)
            self.algo.set_search_space(search_space)

        return self.algo.search(max_iteration, heuristic, n_sample)

    def _build_planning_algorithm(self, support_margin: float):
        search_space = SearchSpace.build_from_env(self.env)
        robot, arrive_radius, collision_checker_resolution = self._extract_robot_info_from_env(support_margin)

        if self.algo_name == "rrt":
            algo = RRT(search_space, robot, arrive_radius, collision_checker_resolution)
        elif self.algo_name == "rrt*":
            algo = RRTStar(search_space, robot, arrive_radius, collision_checker_resolution)
        else:
            raise NotImplementedError()

        return algo

    def _extract_robot_info_from_env(self, support_margin: float):
        """
        if isinstance(self.env.unwrapped, Continuous2DNav):
            env: Continuous2DNav = self.env.unwrapped
            assert env.robot_pos is not None, "reset env first"
            robot_radius = env.robot_radius + support_margin
            arrive_radius = env.arrive_radius
            resolution = np.max(env.floor_ub - env.floor_lb) / 100
        elif issubclass(type(self.env.unwrapped), SafetyGymBase):
            #env: Engine = self.env.unwrapped.env
            env = self.env.unwrapped.env
            assert not env.done, "reset env first"
            robot_radius = env_config[self.env.robot_name]["robot_radius"]
            arrive_radius = env.goal_size
            floor_extends = env.placements_extents
            lb = np.array(floor_extends[:2])
            ub = np.array(floor_extends[2:])
            resolution = np.max(ub - lb) / 100
        else:
            raise NotImplementedError()
        """
        env = self.env
        arrive_radius = env.SOFT_EPS
        lb, ub = env.get_constained_agent_bounds()
        resolution = np.max(ub - lb) / 100
        agent = env.environment.agent.current_state
        initial_state = np.array([agent.x, agent.y, agent.theta, agent.v, agent.steer])

        polygon = True
        if polygon:
            safe_w = 0 # 2 * 0.8
            safe_l = 0 # 2 * 0.5
            robot_w = env.environment.agent.dynamic_model.width
            robot_l = env.environment.agent.dynamic_model.length
            robot = Polygon(initial_state, w=robot_w+safe_w, l=robot_l+safe_l)
            robot.center_state = False
            robot.turning_radius = env.environment.agent.dynamic_model.wheel_base / np.tan(env.environment.agent.dynamic_model.max_steer) # 2.5
            #robot.turning_radius += 0.5
        else:
            robot_radius = env.environment.agent.dynamic_model.width / 2
            robot = Circle(initial_state[:2], robot_radius)

        return robot, arrive_radius, resolution
