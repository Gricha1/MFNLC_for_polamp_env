from typing import List

import numpy as np
#from safety_gym.envs.engine import Engine

#from mfnlc.envs import Continuous2DNav
#from mfnlc.envs.base import SafetyGymBase
from mfnlc.plan.common.geometry import ObjectBase, Circle, Polygon


class SearchSpace:
    def __init__(self,
                 lb: np.ndarray,
                 ub: np.ndarray,
                 initial_state: np.ndarray,
                 goal_state: np.ndarray,
                 obstacles: List[ObjectBase]):
        self.lb = lb
        self.ub = ub
        self.initial_state = initial_state
        self.goal_state = goal_state
        self.obstacles = obstacles

    @classmethod
    def build_from_env(cls, env) -> 'SearchSpace':

        lb, ub = env.get_constained_agent_bounds()
        
        agent = env.environment.agent.current_state
        initial_state = np.array([agent.x, agent.y, agent.theta, agent.v, agent.steer])
        goal = env.environment.agent.goal_state
        goal_state = np.array([goal.x, goal.y, goal.theta, goal.v, goal.steer])

        polygon = True
        with_dubins_curve = True
        if polygon:
            if not with_dubins_curve:
                lb[2] = 0
                ub[2] = 0
            assert -np.pi <= initial_state[2] <= np.pi
            assert -np.pi <= goal_state[2] <= np.pi
            assert len(initial_state) == 5
            assert len(goal_state) == 5
            for i in range(3, 5):
                initial_state[i] = 0
                goal_state[i] = 0
            obstacles_centers = [obst for obst in env.maps[env.map_key]]
            obstacles = [Polygon(obst, w=obst[3], l=obst[4]) for obst in obstacles_centers]
        else:
            obstacles_centers = [obst[:2] for obst in env.maps[env.map_key]]
            obstacle_radius = 5
            obstacles = [Circle(obst, radius=obstacle_radius) for obst in obstacles_centers]
        
        return SearchSpace(lb, ub, initial_state, goal_state, obstacles)

    def sample(self, n_samples: int) -> np.ndarray:
        return np.random.uniform(self.lb, self.ub, (n_samples,) + self.lb.shape)
