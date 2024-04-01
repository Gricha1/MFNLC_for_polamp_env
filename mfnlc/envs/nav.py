import random
from typing import Dict
import copy

import gym
import matplotlib.pyplot as plt
import numpy as np

from mfnlc.envs.base import EnvBase

CUSTOM_DATASET = False
FIXED_HAZARDS = False
DIFFICULTY_LEVEL = 1
OBSTACLES_IN_OBSERVATION = 8
FRAME_STACK = 1
COLLISION_PENALTY = -60
ENV_BOUNDS = False
PLOT_ADD_SUBGOAL_VALUES = False
PLOT_ONLY_START_GOAL_POSE = False
PLOT_SUBGOAL_s_to_sg = True

class Continuous2DNav(EnvBase):

    def __init__(self,
                 no_obstacle=False,
                 end_on_collision=False,
                 fixed_init_and_goal=False):
        super(Continuous2DNav, self).__init__(no_obstacle,
                                              end_on_collision,
                                              fixed_init_and_goal)

        self.arrive_radius = 0.1
        self.robot_radius = 0.1
        self.obstacle_num = 20
        self.obstacle_in_obs = 2
        self.obstacle_radius = 0.15
        self.collision_penalty = -0.01
        self.arrive_reward = 0
        self.step_size = 0.01
        self.robot_name = "Nav"

        self.goal_size = 500
        self.subgoal_size = 100

        self.floor_lb = np.array([-1., -1.], dtype=np.float32)
        self.floor_ub = np.array([1., 1.], dtype=np.float32)

        self.init = None
        self.goal = None
        self.robot_pos = None
        self.obstacle_centers = None
        self.prev_subgoal_num = 0

        self._build_space()
        self._build_sample_space()
        self.prev_vec_to_goal = None

        #self.fig, self.ax = None, None
        self.robot_patch = None
        self.roa_patch = None

    @property
    def hazards_pos(self):
        return self.obstacle_centers

    def _build_space(self):
        action_high = np.ones(2, dtype=np.float32)
        action_low = -action_high
        self.action_space = gym.spaces.Box(action_low, action_high, dtype=np.float32)

        if self.no_obstacle:
            observation_high = 2 * self.floor_ub * np.ones(2, dtype=np.float32)
        else:
            observation_high = 2 * np.ones(2 + self.obstacle_in_obs * 2, dtype=np.float32)
            observation_high = (observation_high.reshape([-1, 2]) * self.floor_ub).flatten()
        observation_low = -observation_high
        self.observation_space = gym.spaces.Box(observation_low, observation_high, dtype=np.float32)

    def _build_sample_space(self):
        self.position_list = []

        x_lb, y_lb = self.floor_lb
        x_ub, y_ub = self.floor_ub
        if self.no_obstacle:
            # no obstacle env is only used for training. Use small map during training.
            grid_num_per_line = int(0.3 * (x_ub - x_lb) / (self.robot_radius * 4))
            x = np.linspace(0.3 * x_lb, 0.3 * x_ub, num=grid_num_per_line, dtype=np.float32)
            y = np.linspace(0.3 * y_lb, 0.3 * y_ub, num=grid_num_per_line, dtype=np.float32)
        else:
            grid_num_per_line = int((x_ub - x_lb) / (self.robot_radius * 4))
            x = np.linspace(0.95 * x_lb, 0.95 * x_ub, num=grid_num_per_line, dtype=np.float32)
            y = np.linspace(0.95 * y_lb, 0.95 * y_ub, num=grid_num_per_line, dtype=np.float32)

        xv, yv = np.meshgrid(x, y)
        for i in range(len(x)):
            for j in range(len(y)):
                self.position_list.append([xv[i, j], yv[i, j]])

    def _generate_map(self):
        if not self.fixed_init_and_goal:
            if self.no_obstacle:
                positions = random.sample(self.position_list, 2)
            else:
                positions = random.sample(self.position_list, 2 + self.obstacle_num)
                self.obstacle_centers = np.array(positions[2:])

            self.init = np.array(positions[0], dtype=np.float32)
            self.goal = np.array(positions[1], dtype=np.float32)
        else:
            if self.goal is None or self.init is None:
                positions = random.sample(self.position_list, 2)
                self.init = np.array(positions[0], dtype=np.float32)
                self.goal = np.array(positions[1], dtype=np.float32)

            if not self.no_obstacle:
                positions = random.sample(self.position_list, self.obstacle_num)

                # Lazy implementation. This may make the obstacle numbers be smaller
                if self.init.tolist() in positions:
                    positions.remove(self.init.tolist())  # noqa
                if self.goal.tolist() in positions:
                    positions.remove(self.goal.tolist())  # noqa
                self.obstacle_centers = np.array(positions)

    def update_env_config(self, config: Dict):
        self.__dict__.update(config)
        self._build_sample_space()
        self._build_space()
        # self.reset()

    def seed(self, seed=None):
        np.random.seed(seed)
        random.seed(seed)

    def reset(self):
        super(Continuous2DNav, self).reset()

        self._generate_map()
        self.robot_pos = self.init

        self.prev_vec_to_goal = None
        self.prev_subgoal_num = 0

        #plt.close("all")
        #if self.fig is not None:
        #    self.fig, self.ax = None, None

        return self.get_obs()

    def goal_obs(self) -> np.ndarray:
        #if self.subgoal is not None:
        #    goal_obs = self.subgoal - self.robot_pos
        #else:
        goal_obs = self.goal - self.robot_pos
        return goal_obs

    def robot_obs(self) -> np.ndarray:
        return np.array([])  # 2d nav does not care about the robot's posture

    def obstacle_obs(self) -> np.ndarray:
        if not self.no_obstacle:
            vec_to_obs = self.obstacle_centers - self.robot_pos
            dist_to_obs = np.linalg.norm(vec_to_obs, ord=2, axis=-1)
            order = dist_to_obs.argsort()[:self.obstacle_in_obs]

            return vec_to_obs[order].flatten()
        else:
            return np.array([])

    def collision_detection(self):
        if self.no_obstacle:
            return False

        closest_dist = np.min(np.linalg.norm(
            self.obstacle_centers - self.robot_pos, axis=-1, ord=2))
        return closest_dist < self.robot_radius + self.obstacle_radius

    def arrive(self):
        return np.linalg.norm(self.goal - self.robot_pos, ord=2) < self.arrive_radius

    def step(self, action: np.ndarray):
        self.robot_pos += action.clip(self.action_space.low,
                                      self.action_space.high) * self.step_size
        self.robot_pos = self.robot_pos.clip(self.floor_lb, self.floor_ub)

        vec_to_goal = self.goal - self.robot_pos
        if self.prev_vec_to_goal is None:
            goal_reward = 0
        else:
            vel_vec = self.prev_vec_to_goal - vec_to_goal
            vec_cos = np.dot(vel_vec, self.prev_vec_to_goal) \
                      / np.sqrt(np.linalg.norm(self.prev_vec_to_goal) + np.linalg.norm(vel_vec))
            goal_reward = vec_cos
        self.prev_vec_to_goal = vec_to_goal

        collision = self.collision_detection()
        arrive = self.arrive()

        if self.end_on_collision and collision:
            done = True
        else:
            done = arrive

        reward = goal_reward + collision * self.collision_penalty + arrive * self.arrive_reward

        self.traj.append(self.robot_pos)

        return self.get_obs(), reward, done, {"collision": collision, "goal_met": arrive}

    def render(self, mode="human"):
        if self.fig is None:
            self.fig, self.ax = plt.subplots(figsize=(10, 10))

            if not self.no_obstacle:
                for obstacle_center in self.obstacle_centers:
                    obstacle_patch = plt.Circle(
                        obstacle_center, radius=self.obstacle_radius, color="blue", alpha=0.5)
                    self.ax.add_patch(obstacle_patch)

            self.robot_patch = plt.Circle(
                self.robot_pos, radius=self.robot_radius, color="red", alpha=0.5)  # noqa
            self.ax.add_patch(self.robot_patch)

            self.roa_patch = plt.Circle(
                self.robot_pos, radius=self.robot_radius, color="cyan", alpha=0.5)
            self.ax.add_patch(self.roa_patch)

            self.ax.scatter(*self.goal, s=self.goal_size, marker='o', color='green', alpha=0.5)

            self.ax.set_xlim(self.floor_lb[0], self.floor_ub[0])
            self.ax.set_ylim(self.floor_lb[1], self.floor_ub[1])
            plt.axis('off')

        if len(self.subgoal_list) != self.prev_subgoal_num:
            self.ax.scatter(*self.subgoal_list[-1], s=self.subgoal_size, marker='o', color='green', alpha=0.5)
            self.prev_subgoal_num = len(self.subgoal_list)

        if len(self.traj) % self.render_config["traj_sample_freq"] == 0 and len(self.traj) > 0:
            self.ax.scatter(*self.traj[-1], s=self.subgoal_size / 3, marker='o', color='gold', alpha=0.5)

        self.robot_patch.center = self.robot_pos

        if self.roa_center is not None:
            self.roa_patch.center = self.roa_center
            self.roa_patch.radius = self.roa_radius

        self.fig.canvas.draw()

        if mode == "human":
            plt.pause(0.001)
        elif mode == "rgb_array":
            data = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
            data = data.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))

            return data


class GCContinuous2DNav(Continuous2DNav):
    def __init__(self,
                 no_obstacle=False,
                 end_on_collision=False,
                 fixed_init_and_goal=False,
                 max_episode_steps=100) -> None:
        super().__init__(no_obstacle=no_obstacle,
                        end_on_collision=end_on_collision,
                        fixed_init_and_goal=fixed_init_and_goal)
        assert self.num_relevant_dim == 2 # goal x, y
        class EnvSpec():
            def __init__(self):
                self.max_episode_steps = max_episode_steps
        self.spec = EnvSpec()

    def robot_goal_obs(self) -> np.ndarray:
        """
            'accelerometer', 'velocimeter', 'gyro', 
            'magnetometer', 'goal_lidar', 'hazards_lidar', 
            'vases_lidar'

            "accelerometer_z" should be 9.81, everything else is 0
        """
        # only gets observation dimensions relevant to robot from safety-gym
        obs = self.env.obs()
        flat_obs = np.zeros(self.robot_obs_size)
        offset = 0

        for k in sorted(self.env.obs_space_dict.keys()):
            if "lidar" in k:
                continue
            k_size = np.prod(obs[k].shape)
            if not "accelerometer" in k:
                continue
            if "accelerometer" in k:
                copy_obs = copy.deepcopy(obs[k])
                copy_obs[:2] = 0 # acc_x, acc_y, acc_z = 0, 0, 9.81
                flat_obs[offset:offset + k_size] = copy_obs.flat
            offset += k_size
        return flat_obs
    
    def get_obs(self, arrive):
        if len(self.state_history) >= self.history_len:
            self.state_history.popleft()
        if len(self.goal_history) >= self.history_len:
            # if the robot meets goal, the goal will be reset immediately
            # this can cause the goal observation has large jumps and affect Lyapunov function
            if not arrive:
                self.goal_history.popleft()
            else:
                print("we should not remove anything because the goal was changed")
                print(f"current goal: {self.env.goal_pos[:self.num_relevant_dim]}")
                print(f"old goal: {self.goal_history[0][:self.num_relevant_dim]}")
                print(f"current pose: {self.env.robot_pos[:self.num_relevant_dim]}")
                distance = np.sqrt(np.power(np.array(self.env.robot_pos[:self.num_relevant_dim]) - np.array(self.goal_history[0][:self.num_relevant_dim]), 2).sum(-1, keepdims=True))
                print(f"distance: {distance} and threshold: {self.env.goal_size}")

        state = np.concatenate([
                               self.env.robot_pos[:self.num_relevant_dim],
                               self.robot_obs(), # absolute robot acc, velocities
                               self.obstacle_obs(), # obsts with respect to obs
                               ])
        goal = np.concatenate([
                               self.env.goal_pos[:self.num_relevant_dim],
                               self.robot_goal_obs(), # absolute goal acc, velocities
                               self.obstacle_goal_obs() # obsts with respect to goal
                               ])
        
        while len(self.state_history) < self.history_len:
            self.state_history.append(state)
        
        while len(self.goal_history) < self.history_len:
            self.goal_history.append(goal)
        
        collision = False
        clearance_is_enough = False
        return {
            "observation": np.concatenate(self.state_history),
            "desired_goal": np.concatenate(self.goal_history),
            "achieved_goal": np.concatenate(self.state_history),
            "collision" : collision,
            "clearance_is_enough": clearance_is_enough,
        }


    def reset(self):
        obs = super().reset()

        #plt.close("all")
        #if self.fig is not None:
        #    self.fig, self.ax = None, None

        return self.get_obs()


    def _build_space(self):
        action_high = np.ones(2, dtype=np.float32)
        action_low = -action_high
        self.action_space = gym.spaces.Box(action_low, action_high, dtype=np.float32)

        max_observation = 2
        observation_high = max_observation * np.ones(
            ((self.num_relevant_dim + self.obstacle_in_obs * self.num_relevant_dim) * self.frame_stack),
            dtype=np.float32)
        observation_low = -observation_high
        self.observation_space = gym.spaces.Dict({
            "observation": gym.spaces.Box(observation_low, observation_high, dtype=np.float32),
            "desired_goal": gym.spaces.Box(observation_low, observation_high, dtype=np.float32),
            "achieved_goal": gym.spaces.Box(observation_low, observation_high, dtype=np.float32),
            "collision": gym.spaces.Box(0.0, 1.0, (1,), np.float32),
            "clearance_is_enough": gym.spaces.Box(0.0, 1.0, (1,), np.float32)
        })


class NavCustomTimeLimit(GCContinuous2DNav):
    def step(self, action):
        assert self._elapsed_steps is not None, "Cannot call env.step() before calling reset()"
        observation, reward, done, info = super().step(action)
        self._elapsed_steps += 1
        if self._elapsed_steps >= self.spec.max_episode_steps:
            info['TimeLimit.truncated'] = not done
            done = True
        info["done"] = done
        return observation, reward, done, info

    def reset(self, **kwargs):
        self._elapsed_steps = 0
        return super().reset(**kwargs)