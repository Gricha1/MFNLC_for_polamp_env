import random
import math
from typing import Dict
import copy
from collections import deque

import gym
import matplotlib.pyplot as plt
import numpy as np

from mfnlc.envs.base import EnvBase

CUSTOM_DATASET = False
FIXED_HAZARDS = False
DIFFICULTY_LEVEL = 1 
OBSTACLES_IN_OBSERVATION = 4 
FRAME_STACK = 1
COLLISION_PENALTY = -120
ENV_BOUNDS = False
PLOT_ADD_SUBGOAL_VALUES = False
PLOT_ONLY_START_GOAL_POSE = False
PLOT_SUBGOAL_s_to_sg = True
PLOT_SUBGOAL = True

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
        self.obstacle_in_obs = OBSTACLES_IN_OBSERVATION
        self.obstacle_radius = 0.09
        self.collision_penalty = -0.01
        self.arrive_reward = 0
        self.step_size = 0.01
        self.robot_name = "Nav"

        # using for default env rendering
        #self.goal_size = 500
        #self.subgoal_size = 100

        if DIFFICULTY_LEVEL == -1: # default
            self.obstacle_num = 20
            self.floor_lb = np.array([-1., -1.], dtype=np.float32)
            self.floor_ub = np.array([1., 1.], dtype=np.float32)
        elif DIFFICULTY_LEVEL == 1:
            self.obstacle_num = 8
            #self.floor_lb = np.array([-2., -2.], dtype=np.float32)
            #self.floor_ub = np.array([2., 2.], dtype=np.float32)
            self.floor_lb = np.array([-1., -1.], dtype=np.float32)
            self.floor_ub = np.array([1., 1.], dtype=np.float32)
        else:
            assert 1 == 0, "didnt implemented"

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

        return self.get_obs(False)

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
        self.num_relevant_dim = 2
        self.frame_stack = FRAME_STACK
        super().__init__(no_obstacle=no_obstacle,
                        end_on_collision=end_on_collision,
                        fixed_init_and_goal=fixed_init_and_goal)
        assert self.num_relevant_dim == 2 # goal x, y
        class EnvSpec():
            def __init__(self):
                self.max_episode_steps = max_episode_steps
        self.spec = EnvSpec()

        # Reward config
        self.collision_penalty = COLLISION_PENALTY
        self.arrive_reward = 0
        self.time_step_reward = -1

        self.subgoal_pos = None
        self.subgoal_s_to_sg_pos = None
        self.obstacle_observation = None
        self.obstacle_goal_observation = None
        self.render_info = {}
        self.render_info["fig"] = None
        self.render_info["ax_states"] = None
        self.plot_subgoal = PLOT_SUBGOAL
        self.plot_only_start_goal_pose = PLOT_ONLY_START_GOAL_POSE # use in envs/train/obstacles/ris/base.pys

        self.hazards_num = self.obstacle_num
        self.state_history = deque([])
        self.goal_history = deque([])
        self.history_len = self.frame_stack

    def obstacle_obs(self) -> np.ndarray:
        if self.no_obstacle:
            self.obstacle_observation = np.zeros(self.num_relevant_dim * self.obstacle_in_obs)
            return self.obstacle_observation

        # get distance to each obstacle upto self.obstacle_in_obs nearest obstacles
        vec_to_obs = (self.hazards_pos - self.robot_pos)[:, :self.num_relevant_dim]
        dist_to_obs = np.linalg.norm(vec_to_obs, ord=2, axis=-1)
        order = dist_to_obs.argsort()[:self.obstacle_in_obs]
        flattened_vec = vec_to_obs[order].flatten()
        # in case of that the obstacle number in environment is smaller than self.obstacle_in_obs
        output = np.zeros(self.obstacle_in_obs * self.num_relevant_dim)
        output[:flattened_vec.shape[0]] = flattened_vec
        self.obstacle_observation = output
        return output
        # obs = self.env.obs()
        # return obs["hazards_lidar"]
    
    def robot_goal_obs(self) -> np.ndarray:
        return np.array([])  # 2d nav does not care about the robot's posture
    
    def obstacle_goal_obs(self) -> np.ndarray:
        """
            get obstacle observation with respect to goal
        """
        if self.no_obstacle:
            self.obstacle_goal_observation = np.zeros(self.num_relevant_dim * self.obstacle_in_obs)
            return self.obstacle_goal_observation

        # get distance to each obstacle upto self.obstacle_in_obs nearest obstacles
        vec_to_obs = (self.hazards_pos - self.goal)[:, :self.num_relevant_dim]
        dist_to_obs = np.linalg.norm(vec_to_obs, ord=2, axis=-1)
        order = dist_to_obs.argsort()[:self.obstacle_in_obs]
        flattened_vec = vec_to_obs[order].flatten()
        # in case of that the obstacle number in environment is smaller than self.obstacle_in_obs
        output = np.zeros(self.obstacle_in_obs * self.num_relevant_dim)
        output[:flattened_vec.shape[0]] = flattened_vec
        self.obstacle_goal_observation = output
        return output        
    
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
                print(f"current goal: {self.goal[:self.num_relevant_dim]}")
                print(f"old goal: {self.goal_history[0][:self.num_relevant_dim]}")
                print(f"current pose: {self.robot_pos[:self.num_relevant_dim]}")
                distance = np.sqrt(np.power(np.array(self.robot_pos[:self.num_relevant_dim]) - np.array(self.goal_history[0][:self.num_relevant_dim]), 2).sum(-1, keepdims=True))
                print(f"distance: {distance} and threshold: {self.arrive_radius}")

        state = np.concatenate([
                               self.robot_pos[:self.num_relevant_dim],
                               self.robot_obs(), # absolute robot acc, velocities
                               self.obstacle_obs(), # obsts with respect to obs
                               ])
        goal = np.concatenate([
                               self.goal[:self.num_relevant_dim],
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


    def reset(self, **kwargs):
        # check env config
        self.state_history.clear()
        self.goal_history.clear()
        if self.no_obstacle:
            assert self.hazards_num == 0, "empty env has no obstacles"
        else:
            assert self.hazards_num > 0, "env with obstacles should have obstacles"
        
        self.subgoal_pos = None
        self.subgoal_s_to_sg_pos = None
        obs = super().reset(**kwargs)
        assert not obs["collision"], "initial state in collision!!!"
        self.previous_min_goal_dist = np.linalg.norm(self.goal_obs(), ord=2)
        self.episode_cost = 0
        return obs


    def step(self, action: np.ndarray):
        # safety gym bug assert
        assert action.shape == self.action_space.low.shape
        info = {}
        
        # step in nav env
        self.robot_pos += action.clip(self.action_space.low,
                                      self.action_space.high) * self.step_size
        #self.robot_pos = self.robot_pos.clip(self.floor_lb, self.floor_ub)

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

        # As of now use safety gym info['cost'] to detect collisions
        info["collision"] = collision
        info["goal_met"] = arrive
        # check env config
        if self.no_obstacle:
            assert collision == False

        if ENV_BOUNDS:
            if self.robot_pos[0] < -2.0 or self.robot_pos[0] > 2.0 or \
                self.robot_pos[1] < -2.0 or self.robot_pos[1] > 2.0:
                collision = True

        reward = self.time_step_reward + self.collision_penalty * collision

        if self.end_on_collision and collision:
            done = True
        else:
            done = arrive or done

        obs = self.get_obs(arrive)
        obs["collision"] = collision
        
        # test
        shift_v = int(obs["observation"].shape[0] / self.frame_stack * (self.frame_stack - 1))
        test_reward = np.sqrt(np.power(np.array(obs["observation"] - obs["desired_goal"])[shift_v : shift_v+2], 2).sum(-1, keepdims=True)) # distance: next_state to goal
        test_arrive = 1.0 * (test_reward <= self.arrive_radius)# terminal condition
        if not arrive == test_arrive:
            assert 1 == 0

        self.traj.append(self.robot_pos)

        info["goal_is_arrived"] = arrive
        info["is_success"] = arrive
        goal_dist = np.linalg.norm(self.goal_obs(), ord=2)
        info["min_goal_distance"] = min(goal_dist, self.previous_min_goal_dist)
        self.previous_min_goal_dist = info["min_goal_distance"]

        # add cost
        info['clearance_is_enough'] = 0
        clearance_distance = self.obstacle_radius + self.robot_radius
        closest_dist = np.min(np.linalg.norm(
            self.obstacle_centers - self.robot_pos, axis=-1, ord=2))
        info['clearance_is_enough'] = float(closest_dist <= self.robot_radius + self.obstacle_radius)
    
        if not collision:
            self.episode_cost += info["clearance_is_enough"]
        else:
            self.episode_cost += math.fabs(self.collision_penalty)

        return obs, reward, done, info

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

    def compute_rewards(self, new_actions, new_next_obs_dict):
        return self.time_step_reward * np.ones_like(new_actions[:, 0])
    
    def set_test_env(self):
        init = 0.9 * self.train_dataset["floor_lb"]
        goal = np.array([0.9, 0.8]) * self.train_dataset["floor_ub"]
        self.update_env_config({
            "robot_locations": [init.tolist()],
            "goal_locations": [goal.tolist()]
        })
    
    def set_eval_env(self):
        self.update_env_config({
            "robot_locations": [],
            "goal_locations": []
        })

    def set_subgoal_pos(self, subgoal_related_pos, s_to_sg=False):
        if s_to_sg:
            if self.subgoal_s_to_sg_pos:
                del self.subgoal_s_to_sg_pos
            self.subgoal_s_to_sg_pos = []
            shift_v = int(subgoal_related_pos[0][0].shape[0] / self.frame_stack * (self.frame_stack - 1))
            self.subgoal_s_to_sg_pos.append(subgoal_related_pos[0][0][0 + shift_v].item())
            self.subgoal_s_to_sg_pos.append(subgoal_related_pos[0][0][1 + shift_v].item())
        else:
            if self.subgoal_pos:
                del self.subgoal_pos
            self.subgoal_pos = []
            shift_v = int(subgoal_related_pos[0][0].shape[0] / self.frame_stack * (self.frame_stack - 1))
            self.subgoal_pos.append(subgoal_related_pos[0][0][0 + shift_v].item())
            self.subgoal_pos.append(subgoal_related_pos[0][0][1 + shift_v].item())

    def custom_render(self, positions_render=False, dubug_info={}, add_subgoal_values=PLOT_ADD_SUBGOAL_VALUES, shape=(600, 600)):
        if positions_render:
            env_min_x, env_max_x = -3, 3
            env_min_y, env_max_y = -3, 3
            if self.render_info["fig"] is None:
                if add_subgoal_values:
                    self.render_info["fig"] = plt.figure(figsize=[6.4*2, 4.8])
                    self.render_info["ax_states"] = self.render_info["fig"].add_subplot(121)
                    self.render_info["ax_subgoal_values"] = self.render_info["fig"].add_subplot(122)
                else:
                    self.render_info["fig"] = plt.figure(figsize=[6.4, 4.8])
                    self.render_info["ax_states"] = self.render_info["fig"].add_subplot(111)
            self.render_info["ax_states"].set_ylim(bottom=env_min_y, top=env_max_y)
            self.render_info["ax_states"].set_xlim(left=env_min_x, right=env_max_x)
            # robot pose
            x = self.robot_pos[0]
            y = self.robot_pos[1]
            circle_robot = plt.Circle((x, y), radius=self.robot_radius, color="g", alpha=0.5)
            self.render_info["ax_states"].add_patch(circle_robot) 
            self.render_info["ax_states"].scatter(x, y, color="red")
            self.render_info["ax_states"].text(x + 0.05, y + 0.05, "s")
            # env_obs = self.env.obs()
            # angle_space = np.linspace(0, 360, env_obs["hazards_lidar"].shape[0] + 1)[:-1]
            # for distance, angle in zip(env_obs["hazards_lidar"], angle_space):
            #     plt.plot([x, x + distance * math.cos(angle)],\
            #             [y, y + distance * math.sin(angle)],\
            #             '-', linewidth = 4, color='red')

            # subgoal
            if self.subgoal_pos is not None and PLOT_SUBGOAL:
                x = self.subgoal_pos[0]
                y = self.subgoal_pos[1]
                circle_robot = plt.Circle((x, y), radius=self.robot_radius, color="orange", alpha=0.5)
                self.render_info["ax_states"].add_patch(circle_robot)
                self.render_info["ax_states"].text(x + 0.05, y + 0.05, "s_g")
                if add_subgoal_values:
                    self.render_info["ax_subgoal_values"].plot(range(len(dubug_info["v_s_sg"])), dubug_info["v_s_sg"])
                    self.render_info["ax_subgoal_values"].plot(range(len(dubug_info["v_sg_g"])), dubug_info["v_sg_g"])
            if PLOT_SUBGOAL_s_to_sg and self.subgoal_s_to_sg_pos is not None:
                x = self.subgoal_s_to_sg_pos[0]
                y = self.subgoal_s_to_sg_pos[1]
                circle_robot = plt.Circle((x, y), radius=self.robot_radius / 3, color="orange", alpha=0.5)
                self.render_info["ax_states"].add_patch(circle_robot)

            # goal
            x = self.goal[0]
            y = self.goal[1]
            circle_robot = plt.Circle((x, y), radius=self.robot_radius, color="y", alpha=0.5)
            self.render_info["ax_states"].add_patch(circle_robot) 
            self.render_info["ax_states"].text(x + 0.05, y + 0.05, "g")
            # for distance, angle in zip(env_obs["goal_lidar"], angle_space):
            #     plt.plot([x, x + distance * math.cos(angle)],\
            #             [y, y + distance * math.sin(angle)],\
            #             '-', linewidth = 4, color='blue')
                
            # add obstacles
            obstacles = [plt.Circle(obs[:2], radius=self.obstacle_radius,  # noqa
                        color="b", alpha=0.5) for obs in self.hazards_pos]
            for obs in obstacles:
                self.render_info["ax_states"].add_patch(obs)
            x = self.robot_pos[0]
            y = self.robot_pos[1]
            self.obstacle_observation = np.reshape(self.obstacle_observation, (int(self.obstacle_observation.shape[0]/ 2), 2))
            for obs_coord in self.obstacle_observation:
                self.render_info["ax_states"].plot([x, x + obs_coord[0]],\
                        [y, y + obs_coord[1]],\
                        '-', linewidth = 2, color='red')
            x = self.goal[0]
            y = self.goal[1]
            self.obstacle_goal_observation = np.reshape(self.obstacle_goal_observation, (int(self.obstacle_goal_observation.shape[0]/ 2), 2))
            for obs_coord in self.obstacle_goal_observation:
                self.render_info["ax_states"].plot([x, x + obs_coord[0]],\
                        [y, y + obs_coord[1]],\
                        '-', linewidth = 2, color='green')
            # debug info
            if len(dubug_info) != 0:
                a0 = dubug_info["a0"]
                a1 = dubug_info["a1"]
                acc_reward = dubug_info["acc_reward"]
                t = dubug_info["t"]
                acc_cost = dubug_info["acc_cost"]
                self.render_info["ax_states"].text(env_max_x - 4.5, env_max_y - 0.3, f"a0:{int(a0*100)/100}")
                self.render_info["ax_states"].text(env_max_x - 3.5, env_max_y - 0.3, f"a1:{int(a1*100)/100}")
                self.render_info["ax_states"].text(env_max_x - 2.5, env_max_y - 0.3, f"R:{int(acc_reward*10)/10}")
                self.render_info["ax_states"].text(env_max_x - 1.5, env_max_y - 0.3, f"C:{int(acc_cost*10)/10}")
                self.render_info["ax_states"].text(env_max_x - 0.5, env_max_y - 0.3, f"t:{t}")

            # render img
            # self.render_info["fig"].savefig("example.png")
            self.render_info["fig"].canvas.draw()
            data = np.frombuffer(self.render_info["fig"].canvas.tostring_rgb(), dtype=np.uint8)
            data = data.reshape(self.render_info["fig"].canvas.get_width_height()[::-1] + (3,))
            self.render_info["ax_states"].clear()
            if add_subgoal_values:
                self.render_info["ax_subgoal_values"].clear()
            return data
        else:
            assert 1 == 0


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