from abc import abstractmethod
from typing import Dict
import copy

import gym
import numpy as np
import cv2
#import safety_gym  # noqa
from gym import Env, GoalEnv
import math

import matplotlib.pyplot as plt

from mfnlc.config import env_config
from collections import deque


class EnvBase(Env):
    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self, no_obstacle: bool,
                 end_on_collision: bool,
                 fixed_init_and_goal: bool):
        self.no_obstacle = no_obstacle
        self.end_on_collision = end_on_collision
        self.fixed_init_and_goal = fixed_init_and_goal
        self.subgoal = None
        self.subgoal_list = []
        self.traj = []
        self.roa_center = None
        self.roa_radius = 0.0

        self.obstacle_radius = 0.15
        self.robot_radius = 0.3

        self.render_config = {
            "traj_sample_freq": -1,
            "follow": False,
            "vertical": False,
            "scale": 4.0
        }

    def reset(self):
        self.subgoal = None
        self.subgoal_list = []
        self.traj = []
        self.roa_center = None
        self.roa_radius = 0.0

    @abstractmethod
    def goal_obs(self) -> np.ndarray:
        pass

    @abstractmethod
    def robot_obs(self) -> np.ndarray:
        pass

    @abstractmethod
    def obstacle_obs(self) -> np.ndarray:
        pass

    @abstractmethod
    def update_env_config(self, config: Dict):
        pass

    def set_subgoal(self, subgoal: np.ndarray, store=False):
        self.subgoal = subgoal
        if store:
            self.subgoal_list.append(subgoal)

    def get_obs(self):
        return np.concatenate([self.goal_obs(),
                               self.robot_obs(),
                               self.obstacle_obs()])

    def set_roa(self, roa_center: np.ndarray, roa_radius: float):
        self.roa_center = roa_center
        self.roa_radius = roa_radius

    def set_render_config(self, config: Dict):
        self.render_config.update(config)


class SafetyGymBase(EnvBase):

    def __init__(self,
                 env_name,
                 no_obstacle=False,
                 end_on_collision=False,
                 fixed_init_and_goal=False) -> None:
        super().__init__(no_obstacle=no_obstacle,
                         end_on_collision=end_on_collision,
                         fixed_init_and_goal=fixed_init_and_goal)

        if self.no_obstacle:
            env_name = env_name[:-4] + "0-v0"
        self.env = gym.make(env_name)
        self.env_name = env_name

        # Count robot relevant observations (ignore lidar)
        self.robot_obs_size = sum([np.prod(self.env.obs_space_dict[obs_name].shape)
                                   for obs_name in self.env.obs_space_dict
                                   if 'lidar' not in obs_name])

        # self.obstacle_in_obs = sum([np.prod(self.env.obs_space_dict[obs_name].shape)
        #                            for obs_name in self.env.obs_space_dict
        #                            if 'hazards_lidar' in obs_name])
        self.obstacle_in_obs = 4
        self.num_relevant_dim = 2  # For x-y relevant observations ignoring z-axis
        self.frame_stack = 1
        # Reward config
        self.collision_penalty = -0.01
        self.arrive_reward = 20

        customized_config = env_config[self.robot_name].get("env_prop", None)
        if customized_config is not None:
            self.update_env_config(customized_config)

        self._build_space()
        self.env.toggle_observation_space()
        self.previous_goal_dist = None

    def _build_space(self):
        self.action_space = self.env.action_space

        max_observation = 10
        if self.no_obstacle:
            observation_high = max_observation * np.ones(
                self.num_relevant_dim + self.robot_obs_size,
                dtype=np.float32)
        else:
            observation_high = max_observation * np.ones(
                self.num_relevant_dim + self.robot_obs_size + self.obstacle_in_obs * self.num_relevant_dim,
                dtype=np.float32)
        observation_low = -observation_high
        self.observation_space = gym.spaces.Box(observation_low, observation_high, dtype=np.float32)

    def seed(self, seed=None):
        self.env.seed(seed)

    def update_env_config(self, config: Dict):
        self.__dict__.update(config)  # Lazy implementation: can introduce unnecessary binding
        self.env.__dict__.update(config)
        self.env.parse(config, update=True)        
        
        assert "robot_base" not in config.keys(), \
            "Do not change robot, this requires to rebuild observation and action space"
        self.env.build_placements_dict()

        self.env.viewer = None
        self.env.world = None
        self.env.clear()
        self.env.done = True

        self.env.clear()
        self._build_space()

    @property
    def robot_name(self):
        all_robot_names = ["Point", "Car", "Doggo", "Polamp"]
        for name in all_robot_names:
            if name in self.env_name:
                return name

    @property
    def robot_pos(self):
        return self.env.robot_pos[:2]

    @property
    def hazards_pos(self):
        return self.env.hazards_pos

    def reset(self):
        # resetting everything in base env (pay attention!)
        super(SafetyGymBase, self).reset()
        self.env.reset()
        self.env.num_steps = 10000

        if self.fixed_init_and_goal and (
                len(self.env.goal_locations) == 0
                or
                len(self.env.robot_locations) == 0
        ):
            self.env.goal_locations = self.goal_obs()[:2].tolist()
            self.env.robot_locations = self.robot_pos[:2].tolist()

        self.previous_goal_dist = None

        return self.get_obs(False)

    def goal_obs(self) -> np.ndarray:
        goal_obs = (self.env.goal_pos - self.env.robot_pos)[:self.num_relevant_dim]
        return goal_obs

    def robot_obs(self) -> np.ndarray:
        # only gets observation dimensions relevant to robot from safety-gym
        obs = self.env.obs()
        flat_obs = np.zeros(self.robot_obs_size)
        offset = 0

        for k in sorted(self.env.obs_space_dict.keys()):
            if "lidar" in k:
                continue
            k_size = np.prod(obs[k].shape)
            flat_obs[offset:offset + k_size] = obs[k].flat
            offset += k_size
        return flat_obs

    def obstacle_obs(self) -> np.ndarray:
        if self.no_obstacle:
            return np.array([])

        # get distance to each obstacle upto self.obstacle_in_obs nearest obstacles
        vec_to_obs = (self.env.hazards_pos - self.env.robot_pos)[:, :self.num_relevant_dim]
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

    def get_goal_reward(self):
        goal_dist = np.linalg.norm(self.goal_obs(), ord=2)
        if self.previous_goal_dist is None:
            goal_reward = 0.0
        else:
            goal_reward = (self.previous_goal_dist - goal_dist) * 10
        self.previous_goal_dist = goal_dist

        return goal_reward

    def step(self, action: np.ndarray):
        s, r, done, info = self.env.step(action)

        # As of now use safety gym info['cost'] to detect collisions
        collision = info.get('cost', 1.0) > 0
        info["collision"] = collision

        arrive = info.get("goal_met", False)

        reward = self.get_goal_reward() + collision * self.collision_penalty + arrive * self.arrive_reward

        if self.end_on_collision and collision:
            done = True
        else:
            done = arrive or done

        obs = self.get_obs()
        if arrive:
            # if the robot meets goal, the goal will be reset immediately
            # this can cause the goal observation has large jumps and affect Lyapunov function
            obs[:self.num_relevant_dim] = np.zeros(self.num_relevant_dim)

        self.traj.append(self.robot_pos)

        return obs, reward, done, info

    def render(self,
               mode="human",
               camera_id=1,
               width=2048,
               height=2048):
        # plot subgoal
        if self.env.viewer is not None:
            for subgoal in self.subgoal_list:
                self.env.render_area(subgoal, 0.1,
                                     np.array([0, 1, 0.0, 0.5]), 'subgoal', alpha=0.5)

            if self.render_config["traj_sample_freq"] > 0:
                for pos in self.traj[::self.render_config["traj_sample_freq"]]:
                    self.env.render_area(pos, 0.05,
                                         np.array([1, 0.5, 0, 0.5]), 'position', alpha=0.5)

            if self.roa_center is not None:
                self.env.render_area(self.roa_center, self.roa_radius,
                                     np.array([0.2, 1.0, 1.0, 0.5]), 'RoA approx', alpha=0.5)

        return self.env.render(mode, camera_id,
                               width=width, height=height,
                               follow=self.render_config["follow"],
                               vertical=self.render_config["vertical"],
                               scale=self.render_config["scale"])
        
class GCSafetyGymBase(SafetyGymBase):    

    def __init__(self,
                 env_name,
                 no_obstacle=False,
                 end_on_collision=False,
                 fixed_init_and_goal=False,
                 max_episode_steps=100) -> None:
        super().__init__(env_name=env_name,
                         no_obstacle=no_obstacle,
                         end_on_collision=end_on_collision,
                         fixed_init_and_goal=fixed_init_and_goal)
        assert self.num_relevant_dim == 2 # goal x, y
        class EnvSpec():
            def __init__(self):
                self.max_episode_steps = max_episode_steps
        self.spec = EnvSpec()
        # Reward config
        self.collision_penalty = -70
        self.arrive_reward = 150
        self.time_step_reward = -1
        self.subgoal_pos = None
        self.obstacle_observation = None
        self.obstacle_goal_observation = None
        self.render_info = {}
        self.render_info["fig"] = None
        self.render_info["ax_states"] = None
        # set difficulty level
        level = 1
        robot_name = "GC" + self.robot_name
        difficulty_config = env_config[robot_name]["difficulty"][level]
        floor_lb, floor_ub = np.array(difficulty_config[1], dtype=np.float32)
        fixed_hazards = env_config[robot_name]["fixed_hazards"]
        hazards_placements = None
        if fixed_hazards:
            if level != 1:
                assert 1 == 0, "didnt implemented"
            hazards_locations = env_config[robot_name]["fixed_hazard_poses"][level]
        else:
            hazards_locations = []
        self.update_env_config({
            "hazards_num": difficulty_config[0],
            "placements_extents": np.concatenate([floor_lb, floor_ub]).tolist(),
            "hazards_keepout": 0.45,
            "hazards_placements": hazards_placements,
            'hazards_locations': hazards_locations,
            "_seed": 42,
        })
        self.obstacle_in_obs = 4
        self.frame_stack = 1
        self.state_history = deque([])
        self.goal_history = deque([])
        self.history_len = self.frame_stack
        print("dataset:", self.env.placements_extents)
        print("hazards_num:", self.env.hazards_num)
        print("placements:", self.env.placements)

    def _build_space(self):
        self.action_space = self.env.action_space

        max_observation = 10
        if self.no_obstacle:
            observation_high = max_observation * np.ones(
                ((self.num_relevant_dim + self.robot_obs_size) * self.frame_stack),
                dtype=np.float32)
        else:
            # observation_high = max_observation * np.ones(
            #     (self.num_relevant_dim + self.robot_obs_size + self.obstacle_in_obs),
            #     dtype=np.float32)
            observation_high = max_observation * np.ones(
                ((self.num_relevant_dim + self.robot_obs_size + self.obstacle_in_obs * self.num_relevant_dim) * self.frame_stack),
                dtype=np.float32)
        observation_low = -observation_high
        self.observation_space = gym.spaces.Dict({
            "observation": gym.spaces.Box(observation_low, observation_high, dtype=np.float32),
            "desired_goal": gym.spaces.Box(observation_low, observation_high, dtype=np.float32),
            "achieved_goal": gym.spaces.Box(observation_low, observation_high, dtype=np.float32),
            "collision": gym.spaces.Box(0.0, 1.0, (1,), np.float32),
            "clearance_is_enough": gym.spaces.Box(0.0, 1.0, (1,), np.float32)
        })
    
    #def compute_rewards(self, achieved_goal, desired_goal, info):
    def compute_rewards(self, new_actions, new_next_obs_dict):
        return self.time_step_reward * np.ones_like(new_actions[:, 0])
            
    def obstacle_goal_obs(self) -> np.ndarray:
        """
            get obstacle observation with respect to goal
        """
        if self.no_obstacle:
            return np.array([])

        # get distance to each obstacle upto self.obstacle_in_obs nearest obstacles
        vec_to_obs = (self.env.hazards_pos - self.env.goal_pos)[:, :self.num_relevant_dim]
        dist_to_obs = np.linalg.norm(vec_to_obs, ord=2, axis=-1)
        order = dist_to_obs.argsort()[:self.obstacle_in_obs]
        flattened_vec = vec_to_obs[order].flatten()
        # in case of that the obstacle number in environment is smaller than self.obstacle_in_obs
        output = np.zeros(self.obstacle_in_obs * self.num_relevant_dim)
        output[:flattened_vec.shape[0]] = flattened_vec
        self.obstacle_goal_observation = output
        return output        
        # obs = self.env.obs()
        # return obs["goal_lidar"] 
    
    def set_subgoal_pos(self, subgoal_related_pos):
        self.subgoal_pos = []
        shift_v = int(subgoal_related_pos[0][0].shape[0] / self.frame_stack * (self.frame_stack - 1))
        self.subgoal_pos.append(subgoal_related_pos[0][0][0 + shift_v].item())
        self.subgoal_pos.append(subgoal_related_pos[0][0][1 + shift_v].item())

    def reset(self, **kwargs):
        # check env config
        self.state_history.clear()
        self.goal_history.clear()
        if self.no_obstacle:
            assert self.hazards_num == 0, "empty env has no obstacles"
        else:
            assert self.hazards_num > 0, "env with obstacles should have obstacles"
        
        self.subgoal_pos = None
        obs = super().reset(**kwargs)
        assert not obs["collision"], "initial state in collision!!!"
        return obs
    
    def step(self, action: np.ndarray):
        s, r, done, info = self.env.step(action)

        # As of now use safety gym info['cost'] to detect collisions
        collision = info.get('cost', 1.0) > 0
        info["collision"] = collision
        # check env config
        if self.no_obstacle:
            assert collision == False

        arrive = info.get("goal_met", False)

        if self.env.robot_pos[0] < -2.0 or self.env.robot_pos[0] > 2.0 or \
            self.env.robot_pos[1] < -2.0 or self.env.robot_pos[1] > 2.0:
            collision = True

        reward = self.time_step_reward + self.collision_penalty * collision

        if self.end_on_collision and collision:
            done = True
        else:
            done = arrive or done

        obs = self.get_obs(arrive)
        obs["collision"] = collision
        # if arrive:
        #     # if the robot meets goal, the goal will be reset immediately
        #     # this can cause the goal observation has large jumps and affect Lyapunov function
        #     #obs[:self.num_relevant_dim] = np.zeros(self.num_relevant_dim)
        #     obs["desired_goal"][:self.num_relevant_dim] = obs["observation"][:self.num_relevant_dim]

        # test
        shift_v = int(obs["observation"].shape[0] / self.frame_stack * (self.frame_stack - 1))
        test_reward = np.sqrt(np.power(np.array(obs["observation"] - obs["desired_goal"])[shift_v : shift_v+2], 2).sum(-1, keepdims=True)) # distance: next_state to goal
        test_arrive = 1.0 * (test_reward <= self.env.goal_size)# terminal condition
        if not arrive == test_arrive:
            assert 1 == 0

        self.traj.append(self.robot_pos)

        info["goal_is_arrived"] = arrive
        info["is_success"] = arrive

        return obs, reward, done, info
    
    def robot_goal_pos(self):
        return np.zeros(shape=self.num_relevant_dim)

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
    
    def custom_render(self, positions_render=False, dubug_info={}, shape=(600, 600)):
        if positions_render:
            env_min_x, env_max_x = -3, 3
            env_min_y, env_max_y = -3, 3
            if self.render_info["fig"] is None:
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
            if self.subgoal_pos is not None:
                x = self.subgoal_pos[0]
                y = self.subgoal_pos[1]
                circle_robot = plt.Circle((x, y), radius=self.robot_radius, color="orange", alpha=0.5)
                self.render_info["ax_states"].add_patch(circle_robot)
                self.render_info["ax_states"].text(x + 0.05, y + 0.05, "s_g")
            # goal
            x = self.env.goal_pos[0]
            y = self.env.goal_pos[1]
            circle_robot = plt.Circle((x, y), radius=self.robot_radius, color="y", alpha=0.5)
            self.render_info["ax_states"].add_patch(circle_robot) 
            self.render_info["ax_states"].text(x + 0.05, y + 0.05, "g")
            # for distance, angle in zip(env_obs["goal_lidar"], angle_space):
            #     plt.plot([x, x + distance * math.cos(angle)],\
            #             [y, y + distance * math.sin(angle)],\
            #             '-', linewidth = 4, color='blue')
                
            # add obstacles
            obstacles = [plt.Circle(obs[:2], radius=self.obstacle_radius,  # noqa
                        color="b", alpha=0.5) for obs in self.env.hazards_pos]
            for obs in obstacles:
                self.render_info["ax_states"].add_patch(obs)
            x = self.env.robot_pos[0]
            y = self.env.robot_pos[1]
            self.obstacle_observation = np.reshape(self.obstacle_observation, (int(self.obstacle_observation.shape[0]/ 2), 2))
            for obs_coord in self.obstacle_observation:
                plt.plot([x, x + obs_coord[0]],\
                        [y, y + obs_coord[1]],\
                        '-', linewidth = 4, color='red')
            x = self.env.goal_pos[0]
            y = self.env.goal_pos[1]
            self.obstacle_goal_observation = np.reshape(self.obstacle_goal_observation, (int(self.obstacle_goal_observation.shape[0]/ 2), 2))
            for obs_coord in self.obstacle_goal_observation:
                plt.plot([x, x + obs_coord[0]],\
                        [y, y + obs_coord[1]],\
                        '-', linewidth = 4, color='green')
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
            return data
        else:
            # camera_name = ('fixednear', 'fixedfar', 'vision', 'track')
            image = self.env.sim.render(shape[0], shape[1], camera_name="fixedfar", mode='offscreen')
            rotated_image = cv2.rotate(image, cv2.ROTATE_180)
            return rotated_image
    

class CustomTimeLimit(GCSafetyGymBase):
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

class ObstacleMaskWrapper(gym.Wrapper):
    """
    ! This wrapper will not change the original observation space
    """

    def get_obs(self):
        return np.concatenate([self.goal_obs(),
                               self.robot_obs()])

    def reset(self, **kwargs):
        super(ObstacleMaskWrapper, self).reset(**kwargs)
        return self.get_obs()

    def step(self, action):
        _, r, d, info = super(ObstacleMaskWrapper, self).step(action)
        return self.get_obs(), r, d, info
