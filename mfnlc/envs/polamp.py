import json
from collections import OrderedDict

import numpy as np
import torch
import random
import argparse
import gym
from mfnlc.envs.base import SafetyGymBase
from gym.envs.registration import register
from gym.spaces import Box

from .bark_ml_ris.polamp_env.lib.utils_operations import generateDataSet
from .bark_ml_ris.goal_polamp_env.env import GCPOLAMPEnvironment

class SafetyGCPOLAMPEnvironment(GCPOLAMPEnvironment):
  def __init__(self, *args, **kwargs):
        self.obs_space_dict = OrderedDict([('kinematic', Box(-np.inf, np.inf, (4 * 5,), np.float32))])
        super().__init__(*args, **kwargs)


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="cross_dataset_simplified") # medium_dataset, hard_dataset, ris_easy_dataset, hard_dataset_simplified
parser.add_argument("--uniform_feasible_train_dataset", default=False)
parser.add_argument("--random_train_dataset",           default=False)

parser.add_argument("--seed",               default=42, type=int) # 42
args = parser.parse_args()

with open("envs/bark_ml_ris/goal_polamp_env/goal_environment_configs.json", 'r') as f:
        goal_our_env_config = json.load(f)
with open("envs/bark_ml_ris/polamp_env/configs/train_configs.json", 'r') as f:
    train_config = json.load(f)
with open("envs/bark_ml_ris/polamp_env/configs/environment_configs.json", 'r') as f:
    our_env_config = json.load(f)
with open("envs/bark_ml_ris/polamp_env/configs/reward_weight_configs.json", 'r') as f:
    reward_config = json.load(f)
with open("envs/bark_ml_ris/polamp_env/configs/car_configs.json", 'r') as f:
    car_config = json.load(f)

if args.dataset == "medium_dataset":
    total_maps = 12
elif args.dataset == "test_medium_dataset":
    total_maps = 3
elif args.dataset == "hard_dataset_simplified_test":
    total_maps = 2
else:
    total_maps = 1
dataSet = generateDataSet(our_env_config, name_folder="envs/bark_ml_ris/"+args.dataset, total_maps=total_maps, dynamic=False)
maps, trainTask, valTasks = dataSet["obstacles"]
goal_our_env_config["dataset"] = args.dataset
goal_our_env_config["uniform_feasible_train_dataset"] = args.uniform_feasible_train_dataset
goal_our_env_config["random_train_dataset"] = args.random_train_dataset
if not goal_our_env_config["static_env"]:
    maps["map0"] = []

args.evaluation = False
environment_config = {
    'vehicle_config': car_config,
    'tasks': trainTask,
    'valTasks': valTasks,
    'maps': maps,
    'our_env_config' : our_env_config,
    'reward_config' : reward_config,
    'evaluation': args.evaluation,
    'goal_our_env_config' : goal_our_env_config,
}
args.other_keys = environment_config

train_env_name = "Safexp-PolampEnvGoal0-v0" 
test_env_name = train_env_name

# Set seed
np.random.seed(args.seed)
random.seed(args.seed)
torch.manual_seed(args.seed)

# register polamp env
env_dict = gym.envs.registration.registry.env_specs.copy()
for env in env_dict:
    if train_env_name in env:
        print("Remove {} from registry".format(env))
        del gym.envs.registration.registry.env_specs[env]
register(
    id=train_env_name,
    #entry_point='mfnlc.envs.bark_ml_ris.goal_polamp_env.env:GCPOLAMPEnvironment',
    entry_point='mfnlc.envs.polamp:SafetyGCPOLAMPEnvironment',
    kwargs={'full_env_name': "polamp_env", "config": args.other_keys}
)

env = gym.make("Safexp-PolampEnvGoal0-v0")

class PolampEnv(SafetyGymBase):
    def __init__(self,
                 no_obstacle=False,
                 end_on_collision=False,
                 fixed_init_and_goal=False) -> None:
        super().__init__('Safexp-PolampEnvGoal1-v0', no_obstacle,
                         end_on_collision, fixed_init_and_goal)
