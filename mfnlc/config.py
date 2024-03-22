from os.path import dirname, abspath
#import os
import mfnlc
#from mfnlc.envs.dataset.utils_operations import *

#dataSet = generateDataSet(our_env_config, name_folder=args.dataset, total_maps=total_maps, dynamic=False)
#maps, trainTask, valTasks = dataSet["obstacles"]
"""
dataset_exists = os.path.isdir("GCPoint_level_1")
print("dataset exists:", dataset_exists)
assert 1 == 0
file = "mfnlc/envs/dataset/GCPoint_level_1/train_map0.txt"
tasks = []
with open(file, "r") as f:
    j = -1
    for line in f.readlines():
        if(j == -1):
            j += 1
            continue
        parameters = line.split('\t')
        print("pars:", parameters)
        assert 1 == 0
        # print(parameters)
        start = []
        goal = []
        for i in range(len(parameters) - 1):
            # print(parameters[i])
            if i > 4:
                goal.append(float(parameters[i]))
            else:
                start.append(float(parameters[i]))
        tasks.append((start, goal))
"""

default_device = "cuda"
ROOT = dirname(abspath(mfnlc.__file__))

env_config = {
    "Nav": {
        "max_step": 200,
        "goal_dim": 2,
        "state_dim": 0,
        "sink": [0, 0],
        "difficulty": {
            # number of obstacle | map size
            1: [8, [[-1, -1], [1, 1]]],
            2: [32, [[-2, -2], [2, 2]]],
            3: [128, [[-4, -4], [4, 4]]]
        }
    },
    "Point": {
        "max_step": 200,
        "goal_dim": 2,
        "state_dim": 0,
        "robot_radius": 0.3,
        "sink": [0.0, 0.0, 0.0, 0.0, 9.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "env_prop": {
            "vases_num": 0,
            "render_lidar_markers": False,
            'floor_display_mode': True
        },
        "difficulty": {
            # number of obstacle | map size
            1: [8, [[-2, -2], [2, 2]]],
            2: [32, [[-4, -4], [4, 4]]],
            3: [128, [[-8, -8], [8, 8]]]
        }
    },
    "Polamp": {
        "max_step": 600,
        "goal_dim": 2,
        "state_dim": 0,
        "robot_radius": 0.3,
        "sink": [0.0, 0.0, 0.0, 0.0, 9.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "env_prop": {
            "vases_num": 0,
            "render_lidar_markers": False,
            'floor_display_mode': True
        },
        "difficulty": {
            # number of obstacle | map size
            1: [8, [[-2, -2], [2, 2]]],
            2: [32, [[-4, -4], [4, 4]]],
            3: [128, [[-8, -8], [8, 8]]]
        }
    },
    "Car": {
        "max_step": 1000,
        "goal_dim": 2,
        "state_dim": 0,
        "robot_radius": 0.3,
        "sink": [0.0, 0.0, 0.0, 0.0, 9.8, 0.0, 0.0, 0.0, 1.0,
                 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0,
                 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "env_prop": {
            "vases_num": 0,
            "render_lidar_markers": False,
            'floor_display_mode': True
        },
        "difficulty": {
            # number of obstacle | map size
            1: [8, [[-2, -2], [2, 2]]],
            2: [32, [[-4, -4], [4, 4]]],
            3: [128, [[-8, -8], [8, 8]]]
        }
    },
    "Doggo": {
        "max_step": 1000,
        "goal_dim": 2,
        "state_dim": 0,
        "robot_radius": 0.3,
        "sink": [0.0, 0.0, 0.47, 0.0, 9.8, 0.0, -0.0, 0.0, 0.01, 1.0, 0.0, 1.0, 0.0, 1.0, 0.01, 1.0, 0.26, 0.97, 0.01,
                 1.0, 0.53, 0.85, 0.01, 1.0, 0.53, 0.85, 0.01, 1.0, 0.26, 0.97, 0.01, 1.0, -0.0, 0.0, 0.0, -0.0, -0.0,
                 0.0, 0.0, -0.0, 0.0, -0.0, -0.0, 0.0, 0.44, -0.24, -0.02, 0.0, 0.13, 0.0, 0.06, 0.0, 0.06, 0.0, 0.13,
                 -0.0, -0.0, 0.0],
        "env_prop": {
            "vases_num": 0,
            "render_lidar_markers": False,
            'floor_display_mode': True
        },
        "difficulty": {
            # number of obstacle | map size
            1: [8, [[-2, -2], [2, 2]]],
            2: [32, [[-4, -4], [4, 4]]],
            3: [128, [[-8, -8], [8, 8]]]
        }
    },
    "GCPoint": {
        "max_step": 600,
        "goal_dim": 2,
        "state_dim": 0,
        "robot_radius": 0.3,
        "sink": [0.0, 0.0, 0.0, 0.0, 9.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "env_prop": {
            "vases_num": 0,
            "render_lidar_markers": False,
            'floor_display_mode': True
        },
        "difficulty": {
            # number of obstacle | map size
            0: [0, [[-2, -2], [2, 2]]],
            1: [8, [[-2, -2], [2, 2]]],
            2: [32, [[-4, -4], [4, 4]]],
            3: [128, [[-8, -8], [8, 8]]]
        },
        "fixed_hazards": True,
        "fixed_hazard_poses": {
            0: [],
            1: [[-1.2, -1], [0, -1.4], 
                [0, 1.1], [1, 1.8], 
                [0, 0], [1.3, 0],
                [-1.8, 0], [-0.8, 1.8]]},
        "custom_dataset": {
            "1": []
        }
    },
    "GCCar": {
        "max_step": 600,
        "goal_dim": 2,
        "state_dim": 0,
        "robot_radius": 0.3,
        "sink": None,
        "env_prop": {
            "vases_num": 0,
            "render_lidar_markers": False,
            'floor_display_mode': True
        },
        "difficulty": {
            # number of obstacle | map size
            0: [0, [[-2, -2], [2, 2]]],
            1: [8, [[-2, -2], [2, 2]]],
            2: [32, [[-4, -4], [4, 4]]],
            3: [128, [[-8, -8], [8, 8]]]
        },
        "fixed_hazards": True,
        "fixed_hazard_poses": {1: [[-1.2, -1], [0, -1.4], 
                                   [0, 1.1], [1, 1.8], 
                                   [0, 0], [1.3, 0],
                                   [-1.8, 0], [-0.8, 1.8]]}
    },
}


def get_path(robot_name, algo, task):
    data_root = f"{ROOT}/mfnlc_data"
    if task == "log":
        return f"{data_root}/{algo}/{robot_name}/{task}"
    elif task == "model":
        ext = "zip" if algo != "cpo" else "onnx"
        return f"{data_root}/{algo}/{robot_name}/model.{ext}"
    elif task == "tclf":
        return f"{data_root}/{algo}/{robot_name}/tclf.pth"
    elif task == "comparison":
        return f"{data_root}/comparison/{robot_name}"
    elif task == "evaluation":
        return f"{data_root}/{algo}/{robot_name}/evaluation"
    elif task == "lv_table":
        return f"{data_root}/lyapunov_td3/{robot_name}/lv_table.pkl"
    elif task == "video":
        return f"{data_root}/{algo}/{robot_name}/evaluation/video_"
