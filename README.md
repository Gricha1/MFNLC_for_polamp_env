# Neural Lyapunov Deep Reinforcement Learning
The repository is taken from:
```commandline
https://github.com/ZikangXiong/MFNLC
```
and is adapted to custom environment

# Neural Lyapunov Deep Reinforcement Learning

Code Repository of IROS 22' paper **Model-free Neural Lyapunov Control for Safe Robot Navigation**

[ArXiv](https://arxiv.org/abs/2203.01190) | [Demos](https://sites.google.com/view/mf-nlc)

https://user-images.githubusercontent.com/73256697/184549907-50287e7f-fc0c-46fa-baf9-58660e8634eb.mp4

## Project Structure

```
├── README.md
├── setup.py
└── shrl
    ├── config.py       # config file, including data path, default devices, ect. 
    ├── envs            # simulation environments
    ├── evaluation      # evaluation utils
    ├── exps            # experiment scripts
    ├── learn           # low-level controller and neural Lyapunov function learning algorithms
    ├── monitor         # high-level monitor
    ├── plan            # high-level planner, RRT & RRT*
    └── tests           # test cases
```

## Install

1. Install necessary dependencies.

```commandline
pip install -e .
```

2. Configure MuJoCo-py by following official [README](https://github.com/openai/mujoco-py).
3. (Optional) Download pretrained models (~35 MB)

```commandline
bash download.sh
```

## Quick Start

Two quick start examples:

1. Co-learning low-level controller and neural Lyapunov function  
   `python exps/train/no_obstacle/lyapunov_td3/[robot-name].py`

2. Pre-compute monitor and evaluate  
   `python exps/hierachical/rrt_lyapunov/[robot-name].py`

One can start tracing code from `exps` folder.

## Bibtex

```bibtex
@inproceedings{Xiong2022ModelfreeNL,
  title={Model-free Neural Lyapunov Control for Safe Robot Navigation},
  author={Zikang Xiong and Joe Eappen and Ahmed H. Qureshi and Suresh Jagannathan},
  booktitle={2022 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year={2022},
}
```


# docker
```commandline
docker run -it --gpus "device=0" --runtime=nvidia -e NVIDIA_DRIVER_CAPABILITIES=compute,utility -v $(pwd):/usr/home/workspace continuumio/miniconda3 /bin/bash -c "conda install python=3.8.5 -y && bash" 
cd /usr/home/workspace
```

# run via Dockerfile
```commandline
cd docker
docker build -t ris_safety_gym_img .
cd MFNLC
docker run -it --gpus "device=0" --runtime=nvidia -e NVIDIA_DRIVER_CAPABILITIES=compute,utility -v $(pwd):/usr/home/workspace ris_safety_gym_img
cd /usr/home/workspace 
cd safety-gym
pip install -e .
cd ..
pip install -e .
install mujoco (it wasnt installed in docker file dont know why)
```

# start existing docker
```commandline
docker exec -it gregory_SafetyRIS bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.mujoco/mujoco210/bin
```

# changed to deps
you need Python 3.8.5

2. safetygym 
```commandline
git clone https://github.com/openai/safety-gym.git
```
in safety-gym/setup.py, comment mujoco_py
#'mujoco_py==2.0.2.7',

```commandline
cd safety-gym
pip install -e .
```

3. mfnlc
```commandline
apt-get update
apt-get install -y \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglew-dev \
    libosmesa6-dev \
    software-properties-common \
    patchelf
pip install onnxruntime
pip install free-mujoco-py

```commandline
cd MFNLC
pip install -e .
      if error with gym install: 
            """
               error in gym setup command: 'extras_require' must be a      
               dictionary whose values are strings or lists of strings con
               taining valid project/version requirement specifiers.
            """
      then:
         pip install setuptools==65.5.0 pip==21
         pip install wheel==0.38.0
```

4. Install Mujoco for mujoco_py lib (Linux)
```commandline
apt-get install build-essential
cd /root
mkdir .mujoco
wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz
tar -xf mujoco210-linux-x86_64.tar.gz -C .mujoco
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.mujoco/mujoco210/bin
```

5. cuda deps (cuda is 11.4 so torch for 11.3 is appropiate)
check if gpu exists:
```commandline
nvidia-smi
```
```commandline
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
```

6. deps for adding video to tensorboard & wandb
```commandline
pip install wandb moviepy opencv-python
```

# troubles
1. No such file or directory: '/home/reedgern/mipt_work_space/sem_3/NIR/other_algs/MFNLC/mfnlc/mfnlc_data/lyapunov_td3/Nav/model.zip.zip'

in Nav.py delete comments:
```commandline
colearn()
# evaluate_controller()
evaluate_lyapunov()
```

2. AttributeError: module 'numpy' has no attribute 'complex'. 
   (NOT NECESSARY IF safety_gym is commented)
in safety-gym/safety_gym/envs/engine.py
change:
```commandline
#z = np.complex(*self.ego_xy(pos))  # X, Y as real, imaginary components
z = self.ego_xy(pos)[0] + self.ego_xy(pos)[1] * 1j
```

3. AssertionError: Bad obs {'goal_lidar': array ...
in safety-gym/safety_gym/envs/engine.py
comment:
```commandline
#assert self.observation_space.contains(obs), f'Bad obs {obs} {self.observation_space}'
```

# set polamp env deps
export PYTHONPATH=$(pwd)/envs/bark_ml_ris:$PYTHONPATH

# start tensorboard
tensorboard --logdir mfnlc/mfnlc_data/ --bind_all

# train RIS point env
```commandline
cd mfnlc
python exps/train/obstacle/ris/point.py
```

