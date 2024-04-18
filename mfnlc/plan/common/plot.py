import matplotlib.pyplot as plt
import numpy as np

from mfnlc.plan.common.geometry import Circle
from mfnlc.plan.common.path import Path
from mfnlc.plan.common.space import SearchSpace
from mfnlc.plan.rrt import Tree


def plot_path_2d(space: SearchSpace = None,
                 path: Path = None,
                 tree: Tree = None,
                 plot_current_pose: bool = False,
                 curret_pose = None,
                 goal_pose = None,
                 obst_poses = None,
                 step_info: dict = None):
    if not(space is None):
        assert space.ub.shape == (2,)

    fig, ax = plt.subplots(figsize=[5, 5])

    env_min_x, env_max_x = -3, 3
    env_min_y, env_max_y = -3, 3
    # map
    ax.set_xlim(env_min_x, env_max_x)
    ax.set_ylim(env_min_y, env_max_y)

    # initial and goal state
    if plot_current_pose: 
        initial = plt.Circle(tuple(curret_pose),
                            radius=0.05, color="g", alpha=0.5)
    else: 
        initial = plt.Circle(tuple(space.initial_state),
                            radius=0.05, color="g", alpha=0.5)
    ax.add_patch(initial)
    if not(space is None): 
        ax.scatter(*space.goal_state, s=500, marker="*", color="gold", alpha=0.5)
    else: # cpo
        ax.scatter(goal_pose[0], goal_pose[1],s=500, marker="*", color="gold", alpha=0.5)

    # obstacle
    if not(space is None): 
        if isinstance(space.obstacles[0], Circle):
            obstacles = [plt.Circle(tuple(obs.state), obs.radius,  # noqa
                                    color="b", alpha=0.5) for obs in space.obstacles]
        else:
            raise NotImplementedError()
    else: # cpo
        obstacles = [plt.Circle(tuple(obs[:2]), 0.3,  # noqa
                                    color="b", alpha=0.5) for obs in obst_poses]

    for obs in obstacles:
        ax.add_patch(obs)

    # tree
    print_tree = False
    if print_tree:
        if tree is not None:
            queue = [tree.root]
            while queue:
                parent = queue.pop(0)
                for child in parent.children:
                    line = np.array([parent.state, child.state])
                    ax.plot(line[:, 0], line[:, 1], marker="x", color="k")
                    queue.append(child)

    # path
    if not (path is None):                    
        if len(path) > 0:
            ax.plot(path[:, 0], path[:, 1], marker="x", color="r")

    # debug info
    if not(step_info is None):
        acc_reward = step_info["acc_reward"]
        t = step_info["t"]
        acc_cost = step_info["acc_cost"]
        ax.text(env_max_x - 2.5, env_max_y - 0.3, f"R:{int(acc_reward*10)/10}")
        ax.text(env_max_x - 1.5, env_max_y - 0.3, f"C:{int(acc_cost*10)/10}")
        ax.text(env_max_x - 0.5, env_max_y - 0.3, f"t:{t}")

    plt.show()

    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data
