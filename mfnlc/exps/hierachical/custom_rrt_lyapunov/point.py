import numpy as np

from mfnlc.exps.hierachical.custom_rrt_lyapunov.base import evaluate, build_lyapunov_table
from mfnlc.exps.check_results import print_all_results, print_level_results

ENV_NAME = "Point-eval"


def rrt_lyapunov(planning_algo):
    level = 1
    i = level
    n_tasks = 50
    #for i in range(1, 4):
    print(f"{ENV_NAME} - RRT + Lyapunov-TD3 - level {i}")
    evaluate(ENV_NAME,
            n_rollout=n_tasks,
            level=i,
            planning_algo=planning_algo,
            planner_max_iter=i * i * 1000,
            n_steps=1000 * i * i,
            arrive_radius=0.3,
            monitor_max_step_size=0.5,
            render=True,
            check_plan=True,
            render_config={
                "traj_sample_freq": 10,
                "follow": False,
                "vertical": True,
                "scale": 7 * i
            },
            video=False,
            seed=0)


def build_lv_table():
    print("start table building ...")
    lb = np.array([-1, -1, -1, -1, 9.8, -1, -1, -1, -1, -1, -1, -1, -1, -1])
    ub = np.array([1, 1, 1, 1, 9.81, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    build_lyapunov_table(ENV_NAME,
                         lb, ub,
                         pgd_max_iter=500,
                         n_radius_est_sample=40)
    print("end table building")


if __name__ == '__main__':
    level = 1
    build_lv_table()
    rrt_lyapunov("rrt*")
    #print_all_results(ENV_NAME, "rrt_lyapunov", "rrt*")
    print_level_results(ENV_NAME, "rrt_lyapunov", level, "rrt*")
    
    input("Press Enter to continue...")
