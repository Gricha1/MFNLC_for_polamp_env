import numpy as np

from mfnlc.exps.hierachical.custom_rrt_lyapunov.base import evaluate, build_lyapunov_table
from mfnlc.exps.check_results import print_all_results, print_level_results

ENV_NAME = "Doggo-eval"


def rrt_lyapunov(planning_algo, level, n_tasks, pretrained):
    i = level
    #for i in range(3, 4):
    print(f"{ENV_NAME} - RRT + Lyapunov-TD3 - level {i}")
    stat, _ = evaluate(ENV_NAME,
            n_rollout=n_tasks,
            level=i,
            planning_algo=planning_algo,
            planner_max_iter=i * i * 1000,
            n_steps=1000 * i * i,
            arrive_radius=0.3,
            monitor_max_step_size=0.5,
            render=True,
            render_config={
                "traj_sample_freq": 10,
                "follow": True,
                "vertical": False,
                "scale": 4
            },
            video=False,
            seed=0, # 123
            pretrained=pretrained)


def build_lv_table(pretrained):
    lb = -np.ones(58)
    lb[4] = 9.8
    ub = np.ones(58)
    ub[4] = 9.81
    build_lyapunov_table(ENV_NAME,
                         lb, ub,
                         pgd_max_iter=1000,
                         n_radius_est_sample=20,
                         pretrained=pretrained)


if __name__ == '__main__':
    level = 1
    n_tasks = 40
    pretrained = True
    if pretrained:
        print("LOAD PRETRAINED WEIGHTS")
    else:
        print("LOAD OWN WEIGHTS")
    print("Validation levels:", level)
    build_lv_table(pretrained)
    stats = rrt_lyapunov("rrt*", level, n_tasks, pretrained)
    print_level_results(ENV_NAME, "rrt_lyapunov", level, "rrt*", with_cost=True)
    input("Press Enter to continue...")
