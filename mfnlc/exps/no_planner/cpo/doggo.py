from mfnlc.exps.check_results import print_all_results, print_level_results
from mfnlc.exps.no_planner.cpo.base import evaluate

ENV_NAME = "Doggo-eval"


def e2e(level, n_tasks, pretrained):
    i = level
    evaluate(ENV_NAME,
                n_rollout=n_tasks,
                level=i,
                render=True,
                n_steps=i * i * 1000)


if __name__ == '__main__':
    level = 1
    n_tasks = 40 # 40
    pretrained = True
    if pretrained:
        print("LOAD PRETRAINED WEIGHTS")
    else:
        print("LOAD OWN WEIGHTS")
    print("Validation levels:", level)
    e2e(level, n_tasks, pretrained)
    print_level_results(ENV_NAME, "cpo", level, with_cost=True)
    #print_all_results(ENV_NAME, "cpo")
