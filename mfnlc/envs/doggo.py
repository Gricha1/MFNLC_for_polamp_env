from mfnlc.envs.base import SafetyGymBase, CustomTimeLimit


class DoggoNav(SafetyGymBase):
    def __init__(self,
                 no_obstacle=False,
                 end_on_collision=False,
                 fixed_init_and_goal=False) -> None:
        super().__init__('Safexp-DoggoGoal1-v0', no_obstacle,
                         end_on_collision, fixed_init_and_goal)
        self.arrive_reward = 100


class GCDoggoNav(CustomTimeLimit):
    def __init__(self,
                 no_obstacle=False,
                 end_on_collision=False,
                 fixed_init_and_goal=False,
                 max_episode_steps=100) -> None:
        super().__init__('Safexp-DoggoGoal1-v0', no_obstacle,
                         end_on_collision, fixed_init_and_goal,
                         max_episode_steps=max_episode_steps)
