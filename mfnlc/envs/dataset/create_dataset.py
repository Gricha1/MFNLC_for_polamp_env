import os

def save_dataset(lst_starts, lst_goals, name, folder_name="cross_dataset_test_level_1/"):
    dataset_exists = os.path.isdir(folder_name)
    if dataset_exists:
        print("changing EXISTING dataset")
    else:
        print("creating NEW dataset")
        os.mkdir(folder_name)
    with open(folder_name + name, 'w') as output:
        output.write(str(len(lst_starts)) + '\n')
        for i in range(len(lst_starts)):
            # print(f"i : {i}")
            start = lst_starts[i]
            goal = lst_goals[i]
            for element in start:
                output.write(str(element) + '\t')
            for element in goal:
                output.write(str(element) + '\t')
            output.write("\n")

def save_map(obstacle, name):
    with open("hard_dataset_simplified_test/" + name, 'w') as output:
    # for obstacle in obstacles:
        output.write(str(len(obstacle)) + '\n')
        for polygon in obstacle:
            for element in polygon:
                output.write(str(element) + '\t')
            output.write("\n")

def task_generator(easy_patern=True, num_tasks=0):
    assert num_tasks > 0
    """
        lst_starts = [ [x_s1, y_s1], [x_s2, y_s2] ]    
        lst_goals = [ [x_g1, y_g1], [x_g2, y_g2] ]    
    """
    lst_starts = []
    lst_goals = []

    task_1_start = [-2, -2]
    task_1_goal = [2, 2]
    task_2_start = [-2, 2]
    task_2_goal = [2, -2]
    lst_starts.append(task_1_start)
    lst_starts.append(task_2_start)
    lst_goals.append(task_1_goal)
    lst_goals.append(task_2_goal)

    assert len(lst_starts) == len(lst_goals)
    return lst_starts, lst_goals
    


print("Start Saving dataset!!! ")
print("..........................")

num_tasks = 1
train_lst_starts, train_lst_goals = task_generator(easy_patern=True, num_tasks=num_tasks)
eval_lst_starts, eval_lst_goals = task_generator(easy_patern=False, num_tasks=num_tasks)
save_dataset(train_lst_starts, train_lst_goals, "train_map0.txt", folder_name="GCPoint_level_1/")
save_dataset(eval_lst_starts, eval_lst_goals, "val_map0.txt", folder_name="GCPoint_level_1/")

"""
# cross dataset test level 1
num_tasks = 15
train_lst_starts, train_lst_goals = task_generator(easy_patern=True, num_tasks=num_tasks)
num_tasks = 3
eval_lst_starts, eval_lst_goals = task_generator(easy_patern=True, num_tasks=num_tasks)
save_dataset(train_lst_starts, train_lst_goals, "train_map0.txt", folder_name="cross_dataset_test_level_1/")
save_dataset(eval_lst_starts, eval_lst_goals, "val_map0.txt", folder_name="cross_dataset_test_level_1/")
"""

"""
# cross dataset test level 2
# train_map0.txt is copied from cross_dataset_simplified/
# this code creates the same dataset but with different starts&goals
num_tasks = 15
train_lst_starts, train_lst_goals = [], []
train_easy_lst_starts, train_easy_lst_goals = task_generator(easy_patern=True, num_tasks=num_tasks)
train_hard_lst_starts, train_hard_lst_goals = task_generator(easy_patern=False, num_tasks=num_tasks)
train_lst_starts.extend(train_easy_lst_starts)
train_lst_starts.extend(train_hard_lst_starts)
train_lst_goals.extend(train_easy_lst_goals)
train_lst_goals.extend(train_hard_lst_goals)
num_tasks = 8
eval_lst_starts, eval_lst_goals = task_generator(easy_patern=False, num_tasks=num_tasks)
save_dataset(train_lst_starts, train_lst_goals, "train_map0.txt", folder_name="cross_dataset_test_level_2/")
save_dataset(eval_lst_starts, eval_lst_goals, "val_map0.txt", folder_name="cross_dataset_test_level_2/")
"""

print("..........................")
print("train dataset num tasks:", len(train_lst_starts))
print("eval dataset num tasks:", len(eval_lst_starts))
print("Success!")