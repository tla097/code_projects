import os
import sys
current_directory = os.getcwd()
print("Current working directory:", current_directory)
pythonpath = os.environ.get('PYTHONPATH', '')
print("PYTHONPATH:", pythonpath)
sys.path.append(current_directory)
directories = [current_directory]
pythonpath = os.pathsep.join(directories)
os.environ['PYTHONPATH'] = pythonpath

pythonpath = os.environ.get('PYTHONPATH', '')
print("PYTHONPATH:", pythonpath)

print("System path:")
for path in sys.path:
    print(path)

# def main():
#     params = {"eps_decay": 100000}
#     name = f"default test"
#     load = f""
#     new_load = "new"
#     params["save"] = name
#     params["load"] = load
#     params["new_load"] = new_load
#     runner = PairWiseRankRunner(**params, des = str(params), test_train="train")
#     target=runner.run()
# num_test_files = 100
# eps = 10000
# test_frequency = 20
# while num_test_files != 10000:
#     params = {'eps_decay' : eps, 'test_frequency':test_frequency, 'test_length' : num_test_files}#
#     name = f"correect_shuffle-e_{eps}:_tf_{test_frequency}:_tl_{num_test_files}"
#     # name = "random_tester_fd"
#     runner= Runner(**params, des=str(params), save=name, test_train="train", new_load="new", start=0.5)
#     exit = runner.run()
    
    
#     if exit:
#         if runner.previous_average <= 70:
#             eps += 10000
#         num_test_files += 50
#         if test_frequency < 200:
#             test_frequency += 20
        
#         eps += 10000




num_test_files = 100
eps = 100000
# test_frequency = 20
# from PairwiseRunner import Runner
# for i in range(5):
#         params = {'eps_decay' : eps}#
#         name = f"log4" + str(i)
#         runner= Runner(**params, des=str(params), save=name, test_train="train", new_load="new")
#         exit = runner.run()
        
        
    
envs = ["an"]
# truncs = [True]
# # pens = [-1, -100]
params = {'eps_decay' : eps}
from list_wise_runner_refactored import ListWiseRunner
for i in range(2):
    for env in envs:
            name = f"long_lw{i}"
            runner= ListWiseRunner(**params, des=str(params), save=name, test_train="train", new_load="new", file="env-comp", env  = env, load=True , start = 0.9)
            exit = runner.run()
            
        #

# runner= ListWiseRunner(**params, des=str(params), save="tester", test_train="train", new_load="new", file="env-comp", env  = "an", load=True , start = 0.9)

# en = runner.test_env

# state0 = en.reset(test=True)
# print(runner.test(en, state0=state0))

# arr = []
    
    
# for i in range(300):
#     state0 = en.reset(test=True)
#     correct, _, total = runner.test(en, state0=state0)
#     print((correct/total))