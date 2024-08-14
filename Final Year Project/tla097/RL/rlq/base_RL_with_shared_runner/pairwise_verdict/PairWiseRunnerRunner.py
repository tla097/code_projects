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
test_frequency = 20
# from PairwiseRunner import Runner
# for i in range(5):
#         params = {'eps_decay' : eps}#
#         name = f"log4" + str(i)
#         runner= Runner(**params, des=str(params), save=name, test_train="train", new_load="new")
#         exit = runner.run()
        
        

# envs= ["orig", "an", "log4"]
        
from pairwise_verdict_runner_refactored import PairWiseVerdictRunner
# for i in range(1):
#         params = {'eps_decay' : eps}#
#         name = f"10000lengthLong" + str(i)
#         runner= PairWiseVerdictRunner(**params, des=str(params), save=name, test_train="train", new_load="new", file="env_comp", test_length=1000, gamma=0.99)
#         exit = runner.run()


runner= PairWiseVerdictRunner(save="orig_values", test_train="train", new_load="new", file="env_comp", test_length=50, gamma=0.99, env="orig")
runner.run()
    
    