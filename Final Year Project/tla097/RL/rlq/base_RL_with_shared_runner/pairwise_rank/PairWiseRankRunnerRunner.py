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
from pairwise_rank_runner_refactored import PairWiseRankRunner

from RL.rlq.base_RL_with_shared_runner.pairwise_verdict.pairwise_verdict_runner_refactored import PairWiseVerdictRunner
from RL.rlq.base_RL_with_shared_runner.listwise.list_wise_runner_refactored import ListWiseRunner

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
envs= ["orig", "log5", "an"]
num_test_files = 100
eps = 10000
test_frequency = 50


params = {"eps_decay" : eps, "test_length" : 1000, "test_frequency" : 15000}

# for i in range(4):
#     for env in envs:
#         if not(i == 0 and env == "orig"):
#             params = {'eps_decay' : eps, 'test_frequency':test_frequency, 'test_length' : num_test_files}#
#             name = f"{str(env)}{str(i)}"
#             runner= PairWiseRankRunner(**params, des=str(params), save=name, test_train="train", new_load="new", file = "envcomp", env=env)
#             exit = runner.run()
            
        
for i in range(3):  
    runner= PairWiseRankRunner(save=f"pwrstart100{i}", test_train="train", new_load="new", file = "testLength100", **params, des = str(params))
    runner.run()

    # runner= PairWiseVerdictRunner(save=f"pwvstart100{i}", test_train="train", new_load="new", file = "testLength100", **params, des = str(params))
    # runner.run()


# params = {"eps_decay" : 100000, "test_length" : 500, "test_frequency" : 5000}

# runner= PairWiseRankRunner(save="pwr5003", test_train="train", new_load="new", file = "testLength500", **params, des = str(params))
# runner.run()

# runner= PairWiseVerdictRunner(save="pwv5003", test_train="train", new_load="new", file = "testLength500", **params, des = str(params))
# runner.run()
    
#     if exit:
#         if runner.previous_average <= 70:
#             eps += 10000
#         num_test_files += 50
#         if test_frequency < 200:
#             test_frequency += 20
        
#         eps += 10000
        
        
        
# for i in range(3):
#     params = {'eps_decay' : eps, 'test_frequency':test_frequency, 'test_length' : num_test_files}
#     name = f"listlog5 - " + str(i)
#     runner= PairWiseRankRunner(**params, des=str(params), save=name, test_train="train", new_load="new")
#     exit = runner.run()
    
    
# from pair_wise_rank_runner_all_normed_logged import PairWiseRankRunner

    
# for i in range(3):
#     params = {'eps_decay' : eps, 'test_frequency':test_frequency, 'test_length' : num_test_files}#
#     name = f"all_normed_logged - " + str(i)
#     runner= PairWiseRankRunner(**params, des=str(params), save=name, test_train="train", new_load="new")
#     exit = runner.run()
    
    
    
    
    
    