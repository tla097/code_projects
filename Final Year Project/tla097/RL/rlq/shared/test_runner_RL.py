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


from RL.rlq.base_RL_with_shared_runner.listwise.list_wise_runner import Runner as lw_runner
from RL.rlq.base_RL_with_shared_runner.pairwise_rank.pair_wise_rank_runner import PairWiseRankRunner as pw_rank_runner
from RL.rlq.base_RL_with_shared_runner.pairwise_verdict.PairwiseRunner import Runner as pw_verdict_runner

rank_envs = ["an", "log5", "orig"]
verdict_envs = ["an", "log4", "orig"]

tf = [True, False]
r = [-1, -100]


for i in range(5):
    for v in tf:
        for re in rank_envs:
            if v:
                for rew in r:
                    params = {"test_train" :"train", "num_actions" : 10, "eps_decay" : 10000, "truncation_env" : v, "env" : re, "penalty" : rew, "file" : "attempt_5"}
                    name = str(i) +"_trunc=_" + str(v)+ "_env=_" + str(re) + "_rew_" + str(rew)
                    params["save"] = name
                    runner = lw_runner(**params, des = str(params))
                    runner.run()
            else:        
                params = {"test_train" :"train", "num_actions" : 10, "eps_decay" : 10000, "truncation_env" : v, "env" : re, "file" : "attempt_5"}
                name = str(i) +"_trunc=_" + str(v)+ "_env=_" + str(re)
                params["save"] = name
                runner = lw_runner(**params, des = str(params))
                runner.run()
                
                
                
                
for i in range(5):
    for v in tf:
        for re in rank_envs:
            if v:
                for rew in r:
                    params = {"test_train" :"train", "num_actions" : 100, "eps_decay" : 100000, "truncation_env" : v, "env" : re, "penalty" : rew, "file" : "eps-100000"}
                    name = str(i) +"_trunc=_" + str(v)+ "_env=_" + str(re) + "_rew_" + str(rew)
                    params["save"] = name
                    runner = lw_runner(**params, des = str(params))
                    runner.run()
            else:        
                params = {"test_train" :"train", "num_actions" : 100, "eps_decay" : 100000, "truncation_env" : v, "env" : re, "file" : "eps-100000"}
                name = str(i) +"_trunc=_" + str(v)+ "_env=_" + str(re)
                params["save"] = name
                runner = lw_runner(**params, des = str(params))
                runner.run()
                
                
                
for i in range(5):
    for v in tf:
        for re in rank_envs:
            if v:
                for rew in r:
                    params = {"test_train" :"train", "num_actions" : 100, "eps_decay" : 1000000, "truncation_env" : v, "env" : re, "penalty" : rew, "file" : "eps-1000000"}
                    name = str(i) +"_trunc=_" + str(v)+ "_env=_" + str(re) + "_rew_" + str(rew)
                    params["save"] = name
                    runner = lw_runner(**params, des = str(params))
                    runner.run()
            else:        
                params = {"test_train" :"train", "num_actions" : 100, "eps_decay" : 1000000, "truncation_env" : v, "env" : re, "file" : "eps-1000000"}
                name = str(i) +"_trunc=_" + str(v)+ "_env=_" + str(re)
                params["save"] = name
                runner = lw_runner(**params, des = str(params))
                runner.run()


# for i in range(5):
#     for re in rank_envs:
#             params = {"test_train" :"train", "eps_decay" : 10000, "env" : re}
#             name = str(i) +"_rank_pw_env=_" + str(re)
#             params["save"] = name
#             runner = pw_rank_runner(**params, des = str(params))
#             runner.run()
            
            
# for i in range(5):
#     for ve in verdict_envs:
#             params = {"test_train" :"train", "eps_decay" : 10000, "env" : ve}
#             name = str(i) +"verd_pw_env=_" + str(ve)
#             params["save"] = name
#             runner = pw_verdict_runner(**params, des = str(params))
#             runner.run()