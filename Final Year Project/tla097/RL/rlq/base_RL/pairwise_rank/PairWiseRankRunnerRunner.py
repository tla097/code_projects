from pair_wise_rank_runner import PairWiseRankRunner

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
num_test_files = 50
eps = 10000
test_frequency = 20
# while num_test_files != 10000:
#     params = {'eps_decay' : eps, 'test_frequency':test_frequency, 'test_length' : num_test_files}#
#     name = f"listlog5"
#     # name = "random_tester_fd"
#     runner= PairWiseRankRunner(**params, des=str(params), save=name, test_train="train", new_load="new")
#     exit = runner.run()
    
    
#     if exit:
#         if runner.previous_average <= 70:
#             eps += 10000
#         num_test_files += 50
#         if test_frequency < 200:
#             test_frequency += 20
        
#         eps += 10000
        
        
        
for i in range(3):
    params = {'eps_decay' : eps, 'test_frequency':test_frequency, 'test_length' : num_test_files}
    name = f"listlog5 - " + str(i)
    runner= PairWiseRankRunner(**params, des=str(params), save=name, test_train="train", new_load="new")
    exit = runner.run()
    
    
from pair_wise_rank_runner_all_normed_logged import PairWiseRankRunner

    
for i in range(3):
    params = {'eps_decay' : eps, 'test_frequency':test_frequency, 'test_length' : num_test_files}#
    name = f"all_normed_logged - " + str(i)
    runner= PairWiseRankRunner(**params, des=str(params), save=name, test_train="train", new_load="new")
    exit = runner.run()
    
    
    
    
    
    