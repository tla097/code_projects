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
eps = 10000
test_frequency = 20
# from PairwiseRunner import Runner
# for i in range(5):
#         params = {'eps_decay' : eps}#
#         name = f"log4" + str(i)
#         runner= Runner(**params, des=str(params), save=name, test_train="train", new_load="new")
#         exit = runner.run()
        
        
        
from PairwiseRunnerAllNormedLogged import Runner
for i in range(5):
        params = {'eps_decay' : eps}#
        name = f"allnormed" + str(i)
        runner= Runner(**params, des=str(params), save=name, test_train="train", new_load="new")
        exit = runner.run()
    
    