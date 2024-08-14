# from list_wise_runner_truncation_trial import Runner


# params = {"test_train" :"train", "num_actions" : 10, "eps_decay" : 10000}
# name = "dont_inc_truc_dummy-10"
# params["save"] = name
# runner = Runner(**params, des = str(params))
# runner.run()



from list_wise_runner import Runner


params = {"test_train" :"train", "num_actions" : 100, "eps_decay" : 10000}
name = "no_trunc-10-orig"
params["save"] = name
runner = Runner(**params, des = str(params))
runner.run()