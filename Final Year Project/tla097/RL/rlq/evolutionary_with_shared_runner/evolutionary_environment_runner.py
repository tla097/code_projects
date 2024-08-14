import copy
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


from shared_evo_env import Evolultionary_Environment
from list_wise_evo_runner import ListwiseEvoEnvRunner, ListwiseEnv
from pairwise_verdict_evo_runner import PairWiseVerdictRunner, PairWiseEnv
from pairwise_rank_evo_runner import  PairWiseRankRunner, PairWiseRankEnv





# crossover = ["ar", "n_ar", "cut", "n_cut", None]
# mutation= [None, "random", "availability"]
# runners = [(ListwiseEvoEnvRunner, ListwiseEnv, "lw", 150), (PairWiseVerdictRunner, PairWiseEnv, "pwv", 20), (PairWiseRankRunner, PairWiseRankEnv, "pwr", 20)]
# env_params = {"env" : "an", "num_actions" : 100, "penalty" :-1}

runners = [(PairWiseVerdictRunner, PairWiseEnv, "pwv", 20)]

params = {"eps_decay" : 100000}

crossover = ["ar"]
mutation= ["random"]
env_params = {"env" : "an", "num_actions" : 100, "penalty" :-1, "test_length" : 1000}




for runner in runners:
    for cross in crossover:
        for mut in mutation:
            if not((cross is None) and (mut is None)):
                m = mut
                c = cross
                if cross is None:
                    c = "none"
                if mut is None:
                    m = "none"
                save = str(runner[2]) + "_mut_" + m + "_cross_" + c
                envir = Evolultionary_Environment(SAVE = save, EvoRunner=runner[0], environment_params=env_params, Environment=runner[1], crossover_type=cross, mutation_type=mut, crossover_rate=0.7, random_mutation_rate_threshold=0.1, random_mutation_rate=1, eps_decay=100000, test_length=1000, num_rounds_ran=runner[3])
                envir.run(description="crossover_rate=0.7, random_mutation_rate_threshold=0.1, random_mutation_rate=1, eps_decay=100000, test_length=1000")
            
               
               








# envir = Evolultionary_Environment(SAVE="crossover_testing", EvoRunner=PairWiseVerdictRunner, environment_params=env_params, Environment = PairWiseEnv, crossover_type="n_ar", random_mutation_rate_threshold=0.01, crossover_rate= 0.6, random_mutation_rate=1, mutation_type=None)
# envir.run("rejig testing")