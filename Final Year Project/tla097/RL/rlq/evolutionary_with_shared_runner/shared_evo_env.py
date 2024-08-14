import copy
import importlib
import math
from pathlib import Path
import pickle
import random
import threading
import time
import numpy as np

from sklearn.cross_decomposition import CCA

import torch.nn.functional as F

import torch.optim as optim
from RL.rlq.shared.DQN import DQN3
from sklearn.preprocessing import StandardScaler
import networkx as nx

from collections import deque


from RL.rlq.shared.memory_nodes import Memory_Nodes_Stack
import torch
from RL.rlq.shared.ReplayMemory import ReplayMemory, Transition

class Evolultionary_Environment:
    def __init__(self, SAVE, num_actions=100, children_ratio=0.5, pop_size=2, max_generations=100, mutation_type = None, num_samples = 200, num_events = 200, random_mutation_rate = 1, random_mutation_rate_threshold = 0.4, uniform_low = 0.95, uniform_high = 1.05, crossover_rate = 0.3, thread = 1, crossover_type = None, just_evo = None, agent = None, EvoRunner = None, Environment = None, runner_params = None, environment_params = None, eps_decay = 10000, test_length = 50, num_rounds_ran = 20) -> None:
        self.SAVE = SAVE
        self.pop_size = 3
        self.mems = [None] * self.pop_size
        
        self.num_actions = num_actions
        
        self.EvoRunner = EvoRunner
        
        # self.pop_size = 2
        self.max_generations = max_generations
        self.children_number = 2
        
        
        self.runner_params= {"test_length":test_length, "eps_decay":eps_decay, "test_train":"train", "num_actions":num_actions, "random_mutation_rate":random_mutation_rate, "random_mutation_rate_threshold":random_mutation_rate_threshold, "uniform_low":uniform_low, "uniform_high":uniform_high, "num_samples" : num_samples, "num_events" : num_events}

        self.environment_params = environment_params
        self.best = EvoRunner( save = "best", **self.runner_params)
        
        self.start = time.time()
        self.folder_path = "RL/rlq/models/evo/_aNewTests/1/" + self.SAVE
        
        self.mutation_type = mutation_type
        self.num_actions = num_actions
        
        
        self.crossover_rate = crossover_rate
        
        self.thread = thread
        self.crossover_type  = crossover_type
        
        self.just_evo = just_evo
        
        self.environment = Environment
        
        self.num_rounds_ran = num_rounds_ran
        
        self.last_10_generations = deque(maxlen=10)
        
        
    def save_best_citizen(self):
        # saving the best
        if (self.population[0].fitness) > self.best.fitness:
            self.best = self.population[0]
            with open(self.self.folder_path + "/best_thread/policy.pkl", "wb") as p:
                pickle.dump(self.policy_net, p)
            with open(self.folder_path + "/best_thread/target.pkl", "wb") as p:
                pickle.dump(self.policy_net, p)
            with open(self.folder_path + "/best_thread/optimiser.pkl", "wb") as p:
                pickle.dump(self.optimizer, p)
        
        
    def initialise_population(self, pop_size):
        ## initial population
        self.population:list[self.EvoRunner] = [None] * pop_size
        for k in range(pop_size):
            # params = {"test_train": "train","save": self.SAVE,"num_actions": self.num_actions,"random_mutation_rate": self.random_mutation_rate,"random_mutation_rate_threshold": self.random_mutation_rate_threshold}
            runner = self.EvoRunner(save=self.SAVE, **self.runner_params)
            runner.num_rounds= self.num_rounds_ran
            self.population[k] = runner
            self.population[k].memory = Memory_Nodes_Stack()            
            folder_path = self.folder_path + "/thread" + str(k)
            if k == 0:
                pth = self.folder_path + "/best_thread"
            else:
                pth = folder_path
            self.population[k].set_save_location(pth, -1)
            self.population[k].calculate_fitness(initial_fitness = True, just_evo = self.just_evo, save = False)
            
      
            
            # self.population[k].memory = ReplayMemory(10000)
            
        self.population = sorted(self.population, key=lambda x: x.fitness, reverse=True)
        
        self.best = self.population[0]
        
        
        
        
    def  linear_threads(self, env, generation):
        
        # self.gen_envs = [(env.reset(), env.arr) for _ in range(11)]
        self.gen_envs = []
        for _ in range(self.num_rounds_ran):
            reset = env.reset()
            obs = env.observation
            self.gen_envs.append((reset, obs))
        reset = env.reset(test = True)
        obs = env.observation
        self.gen_envs.append((reset, obs))
        start = time.time()
        
        folder_path = ""
            
        
        ## train all models for 10 rounds
        threads = []
        pth= ""
        for index,runner  in enumerate(self.population):
            
            folder_path = self.folder_path + "/thread" + str(index)
            runner.name = str(index)
            runner.envs = self.gen_envs
            if index == 0:
                pth = self.folder_path + "/best_thread"
            else:
                pth = folder_path
            runner.set_save_location(pth, generation)
        
            runner.run(generation)
            
        
        
    def run_threads(self, env, generation):
        # self.gen_envs = [(env.reset(), env.arr) for _ in range(11)]
        self.gen_envs = []
        for _ in range(self.num_rounds_ran):
            reset = env.reset()
            obs = env.observation
            self.gen_envs.append((reset, obs))
            
        reset = env.reset(test=True)
        obs = env.observation
        self.gen_envs.append((reset, obs))
        
        
        # self.gen_envs = [(env.reset(), env.observation) for _ in range(151)]
        
        start = time.time()
        
        folder_path = ""
            
        
        ## train all models for 10 rounds
        threads = []
        pth= ""
        for index,runner  in enumerate(self.population):
            
            folder_path = self.folder_path + "/thread" + str(index)
            runner.name = str(index)
            runner.envs = self.gen_envs
            if index == 0:
                pth = self.folder_path + "/best_thread"
            else:
                pth = folder_path
            runner.set_save_location(pth, generation)
        
            thread = threading.Thread(target=runner.run, args=(generation,), name=str(index))
            threads.append(thread)
            thread.start()
            
        for thread in threads:
            thread.join()
                
    def create_files(self, description):
        if not Path(self.folder_path).exists():
            print(f"The file {self.folder_path} does not exist - creating folder.")
            Path(self.folder_path).mkdir(parents=True, exist_ok=True)
            
            with open(self.folder_path + "/description.txt", "w") as f:
                f.write(description)
        
            
            for i in range(1, self.pop_size):
                Path(self.folder_path + "/thread" + str(i)).mkdir(parents=True, exist_ok=True)
                
            Path(self.folder_path + "/best_thread").mkdir(parents=True, exist_ok=True)
    
    def run(self, description):
        self.create_files(description)    
        env = self.environment(**self.environment_params)
        self.initialise_population(self.pop_size)
        
        
        for i in range(self.max_generations):
            
            # self.run_threads(env, i)
            self.linear_threads(env,i)
            parents = self.tournament_selection()
            children = self.variation(parents)
            self.reproduction(children)
            self.save_best_citizen()
                            
            var = f"generation {i} time = {time.time() - self.start}"
            with open(self.folder_path + "/time.txt", "a") as f:
                f.write(f"generation {i} time = {var}\n")
            
        
        
            if i > 100:
                if sum(self.last_10_generations)/10 >= self.best.fitness:
                    print("AVERAGE REACHED")
                    break        
        
            self.last_10_generations.append(self.best.fitness)
            print(f"generation {i} time = {var}")
            
            
        
        
    def run_just_evo(self, description):
        self.create_files(description)    
        
        env = self.environment(self.environment_params)
        
        self.initialise_population(self.pop_size)
        for i in range(self.max_generations):
            
            self.gen_envs = []
            for _ in range(1):
                reset = env.reset()
                obs = env.observation
                self.gen_envs.append((reset, obs))
                
            for citizen in self.population:
                citizen.envs = self.gen_envs
            
            parents = self.tournament_selection()
            children = self.variation(parents)
            self.reproduction(children)
            
            self.save_best_citizen()
                            
            var = f"generation {i} time = {time.time() - self.start}"
            with open(self.folder_path + "/time.txt", "a") as f:
                f.write(f"generation {i} time = {var}\n")
            
        print(f"generation {i} time = {var}")
                
                
    def reproduction(self, children):
        print("reproduction")
        
        population = self.population + children
        self.population = sorted(population, key=lambda x: x.fitness, reverse=True)
        
        for chap in self.population:
            print(chap.fitness)
            chap.memory.reset_stack()

            
        print("\n")
        c = -1
        if len(children) != 0: 
            self.population = self.population[:-len(children)]
        for citizen in self.population:
            c +=1
            
            # print(citizen.fitness)
            save = "`"
            if c == 0:
                save = self.folder_path + "/best_thread/fitness_calc.txt"
            else:
                save = self.folder_path + "/thread" + str(c) + "/fitness_calc.txt"
                
            with open(save, "a") as f:
                f.write(f"{citizen.fit_string}\n")
                
                
            folder_path = self.folder_path + "/thread" + str(c)
            if c == 0:
                pth = self.folder_path + "/best_thread"
            else:
                pth = folder_path
            self.population[c].set_save_location(pth, -1)
            
            
        self.best = self.population[0]
            
        for chap in self.population:
            print(chap.fitness)


                
    def tournament_selection(self):
        
        for citizen in self.population:
            citizen.reset_selected()
        
        print("selection")
        parents = []
        size = 2
        for i in range(self.children_number):
            counter = 0
            while True:
                counter +=1
                fighters= random.sample(self.population, size)
                if counter == 10:
                    input()
                print(21313213)
                print(self.population)
                if not(fighters[0].selected) and not(fighters[1].selected):
                    break 
            
            best = fighters[0]
            
            for fighter in fighters:
                if fighter.fitness > best.fitness:
                    best = fighter
                    
            best.set_selected()
            parents.append(best)
            
        for citizen in self.population:
            citizen.reset_selected()
        
        return parents
    
    
    # def correlation_cut(self, parent1, parent2, layers):
    #     n1_activations, n2_activations= self.get_standardised_activations(parent1, parent2)
    #     # get correlation table
    #     zipped_correlation = [(x, y) for x, y in zip(*self.get_corr(n1_activations, n2_activations))]
        
    #     self.cut_point_crossover(parent1, parent2, layers, zipped_correlation)
    
    
    def niiave_cut_cross(self, parent1, parent2, layers):
        # get policy net state dicts
        parent1_policy_sd = parent1.policy_net.state_dict()
        parent2_policy_sd = parent2.policy_net.state_dict()
                
        # get target net state dicts
        parent1_target_sd = parent1.target_net.state_dict()
        parent2_target_sd = parent2.target_net.state_dict()
                
        # get cut location
        cut_location = random.randrange(60, 68)
        
        # do the crossover
        parent1_policy_sd, parent2_policy_sd = self.do_cut_cross(parent1_policy_sd, parent2_policy_sd, layers, cut_location)
        parent1_target_sd, parent2_target_sd = self.do_cut_cross(parent1_target_sd, parent2_target_sd, layers, cut_location)
        
        # create child from these
        self.create_child(parent1, parent1_policy_sd, parent1_target_sd)
        self.create_child(parent2, parent2_policy_sd, parent2_target_sd)
        
        
    
    def pairwise_cross_corr_cut(self, parent1, parent2):
        activations1, activations2 = self.get_standardised_activations(parent1, parent2)
        pairs = self.pairwise_cca_pairs(activations1, activations2)
        self.cut_point_crossover(parent1, parent2, [1], pairs)
    
    def pairwise_cross_corr_arithmetic(self, parent1, parent2):
        activations1, activations2 = self.get_standardised_activations(parent1, parent2)
        pairs = self.pairwise_cca_pairs(activations1, activations2)
        p1, p2 = self.arithmetic_cross(parent1, parent2, pairs, [1])
        
        return p1, p2
        
    def arithmetic_cross(self, parent1, parent2, zipped_correlation, layers):
        
        # get policy net state dicts
        parent1_policy_sd = parent1.policy_net.state_dict()
        parent2_policy_sd = parent2.policy_net.state_dict()
        
                
        # get target net state dicts
        parent1_target_sd = parent1.target_net.state_dict()
        parent2_target_sd = parent2.target_net.state_dict()
        
        
        # permute parent 1
        parent1_policy_sd = self.permute_layers(zipped_correlation, parent1_policy_sd, layers)
        parent1_target_sd = self.permute_layers(zipped_correlation, parent1_target_sd, layers)
        
        t = random.uniform(-0.25, 1.25)
        
        # do arithmetic crossover 
        parent1_policy_sd, parent2_policy_sd = self.do_arithmetic_cross(parent1_policy_sd, parent2_policy_sd, layers, t)
        
        parent1_target_sd, parent2_target_sd = self.do_arithmetic_cross(parent1_target_sd, parent2_target_sd, layers, t)
         
        
        # create child from these
        p1 = self.create_child(parent1, parent1_policy_sd, parent1_target_sd) 
        p2 = self.create_child(parent2, parent2_policy_sd, parent2_target_sd)

        return p1, p2
        
    def naive_arithmetic_cross(self, parent1, parent2, layers):

         # get policy net state dicts
        parent1_policy_sd = parent1.policy_net.state_dict()
        parent2_policy_sd = parent2.policy_net.state_dict()
                
        # get target net state dicts
        parent1_target_sd = parent1.target_net.state_dict()
        parent2_target_sd = parent2.target_net.state_dict()
        
        
        t = random.uniform(-0.25, 1.25)
        
        # do arithmetic crossover 
        parent1_policy_sd, parent2_policy_sd = self.do_arithmetic_cross(parent1_policy_sd, parent2_policy_sd, layers, t)
        parent1_target_sd, parent2_target_sd = self.do_arithmetic_cross(parent1_target_sd, parent2_target_sd, layers, t)
        
        # create child from these
        newParent1 = self.create_child( parent1_policy_sd, parent1_target_sd)
        newParent2 = self.create_child(parent2, parent2_policy_sd, parent2_target_sd)
        
        return newParent1, newParent2
            
            
    def do_arithmetic_cross(self, parent1_state_dict, parent2_state_dict, layers, t):
        for layer in layers:
            for key in [f"layer{str(layer)}.weight", f"layer{str(layer)}.bias", "last_layer.weight"]:
                parent1_state_dict[key] = (1 -t)*parent1_state_dict[key] + t*parent2_state_dict[key]
                parent2_state_dict[key] = (1-t)*parent2_state_dict[key] + t*parent1_state_dict[key]

        return parent1_state_dict, parent2_state_dict
        
        
    
    
    def pairwise_cca_pairs(self, activations1, activations2):
        
        ha = activations1.T
        hb = activations2.T

        sh = activations1.shape[1]
        
        mat = np.zeros((sh,sh))
        for j in range(sh):
            for i in range(sh):
                if np.var(ha[i]) == 0:
                    ha[i][5] += 1e-8
                if np.var(hb[j]) == 0:
                    hb[j][5] += 1e-8
                    
                res = np.dot(ha[i], hb[j]) / math.sqrt(np.var(ha[i]) * np.var(hb[j]))
                mat[i,j] = res
            
                


        # import numpy as np

        # Given 2D NumPy array

        # Convert the array into a set of weighted edges
        weighted_edges = []

        # Assign node labels to each element
        vertical_nodes = [('ha' + str(i)) for i in range(sh)]
        horizontal_nodes = [('hb' + str(i)) for i in range(sh)]

        # Iterate over each element in the array and create edges
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                source_node = vertical_nodes[i]
                target_node = horizontal_nodes[j]
                weight = mat[i, j]
                weighted_edges.append((source_node, target_node, weight))

        # print(weighted_edges)
        
        # sorted_pairs = sorted(weighted_edges, key=lambda x: (x[0].startswith('ha'), x))
        # # sorted_pairs = sorted(sorted_pairs, key=lambda x: int(x[0][2:]))
        # print(sorted_pairs)



        G = nx.Graph()
        G.add_nodes_from(vertical_nodes, bipartite=0)  # Set the node attribute "bipartite" for the top nodes
        G.add_nodes_from(horizontal_nodes, bipartite=1)  # Set the node attribute "bipartite" for the bottom nodes
        G.add_weighted_edges_from(weighted_edges)
        matching = nx.algorithms.matching.max_weight_matching(G)
        
        
        swapped_pairs = [(int(y[2:]), int(x[2:])) if y.startswith('ha') else (int(x[2:]), int(y[2:])) for x, y in matching]

        return swapped_pairs
    

    
    def get_standardised_activations(self, parent1, parent2):
        
        n1_activations = np.zeros((128, parent1.BATCH_SIZE))
        n2_activations = np.zeros((128, parent2.BATCH_SIZE))
       
        for i in range(128):
            random_tensor = parent1.memory.sample(1)[0].state
            n1_tensor = F.relu(parent1.policy_net.layer1(random_tensor))
            n2_tensor = F.relu(parent2.policy_net.layer1(random_tensor))
            
            n1_activations[i] = n1_tensor.detach().cpu().numpy()
            n2_activations[i] = n2_tensor.detach().cpu().numpy()
            
        # scaler = StandardScaler()# Fit data on the scaler object
        # n1_activations = scaler.fit_transform(n1_activations)
        # n2_activations = scaler.fit_transform(n2_activations)

        return n1_activations, n2_activations
    
    
    def cca_cut(self, parent1, parent2, layers):
        n1_activations, n2_activations = self.get_standardised_activations(parent1, parent2)
        cca = CCA(n_components=128)
        cca.fit(n1_activations, n2_activations)
        X_c, Y_c = cca.transform(n1_activations, n2_activations)
        zipped_correlation = [(x, y) for x, y in zip(*self.get_ordered_indices(X_c, Y_c))]
        self.cut_point_crossover(parent1, parent2, layers, zipped_correlation)
        
        
    def correlation_cut(self, parent1, parent2, layers):
        n1_activations, n2_activations= self.get_standardised_activations(parent1, parent2)
        # get correlation table
        zipped_correlation = [(x, y) for x, y in zip(*self.get_corr(n1_activations, n2_activations))]
        
        self.cut_point_crossover(parent1, parent2, layers, zipped_correlation)
        
        
        
        
        
    
    
    def cut_point_crossover(self, parent1, parent2, layers, zipped_correlation): 
        # get policy net state dicts
        parent1_policy_sd = parent1.policy_net.state_dict()
        parent2_policy_sd = parent2.policy_net.state_dict()
                
        # get target net state dicts
        parent1_target_sd = parent1.target_net.state_dict()
        parent2_target_sd = parent2.target_net.state_dict()
        
        # permute parent 1
        parent1_policy_sd = self.permute_layers(zipped_correlation, parent1_policy_sd, layers)
        parent1_target_sd = self.permute_layers(zipped_correlation, parent1_target_sd, layers)
                
        # get cut location
        cut_location = random.randrange(60, 68)
        
        # do the crossover
        parent1_policy_sd, parent2_policy_sd = self.do_cut_cross(parent1_policy_sd, parent2_policy_sd, layers, cut_location)
        parent1_target_sd, parent2_target_sd = self.do_cut_cross(parent1_target_sd, parent2_target_sd, layers, cut_location)
        
        # create child from these
        self.create_child(parent1, parent1_policy_sd, parent1_target_sd)
        self.create_child(parent2, parent2_policy_sd, parent2_target_sd)
        
        
        
        
        
    def do_cut_cross(self, state_dict1, state_dict2, layers, cut_location):
        
        device = torch.device(str("cuda") if torch.cuda.is_available() else "cpu")
        
        for layer in layers:
            
        #     if f"layer{layer + 1}.weight" in state_dict.keys():
        #     keys = [f"layer{layer}.weight", f"layer{layer}.bias",f"last{layer + 1}.weight"]
        # else:
        #     keys = [f"layer{layer}.weight", f"layer{layer}.bias",f"last_layer.weight"]
            for key in [f"layer{layer}.weight", f"layer{layer}.bias", f"last_layer.weight"]:
                nn_layer1 = state_dict1[key]
                nn_layer2 = state_dict2[key]
                shape = nn_layer1.shape
                
                zeros_first = torch.zeros(shape, device= device)  # Shape: (4, 2)
                ones_first = torch.ones(shape, device = device)
                
                ones_first[cut_location:] = 0
                zeros_first[cut_location:] = 1
                
                first_half1 = nn_layer1 * ones_first
                second_half1 = nn_layer1 * zeros_first
                
                first_half2 = nn_layer2 * ones_first
                second_half2 = nn_layer2 * zeros_first
                
                state_dict1[key] = first_half1 + second_half2
                state_dict2[key] = first_half2 + second_half1
                
            return state_dict1, state_dict2
        
        
        
    def swapping_mutate(self, parent1):
        policy_state_dict1 = parent1.policy_net.state_dict()
        
        rand1 = random.randint(0, 127)
        rand2 = random.randint(0, 127)
        
        policy_state_dict = self.swap_nodes(rand1, rand2, policy_state_dict1, 1)
        
        parent1.policy_net.load_state_dict(policy_state_dict)
        
        
    def single_swap_crossover(self, parent1, parent2, layers):
        
        node_to_swap = random.randint(0, 127)
        
        policy1 = parent1.policy_net.state_dict()
        policy2 = parent2.policy_net.state_dict()
        
        target1 = parent1.target_net.state_dict()
        target2 = parent2.target_net.state_dict()
        
        policy1 = self.permute_layers([(127, node_to_swap)], policy1, layers)
        policy2 = self.permute_layers([(127, node_to_swap)], policy2, layers)
        
        target1 = self.permute_layers([(127, node_to_swap)], target1, layers)
        target2 = self.permute_layers([(127, node_to_swap)], target2, layers)
        
        policy1, policy2 = self.do_cut_cross(policy1, policy2,layers, 127)
        target1, target2 = self.do_cut_cross(target1, target2,layers, 127)
        
        self.create_child(parent1, policy1, policy1)
        self.create_child(parent2, policy2, target2)
    
    
    def crossover_caa(self, parent1, parent2, layers):
        
        comparison_input = parent1.memory.sample(1)[0].state
        
        print(comparison_input.detach().cpu().numpy().shape)
        
        n1_activations = np.zeros((128, parent1.BATCH_SIZE))
        n2_activations = np.zeros((128, parent2.BATCH_SIZE))
        
        n1_tensor = F.relu(parent1.policy_net.layer1(comparison_input))
        n2_tensor = F.relu(parent2.policy_net.layer1(comparison_input))
            
        n1_activations[0] = n1_tensor.detach().cpu().numpy()
        n2_activations[0] = n2_tensor.detach().cpu().numpy()
       
        for i in range(1,128):
            random_tensor = parent1.memory.sample(1)[0].state
            n1_tensor = F.relu(parent1.policy_net.layer1(random_tensor))
            n2_tensor = F.relu(parent2.policy_net.layer1(random_tensor))
            
            n1_activations[i] = n1_tensor.detach().cpu().numpy()
            n2_activations[i] = n2_tensor.detach().cpu().numpy()
        
        scaler = StandardScaler()# Fit your data on the scaler object
        relu_layer_one = scaler.fit_transform(n1_activations)
        relu_layer_two = scaler.fit_transform(n2_activations)
        
        
        cca = CCA(n_components=128)
        try:
            cca.fit(relu_layer_one, relu_layer_two)
        except:
            print(relu_layer_one, relu_layer_two)
        X_c, Y_c = cca.transform(relu_layer_one, relu_layer_two)
        
        la, lb = self.get_ordered_indices(X_c, Y_c)
        
        print(sorted(la))
        
        print(len(la) != len(set(la)))
        print(set(la))
        
        print(la)
        
        print(len(lb) != len(set(lb)))
        
        zipped_correlation = [(x, y) for x, y in zip(la, lb)]
        
        # get policy net state dicts
        parent1_sd = parent1.policy_net.state_dict()
        parent2_sd = parent2.policy_net.state_dict()
        
        # permute parent 1
        parent1_sd = self.permute_layers(zipped_correlation, parent1_sd, layers)
        
        # t = random.uniform(-0.5, 0.5)
        
        t = self.low_cross
        
        # return crossver parents state_dict
        parent1_policy_sd, parent2_policy_sd = self.arithmetic_cross(parent1_sd, parent2_sd, layers, t)

        # now do with target_net
        target1 = parent1.target_net.state_dict()
        target2 = parent2.target_net.state_dict()
        
        # permute_target_1_weights
        target1 = self.permute_layers(zipped_correlation, target1, layers)
        
        # target_network_child_weights
        target1_child_sd, target2_child_sd = self.arithmetic_cross(target1, target2, layers, t)
        
        # create child as copy of parent
        self.create_child(parent1, parent1_policy_sd, target1_child_sd)
        self.create_child(parent2, parent2_policy_sd, target2_child_sd)
    

    def crossover(self, parent1, parent2, layers : int):
            
        n1_activations = np.zeros((10, parent1.BATCH_SIZE))
        n2_activations = np.zeros((10, parent2.BATCH_SIZE))
       
        for i in range(10):
            random_tensor = parent1.memory.sample(1)[0].state
            n1_tensor = F.relu(parent1.policy_net.layer1(random_tensor))
            n2_tensor = F.relu(parent2.policy_net.layer1(random_tensor))
            
            n1_activations[i] = n1_tensor.detach().cpu().numpy()
            n2_activations[i] = n2_tensor.detach().cpu().numpy()
            
            
            
        self.pairwise_cca(n1_activations, n2_activations)
            
            
            
        # w1, w2 = self.get_corr(n1_activations, n2_activations)
        
        
        # get correlation table
        zipped_correlation = [(x, y) for x, y in zip(*self.get_corr(n1_activations, n2_activations))]
        
        print(zipped_correlation)
        
        # input()
        
        # get policy net state dicts
        parent1_sd = parent1.policy_net.state_dict()
        parent2_sd = parent2.policy_net.state_dict()
        
        # permute parent 1
        parent1_sd = self.permute_layers(zipped_correlation, parent1_sd, layers)
        
        # t = random.uniform(-0.5, 0.5)
        
        t = self.low_cross
        
        # return crossver parents state_dict
        parent1_policy_sd, parent2_policy_sd = self.arithmetic_cross(parent1_sd, parent2_sd, layers, t)

        # now do with target_net
        target1 = parent1.target_net.state_dict()
        target2 = parent2.target_net.state_dict()
        
        # permute_target_1_weights
        target1 = self.permute_layers(zipped_correlation, target1, layers)
        
        # target_network_child_weights
        target1_child_sd, target2_child_sd = self.arithmetic_cross(target1, target2, layers, t)
        
        # create child as copy of parent
        self.create_child(parent1, parent1_policy_sd, target1_child_sd)
        self.create_child(parent2, parent2_policy_sd, target2_child_sd)
        
        
    def create_child(self, old_parent, policy_state_dict, target_state_dict):
        
        newParent = self.EvoRunner(self.runner_params)
        newParent.folder_path = old_parent.folder_path
        newParent.eps_threshold = old_parent.eps_threshold
        newParent.total_steps = old_parent.total_steps
        
        old_parent.memory.make_child()
        newParent.memory = Memory_Nodes_Stack(parent=old_parent.memory)
        
        newParent.policy_net.load_state_dict(policy_state_dict)
        newParent.target_net.load_state_dict(target_state_dict)
        newParent.optimizer = optim.AdamW(old_parent.policy_net.parameters(), lr=old_parent.LR, amsgrad=True)
        
        newParent.envs= old_parent.envs
        newParent.fitness = old_parent.fitness
        
        return newParent
        
    
    
        
        
        # plan continue accewing data
        # continue this and set it off
        
    
    
    def crossover_attempt(self, parent1, parent2, layers = [1]):
        if random.random() <= self.crossover_rate:
            return self.crossover(parent1, parent2, layers=layers)
            
            
            
            
              
    
    def swap_nodes(self, swap1, swap2, state_dict, layer):
        
        # for keys in state_dict.keys():
        #     print(keys)
        # input()
        
        if f"layer{layer + 1}.weight" in state_dict.keys():
            keys = [f"layer{layer}.weight", f"layer{layer}.bias",f"last{layer + 1}.weight"]
        else:
            keys = [f"layer{layer}.weight", f"layer{layer}.bias",f"last_layer.weight"]
            
        for key in keys:
            tor = state_dict[key]
            if key != keys[2]:
                try:
                    tor[[swap1,swap2]] = tor[[swap2,swap1]]
                except:
                    tor[:, [swap1,swap2]] = tor[:, [swap2,swap1]]
            else:
                for out in state_dict[keys[2]]:
                    out[[swap1,swap2]] = out[[swap2,swap1]]
        return state_dict


    def permute_layers(self, swaps, state_dict, layers):
        for layer in layers:
            already_swapped = []
            to_change = {}
            for index1, index2 in swaps:
                if index1 in to_change.keys():
                    index1 = to_change[index1]
                if index1 != index2:
                    if (index1, index2) not in already_swapped:
                        if index1 != index2:
                            if (index1, index2) not in already_swapped:
                                state_dict = self.swap_nodes(index1, index2, state_dict, layer)
                                to_change[index2] = index1
                                already_swapped.append((index2, index1))
        return state_dict
                    
    def standardize_data(data):
        mean = np.mean(data, axis=0)
        std_dev = np.std(data, axis=0)
        standardized_data = (data - mean) / std_dev
        return standardized_data
    
    
    
    
    

        
    def variation(self, parents):
        print("variation")
        
        with open("tes1.txt", "w") as f:
            f.write("parent fitness\n")
        with open("tes2.txt", "w") as f:
            f.write("parent fitness\n")
            
        with open("ARR1.txt", "w") as f:
            f.write("parent\n")
        with open("ARR2.txt", "w") as f:
            f.write("parent\n")
        # children = copy.deepcopy(parents)
        print(f"length of parents {len(parents)}")
        children = [None] * len(parents)
        for i, parent in enumerate(parents):
            with open(str(parent.folder_path) + '/_model.pkl', 'wb') as f:
                pickle.dump(parent, f)

            with open(str(parent.folder_path) + '/_model.pkl', 'rb') as f:
                child = pickle.load(f)
                
            
            parent.memory.make_child()
            child.memory = Memory_Nodes_Stack(parent=parent.memory)
            
            
            children[i] = child
            

            

        children_to_return = []
        if self.mutation_type is not None:
            for child in children:
                if self.mutation_type == "random":
                    
                    # if child.mutate_attempt_random_weight_change():
                    if random.random() <= child.random_mutation_rate:
                        newParent= self.create_child(child, child.policy_net.state_dict(), child.target_net.state_dict())
                        children_to_return.append(newParent)
                        print("mutated")
                        
                elif self.mutation_type == "availability":
                    if len(child.memory) != 0:
                        if child.mutate_attempt(self.best):
                            children_to_return.append(child)
                            print("mutated")
            
        if len(children) >= 2:
            if self.crossover_type is not None:
                if self.crossover_type == "cut":
                    for i in range(int(self.children_number / 2)):
                        if self.crossover_rate >= random.random():
                            child1, child2 = tuple(random.sample(children, 2))
                            if len(child1.memory) != 0:
                                if len(child2.memory) != 0:
                                    self.pairwise_cross_corr_cut(child1, child2)
                                    children_to_return.append(child1)
                                    children_to_return.append(child2)
                                    
                            
                elif self.crossover_type == "ar":
                    for i in range(int(self.children_number / 2)):
                        if self.crossover_rate >= random.random():
                            child1, child2 = tuple(random.sample(children, 2))
                            if len(child1.memory) != 0:
                                if len(child2.memory) != 0:
                                    p1, p2 = self.pairwise_cross_corr_arithmetic(child1, child2)
                                    children_to_return.append(p1)
                                    children_to_return.append(p2)
                            
                elif self.crossover_type == "n_cut":
                    for i in range(int(self.children_number / 2)):
                        if self.crossover_rate >= random.random():
                            child1, child2 = tuple(random.sample(children, 2))
                            if len(child1.memory) != 0:
                                if len(child2.memory) != 0:
                                    self.niiave_cut_cross(child1, child2, [1])
                                    children_to_return.append(child1)
                                    children_to_return.append(child2)
                            
                elif self.crossover_type == "n_ar":
                    for i in range(int(self.children_number / 2)):
                        if self.crossover_rate >= random.random():
                            child1, child2 = tuple(random.sample(children, 2))
                            if len(child1.memory) != 0:
                                if len(child2.memory) != 0:
                                    self.naive_arithmetic_cross(child1, child2, [1])
                                    children_to_return.append(child1)
                                    children_to_return.append(child2)
                        
                        
                        
        
        for l in range(len(children_to_return)):    
            for k in range(len(children_to_return)):
                if l != k:
                    if id(children_to_return[l]) == id(children_to_return[k]):
                        new_child = copy.deepcopy(children_to_return[l])
                        children_to_return[l] = new_child
                        print("into action")
                        input() 
                
        
        for child in children_to_return:
            pre_fit = child.fitness
            print(f"pre fitness = {child.fitness}")
            child.calculate_fitness(just_evo = self.just_evo, save = False)
            post_fit = child.fitness
            print(f"post crossover = {child.fitness}")
            
            string = ""
            if post_fit > pre_fit:
                string ="--------------BETTER"
            with open(self.folder_path + "/child_comparison.txt", "a") as f:
                f.write(f"parent fitness {pre_fit} - child fitness {post_fit} {string}\n")
                
            print(f"parent fitness {pre_fit} - child fitness {post_fit} {string}\n")
        return children_to_return
        
        
    
# test1 = Evo_runner()
# test2 = Evo_runner()
# test2.memory= test1.memory
# test1.memory.push(1,1,1,1)

# print(test2.memory.sample(1))

# start = time.time()
# environment = Evolultionary_Environment("with_prob_mutation", num_actions=10)
# environment.run("test")

# end = time.time() - start

    # def get_ordered_indices(self, wa, wb):

    #     la = []
    #     lb = []
        
    #     la_indicies  = []
    #     lb_indicies = []

    #     for k in range(min(len(wa), len(wb))):
            
    #         wa_mean = np.mean(wa[k])
    #         wb_mean = np.mean(wb[k])

    #         wa[k][la_indicies] = wa_mean
    #         wb[k][lb_indicies] = wb_mean

    #         sk_minus = abs(min(wa[k]) + min(wb[k]))
    #         sk_plus = abs(max(wa[k]) + max(wb[k]))

    #         la_index = lb_index = 0 

    #         if sk_plus > sk_minus:
    #             wa[k][la_indicies] = -np.inf
    #             wb[k][lb_indicies] = -np.inf
    #             la_index = np.argmax(wa[k])
    #             lb_index = np.argmax(wb[k])
    #         else:
    #             wa[k][la_indicies] = np.inf
    #             wb[k][lb_indicies] = np.inf
    #             la_index = np.argmin(wa[k])
    #             lb_index = np.argmin(wb[k])
                
                
    #         la.append(la_index)
    #         la_indicies.append(la_index)
    #         lb.append(lb_index)
    #         lb_indicies.append(lb_index)

    #     return la, lb
    
    
    # def get_corr(self, relu_original_one, relu_original_two):
    
    #     axis_number = 0
    #     semi_matching = False
    #     n = relu_original_one.shape[1]
    #     list_neurons_x = []
    #     list_neurons_y = []
        
    #     scaler = StandardScaler()# Fit your data on the scaler object
    #     relu_layer_one = scaler.fit_transform(relu_original_one)
    #     relu_layer_two = scaler.fit_transform(relu_original_two)

    #     corr_matrix_nn = np.empty((n,n))

    #     for i in range(n):
    #         for j in range(n):
    #             corr = np.corrcoef(relu_layer_one[:,i], relu_layer_two[:,j])[0,1]
    #             corr_matrix_nn[i,j] = corr

    #     corr_matrix_nn[np.isnan(corr_matrix_nn)] = -1
        
    #     #argmax_columns = np.argmax(corr_matrix_nn, axis=axis_number)
    #     argmax_columns = np.flip(np.argsort(corr_matrix_nn, axis=axis_number), axis=axis_number)
    #     dead_neurons = np.sum(corr_matrix_nn, axis=axis_number) == n*(-1) # these are neurons that always output 0 (dead relu)
    #     for index in range(n):
    #         if dead_neurons[index] == False:
    #             if semi_matching:
    #                 if axis_number == 0:
    #                     list_neurons_y.append(index)
    #                     list_neurons_x.append(argmax_columns[0,index])
    #                 elif axis_number == 1:
    #                     list_neurons_x.append(index)
    #                     list_neurons_y.append(argmax_columns[index,0])
                        
    #             elif semi_matching == False:
                    
    #             # do not allow same matching
    #                 for count in range(n):

    #                     if axis_number == 0:
    #                         if argmax_columns[count,index] not in list_neurons_x:
    #                             list_neurons_y.append(index)
    #                             list_neurons_x.append(argmax_columns[count,index])
    #                             break
    #                     elif axis_number == 1:
    #                         if argmax_columns[index,count] not in list_neurons_y:
    #                             list_neurons_x.append(index)
    #                             list_neurons_y.append(argmax_columns[index,count])
    #                             break
        
    #     # randomly pair the unpaired neurons
    #     for index in range(n):
    #         if index not in list_neurons_x and len(list_neurons_x) < n:
    #             list_neurons_x.append(index)
    #         if index not in list_neurons_y and len(list_neurons_y) < n:
    #             list_neurons_y.append(index)
        
    #     return list_neurons_x, list_neurons_y

        
            

class Test_Class():
    def __init__(self) -> None:
        self.s = 0
    def run(self):
        self.s += 1
        
    def print_s(self):
        print(self.s)
        


# # Define a shared variable
# shared_variable = 0

# # Define a lock
# lock = threading.Lock()

# # Function to increment the shared variable safely
# def increment_shared_variable():
#     global shared_variable
#     # Acquire the lock
#     lock.acquire()
#     try:
#         shared_variable += 1
#         print("Shared variable incremented to:", shared_variable)
#     finally:
#         # Release the lock
#         lock.release()

# # Create threads to increment the shared variable
# threads = []
# for _ in range(100):
#     t = threading.Thread(target=increment_shared_variable)
#     threads.append(t)
#     t.start()

# # Wait for all threads to finish
# for t in threads:
#     t.join()

# print("Final value of shared variable:", shared_variable)


# test = Test_Class()

# thread = threading.Thread(target=test.run, name=str(0))
# thread.start()
# thread.join()

# test.print_s()
