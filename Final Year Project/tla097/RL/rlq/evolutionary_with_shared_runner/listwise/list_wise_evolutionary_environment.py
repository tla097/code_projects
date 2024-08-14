import copy
import importlib
from pathlib import Path
import pickle
import random
import threading
import time
import numpy as np

from sklearn.cross_decomposition import CCA

import torch.nn.functional as F

import torch.optim as optim

from DQN import DQN3

from sklearn.preprocessing import StandardScaler



from memory_nodes import Memory_Nodes_Stack

import torch

from list_wise_evolution_runner import Evo_runner
from RL.rlq.shared.ReplayMemory import ReplayMemory, Transition

from listwise_env import ListwiseEnv

class Evolultionary_Environment:
    def __init__(self, SAVE = "test", num_actions = 10, children_ratio = 0.5, pop_size = 2, max_generations=250, low_cross = -0.25, high_cross= 1.25) -> None:
        self.SAVE = SAVE
        
        self.pop_size = 2
        self.mems = [None] * self.pop_size
        
        self.num_actions = num_actions
        
        self.pop_size = 12
        self.max_generations = max_generations
        self.children_number = 6
        
        self.folder_path = "RL/rlq/models/evo/" + self.SAVE
        self.best = Evo_runner()
        
        self.start = time.time()
        
        self.low_cross = low_cross
        self.high_cross= high_cross
        
        
    def save_best_citizen(self):
        # saving the best
        if (self.population[0].fitness) > self.best.fitness:
            self.best = self.population[0]
            torch.save(self.best.policy_net.state_dict(), self.folder_path + "/best_thread/" + "policy.pth")
            torch.save(self.best.target_net.state_dict(), self.folder_path + "/best_thread/" + "target.pth")
            torch.save(self.best.optimizer.state_dict(), self.folder_path + "/best_thread/" + "optimiser.pth")
        
        
    def initialise_population(self, pop_size):
        ## initial population
        self.population:list[Evo_runner] = [None] * pop_size
        for k in range(pop_size):
            params = {"test_train":"train", "save":self.SAVE, "num_actions":self.num_actions}
            self.population[k] = Evo_runner(test_train="train", save=self.SAVE, num_actions=self.num_actions)
            self.population[k].memory = Memory_Nodes_Stack()
            self.population[k].calculate_fitness(initial_fitness = True)
            
            
            # self.population[k].memory = ReplayMemory(10000)
            
        self.population = sorted(self.population, key=lambda x: x.fitness, reverse=True)
        
        self.best = self.population[self.pop_size - 1]
        
        
    def run_threads(self, env, generation):
        # self.gen_envs = [(env.reset(), env.arr) for _ in range(11)]
        self.gen_envs = []
        for _ in range(151):
            reset = env.reset()
            obs = env.observation
            self.gen_envs.append((reset, obs))
        
        
        # self.gen_envs = [(env.reset(), env.observation) for _ in range(151)]
        
        start = time.time()
        
        folder_path = ""
            
        
        ## train all models for 10 rounds
        threads = []
        pth= ""
        for index,runner  in enumerate(self.population):
            
            folder_path = self.folder_path + "/thread" + str(index) +"/"
            runner.name = str(index)
            runner.envs = self.gen_envs
            if index == 0:
                pth = self.folder_path + "/best_thread/"
            else:
                pth = folder_path
            runner.set_save_location(pth, generation)
            thread = threading.Thread(target=runner.run, name=str(index))
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
        
        env = ListwiseEnv(num_actions=self.num_actions)
        
        self.initialise_population(self.pop_size)
        
        
        for i in range(self.max_generations):
            
            self.run_threads(env, i)
            parents = self.tournament_selection()
            children = self.variation(parents)
            self.reproduction(children)
            
            self.save_best_citizen()
                            
            var = f"generation {i} time = {time.time() - self.start}"
            with open(self.folder_path + "/time.txt", "a") as f:
                f.write(f"generation {i} time = {var}\n")
            
        print(f"generation {i} time = {var}")
        
        
    def run_just_evo(self, description):
        self.create_files(description)    
        
        env = ListwiseEnv(num_actions=self.num_actions)
        
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
                
                
    def reproduction(self, children : Evo_runner):
        print("reproduction")
        population = self.population + children
        population = sorted(population, key=lambda x: x.fitness, reverse=True)
        
        for chap in population:
            print(chap.fitness)

            
        print("\n")
        c = -1
        if len(children) != 0: 
            self.population = population[:-len(children)]
        for citizen in self.population:
            c +=1
            citizen.memory.reset_stack()
            # print(citizen.fitness)
            save = "`"
            if c == 0:
                save = self.folder_path + "/best_thread/fitness_calc.txt"
            else:
                save = self.folder_path + "/thread" + str(c) + "/fitness_calc.txt"
                
            with open(save, "a") as f:
                f.write(f"fitness = {citizen.fitness}\n")

                
    def tournament_selection(self):
        
        print("selection")
        parents = []
        size = 2
        for i in range(self.children_number):
            
            while True:
                fighters= random.sample(self.population, size)
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
    
    def get_ordered_indices(self, wa, wb):

        la = []
        lb = []
        
        la_indicies  = []
        lb_indicies = []

        for k in range(min(len(wa), len(wb))):
            
            wa_mean = np.mean(wa[k])
            wb_mean = np.mean(wb[k])

            wa[k][la_indicies] = wa_mean
            wb[k][lb_indicies] = wb_mean

            sk_minus = abs(min(wa[k]) + min(wb[k]))
            sk_plus = abs(max(wa[k]) + max(wb[k]))

            la_index = lb_index = 0 

            if sk_plus > sk_minus:
                wa[k][la_indicies] = -np.inf
                wb[k][lb_indicies] = -np.inf
                la_index = np.argmax(wa[k])
                lb_index = np.argmax(wb[k])
            else:
                wa[k][la_indicies] = np.inf
                wb[k][lb_indicies] = np.inf
                la_index = np.argmin(wa[k])
                lb_index = np.argmin(wb[k])
                
                
            la.append(la_index)
            la_indicies.append(la_index)
            lb.append(lb_index)
            lb_indicies.append(lb_index)

        return la, lb
    
    
    def get_corr(self, relu_original_one, relu_original_two):
    
        axis_number = 0
        semi_matching = False
        n = relu_original_one.shape[1]
        list_neurons_x = []
        list_neurons_y = []
        
        scaler = StandardScaler()# Fit your data on the scaler object
        relu_layer_one = scaler.fit_transform(relu_original_one)
        relu_layer_two = scaler.fit_transform(relu_original_two)

        corr_matrix_nn = np.empty((n,n))

        for i in range(n):
            for j in range(n):
                corr = np.corrcoef(relu_layer_one[:,i], relu_layer_two[:,j])[0,1]
                corr_matrix_nn[i,j] = corr

        corr_matrix_nn[np.isnan(corr_matrix_nn)] = -1
        
        #argmax_columns = np.argmax(corr_matrix_nn, axis=axis_number)
        argmax_columns = np.flip(np.argsort(corr_matrix_nn, axis=axis_number), axis=axis_number)
        dead_neurons = np.sum(corr_matrix_nn, axis=axis_number) == n*(-1) # these are neurons that always output 0 (dead relu)
        for index in range(n):
            if dead_neurons[index] == False:
                if semi_matching:
                    if axis_number == 0:
                        list_neurons_y.append(index)
                        list_neurons_x.append(argmax_columns[0,index])
                    elif axis_number == 1:
                        list_neurons_x.append(index)
                        list_neurons_y.append(argmax_columns[index,0])
                        
                elif semi_matching == False:
                    
                # do not allow same matching
                    for count in range(n):

                        if axis_number == 0:
                            if argmax_columns[count,index] not in list_neurons_x:
                                list_neurons_y.append(index)
                                list_neurons_x.append(argmax_columns[count,index])
                                break
                        elif axis_number == 1:
                            if argmax_columns[index,count] not in list_neurons_y:
                                list_neurons_x.append(index)
                                list_neurons_y.append(argmax_columns[index,count])
                                break
        
        # randomly pair the unpaired neurons
        for index in range(n):
            if index not in list_neurons_x and len(list_neurons_x) < n:
                list_neurons_x.append(index)
            if index not in list_neurons_y and len(list_neurons_y) < n:
                list_neurons_y.append(index)
        
        return list_neurons_x, list_neurons_y
    
    def get_standardised_activations(self, parent1, parent2):
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
            
        scaler = StandardScaler()# Fit data on the scaler object
        n1_activations = scaler.fit_transform(n1_activations)
        n2_activations = scaler.fit_transform(n2_activations)

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
        parent1_policy_sd, parent2_target_sd = self.do_cut_cross(parent1_policy_sd, parent2_policy_sd, layers, cut_location)
        parent1_target_sd, parent2_target_sd = self.do_cut_cross(parent1_target_sd, parent2_target_sd, layers, cut_location)
        
        # create child from these
        self.create_child(parent1, parent1_policy_sd, parent1_target_sd)
        self.create_child(parent2, parent2_policy_sd, parent2_target_sd)
        
        
        
        
        
    def do_cut_cross(self, state_dict1, state_dict2, layers, cut_location):
        
        device = torch.device(str("cuda") if torch.cuda.is_available() else "cpu")
        
        for layer in layers:
            for key in [f"layer{layer}.weight", f"layer{layer}.bias"]:
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
    

    def crossover(self, parent1 : Evo_runner, parent2: Evo_runner, layers : int):
        comparison_input = parent1.memory.sample(1)[0].state
        
        print(comparison_input.detach().cpu().numpy().shape)
        
        n1_activations = np.zeros((10, parent1.BATCH_SIZE))
        n2_activations = np.zeros((10, parent2.BATCH_SIZE))
        
        n1_tensor = F.relu(parent1.policy_net.layer1(comparison_input))
        n2_tensor = F.relu(parent2.policy_net.layer1(comparison_input))
            
        n1_activations[0] = n1_tensor.detach().cpu().numpy()
        n2_activations[0] = n2_tensor.detach().cpu().numpy()
       
        for i in range(1,10):
            random_tensor = parent1.memory.sample(1)[0].state
            n1_tensor = F.relu(parent1.policy_net.layer1(random_tensor))
            n2_tensor = F.relu(parent2.policy_net.layer1(random_tensor))
            
            n1_activations[i] = n1_tensor.detach().cpu().numpy()
            n2_activations[i] = n2_tensor.detach().cpu().numpy()
            
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
        
        
    def create_child(self, parent : Evo_runner,policy_state_dict, target_state_dict):
        parent.policy_net.load_state_dict(policy_state_dict)
        parent.target_net.load_state_dict(target_state_dict)
        parent.optimizer = optim.AdamW(parent.policy_net.parameters(), lr=parent.LR, amsgrad=True)
        
    
    def arithmetic_cross(self, parent1_state_dict, parent2_state_dict, layers, t):
        
        print(parent1_state_dict)
        for layer in layers:
            str_layer = f"layer{str(layer)}.weight"
            parent1_state_dict[str_layer] = (1 -t)*parent1_state_dict[str_layer ] + t*parent2_state_dict[str_layer]
            parent2_state_dict[str_layer] = (1-t)*parent2_state_dict[str_layer] + t*parent1_state_dict[str_layer]

        return parent1_state_dict, parent2_state_dict
        
        
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
            keys = [f"layer{layer}.weight", f"layer{layer}.bias",f"layer{layer + 1}.weight"]
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
                        print(index1, index2)
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
        # children = copy.deepcopy(parents)
        print(f"length of parents {len(parents)}")
        children = [None] * len(parents)
        for i, parent in enumerate(parents):
            
            with open('model.pkl', 'wb') as f:
                pickle.dump(parent, f)

            with open('model.pkl', 'rb') as f:
                child = pickle.load(f)
                
            parent.memory.make_child()
            child.memory = Memory_Nodes_Stack(parent=parent.memory)
            children[i] = child
        c = -1
        # returning_children = []
        for child in children:
            # c +=1
            # if c == 0:
            #     child.set_save_location(self.folder_path + "/best_thread/", "child" + str(c))
            # else:
            #     child.set_save_location(self.folder_path + "/thread" + str(i) + "/", "child" + str(c))
            
            # if child.mutate_attempt_random_weight_change():
            # returning_children.append(child)
                
            if child.mutate_attempt():
                returning_children.append(child)
            
            pre_fit = child.fitness
            print(f"pre fitness = {child.fitness}")
            child.calculate_fitness()
            post_fit = child.fitness
            
            with open(self.folder_path + "/child_comparison.txt", "a") as f:
                f.write(f"parent fitness {pre_fit} - child fitness {post_fit}\n")
            
            
            
            # print(f"child fitness {child.fitness}")
            
            # input()
            
        return returning_children
    
# test1 = Evo_runner()
# test2 = Evo_runner()
# test2.memory= test1.memory
# test1.memory.push(1,1,1,1)

# print(test2.memory.sample(1))

# start = time.time()
# environment = Evolultionary_Environment("with_prob_mutation", num_actions=10)
# environment.run("test")

# end = time.time() - start

        
            

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
