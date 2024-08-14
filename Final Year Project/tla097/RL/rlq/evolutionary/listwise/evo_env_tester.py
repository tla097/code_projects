import pickle
import random
import threading

import torch
from shared.memory_nodes import Memory_Nodes_Stack
from list_wise_evolutionary_environment import Evolultionary_Environment
# from '\RL\rlq\list_wise_evolution_runner' import Evo_runner


import importlib.util

from list_wise_evolution_runner import Evo_runner



import torch.nn as nn

class EvolutionaryEnvTester(Evolultionary_Environment):
    def __init__(self, SAVE, num_actions=10, children_ratio=0.5, pop_size=2, max_generations=250, mutation_type = None, num_samples = 200, num_events = 200, random_mutation_rate = 1, random_mutation_rate_threshold = 0.4, uniform_low = 0.95, uniform_high = 1.05, cross = 0.3, thread = 1, crossover_type = None, just_evo = None) -> None:
        super().__init__(SAVE, num_actions, children_ratio, pop_size, max_generations, low_cross=cross)
        
        
        self.folder_path = "RL/rlq/models/evo/tests/" + self.SAVE
        
        self.mutation_type = mutation_type
        self.num_samples = num_samples
        self.num_events = num_events
        self.random_mutation_rate =  random_mutation_rate
        self.random_mutation_rate_threshold = random_mutation_rate_threshold
        self.uniform_low = uniform_low
        self.uniform_high = uniform_high
        self.num_actions = num_actions
        
        self.cross = cross
        
        self.crossover_rate = 1
        
        self.thread = thread
        self.crossover_type  = crossover_type
        
        self.just_evo = just_evo
        
    def initialise_population(self, pop_size):
        ## initial population
        self.population:list[Evo_runner] = [None] * pop_size
        for k in range(pop_size):
            # params = {"test_train": "train","save": self.SAVE,"num_actions": self.num_actions,"random_mutation_rate": self.random_mutation_rate,"random_mutation_rate_threshold": self.random_mutation_rate_threshold}

            self.population[k] = EvoListWiseTester(test_train="train", save=self.SAVE, num_actions=self.num_actions, random_mutation_rate=self.random_mutation_rate, random_mutation_rate_threshold=self.random_mutation_rate_threshold, uniform_low=self.uniform_low, uniform_high=self.uniform_high)
            self.population[k].memory = Memory_Nodes_Stack()
            self.population[k].calculate_fitness(initial_fitness = True, just_evo = self.just_evo)
            
            
            # self.population[k].memory = ReplayMemory(10000)
            
        self.population = sorted(self.population, key=lambda x: x.fitness, reverse=True)
        
        self.best = self.population[0]
        
    def variation(self, parents):
        print("variation")
        # children = copy.deepcopy(parents)
        print(f"length of parents {len(parents)}")
        children = [None] * len(parents)
        for i, parent in enumerate(parents):
            
            with open(str(self.SAVE) + '_model.pkl', 'wb') as f:
                pickle.dump(parent, f)

            with open(str(self.SAVE) + '_model.pkl', 'rb') as f:
                child = pickle.load(f)
                
            parent.memory.make_child()
            child.memory = Memory_Nodes_Stack(parent=parent.memory)
            children[i] = child
            
            
        
        if self.crossover_type is not None:
            if self.crossover_type == "cca":
                for i in range(int(self.children_number / 2)):
                    if self.crossover_rate >= random.random():
                        child1, child2 = tuple(random.sample(children, 2))
                        self.crossover_caa(child1, child2, [1])
            elif self.crossover_type == "corrolation":
                for i in range(int(self.children_number / 2)):
                    if self.crossover_rate >= random.random():
                        child1, child2 = tuple(random.sample(children, 2))
                        self.crossover(child1, child2, [1])
            elif self.crossover_type == "ccacut":
                for i in range(int(self.children_number / 2)):
                    if self.crossover_rate >= random.random():
                        child1, child2 = tuple(random.sample(children, 2))
                        self.cca_cut(child1, child2, [1])
            elif self.crossover_type == "corrcut":
                for i in range(int(self.children_number / 2)):
                    if self.crossover_rate >= random.random():
                        child1, child2 = tuple(random.sample(children, 2))
                        self.correlation_cut(child1, child2, [1])
            elif self.crossover_type == "single_swap":
                for i in range(int(self.children_number / 2)):
                    if self.crossover_rate >= random.random():
                        child1, child2 = tuple(random.sample(children, 2))
                        self.single_swap_crossover(child1, child2, [1])
                        
                                
        if self.mutation_type is not None:
            for child in children:
                if self.mutation_type == "random":
                    if child.mutate_attempt_random_weight_change():
                        print("mutated")
                elif self.mutation_type == "availability":
                    if child.mutate_attempt():
                        print("mutated")
                elif self.mutation_type == "just_evo":
                    if child.mutate_attempt_just_evo(self.best):
                        print("mutated")
                                        
                
                
        
        
        # for child in children:
        #     self.swapping_mutate(child)
        for child in children:
            pre_fit = child.fitness
            print(f"pre fitness = {child.fitness}")
            child.calculate_fitness(just_evo = self.just_evo)
            post_fit = child.fitness
            print(f"post crossover = {child.fitness}")
            
            with open(self.folder_path + "/child_comparison.txt", "a") as f:
                f.write(f"parent fitness {pre_fit} - child fitness {post_fit}\n")
            
        return children
        
        
class EvoListWiseTester(Evo_runner):
    def __init__(self, tv=0, ask=0, mini_batch=1, added_comparison=2, comparison="Duration", layers=0, activation="relu", load="", save="test", des="des", results="res", new_load="new", gamma=0.99, reward=0.5, eps_decay=10000, lr=0.0001, test_train="test", memory: Memory_Nodes_Stack = None, name="test", num_actions=10, fitness=-100, parent_sd="", num_events = 100, random_mutation_rate = 0.5, num_samples = 100, random_mutation_rate_threshold = 0.05,  uniform_low = 0.95, uniform_high = 1.05):
        super().__init__(tv, ask, mini_batch, added_comparison, comparison, layers, activation, load, save, des, results, new_load, gamma, reward, eps_decay, lr, test_train, memory, name, num_actions, fitness, parent_sd)
        self.num_events = num_events
        self.random_mutation_rate = random_mutation_rate
        self.num_samples = num_samples
        self.random_mutation_rate_threshold  = random_mutation_rate_threshold
        
        self.uniform_low = uniform_low
        self.uniform_high = uniform_high
        
        
    
    def check_mutate(self, event):
        return random.random() <= self.get_mutation_prob(event)
    
    def mutate_attempt(self):
        samples = []
        for _ in range(self.num_samples):
            event = None
            none_present = True
            while none_present:
                none_present = True
                event = self.memory.sample(1)[0]
                none_present = event.next_state is None
            samples.append(event)
            
        
        result = False
        for i in range(self.num_samples):
            event = samples[i]
            if self.check_mutate(event):
                result = True
                self.mutate(event)
        return result
    
    
    def mutate(self, event):
        
        state_batch = torch.cat((event.state, event.state), dim=0)
        next_state_batch  = torch.cat((event.next_state, event.next_state), dim=0)
        
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                    next_state_batch)), device=self.device, dtype=torch.bool)
        
        non_final_next_states = [s for s in next_state_batch
                                                    if s is not None]
        
        non_final_next_states = torch.stack(non_final_next_states)
        
        current_best_action = self.policy_net(event.state).max(1).indices.view(1, 1)
        action_to_improve = self.pick_action(event)
        
        if current_best_action == action_to_improve:
            print("same")
            return
            
        
        action_batch = torch.cat((current_best_action, action_to_improve),dim=0)
        
        positive_reward = torch.tensor([[1]], device=self.device, dtype=torch.float32)
        negative_reward = torch.tensor([[-1]], device=self.device, dtype=torch.float32)
        reward_batch = torch.cat((negative_reward, positive_reward), dim=1).squeeze(0)


        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net                            
        for i in range(self.num_events):
            state_action_values = self.policy_net(state_batch).gather(1, action_batch)
            
            # print(state_action_values)
            

            # Compute V(s_{t+1}) for all next states.
            # Expected values of actions for non_final_next_states are computed based
            # on the "older" target_net; selecting their best reward with max(1).values
            # This is merged based on the mask, such that we'll have either the expected
            # state value or 0 in case the state was final.
            next_state_values = torch.zeros(2, device=self.device)
            with torch.no_grad():
                next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values
            # Compute the expected Q values
            expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch
            
            # Compute Huber loss
            criterion = nn.SmoothL1Loss()
            loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

            # Optimize the model
            self.optimizer.zero_grad()
            loss.backward()
            # In-place gradient clipping
            torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
            self.optimizer.step()
            
            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′
            target_net_state_dict = self.target_net.state_dict()
            policy_net_state_dict = self.policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*self.TAU + target_net_state_dict[key]*(1-self.TAU)
            self.target_net.load_state_dict(target_net_state_dict)
            
            
            state_action_values = self.policy_net(state_batch).gather(1, action_batch)
            # print(state_action_values)
            if state_action_values[0].item() < state_action_values[1].item():
                break
        
        
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        print("mutated")

                
    def mutate_attempt_random_weight_change(self):
        if random.random() <= self.random_mutation_rate:
            self.mutate_random_weight_change()
            return True
        else:
            return False
        
    def mutate_random_weight_change(self):
        state_dict = self.policy_net.state_dict()
        mutate_prob = self.random_mutation_rate_threshold
        for key in state_dict:
            shape = state_dict[key].shape
            random_tensor = torch.rand(shape, device=self.device)
            random_tensor = torch.where(random_tensor > mutate_prob, 
                            torch.full_like(random_tensor, 1, device=self.device), 
                            torch.tensor([random.uniform(self.uniform_low, self.uniform_high) for _ in range(random_tensor.numel())], device=self.device).reshape(shape))
            
            
            state_dict[key] = state_dict[key] * random_tensor
            print("mutated")
        self.policy_net.load_state_dict(state_dict)


## train all models for 10 rounds
threads = []




def availability():
    mutation_types = ["availability"]
    list_num_samples = [10, 50, 100, 150, 200]
    list_num_events = [10, 50, 100, 150, 200]
    i = 0
    for ns in list_num_samples:
        for ne in list_num_events:
            i += 1
            params = {"mutation_type":"availability", "num_samples":int(ns), "num_events":int(ne),"SAVE":"availablility_again_2"+str(i), "max_generations":20,  "thread" : "availability_mutate", "thread" : "availability"}
            runner = EvolutionaryEnvTester(**params)
            description = str(params)
            runner.run(description)
            
       
def random_mutate1():  
    i = 0   
    list_random_mutation_rate_threshold = [0.001, 0.01, 0.05]
    list_uniform = [(0.7, 1.3), (0.8, 1.2), (0.9, 1.1), (0.95, 1.05), (0.99, 1.01)]
    for lrmrt in list_random_mutation_rate_threshold:
        for lul, luh in list_uniform:
                i += 1
                params = {"mutation_type":"random","random_mutation_rate_threshold":float(lrmrt), "uniform_low":float(lul), "uniform_high":float(luh), "SAVE":"random_mutate1" + str(i), "max_generations":20, "thread" : "random1"}
                runner = EvolutionaryEnvTester(**params)
                description = str(params)
                runner.run(description)
                
                
def random_mutate2():   
    i = 0  
    list_random_mutation_rate_threshold = [0.1, 0.2, 0.3, 0.5]
    list_uniform = [(0.7, 1.3), (0.8, 1.2), (0.9, 1.1), (0.95, 1.05), (0.99, 1.01)]
    for lrmrt in list_random_mutation_rate_threshold:
        for lul, luh in list_uniform:
            i += 1
            params = {"mutation_type":"random", "random_mutation_rate_threshold":float(lrmrt), "uniform_low":float(lul), "uniform_high":float(luh), "SAVE":"random_mutate2" + str(i), "max_generations":20, "thread" : "random2"}
            runner = EvolutionaryEnvTester(**params)
            description = str(params)
            runner.run(description)
    


# mutation_types = ["prob", "random"]
# list_num_samples = [10, 50, 100, 150, 200]
# list_num_events = [10, 50, 100, 150, 200]
# list_random_mutation_rate = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6]
# list_random_mutation_rate_threshold = [0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5]
# list_uniform_low = [0.8, 0.9, 0.95, 0.99]
# list_uniform_high = [1.01, 1.05, 1.1, 1.2]
# pth= ""
# i = 0
# for mt in mutation_types:
#     for ns in list_num_samples:
#         for ne in list_num_events:
#             i += 1
#             if i > 24:
#                 params = {"mutation_type":"availability_mutate", "num_samples":int(ns), "num_events":int(ne),"SAVE":"availablility_again"+str(i), "max_generations":20,  "thread" : "availability_mutate"}
#                 runner = EvolutionaryEnvTester(**params)
#                 description = str(params)
#                 runner.run(description)
            
            
# ava = threading.Thread(target=availability)
# random1 = threading.Thread(target=random_mutate1)
# random2 = threading.Thread(target=random_mutate2)

# ava.start()
# # random1.start()
# # random2.start()

# ava.join()
# random1.join()
# random2.join()


# ava.start()
# # ran.start()
# ava.join()
# # ran.join()

# print("hello")

#             for rmr in list_random_mutation_rate:
#                 for lrmrt in list_random_mutation_rate_threshold:
#                     for lul in list_uniform_low:
#                         for luh in list_uniform_high:
#                             i += 1
#                             params = {"mutation_type":mt, "num_samples":int(ns), "num_events":int(ne), "random_mutation_rate":float(rmr), "random_mutation_rate_threshold":float(lrmrt), "uniform_low":float(lul), "uniform_high":float(luh), "SAVE":str(i), "max_generations":20}
#                             runner = EvolutionaryEnvTester(**params)
#                             description = str(params)
#                             runner.run(description)


# low = [0.2, 0.3, 0.4, 0.,5, 0.6, 0.7, 0.8, 0.9, 1.1, 1.2, 1.3, 1.4, 1.5]
# for lowc in low:
#     runner = EvolutionaryEnvTester(SAVE="crossover_test_" + str(lowc), cross=lowc, max_generations=20)
#     description = str(lowc)
#     runner.run(description)
    
    
# lowc = 0.1

runner = EvolutionaryEnvTester(SAVE="just_evo_comp_4" , crossover_type=None, mutation_type= "just_evo", random_mutation_rate=1, random_mutation_rate_threshold=0.5, uniform_low=0.8, uniform_high=1.2, just_evo = None, num_events=20, num_samples=100)
description = " samples 100 event 20"
runner.run(description)














# plan

# set a bunch of tests running on lots of threads
# try to add different mutation attempts 
# dry different crossover attempts





