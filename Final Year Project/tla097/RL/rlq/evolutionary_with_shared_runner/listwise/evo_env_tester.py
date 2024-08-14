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
        
        
