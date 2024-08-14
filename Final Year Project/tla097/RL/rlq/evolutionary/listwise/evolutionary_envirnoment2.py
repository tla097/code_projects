import copy
from pathlib import Path
import random
import threading
import time

import torch
from RL.rlq.list_wise_evolution_runner import Evo_runner

from RL.rlq.shared.ReplayMemory import ReplayMemory, Transition

from listwise_env import ListwiseEnv

class Evolultionary_Environment:
    def __init__(self, SAVE = "test") -> None:
        self.SAVE = SAVE
        
        self.pop_size = 2
        self.mems = [None] * self.pop_size
        
        for k in range(self.pop_size):
            self.mems[k]= ReplayMemory(10000)
    
    def run(self, description):
        
        children_ratio = 1
        pop_size = self.pop_size
        max_generations = 250
        self.children_number = int(children_ratio * pop_size)
        
        
        self.folder_path = "RL/rlq/models/evo/" + self.SAVE
        
        overall_best_location = self.folder_path + "/overall_best"
        generation_best_location = ""
        
        self.best = Evo_runner()
        
        if not Path(self.folder_path).exists():
            print(f"The file {self.folder_path} does not exist - creating folder.")
            Path(self.folder_path).mkdir(parents=True, exist_ok=True)
            
            with open(self.folder_path + "/description.txt", "w") as f:
                f.write(description)
        
            
            for i in range(1, self.pop_size):
                Path(self.folder_path + "/thread" + str(i)).mkdir(parents=True, exist_ok=True)
                
            Path(self.folder_path + "/best_thread").mkdir(parents=True, exist_ok=True)
            
            
        
        env = ListwiseEnv()
        ## initial population
        self.population = [None] * pop_size
        for k in range(pop_size):
            self.population[k] = Evo_runner(test_train="train", save=self.SAVE, eps_decay=1000000, memory=self.mems[k], reward=1)            
        for i in range(max_generations):
            
            
            
            # self.gen_envs = [(env.reset(), env.arr) for _ in range(11)]
            self.gen_envs = [(env.reset(), env.observation) for _ in range(11)]
            
            start = time.time()
            
            folder_path = ""
            
            generation_best_location = self.folder_path + "/best_thread/"
                
            
            ## train all models for 10 rounds
            threads = []
            for index,runner in enumerate(self.population):
                
                folder_path = self.folder_path + "/thread" + str(index) +"/"
                
                runner.name = str(index)
                
                runner.envs = self.gen_envs
                
                if index == 0:
                    pth = generation_best_location
                else:
                    pth = folder_path
                runner.set_save_location(pth, i)
                thread = threading.Thread(target=runner.run, name=str(index))
                threads.append(thread)
                thread.start()
                
            for thread in threads:
                thread.join()
                
            
            # print("lol")
                
            parents = self.tournament_selection()
            children = self.variation(parents)
            self.reproduction(parents)
            
            
            # saving the best
            if (self.population[pop_size - 1].fitness) > self.best.fitness:
                self.best = self.population[pop_size - 1]
                torch.save(self.best.policy_net.state_dict(), generation_best_location + "policy.pth")
                torch.save(self.best.target_net.state_dict(), generation_best_location + "target.pth")
                torch.save(self.best.optimizer.state_dict(), generation_best_location + "optimiser.pth")
                
                
        var = f"generation {i} time = {time.time() - start}"
        with open(self.folder_path, "a") as f:
            f.write(f"generation {i} time = {var}")
            
        print(f"generation {i} time = {var}")
                
                
    def reproduction(self, children):
        print("reproduction")
        population = self.population + children
        population = sorted(population, key=lambda x: x.fitness, reverse=True)

            
        print("\n")
        self.population = population[:-len(children)]
        
        # input()
                
                
    def tournament_selection(self):
        
        print("selection")
        parents = []
        size = 2
        for i in range(self.children_number):
            
            while True:
                fighters= random.sample(self.population, size)
                if not(fighters[0].selected) and not(fighters[1].selected):
                    break 
            
            best = Evo_runner()
            
            for fighter in fighters:
                fighter.set_selected()
                if fighter.fitness > best.fitness:
                    best = fighter
                    
                    
            parents.append(best)
            
        for citizen in self.population:
            citizen.reset_selected()
        
        return parents
    
    
    def variation(self, parents):
        print("variation")
        # children = copy.deepcopy(parents)
        children = [None] * len(parents)
        for i, parent in enumerate(parents):
            torch.save(parent.target_net.state_dict(), 'target.pth')
            torch.save(parent.policy_net.state_dict(), 'policy.pth')
            torch.save(parent.optimizer.state_dict(), 'optimiser.pth')
            
            
            
            
            child = Evo_runner("train", reward=1)
            child.EPS_START = parent.eps_threshold
            
            child.memory = parent.memory
            
            ##################
            # child.memories["Parent"] = 
            
            ##################
            
            child.memory = parent.memory
            child.envs = self.gen_envs
            
            child.target_net.load_state_dict(torch.load("target.pth"))
            child.target_net.to(child.device)

            child.policy_net.load_state_dict(torch.load("policy.pth"))
            child.policy_net.to(child.device)
            
            child.optimizer.load_state_dict(torch.load("optimiser.pth"))
            
            children[i] = child
            
            
        # for i in range(int(self.children_number / 2)):
        #     if self.crossover_rate >= random.random():
        #         child1, child2 = tuple(random.sample(children, 2))
        #         self.order_crossover(child1, child2)
        c = -1
        for child in children:
            c +=1
            child.set_save_location(self.folder_path + "/thread" + str(i) + "/", "child" + str(c))
            child.mutate_attempt_random_weight_change()
            child.calculate_fitness()
            
        return children
    
# test1 = Evo_runner()
# test2 = Evo_runner()
# test2.memory= test1.memory
# test1.memory.push(1,1,1,1)

# print(test2.memory.sample(1))

start = time.time()
environment = Evolultionary_Environment("TEST2_popsize6_random_weight_change_list_wise")
environment.run("test")

end = time.time() - start

        
            

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
