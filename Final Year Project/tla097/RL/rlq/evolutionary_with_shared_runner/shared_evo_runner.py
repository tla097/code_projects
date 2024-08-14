import copy
from datetime import datetime
from pathlib import Path
import random
import threading
import time
import numpy as np
import math

import torch

from RL.rlq.shared.BaseRunner import BaseRunner
from RL.rlq.shared.ReplayMemory import ReplayMemory, Transition
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler


from RL.rlq.shared.memory_nodes import Memory_Nodes_Stack

class Evo_runner(BaseRunner):
    
    def __init__(self, tv=0, ask=0, mini_batch=1, added_comparison=2, comparison="Duration", layers=0, activation="relu", load="", save="test", des="des", results="res", new_load="new", gamma=0.99, reward=0.5, eps_decay=10000, lr=0.0001, test_train="test", memory: Memory_Nodes_Stack = None, name="test", num_actions=10, fitness=-100, parent_sd="", num_events = 100, random_mutation_rate = 0.5, num_samples = 100, random_mutation_rate_threshold = 0.05,  uniform_low = 0.95, uniform_high = 1.05):
        super().__init__(tv, ask, mini_batch, added_comparison, comparison, layers, activation, load, save, des, results, new_load, gamma, reward, eps_decay, lr, test_train)
        
        self.fit_string = ""
        self.memory = memory
        self.fitness= -100
        self.selected = False
        self.name = name
        self.mutation_rate = 0.2
        self.envs = None        
        self.num_events = num_events
        self.random_mutation_rate = random_mutation_rate
        self.num_samples = num_samples
        self.random_mutation_rate_threshold  = random_mutation_rate_threshold
        self.uniform_low = uniform_low
        self.uniform_high = uniform_high
        
        self.save_frequency = math.pi
        self.test_frequency = math.pi
        
        
        
              
        
        
        # def __init__(self, tv=0, ask=0, mini_batch=1, added_comparison=1, comparison="Duration", layers=1, activation="relu", load="", save="test", des="des", results="res", new_load="n", gamma=0.99, reward=0.5, eps_decay=1000000, lr=0.0001, low_reward=0, high_reward=1, mid_low_reward=0, test_train="test", batch_size=128, mem_size=10000, tau=0.005, start=0.9,  path="RL/rlq/models/evo/test", memory = None, name = "-1"):
        #     super().__init__(tv, ask, mini_batch, added_comparison, comparison, layers, activation, load, save, des, results, new_load, gamma, reward, eps_decay, lr, low_reward, high_reward, mid_low_reward, test_train, batch_size, mem_size, tau, start)
        
        
    def return_memory(self):
        return self.memory
        
        
    def run(self, gen = ""):
        print("runner running")
        for epoch in range(20):
            k = 0
            state = self.envs[epoch][0]
            arr = copy.copy(self.envs[epoch][1])
            self.env.reset()
            self.env.observation = arr
            self.start_time= time.time()
            self.train_round(epoch, self.folder_path + "/results.txt", self.env, state0=state)
        self.calculate_fitness(gen = gen)
            
        
        
    
    def set_selected(self):
        self.selected = True
        
    def reset_selected(self):
        self.selected = False
        
    def set_save_location(self, save_location, generation):
        self.folder_path = save_location
        
        
    # def test(self, mid_run=True, k=-1, stage=""):
    #     self.mutate_attempt()
        # result = super().test(mid_run, k, stage)
    
    def calculate_fitness(self, initial_fitness = False, just_evo = None, save = True, gen = ""):
        print("fitness")
        if initial_fitness:
            state0 = self.env.reset()
        elif just_evo is None:
            arr = copy.copy(self.envs[self.num_rounds_ran][1])
            state0 = self.env.reset(arr= arr, test = True)
            # self.env.observation  = arr
            
            # print(state)
            # print(arr)
        # else:
            # state = self.envs[0][0]
            # arr = copy.copy(self.envs[0][1])
            # self.env.reset()
            # self.env.arr  = arr       
            
        
        correct, actual_count, total_count = self.test(test_env=self.env, state0=state0)
        mae = self.base_env.calculate_accuracy(self.env)
        npra = self.base_env.calc_normalised_rank_point_average(self.env)
        afpd = self.base_env.calc_APFD(self.env)
        
        time_taken= time.time() - self.start_time
        
        string1 = f"\ngeneration = {gen} - total steps = {self.total_steps} - {self.SAVE} - round steps = {actual_count}   "
        string2 = f""
        string3 = f"correct = {correct/total_count:.5f} - mae = {mae}, AFPD = {afpd}, NPRA = {npra} -time taken = {time_taken}"

        string = string1 + string2+ string3

        
        print(self.folder_path + "/fitness.txt")
        if save:
            with open(self.folder_path + "/fitness.txt", 'a') as f:
                f.write(f"{string}")
            
        print(f"Test Complete\n{string}")
        self.fitness = correct/total_count
        self.fit_string = string
        print(f"fitness {self.name} - {self.fitness} {actual_count} steps")
        
        
    def check_mutate(self, event, best_model):
        return random.random() <= self.get_mutation_prob(event, best_model)
    
    def mutate_attempt(self, best_model):
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
            if self.check_mutate(event, best_model):
                result = True
                self.mutate(event, best_model)
        return result
                
    def mutate_attempt_random_weight_change(self):
        if random.random() <= self.random_mutation_rate:
            self.mutate_random_weight_change()
            return True
        else:
            return False
        
                
    def set_state_action_list(self, event, best_model):
        action_list = list(range(self.n_actions))
        
        tensor_action_list= torch.tensor(action_list, dtype=torch.int64, device=self.device).unsqueeze(0)
        state_action_values = best_model.policy_net(event.state).gather(1, tensor_action_list).cpu()
        self.list_state_action_values = state_action_values.detach().numpy()[0]
        
        # print(f"state action values: {self.list_state_action_values}\n")
        
        #rescale
        self.list_state_action_values = self.list_state_action_values * -1
        normalized_arr = (self.list_state_action_values - self.list_state_action_values.min()) / (self.list_state_action_values.max() - self.list_state_action_values.min())
        self.list_state_action_values = normalized_arr * (2 - 1) + 1

    def get_mutation_prob(self, event, best_model):
        
        self.set_state_action_list(event, best_model)
        
        current_action = event.action.item()
        
        
        current_action_value = self.list_state_action_values[current_action]
        # swapped for -1
        greater_than_condition = self.list_state_action_values <= current_action_value
        lesser_than_condition = self.list_state_action_values >= current_action_value

        greater = np.sum(greater_than_condition)
        lesser = np.sum(lesser_than_condition)
        
        result = greater / (lesser + greater)
        
        
        print(f"mutation prob: {result}\n")
        return result
        # return 1
        
                     

        
        # state_action_values2 = self.policy_net(event.state).gather(1, other)
        
        # print(state_action_values2)
        
        ###### plan for tomorrow:
        # do the better action selection
        # do the probability of picking other action
        # try it with listwise
        # figure out a way to work out probability of mutation
        
        # self.policy_net(event.state).max(1).indices.view(1, 1).to(self.device)
        
        # with torch.no_grad():
        #     next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values
    
    def pick_action(self, event):
        current_action = event.action.item()
        pi_div_a = self.list_state_action_values[current_action] / self.list_state_action_values
        
        top = np.exp(pi_div_a)
        bottom = np.sum(top)
        
        probability = top/bottom        
        
        random_number = random.random()        
        total = 0
        for i, element in enumerate(probability):
            total = total + element
            # print(f"total = {total}")
            if random_number <= total:
                result = torch.tensor([[i]], device=self.device, dtype=torch.int64)
                return result
            

            
            
            
    ## start with all random q values
    
    ## every 10 rounds after a test we find the one that performed the best
    ## pick the top 3 best running ones compared to that and then mutate it and make it into a new network
    ##      add that network to the population
    
    ## values are getting too high- need to find a way to reduce it, maybe have negative rewards
    
    ## try crossover next if not working
    
    
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
        
        for k in range(10):
            target_net_state_dict = self.target_net.state_dict()
            policy_net_state_dict = self.policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*self.TAU + target_net_state_dict[key]*(1-self.TAU)
            self.target_net.load_state_dict(target_net_state_dict)


    def mutate(self, event, mutate_attempt_random_weight_change):
        
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

                
        
        # input()
            
            # print(f"previous action = {event.action}")
            # print(f"now action {self.policy_net(event.state).max(1).indices.view(1, 1).to(self.device)}")
                
            
            
        
    """plan
        fitness of a particular network is down to its average test performance
        
        how often to try to mutate?
        every test
        
        if the network needs mutation then on every state calc the prob and chuck in a penalty for the current action
        randomly pick another one and then give a reward to that
        
        do prob of changing for the states
        do prob of picking other states
        
        make environment object with lots of networks in"""
        
        
        
        
# params = {"lr":0.0001, "low_reward":0, "high_reward":1, "reward":0.5, "mid_low_reward":0, "test_train" :"train", "eps_decay":10000000, "layers" : 2}
# name = f"evo_test"
# load = f"new"
# new_load = "new"
# params["save"] = name
# params["load"] = load
# params["new_load"] = new_load
# runner = Evo_runner(**params, des = str(params))
# t15 = threading.Thread(target=runner.run, name=name)
# t15.start()
# t15.join()



        


# ## train all models for 10 rounds
# threads = []




# def availability():
#     mutation_types = ["availability"]
#     list_num_samples = [10, 50, 100, 150, 200]
#     list_num_events = [10, 50, 100, 150, 200]
#     i = 0
#     for ns in list_num_samples:
#         for ne in list_num_events:
#             i += 1
#             params = {"mutation_type":"availability", "num_samples":int(ns), "num_events":int(ne),"SAVE":"availablility_again_2"+str(i), "max_generations":20,  "thread" : "availability_mutate", "thread" : "availability"}
#             runner = EvolutionaryEnvTester(**params)
#             description = str(params)
#             runner.run(description)
            
       
# def random_mutate1():  
#     i = 0   
#     list_random_mutation_rate_threshold = [0.001, 0.01, 0.05]
#     list_uniform = [(0.7, 1.3), (0.8, 1.2), (0.9, 1.1), (0.95, 1.05), (0.99, 1.01)]
#     for lrmrt in list_random_mutation_rate_threshold:
#         for lul, luh in list_uniform:
#                 i += 1
#                 params = {"mutation_type":"random","random_mutation_rate_threshold":float(lrmrt), "uniform_low":float(lul), "uniform_high":float(luh), "SAVE":"random_mutate1" + str(i), "max_generations":20, "thread" : "random1"}
#                 runner = EvolutionaryEnvTester(**params)
#                 description = str(params)
#                 runner.run(description)
                
                
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
    







    
