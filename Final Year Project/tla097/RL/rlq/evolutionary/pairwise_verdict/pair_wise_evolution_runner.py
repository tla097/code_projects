import copy
from datetime import datetime
from pathlib import Path
import random
import threading
import time
import numpy as np


import torch.optim as optim

import torch

from new_runner import Runner, Transition


from memory_nodes import Memory_Nodes_Stack

import torch.nn as nn

class Evo_runner(Runner):
    
    def __init__(self, tv=0, ask=0, mini_batch=1, added_comparison=1, comparison="Duration", layers=0, activation="relu", load="", save="test", des="des", results="res", new_load="new", gamma=0.99, reward=0.5, eps_decay=10000, lr=0.0001, test_train="test", memory : Memory_Nodes_Stack= None, name = "test", fitness = -100,parent_sd= "", params = {}):
        super().__init__(tv, ask, mini_batch, added_comparison, comparison, layers, activation, load, save, des, results, new_load, gamma, reward, eps_decay, lr, test_train=test_train)
        
        self.memory = memory
        
        self.fitness= fitness
        
        self.selected = False
        
        self.name = name
        
        self.mutation_rate = 0.2
        
        self.envs = None
        
        self.parent_sd = parent_sd
        
        self.keep = 0
        
        self.params = params
        
        
        
              
        
        
        # def __init__(self, tv=0, ask=0, mini_batch=1, added_comparison=1, comparison="Duration", layers=1, activation="relu", load="", save="test", des="des", results="res", new_load="n", gamma=0.99, reward=0.5, eps_decay=1000000, lr=0.0001, low_reward=0, high_reward=1, mid_low_reward=0, test_train="test", batch_size=128, mem_size=10000, tau=0.005, start=0.9,  path="RL/rlq/models/evo/test", memory = None, name = "-1"):
        #     super().__init__(tv, ask, mini_batch, added_comparison, comparison, layers, activation, load, save, des, results, new_load, gamma, reward, eps_decay, lr, low_reward, high_reward, mid_low_reward, test_train, batch_size, mem_size, tau, start)
        
        
    def return_memory(self):
        return self.memory
        
        
    def run(self):
        
        print("runner running")
        for epoch in range(20):
            k = 0
            episode_rewards = []
            episode_averages = []
            
            state = self.envs[epoch][0]
            arr = copy.copy(self.envs[epoch][1])
            self.env.reset()
            self.env.arr = arr
            # state = torch.tensor(self.env.reset(), dtype=torch.float32, device=self.device, requires_grad=True).unsqueeze(0)
            state = torch.tensor(state, dtype=torch.float32, device=self.device, requires_grad=True).unsqueeze(0)
            self.start_time= time.time()
            
            while True:
                k +=1
                self.keep += 1
                action = self.select_action(state)
                observation, reward, done, _ = self.env.step(action)
                # observation, reward, done, truncated = self.env.step(action)
                
                reward_to_use = torch.tensor([reward], device=self.device)
                action_to_use = action
                observation = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
                
                # Store the transition in memory
                self.memory.push(state, action_to_use, observation, reward_to_use)
                state = observation    
                    
                    

                

                
                
                # for evo
                # self.episode_memory.push(state, action_to_use, observation, reward_to_use)
                
                episode_rewards.append(reward)
                
                
                
                
                if self.keep <= 100:
                    episode_averages.append(sum(episode_rewards)/k)
                else:
                    episode_averages.append(sum(episode_rewards[-100:])/100)
                    
                if self.keep % 200 == 0:
                    episode_averages = episode_averages[-100:]
                    
                if self.keep % 100 == 0:
                    avg= sum(episode_averages[-100:])/100
                    curr_time=time.time() - self.start_time
                    with open(self.save_location + "/results.txt", 'a') as f:
                        f.write(f"average = {avg:.10f}. Time = {curr_time:.4f}\n")


                # print(1)
                # Perform one step of the optimization (on the policy network)
                self.optimize_model()
                
                # Soft update of the target network's weights
                # θ′ ← τ θ + (1 −τ )θ′
                target_net_state_dict = self.target_net.state_dict()
                policy_net_state_dict = self.policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key]*self.TAU + target_net_state_dict[key]*(1-self.TAU)
                self.target_net.load_state_dict(target_net_state_dict)
                
                
                # if self.keep % 100 == 0:
                    # print(self.last_sd == str(self.target_net.state_dict()))
                    # self.last_sd = str(self.target_net.state_dict())
                    # print(self.last_sd)
                    
                if done:
                    with open(self.save_location + "results.txt", 'a') as f:
                        f.write(f"SORTED WITH {k} steps\n\n")
                    print(f"SORTED WITH {k} steps")
                    
                    torch.save(self.policy_net.state_dict(), self.save_location + "policy_" + self.SAVE + ".pth")
                    torch.save(self.target_net.state_dict(), self.save_location + "target_" + self.SAVE + ".pth")
                    torch.save(self.optimizer.state_dict(), self.save_location + "optimiser_" + self.SAVE + ".pth")
                    break
                
        if self.keep >= 700:
            print(self.keep)
            print("hello")    
        self.calculate_fitness()
        
    def optimize_model(self):
            
                ### code I have adapted from elsewhere
        if len(self.memory) < self.BATCH_SIZE:
            return
        transitions = self.memory.sample(self.BATCH_SIZE)
        

        batch = Transition(*zip(*transitions))


        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=self.device, dtype=torch.bool)

        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
     
        
        # print(non_final_next_states)
        state_batch = torch.cat(batch.state)
        # print(batch.action)
        action_batch = torch.cat(batch.action)
        # print(action_batch)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1).values
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.BATCH_SIZE, device=self.device)
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
        
    def set_selected(self):
        self.selected = True
        
    def reset_selected(self):
        self.selected = False
        
    def set_save_location(self, save_location, generation):
        with open(save_location + "results.txt", "a") as f:
            f.write(f"\n generation - {generation}")
        self.save_location = save_location
    
    def calculate_fitness(self, initial_fitness = False, state = None, just_evo = None):
        
        
        print("fitness")
        
        # counter = 0
        # incorrect = 0
        # correct = 0
        # same = 0
        
        rewards = 0
        counter = 0
        
        if initial_fitness:
            state = self.env.reset()
            
        elif just_evo is None:
            state = self.envs[20][0]
            arr = copy.copy(self.envs[20][1])
            self.env.reset()
            self.env.arr  = arr
        else:
            state = self.envs[0][0]
            arr = copy.copy(self.envs[0][1])
            self.env.reset()
            self.env.arr  = arr
            
        # state = self.env.reset()
        print(f"state sum  = {sum(state)}")
        state = torch.tensor(state, dtype=torch.float32, device=self.device, requires_grad=False).unsqueeze(0)
        
        
        
        # print(self.env.observation)
        # print(self.env.observation)
        
        
        # input()
        
        while True:
            counter += 1
            # print(f"sum = {sum(param.sum().item() for param in self.policy_net.state_dict().values())}")
            # print(f"id= {id(self.policy_net)}")
            # if self.parent_sd != "":
            #     print("in thingy")
            #     self.policy_net.load_state_dict(self.parent_sd)
            action = self.policy_net(state).max(1).indices.view(1, 1)
            # print(f"action = {action}")
            # print(f"action = {action}")
            # state, reward, done, truncated = self.env.step(action)
            observation, reward, done, _ = self.env.step(action)
            # print(f"state = {state}")
            # print(dummies)
            
            # print(self.dummies)
            
            
            # if just_evo is not None:#
            
            if just_evo is not None:
                reward_to_use = torch.tensor([reward], device=self.device)
                action_to_use = action
            
                if not done:
                    push_observation = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
                    
                # Store the transition in memory
                self.memory.push(state, action_to_use, push_observation, reward_to_use)
                
            state = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
                
            
            # print(696969)
            
            
            
            # print(state)
            rewards += reward
            
            
            if done:
                self.fitness = rewards/counter
                break
            
        
        print(self.fitness)
        

        print(counter)

        
        
        # while True:
        #     counter += 1
            
        #     action = self.target_net(state).max(1).indices.view(1, 1)
        #     state, reward, done, truncated = self.env.step(action)
            
        #     # state, reward, done, truncated, dummies = self.env.step(action, dummies=self.dummies)
        #     # print(state)
            
        #     # self.dummies = dummies
        #     # print(dummies)
        #     # print(state)
            
        #     if truncated:
        #         self.fitness = (rewards/counter) * 0.1 * counter - 10
        #         print("finished early")
        #         break
            
        #     # state[action.item() * 8 + 7] = 1

                
        #     state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            
            
        #     # print(state)
        #     rewards += reward
            
        #     if done:
        #         self.fitness = rewards/counter
        #         break
            
            
        
        
        print(f"fitness {self.name} - {self.fitness} {counter} steps")
        
        # if not initial_fitness:
        #     with open(self.save_location + "/fitness_calc.txt", 'a') as f:
        #         f.write(f"fitness = {self.fitness} in {counter} steps\n")
            
    
    def check_mutate(self, event):
        return random.random() <= self.get_mutation_prob(event)
    
    def check_mutate_just_evo(self,  event, best):
        return random.random() <= self.get_mutation_prob_just_evo(event, best)
    
    def get_mutation_prob_just_evo(self, event, best):
        self.set_state_action_list_just_evo(event, best)
        
        current_best_action =best.policy_net(event.state).max(1).indices.view(1, 1)
        current_best_action_value = self.best_list_state_action_values[current_best_action]
        
        greater_than_condition = self.own_list_state_action_values >= current_best_action_value
        lesser_than_condition = self.own_list_state_action_values <= current_best_action_value

        greater = np.sum(greater_than_condition)
        lesser = np.sum(lesser_than_condition)
        
        result = greater / (lesser + greater)
        
        result = result
        print(f"mutation prob: {result}\n")
        input()
        return result
        
        
        
        
    
    def mutate_attempt(self):
        samples = []
        SAMPLE_NUM = 200
        for _ in range(SAMPLE_NUM):
            event = None
            none_present = True
            while none_present:
                none_present = True
                event = self.memory.sample(1)[0]
                none_present = event.next_state is None
            samples.append(event)
            
        
        result = False
        for i in range(SAMPLE_NUM):
            event = samples[i]
            if self.check_mutate(event):
                result = True
                self.mutate(event)
        return result
                
                # print(state_action_values)
                
                
    def mutate_attempt_just_evo(self, best):
        samples = []
        SAMPLE_NUM = self.num_samples
        print(len(self.memory))
        input()
        if SAMPLE_NUM > len(self.memory):
            SAMPLE_NUM = len(self.memory)
        
        for _ in range(SAMPLE_NUM):
            event = None
            none_present = True
            while none_present:
                none_present = True
                event = self.memory.sample(1)[0]
                none_present = event.next_state is None
            # event = self.memory.sample(1)[0]
            samples.append(event)
            
        
        result = False
        for i in range(SAMPLE_NUM):
            event = samples[i]
            if self.check_mutate_just_evo(event, best):
                result = True
                self.mutate_just_evo(event, best)
            return result
                
                # print(state_action_values)
                
    def mutate_attempt_random_weight_change(self):
        if random.random() < 2:
            self.mutate_random_weight_change()
            return True
        else:
            return False
        
    def set_state_action_list_just_evo(self, event, best):
        action_list = list(range(2))
        self.tensor_action_list= torch.tensor(action_list, dtype=torch.int64, device=self.device).unsqueeze(0)
        
        state_action_values = best.policy_net(event.state).gather(1, self.tensor_action_list).cpu()
        self.best_list_state_action_values = state_action_values.detach().numpy()[0]

        
        self.best_list_state_action_values = self.best_list_state_action_values * 100
        
        if np.max(self.best_list_state_action_values) > 0:
            self.best_list_state_action_values = self.best_list_state_action_values - 2*np.max(self.best_list_state_action_values)
            
            
        state_action_values = self.policy_net(event.state).gather(1, self.tensor_action_list).cpu()
        self.own_list_state_action_values = state_action_values.detach().numpy()[0]
        
        # print(f"state action values: {self.list_state_action_values}\n")
        
        self.own_list_state_action_values = self.own_list_state_action_values * 100
        
        if np.max(self.own_list_state_action_values) > 0:
            self.own_list_state_action_values = self.own_list_state_action_values - 2*np.max(self.own_list_state_action_values)
        
    def set_state_action_list(self, event):
        action_list = list(range(self.num_actions))
        tensor_action_list= torch.tensor(action_list, dtype=torch.int64, device=self.device).unsqueeze(0)
        state_action_values = self.policy_net(event.state).gather(1, tensor_action_list).cpu()
        self.list_state_action_values = state_action_values.detach().numpy()[0]
        
        # print(f"state action values: {self.list_state_action_values}\n")
        
        self.list_state_action_values = self.list_state_action_values * 100
        
        if np.max(self.list_state_action_values) > 0:
            self.list_state_action_values = self.list_state_action_values - 2*np.max(self.list_state_action_values)
            
        
        
        # self.list_state_action_values = self.list_state_action_values - np.max(self.list_state_action_values) - 0.01
        # print(f"state action values: {self.list_state_action_values}\n")

    def get_mutation_prob(self, event):
        
        self.set_state_action_list(event)
        
        current_best_action = event.action.item()
        
        current_best_action_value = self.list_state_action_values[current_best_action]
        greater_than_condition = self.list_state_action_values >= current_best_action_value
        lesser_than_condition = self.list_state_action_values <= current_best_action_value

        greater = np.sum(greater_than_condition)
        lesser = np.sum(lesser_than_condition)
        
        result = greater / (lesser + greater)
        # print(f"mutation prob: {result}\n")
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
        current_best_action = event.action.item()
        pi_div_a = self.list_state_action_values[current_best_action] / self.list_state_action_values
        top = np.exp(pi_div_a)
        
        infs = []
        for i, el in enumerate(top): 
            if el == np.inf:
                print("inf")
                infs.append(i)
                top[i] = 0
                
        if len(infs) != 0:
            inf_prob = 0.8 / len(infs)
            bottom = sum(top)

            probability = np.empty(top.shape)
            print(probability)
            for i, el in enumerate(top):
                if i in infs:
                    probability[i] = (inf_prob)
                else:
                    probability[i] = (top[i] * 0.2/bottom)
        else:
            bottom = np.sum(top)
            probability = top/bottom

        random_number = random.random()
        
        total = 0
        for i, element in enumerate(probability):
            total = total + element
            print(f"total = {total}")
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
        probability_of_zero = self.ra
        for key in state_dict:
            shape = state_dict[key].shape
            random_tensor = torch.rand(shape, device=self.device)
            random_tensor = torch.where(random_tensor > probability_of_zero, 
                            torch.full_like(random_tensor, 1, device=self.device), 
                            torch.tensor([random.uniform(0.8, 1.2) for _ in range(random_tensor.numel())], device=self.device).reshape(shape))
            state_dict[key] = state_dict[key] * random_tensor
            print("mutated")
        self.policy_net.load_state_dict(state_dict)
        
        
    def pick_action_just_evo(self, event, best):
        current_best_action = best.policy_net(event.state).max(1).indices.view(1, 1)
        pi_div_a = self.best_list_state_action_values[current_best_action] / self.own_list_state_action_values
        top = np.exp(pi_div_a)
        
        infs = []
        for i, el in enumerate(top): 
            if el == np.inf:
                print("inf")
                infs.append(i)
                top[i] = 0
                
        if len(infs) != 0:
            inf_prob = 0.8 / len(infs)
            bottom = sum(top)

            probability = np.empty(top.shape)
            print(probability)
            for i, el in enumerate(top):
                if i in infs:
                    probability[i] = (inf_prob)
                else:
                    probability[i] = (top[i] * 0.2/bottom)
        else:
            bottom = np.sum(top)
            probability = top/bottom

        random_number = random.random()
        
        total = 0
        for i, element in enumerate(probability):
            total = total + element
            print(f"total = {total}")
            if random_number <= total:
                result = torch.tensor([[i]], device=self.device, dtype=torch.int64)
                return result
         
        
        
        
    def mutate_just_evo(self, event, best):
        
        state_batch = torch.cat((event.state, event.state), dim=0)
        next_state_batch  = torch.cat((event.next_state, event.next_state), dim=0)
        
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                    next_state_batch)), device=self.device, dtype=torch.bool)
        
        non_final_next_states = [s for s in next_state_batch
                                                    if s is not None]
        
        non_final_next_states = torch.stack(non_final_next_states)
        
        
        # best_best_current_best_action = best.policy_net(event.state).max(1).indices.view(1, 1)
        current_best_action = self.policy_net(event.state).max(1).indices.view(1, 1)
        action_to_improve = self.pick_action_just_evo(event, best)
        
        if current_best_action == action_to_improve:
            print("same")
            return
            
        
        action_batch = torch.cat((action_to_improve, action_to_improve),dim=0)
        
        positive_reward = torch.tensor([[1]], device=self.device, dtype=torch.float32)
        negative_reward = torch.tensor([[-1]], device=self.device, dtype=torch.float32)
        reward_batch = torch.cat((positive_reward, positive_reward), dim=1).squeeze(0)


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
            print(state_action_values)
            # if state_action_values[0].item() < state_action_values[1].item():
            #     break
            if self.policy_net(event.state).max(1).indices.view(1, 1) == action_to_improve:
                break
        
        state_action_values = self.policy_net(state_batch).gather(1, self.tensor_action_list)
        print(f"state_action_values{state_action_values}")
        print(f"action to improve {action_to_improve}")
        
        input()
        # input()
        print("mutated")
        


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
        for i in range(100):
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
            
            # if i == 99:
            #     input()
        
        ts_2 = str(self.target_net.state_dict())
        
        
        # with open("target_net_before.txt", 'w') as f:
        #     f.write(t_s)
            
        # with open("target_net_after.txt", 'w') as f:
        #     f.write(ts_2)
            
            
        
        # print(state_action_values)
        
        
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        print(f"state_action_values{state_action_values}")
        print(f"action to improve {action_to_improve}")
        
        input()
        
        
        # print(state_action_values)
        
        
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


# # Define the shape of the tensor
# shape = ([128, 80])  # Example shape, change as needed

# # Define the probability of an element being 0
# probability_of_zero = 0.05  # Example probability, change as needed

# # Create a random tensor of the specified shape
# random_tensor = torch.rand(shape)

# # Apply threshold to set elements to 0 or to a random number

# print(random_tensor)

# random_tensor = torch.where(random_tensor > probability_of_zero, 
#                             torch.full_like(random_tensor, 1), 
#                             torch.tensor([random.uniform(0.8, 1.2) for _ in range(random_tensor.numel())]).reshape(shape))

# # Print the random tensor

# # Print the random tensor
# print(random_tensor)
# input()

"""

torch.Size([128, 80])
torch.Size([128])
torch.Size([10, 128])
torch.Size([10])
"""


    
    