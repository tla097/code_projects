from datetime import datetime
import os
import time
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from pathlib import Path

from DQN import DQN, DQN3
from RL.rlq.shared.ReplayMemory import ReplayMemory
from RL.rlq.shared.ReplayMemory import Transition
from listwise_env import ListwiseEnv

import threading
import os



class Runner():
    
    def __init__(self, tv = 0, ask= 0, mini_batch=1, added_comparison= 1, comparison= "Duration", layers =1, activation= "relu", load = "", save= "test", des= "des", results = "res", new_load = "new", gamma = 0.99, reward = 0.5, eps_decay = 10000, lr = 1e-4, test_train="test"):
        self.device = torch.device(str("cuda") if torch.cuda.is_available() else "cpu")
        
        self.incorrect = self.correct = self.counter = self.same = self.steps_done = self.eps_threshold = 0
        
        self.env = ListwiseEnv()
        self.memory = ReplayMemory(10000)
        
        self.BATCH_SIZE = 128
        gamma = float(gamma)
        self.GAMMA = gamma
        self.EPS_START = 0.9
        self.EPS_END = 0.05
        self.EPS_DECAY = eps_decay
        self.TAU = 0.005
        self.LR = lr
        self.test_train = test_train
        
        self.num_rounds = 200000000
        
        n_actions = 10
        n_observations = 10*8
        
        print(n_observations)
        self.layers = layers
        self.activation = activation
        
        self.policy_net = DQN3(n_observations, n_actions, layers, activation)
        self.policy_net = self.policy_net.to(self.device)


        
        
        self.target_net = DQN3(n_observations, n_actions, layers, activation)        
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net = self.target_net.to(self.device)

        print(423984723749832)

        for i in self.target_net.named_parameters():
            print(f"{i[0]} -> {i[1].device}")
        print(next(self.policy_net.parameters()).device)
                
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.LR, amsgrad=True)
        
        self.folder_path = "RL/rlq/models/_"
        
        self.terminal_values = tv
        
        self.ask = ask
        self.des = des
        self.SAVE = save
        self.results = results
        self.new_load = new_load
        self.TRAINING_RESULTS = results
        self.minibatch = mini_batch
        self.added_comparison = added_comparison
        self.LOAD = load
        
        self.count = 0
        
    def run(self):
        self.start_time = time.time()
        test_train = " "
        while test_train not in ["test", "train"]:
            test_train = self.test_train
            
        if test_train == "test":
            self.test(False)
        else:
            self.train()
            
    def trial(self):
        self.TRAINING_RESULTS = "test"
        self.folder_path = "RL/rlq/models/_test"
        self.start_time = time.time()
        envList = self.env.excel.values.tolist()
        envList = envList[:500] 
        random.shuffle(envList)
        self.quickSortIterative(envList, 0, 499, True)
        
            
    
    def test(self, mid_run= True, k= -1, stage = ""):
        
        self.folder_path = "RL/rlq/models/_" + self.SAVE
        
        path = ""
        
        if not mid_run:
            if self.ask:
                self.SAVE = input(f"Which file are you testing? - {self.terminal_values}: ")
                
            self.target_net.load_state_dict(torch.load(self.folder_path + "/target_" + self.SAVE + ".pth"))
            self.target_net.eval()
            
            self.policy_net.load_state_dict(torch.load(self.folder_path + "/policy_" + self.SAVE + ".pth"))
            
            self.optimizer.load_state_dict(torch.load(self.folder_path + "/optimiser_" + self.SAVE + ".pth"))
            
            self.TEST_NAME = input("Test name?: ")
            
            path = self.folder_path + "/tests/_"+ self.SAVE + ".txt"
            
            with open(path, 'a') as f:
                f.write(f"\nTime = {str(datetime.utcfromtimestamp(self.start_time).date()):}\n")
        else:
            path = self.folder_path +"/tests/_in_computation_"+ self.SAVE + ".txt"
            
        state = torch.tensor(self.env.reset(), dtype=torch.float32, device=self.device).unsqueeze(0)
        
        # print(state)
        print(f"Training now: file = {self.SAVE}\n")
        
        
        counter = 0
        incorrect = 0
        correct = 0
        same = 0
        
        
        
        while True:
            counter += 1
            
            action = self.policy_net(state).max(1).indices.view(1, 1)
            state, reward, done, truncated = self.env.step(action)
            
            if truncated:
                break
            
            state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
                
            
            # print(696969)
            
            
            
            # print(state)
            correct += reward
            
            
            if done:
                state = torch.tensor(self.env.reset(), dtype=torch.float32, device=self.device).unsqueeze(0)
                break
            
        # print(self.env.final_order)
        # print(correct/counter)
                
        string1 = f"\nROUND {stage} - {counter} steps "
        string2 = f"reward = {correct/counter} - {truncated}"
        
        string = string1 + string2
        time_taken = time.time() - self.start_time
        
        if k == -1:
            k = ""
        
        with open(path, "a") as f:
            f.write(f"{string} - ")
            f.write(f"time taken = {time_taken}")
            
        with open(self.folder_path + "/results/_" + self.TRAINING_RESULTS + ".txt", 'a') as f:
            f.write(f"{string2}")
            f.write(f"time taken = {time_taken}")
            
        print(f"Test Complete\n{string}")
    
    def train(self):
        
        if self.ask:
            self.SAVE = input(f"Where are you saving the model?- {self.terminal_values}: ")
            
        self.folder_path = "RL/rlq/models/_" + self.SAVE
        
        
        if not Path(self.folder_path).exists():
            print(f"The file {self.folder_path} does not exist - creating folder.")
            
            if self.ask:
                description = input(f"Provide a description of this model - {self.terminal_values}: \n")
            else:
                description = self.des
            
            Path(self.folder_path).mkdir(parents=True, exist_ok=True)
            
            
            
            with open(self.folder_path + "/description_" + self.SAVE, "w") as f:
                f.write(description)
        
            
            Path(self.folder_path + "/results").mkdir(parents=True, exist_ok=True)
            Path(self.folder_path + "/tests").mkdir(parents=True, exist_ok=True)
            
        
        if self.ask:
            wrong = True
            new_load = " "
            while new_load not in ["new", "load"]:
                new_load = input(f"New or Load?(new/load) - {self.terminal_values}: ")
                
            if new_load == "load":
                self.LOAD = input(f"Which file are you loading from?: {self.terminal_values} ")
                
            self.TRAINING_RESULTS = input(f"Results name? - {self.terminal_values}:  ")
                
        else:
            new_load = self.new_load
                
        if new_load == "load":
            self.target_net.load_state_dict(torch.load("RL/rlq/models/_" + self.LOAD + "/target_" + self.LOAD + ".pth"))
            self.target_net.eval()

            self.policy_net.load_state_dict(torch.load("RL/rlq/models/_" + self.LOAD + "/policy_" + self.LOAD + ".pth"))
            self.policy_net.eval()
            
            self.optimizer.load_state_dict(torch.load("RL/rlq/models/_" + self.LOAD + "/optimiser_" + self.LOAD + ".pth"))
        
        envList = self.env.excel.values.tolist()
        
        
        with open(self.folder_path + "/results/_" + self.TRAINING_RESULTS + ".txt", 'a') as f:
                f.write(f"{self.SAVE}---------------------------------------------")
                f.write(f"\n\nTime = {str(datetime.utcfromtimestamp(self.start_time).date()):}\n")
                
                
        
        # self.test(True, 0, k)
        # last_state = torch.tensor(self.env.reset(), dtype=torch.float32, device=self.device).unsqueeze(0)
        episode_rewards = []
        episode_averages = []
        
            
        with open(self.folder_path +"/tests/_in_computation_"+ self.SAVE + ".txt", 'a') as f:
            f.write(f"\n\nTime = {str(datetime.utcfromtimestamp(self.start_time).date()):}")
            
        
        
        keep = 0
        # self.test(True, stage=0)
        for rounds in range(1, self.num_rounds + 1):
            print(self.eps_threshold)
            print((f"ROUND {rounds}\n"))
            
            with open(self.folder_path + "/results/_" + self.TRAINING_RESULTS + ".txt", 'a') as f:
                f.write(f"ROUND {rounds}\n")
            
            state = torch.tensor(self.env.reset(), dtype=torch.float32, device=self.device).unsqueeze(0)
            
            episode_rewards = []
            episode_averages = []
            k = 0
            
            while True: 
                k += 1
                keep += 1
                
                action = self.select_action(state)
                                
                observation, reward, done, truncated = self.env.step(action)

                reward_to_use = torch.tensor([reward], dtype=torch.float32, device=self.device)
                action_to_use = action

                if observation is not None:
                    print("not none")
                    observation = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
                    self.memory.push(state, action_to_use, observation, reward_to_use)
                    state = observation.to(self.device)
                else:
                    self.memory.push(state, action_to_use, observation, reward_to_use)
                    done = True
                    
                
                # print(f"reward {reward}")

                
                episode_rewards.append(reward)
                
                
                
                # print(f"rew = {reward}")
                
                
                if k <= 100:
                    episode_averages.append(sum(episode_rewards)/k)
                # else:
                #     episode_averages.append(sum(episode_rewards[-100:])/100)
                    
                # if k % 200 == 0:
                #     episode_averages = episode_averages[-100:]
                    
                # if k % 100 == 0:
                #     avg= sum(episode_averages[-100:])/100
                #     curr_time=time.time() - self.start_time
                #     with open(self.folder_path + "/results/_" + self.TRAINING_RESULTS + ".txt", 'a') as f:
                #         f.write(f"average = {avg:.10f}. Time = {curr_time:.4f}\n")

                # Perform one step of the optimization (on the policy network)
                self.optimize_model()
                
                # Soft update of the target network's weights
                # θ′ ← τ θ + (1 −τ )θ′
                target_net_state_dict = self.target_net.state_dict()
                policy_net_state_dict = self.policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key]*self.TAU + target_net_state_dict[key]*(1-self.TAU)
                self.target_net.load_state_dict(target_net_state_dict)
                    
                    
                if keep % 1000 == 0:
                    torch.save(self.policy_net.state_dict(), self.folder_path + "/policy_" + self.SAVE + ".pth")
                    torch.save(self.target_net.state_dict(), self.folder_path + "/target_" + self.SAVE + ".pth")
                    torch.save(self.optimizer.state_dict(), self.folder_path + "/optimiser_" + self.SAVE + ".pth")
                    
                    
                if keep % 200 == 0:
                    self.test(True, stage=rounds)
                    
                    
                    
                if done:
                    
                    avg= sum(episode_averages[-k:])/k
                    curr_time=time.time() - self.start_time
                    with open(self.folder_path + "/results/_" + self.TRAINING_RESULTS + ".txt", 'a') as f:
                        f.write(f"average = {avg:.10f}. Time = {curr_time:.4f} - tuncated = {truncated}\n")
                    
                    with open(self.folder_path + "/results/_" + self.TRAINING_RESULTS + ".txt", 'a') as f:
                        f.write(f"DONE WITH {k} steps- Truncated = {truncated}\n\n")
                    print(f"DONE WITH {k} steps")
                    
                    torch.save(self.policy_net.state_dict(), self.folder_path + "/policy_" + self.SAVE + ".pth")
                    torch.save(self.target_net.state_dict(), self.folder_path + "/target_" + self.SAVE + ".pth")
                    torch.save(self.optimizer.state_dict(), self.folder_path + "/optimiser_" + self.SAVE + ".pth")
                    
                    
                    if rounds % 5 == 0:
                        pass
                        # self.test(True, stage=rounds)
                        
                    break
                    
                
                
                
                        
        # self.test(True, k, self.counter)

    
    
    def select_action(self, state):
        sample = random.random()
        
        # sample = -100
        
        self.eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
            math.exp(-1. * self.steps_done / self.EPS_DECAY)
        self.steps_done += 1
        
        # print(self.steps_done)
        # print(self.eps_threshold)
        if sample > self.eps_threshold:
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return (self.policy_net(state).max(1).indices.view(1, 1))
        else:
            c = [random.choice(range(10))]
            return torch.tensor([c], device=self.device, dtype=torch.long)
        
        
    # def do_round(self, val1, val2):
        
    #     action = self.select_action(self.env.last_state[0][:-2] + self.env.last_state[1][:-2])
    #     reward = 0         
            
        
            
            
    #     state = torch.tensor(self.env.last_state[0][:-2] + self.env.last_state[1][:-2], dtype=torch.float32, device=self.device).unsqueeze(0)
    #     reward_to_use = torch.tensor([reward], device=self.device)
    #     action_to_use = torch.tensor(action, device=self.device)

    #     # Store the transition in memory
    #     self.memory.push(state, action_to_use,sent_obs, reward_to_use)
    #     self.count += 1
    #     self.counter += 1
    #     # print(episode_durations)
    #     self.episode_rewards.append(reward)
    #     # print(episode_rewards)
    #     # input()
    #     if len(self.episode_rewards) <= 100:
    #         self.episode_averages.append(sum(self.episode_rewards)/self.counter)
    #     else:
    #         self.episode_averages.append(sum(self.episode_rewards[-100:])/100)
            
    #     if self.count % 200 == 0:
    #         self.episode_averages = self.episode_averages[-100:]
            
    #     if self.count % 100 == 0:
    #         avg= sum(self.episode_averages[-100:])/100
    #         curr_time=time.time() - self.start_time
    #         with open(self.folder_path + "/results/_" + self.TRAINING_RESULTS + ".txt", 'a') as f:
    #             f.write(f"----COUNT: {self.count}------\n")
    #             f.write(f"{avg}. Time = {curr_time:.4f}\n")
                
    #     if self.count % 100000 == 0:
    #         self.test(True)

    #     # Perform one step of the optimization (on the policy network)
    #     self.optimize_model()
        
        

    #     # Soft update of the target network's weights
    #     # θ′ ← τ θ + (1 −τ )θ′
    #     target_net_state_dict = self.target_net.state_dict()
    #     policy_net_state_dict = self.policy_net.state_dict()
    #     for key in policy_net_state_dict:
    #         target_net_state_dict[key] = policy_net_state_dict[key]*self.TAU + target_net_state_dict[key]*(1-self.TAU)
    #     self.target_net.load_state_dict(target_net_state_dict)
        
        
        
        
    #     # if count % 100 == 0:
    #     #     # plot_durations()


    #     self.env.last_state = (val1, val2)
    #     self.env.last_action = action
            
            
    #     if self.counter % 1000 == 0:
    #         torch.save(self.policy_net.state_dict(), self.folder_path + "/policy_" + self.SAVE + ".pth")
    #         torch.save(self.target_net.state_dict(), self.folder_path + "/target_" + self.SAVE + ".pth")
    #         torch.save(self.optimizer.state_dict(), self.folder_path + "/optimiser_" + self.SAVE + ".pth")
    #     return action
    
    def optimize_model(self):
        
        ### code I have adapted from elsewhere
        if len(self.memory) < self.BATCH_SIZE:
            return
        transitions = self.memory.sample(self.BATCH_SIZE)
        # input()
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))
        # print(999999999999999999999999999999999999)
        # # print(batch)
        
        # print(batch.reward)

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        
        
        ####################
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=self.device, dtype=torch.bool)
        ###########################
        
        # non_final_mask = torch.tensor(tuple(map(lambda r: r != 0,
        #                                       batch.reward)), device=device, dtype=torch.bool)
        
        # print(4809324793287432987432897)
        # print(non_final_mask)
        #################################################################### 
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])##
        ######################################################################
        
        # non_final_next_states = torch.cat([r for r in batch.next_state
        #                                             if s is not None])##
        
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
        
# class Runner_with_extra_feature(Runner):
#     def __init__(self, tv, ask, mini_batch, added_comparison, comparison, added_feature, layers, activation, load = "", save= "", des= "", results = "", new_load = "", eps_decay = 10000):
#         super().__init__(tv, ask, mini_batch, added_comparison, comparison, layers, activation, load, save, des, results, new_load, eps_decay)
#         self.added_feature = added_feature
        
#         self.env = MyPairEnv_with_extra_feature(comparison, added_feature)
#         self.env.set_up_env()
#         self.env.reset()
        
#         self.policy_net = DQN3(18, 2, self.layers, activation).to(self.device)
#         self.target_net = DQN3(18, 2, self.layers, activation).to(self.device)
#         self.target_net.load_state_dict(self.policy_net.state_dict())
    
#     def make_selection(self, val1, val2):
#         state = torch.tensor(val1[:-2] + val2[:-2], dtype=torch.float32, device=self.device).unsqueeze(0)
#         selection = self.policy_net(state).max(1).indices.view(1, 1)
#         # print(selection)
#         check_reward = 0
#         if selection:
#             check_reward = self.env.give_reward((val1[9], val1[10]), (val2[9], val2[10]), self.added_comparison)
#         else:
#             check_reward = self.env.give_reward((val2[9], val2[10]), (val1[9], val1[10]), self.added_comparison)
            
#         if check_reward == 0:
#             self.incorrect += 1
#         else:
#             self.correct += 1
            
#         self.counter += 1
        
#         # print(self.counter)
        
#         return selection
    
#     def do_round(self, val1, val2):
        
#         action = self.select_action(self.env.last_state[0][:-2] + self.env.last_state[1][:-2])
#         reward = 0
#         # print(action)
#         if action:
#             reward = self.env.give_reward((self.env.last_state[0][9], self.env.last_state[0][10]), (self.env.last_state[1][9], self.env.last_state[1][10]), self.added_comparison)
#         else:
#             reward = self.env.give_reward((self.env.last_state[1][9], self.env.last_state[1][10]), (self.env.last_state[0][9], self.env.last_state[0][10]), self.added_comparison)


#         if self.terminal_values:
#             if reward == 0:
#                 sent_obs = None
#             else:   
#                 sent_obs = torch.tensor(val1[:-2] + val2[:-2], dtype=torch.float32, device=self.device).unsqueeze(0)
#         else:
#             sent_obs = torch.tensor(val1[:-2] + val2[:-2], dtype=torch.float32, device=self.device).unsqueeze(0)
            
            
        
            
            
#         state = torch.tensor(self.env.last_state[0][:-2] + self.env.last_state[1][:-2], dtype=torch.float32, device=self.device).unsqueeze(0)
#         reward_to_use = torch.tensor([reward], device=self.device)
#         action_to_use = torch.tensor(action, device=self.device)

#         # Store the transition in memory
#         self.memory.push(state, action_to_use,sent_obs, reward_to_use)
#         self.count += 1
#         self.counter += 1
#         # print(episode_durations)
#         self.episode_rewards.append(reward)
#         # print(episode_rewards)
#         # input()
#         if len(self.episode_rewards) <= 100:
#             self.episode_averages.append(sum(self.episode_rewards)/self.counter)
#         else:
#             self.episode_averages.append(sum(self.episode_rewards[-100:])/100)
            
#         if self.count % 200 == 0:
#             self.episode_averages = self.episode_averages[-100:]
            
#         if self.count % 100 == 0:
#             avg= sum(self.episode_averages[-100:])/100
#             curr_time=time.time() - self.start_time
#             with open(self.folder_path + "/results/_" + self.TRAINING_RESULTS + ".txt", 'a') as f:
#                 f.write(f"----COUNT: {self.count}------\n")
#                 f.write(f"{avg}. Time = {curr_time:.4f}\n")
                
#         if self.count % 100000 == 0:
#             self.test(True)

#         # Perform one step of the optimization (on the policy network)
#         self.optimize_model()
        
        

#         # Soft update of the target network's weights
#         # θ′ ← τ θ + (1 −τ )θ′
#         target_net_state_dict = self.target_net.state_dict()
#         policy_net_state_dict = self.policy_net.state_dict()
#         for key in policy_net_state_dict:
#             target_net_state_dict[key] = policy_net_state_dict[key]*self.TAU + target_net_state_dict[key]*(1-self.TAU)
#         self.target_net.load_state_dict(target_net_state_dict)

#         self.env.last_state = (val1, val2)
#         self.env.last_action = action
            
#         if self.counter % 1000 == 0:
#             torch.save(self.policy_net.state_dict(), self.folder_path + "/policy_" + self.SAVE + ".pth")
#             torch.save(self.target_net.state_dict(), self.folder_path + "/target_" + self.SAVE + ".pth")
#             torch.save(self.optimizer.state_dict(), self.folder_path + "/optimiser_" + self.SAVE + ".pth")
#         return action
    
def add_thread_f():
    
    name = input("Name:? ")
    ask = input("Do you want to be asked questions?: (y/n)")
    
    if ask == "y":
        # comparison = input("comparison?: ")
        if name != "cancel": # and comparison != "cancel":
            load = input("load from where?: ")
            # extra_feature = input("Extra_feature?(y/n): ")
            eps = int(input("eps decay?(basic is 10000): "))
            # added_comparison = int(input("compare with something else? (1/0): "))
            des = input("description: ")
            res = input("results name: ")
            nl = input("new or load? (new/load): ")
            layers = int(input("Num hidden layers: "))
            # activation = input("Activation(relu or lrelu): ")
            gamma = input("Gamma? (default 0.99): ")
            reward = input("Reward for comp(default = 0.5): ")
            lr = float(input("learning rate?(1e-4): "))
            test_train = input("test/train: ")
            runner = Runner(layers=layers,  load=load, save=name, des=des, eps_decay=eps, results=res, new_load=nl, gamma=gamma, reward=reward, test_train=test_train)   
            return threading.Thread(target=runner.run, name=str(name))
        else:
            return None
    else:
        runner = Runner(save=name)   
        return threading.Thread(target=runner.run, name=str(name))
        
    
# def get_input_with_timeout():
#     while True:
#         if msvcrt.kbhit():  # Check if a key is pressed
#             user_input = msvcrt.getch().decode("utf-8")
#             return user_input

#         time.sleep(0.1)  # Sleep for a short duration to avoid high CPU usage
    
    

    
# thread_list = []
# current_index = -1
# def go():
#     global thread_list
#     global current_index
#     while True: 
#         user_input = get_input_with_timeout()
#         if user_input is not None:
#             print(f"You entered: {user_input}")
#             with my_lock:
#                 t = add_thread_f()
#                 thread_list.append((t, 0))
#                 current_index +=1
                
#         time.sleep(0.1)
                
# def loop():
#     while True:
#         if current_index == -1:
#             pass
#         elif not thread_list[current_index][1]:
#             thread_list[current_index][0].start()
#             thread_list[current_index][0].join()
#             thread_list[current_index][1] = None, 1
            
#         time.sleep(0.1)
    
# g = threading.Thread(target=go, name=str("go"))
# l = threading.Thread(target=loop, name=str("loop"))
# g.start()
# l.start()
# g.join()
# l.join()




my_lock = threading.Lock()
    

def get_input_with_timeout():
    while True:
        user_input = input("")
        return user_input
  # Sleep for a short duration to avoid high CPU usage

# Set the timeout to 5 seconds

list = []
ind = -1
def main():
    global list
    global ind
    while True:
        user_input = get_input_with_timeout()

        if user_input is not None:
            # i =threading.Thread(target=test_go, name=str("test_go"), args=user_input)
            
            with my_lock:
                i = add_thread_f()
                if i is not None:
                    list.append((i, 0, 0))
                    ind +=1
        time.sleep(0.1)
        
def test_go(num):
    while True:
        print("RUNNING" + num)
        # time.sleep(0.1)
        
def go():
    while True:
        if ind == -1:
            pass
        elif not list[ind][1]:
            list[ind][0].start()
            with my_lock:
                temp = list[ind][0]
                list[ind] = temp, 1, 0
        time.sleep(0.1)
    
def joins():
    while True:
        if ind != -1:
            if list[ind][1] and not list[ind][2]:
                list[ind][0].join()
                with my_lock:
                    list[ind] = None, 1, 1
        time.sleep(0.1)
    
    
    
    
    
# m = threading.Thread(target=main, name="main")
# p = threading.Thread(target=go, name="print")
# j = threading.Thread(target=joins, name="joins")

# m.start()
# p.start()
# j.start()



# params = {"lr":0.0001, "test_train" :"train", "eps_decay":1000000, "layers" : 2}
# name = f"lw_test"
# params["save"] = name
# runner = Runner(**params, des = str(params))
# t1 = threading.Thread(target=runner.run, name=name)
# t1.start()
# t1.join()


# params = {"lr":0.001, "test_train" :"train", "eps_decay":1000000, "layers" : 2}
# name = f"list_wise-lr_{params['lr']}-eps_{params['eps_decay']}-layers_{params['layers']}"
# params["save"] = name
# runner = Runner(**params, des = str(params))
# t2= threading.Thread(target=runner.run, name=name)
# t2.start()

# params = {"lr":0.01, "test_train" :"train", "eps_decay":1000000, "layers" : 2}
# name = f"list_wise-lr_{params['lr']}-eps_{params['eps_decay']}-layers_{params['layers']}"
# params["save"] = name
# runner = Runner(**params, des = str(params))
# t3 = threading.Thread(target=runner.run, name=name)
# t3.start()

# params = {"lr":0.00001, "test_train" :"train", "eps_decay":1000000, "layers" : 2}
# name = f"list_wise-lr_{params['lr']}-eps_{params['eps_decay']}-layers_{params['layers']}"
# params["save"] = name
# runner = Runner(**params, des = str(params))
# t4 = threading.Thread(target=runner.run, name=name)
# t4.start()

# params = {"lr":0.000001, "test_train" :"train", "eps_decay":1000000, "layers" : 2}
# name = f"list_wise-lr_{params['lr']}-eps_{params['eps_decay']}-layers_{params['layers']}"
# params["save"] = name
# runner = Runner(**params, des = str(params))
# t5 = threading.Thread(target=runner.run, name=name)
# t5.start()


# t1.start()
# t2.start()
# t1.join()
# t2.join()



# # runner = Runner(save=name)   
# # t1 = threading.Thread(target=runner.run, name=str(name))

# t1.start()
# t2.start()
# t3.start()
# t4.start()
# t5.start()

# t1.join()
# t2.join()
# t3.join()
# t4.join()
# t5.join()


# m.join()
# p.join()
# j.join()

runner = Runner(save="lw_demo", test_train="train", eps_decay=10000)
runner.run()
# runner.trial()      
    
    

    

# # threadList = []
# # add_thread = 1
# # while add_thread:
# #     name = input("Name:? ")
# #     comparison = input("comparison?: ")
# #     load = input("load from where?: ")
# #     extra_feature = input("Extra_feature?(y/n): ")
# #     eps = int(input("eps decay?(basic is 10000): "))
# #     added_comparison = int(input("compare with something else? (1/0): "))
# #     des = input("description: ")
# #     res = input("results name: ")
# #     nl = input("new or load? (new/load): ")
# #     layers = int(input("Num hidden layers: "))
# #     activation = input("Activation(relu or lrelu): ")
    
#     runner = None
#     if extra_feature == "y":
#         what_feature = input("What feature?: ")
#         runner = Runner_with_extra_feature(0, 0, 1, added_comparison, comparison, what_feature, layers, activation, load, name, des, res, nl, eps_decay = eps)
#     else:
#         runner = Runner(0, 0, 1, 1, comparison, layers, activation, load, name, des, res, nl, eps_decay= eps)
        
    
        
#     t = threading.Thread(target=runner.run, name=str(name))
    
#     threadList.append(t)
    
#     add_thread = int(input("another thread?(1/0): "))

    
# for thread in threadList:
#     thread.start()
    
# for thread in threadList:
#     thread.join()
    
    
    



# Set the timeout to 5 seconds


        
    

# runner.run()


 
# # tv = threading.Thread(target=tvRunner.run, name='tv')
# # ntv = threading.Thread(target=nonTvRunner.run, name='ntv')
# # mbtv = threading.Thread(target=minibatchTvRunner.run, name='mbtv')
# mbntv = threading.Thread(target=minibatchNTvRunner.run, name='mbntv')

# # tr

# # tv.start()
# # ntv.start()
# # mbtv.start()
# mbntv.start()

# # tv.join()
# # ntv.join()
# # mbtv.join()
# mbntv.join()


    
    
