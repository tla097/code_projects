import copy
from datetime import datetime
import os
import pickle
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
from pair_wise_env_2_all_normed_logged import CIPairWiseEnv

import threading
import os



class Runner():
    
    def __init__(self, tv = 0, ask= 0, mini_batch=1, added_comparison= 1, comparison= "Duration", layers =1, activation= "relu", load = "", save= "test", des= "des", results = "res", new_load = "n", gamma = 0.99, reward = 0.5, eps_decay = 10000, lr = 1e-4, low_reward = 0, high_reward = 1, mid_low_reward = 0, test_train = "test", batch_size = 128, mem_size = 10000, tau = 0.005, start= 0.9, file = "default"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.device = "cpu"
        print(self.device)

        self.incorrect = self.correct = self.counter = self.same = self.steps_done = self.eps_threshold = 0
        
        self.env = CIPairWiseEnv(reward_val=reward, low_reward=low_reward, high_reward=high_reward, mid_low_reward = mid_low_reward)
        self.memory = ReplayMemory(100000)
        
        self.BATCH_SIZE = batch_size
        gamma = float(gamma)
        self.GAMMA = 0.99
        self.EPS_START = start
        self.EPS_END = 0.05
        self.EPS_DECAY = eps_decay
        self.TAU = 0.5
        self.LR = lr
        
        self.num_rounds = 20000
        
        self.test_train = test_train
        
        self.n_actions = 2
        n_observations = self.env.observation_len
        
        print(n_observations)
        self.layers = layers
        self.activation = activation
        
        self.policy_net = DQN3(n_observations, self.n_actions, layers, activation)
        
        self.target_net = DQN3(n_observations, self.n_actions, layers, activation)
        self.policy_net.to(self.device)
        self.target_net.to(self.device)
        
        self.target_net.load_state_dict(self.policy_net.state_dict())
        

        self.target_net = self.target_net.to(self.device)
        
        
        self.target_net.train()
        self.policy_net.train()
        
        self.file = file
        
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.LR, amsgrad=True)
        
        self.folder_path = "RL/rlq/models/pairwise/" + self.file
        
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
        # test_train = " "
        # while test_train not in ["test", "train"]:
        #     # test_train = input(f"test or train?(test/train) - {self.terminal_values}: ")
            
        if self.test_train == "test":
            self.test(False)
        else:
            print("train")
            self.train()
            
            
            
    def trial(self):
            
        self.folder_path = "RL/rlq/models/param_tests/_" + self.SAVE
        
        
        if not Path(self.folder_path).exists():
            print(f"The file {self.folder_path} does not exist - creating folder.")
            
            description = self.des
            
            Path(self.folder_path).mkdir(parents=True, exist_ok=True)
            
            
            
            with open(self.folder_path + "/description_" + self.SAVE, "w") as f:
                f.write(description)
            
        self.y = 0
            
        
        for rounds in range(1, 10 + 1):
            print(f"{self.SAVE} {self.eps_threshold}")
            print((f"ROUND {rounds}\n"))
            
            state = torch.tensor(self.env.reset(), dtype=torch.float32, device=self.device).unsqueeze(0)
            episode_rewards = []
            k = 0
                
                
            while True: 
                k += 1
                self.y +=1
                
                action = self.select_action(state)
                
                observation, reward, done, _ = self.env.step(action) 

                observation = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
                reward_to_use = torch.tensor([reward], device=self.device)
                action_to_use = action

                # Store the transition in memory
                self.memory.push(state, action_to_use, observation, reward_to_use)
                state = observation
                


                # print(1)
                # Perform one step of the optimization (on the policy network)
                self.optim_test()
                
                # Soft update of the target network's weights
                # θ′ ← τ θ + (1 −τ )θ′
                # if k % 100 == 0:
                #     self.target_net.load_state_dict(self.policy_net.state_dict())
                target_net_state_dict = self.target_net.state_dict()
                policy_net_state_dict = self.policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key]*self.TAU + target_net_state_dict[key]*(1-self.TAU)
                self.target_net.load_state_dict(target_net_state_dict)
                
                
                
                # print(str(this_state_dict) ==str(last_state))                
                    
                    
                if done:
                    with open(self.folder_path + "/" + self.TRAINING_RESULTS + ".txt", 'a') as f:
                        f.write(f"SORTED WITH {k} steps\n\n")
                    print(f"SORTED WITH {k} steps")
                    
                    torch.save(self.policy_net.state_dict(), self.folder_path + "/policy_" + self.SAVE + ".pth")
                    torch.save(self.target_net.state_dict(), self.folder_path + "/target_" + self.SAVE + ".pth")
                    torch.save(self.optimizer.state_dict(), self.folder_path + "/optimiser_" + self.SAVE + ".pth")
                    
                    break
        
                            
        
            
    
    def test(self, mid_run= True, k= -1, stage = ""):
        
        print(mid_run)
        
        if not mid_run:
            if self.ask:
                self.SAVE = input(f"Which file are you testing? - {self.terminal_values}: ")
                
            self.folder_path = "RL/rlq/models/" + self.file + "//" + self.SAVE
                
            self.target_net.load_state_dict(torch.load(self.folder_path + "/target" + self.SAVE + ".pth"))
            self.target_net.eval()
            
            self.policy_net.load_state_dict(torch.load(self.folder_path + "/policy" + self.SAVE + ".pth"))
            
            self.optimizer.load_state_dict(torch.load(self.folder_path + "/optimiser" + self.SAVE + ".pth"))
            
            # self.TEST_NAME = input("Test name?: ")
            
            path = self.folder_path + "/tests/_post_tests.txt"
            
            with open(path, 'a') as f:
                f.write(f"\nTime = {str(datetime.utcfromtimestamp(self.start_time).date()):}\n")
        else:
            path = self.folder_path +"/tests/_in_computation.txt"
            
        state = torch.tensor(self.env.reset(test=True), dtype=torch.float32, device=self.device).unsqueeze(0)
        print(f"Testing now: file = {self.SAVE}\n")
        
        
        counter = 0
        incorrect = 0
        correct = 0
        same = 0
        
        
        
        while True:
            counter += 1
            
            action = self.policy_net(state).max(1).indices.view(1, 1)
            
            state, reward, done, _ = self.env.step(action)
            state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            
            # print(state)
            if reward == self.env.high_reward:
                correct += 1
            else:
                incorrect += 1
            
            if done:
                break
            
            
            # "low_reward":-100, "high_reward":-1, "reward":-100, "mid_low_reward":-50,
                
            
       	
        
        string1 = f"\nROUND {stage} - {self.SAVE}"
        string2 = f"incorrect = {incorrect/counter:.5f} - "
        string3 = f"correct = {correct/counter:.5f}"
        
        string = string1 + string2 + string3
        time_taken = time.time() - self.start_time
        
        if k == -1:
            k = ""
        
        with open(path, "a") as f:
            f.write(f"{string} - ")
            f.write(f"time taken = {time_taken:.4f}")
            
        with open(self.folder_path + "/results/" + self.TRAINING_RESULTS + ".txt", 'a') as f:
            f.write(f"{string2 + string3}")
            f.write(f"time taken = {time_taken:.4f}")
            
        print(f"Test Complete\n{string}")
    
    def train(self):
        
        if self.ask:
            self.SAVE = input(f"Where are you saving the model?- {self.terminal_values}: ")
                
        if not Path(self.folder_path).exists():
            print(f"The file {self.folder_path} does not exist - creating folder.")
            Path(self.folder_path).mkdir(parents=True, exist_ok=True)
            
            
        self.folder_path = self.folder_path + "/" + self.SAVE
        
        if not Path(self.folder_path).exists():
            print(f"The file {self.folder_path} does not exist - creating folder.")
            
            if self.ask:
                description = input(f"Provide a description of this model - {self.terminal_values}: \n")
            else:
                description = self.des
            
            Path(self.folder_path).mkdir(parents=True, exist_ok=True)
            
            
            
            with open(self.folder_path + "/description.txt", "w") as f:
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
                                
            with open(self.folder_path + "/target.pkl", "rb") as p:
                    self.target_net = pickle.load(p)
            self.target_net.to(self.device)
            
            

            with open(self.folder_path +  "/policy.pkl", "rb") as p:
                    self.policy_net = pickle.load(p)            
            self.policy_net.to(self.device)
            
                    
            with open(self.folder_path + "/optimiser.pkl", "rb") as p:
                    self.optimizer = pickle.load(p)
                    
            self.optimizer.to(self.device)
        
        with open(self.folder_path + "/results/" + self.TRAINING_RESULTS + ".txt", 'a') as f:
                f.write(f"{self.SAVE}---------------------------------------------")
                f.write(f"\n\nTime = {str(datetime.utcfromtimestamp(self.start_time).date()):}\n")
                
                
        
            
        with open(self.folder_path +"/tests/_in_computation"+ self.SAVE + ".txt", 'a') as f:
            f.write(f"\n\nTime = {str(datetime.utcfromtimestamp(self.start_time).date()):}")
            
            self.y = 0
            
            
            
        self.test(True, stage=0)
            
        
        for rounds in range(1, self.num_rounds + 1):
            print(f"{self.SAVE} {self.eps_threshold}")
            print((f"ROUND {rounds}\n"))
            
            with open(self.folder_path + "/results/" + self.TRAINING_RESULTS + ".txt", 'a') as f:
                f.write(f"ROUND {rounds}\n")
            
            state = torch.tensor(self.env.reset(), dtype=torch.float32, device=self.device, requires_grad=True).unsqueeze(0)
            episode_rewards = []
            episode_averages = []
            k = 0


            if rounds % 20 == 0:
                self.test(True, stage=rounds)
                
                with open(self.folder_path + "/policy.pkl", "wb") as p:
                    pickle.dump(self.policy_net, p)
                with open(self.folder_path + "/target.pkl", "wb") as p:
                    pickle.dump(self.policy_net, p)
                with open(self.folder_path + "/optimiser.pkl", "wb") as p:
                    pickle.dump(self.optimizer, p)
                
            while True: 
                k += 1
                self.y +=1
                
                action = self.select_action(state)
                
                observation, reward, done, _ = self.env.step(action) 

                observation = torch.tensor(observation, dtype=torch.float32, device=self.device, requires_grad=True).unsqueeze(0)
                pre_obs = observation
                reward_to_use = torch.tensor(reward, device=self.device, dtype=torch.float32,requires_grad=True)
                reward_to_use= reward_to_use.unsqueeze(0)
                # print(reward_to_use)
                action_to_use = action

                # Store the transition in memory
                self.memory.push(state, action_to_use, observation, reward_to_use)
                episode_rewards.append(reward)
                
                pre_state = state
                state = observation
                
                if k <= 100:
                    episode_averages.append(sum(episode_rewards)/k)
                else:
                    episode_averages.append(sum(episode_rewards[-100:])/100)
                    
                if k % 200 == 0:
                    episode_averages = episode_averages[-100:]
                    
                if k % 100 == 0:
                    avg= sum(episode_averages[-100:])/100
                    curr_time=time.time() - self.start_time
                    with open(self.folder_path + "/results/" + self.TRAINING_RESULTS + ".txt", 'a') as f:
                        f.write(f"average = {avg:.10f}. Time = {curr_time:.4f}\n")


                # print(1)
                # Perform one step of the optimization (on the policy network)
                self.optimize_model()
                
                # Soft update of the target network's weights
                # θ′ ← τ θ + (1 −τ )θ′
                # if k % 100 == 0:
                #     self.target_net.load_state_dict(self.policy_net.state_dict())
                target_net_state_dict = self.target_net.state_dict()
                policy_net_state_dict = self.policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key]*self.TAU + target_net_state_dict[key]*(1-self.TAU)
                self.target_net.load_state_dict(target_net_state_dict)
                
                
                
                # print(str(this_state_dict) ==str(last_state))                
                    
                    
                if done:
                    with open(self.folder_path + "/results/" + self.TRAINING_RESULTS + ".txt", 'a') as f:
                        f.write(f"SORTED WITH {k} steps\n\n")
                    print(f"SORTED WITH {k} steps")
                    
                    # # action_to_use = action_to_use.squeeze(0)

                    # state_action_values = self.policy_net(pre_state).gather(1, action_to_use)
                    
                    # next_state_values = torch.zeros(1, device=self.device)
                    # with torch.no_grad():
                    #     next_state_values[True] = self.target_net(pre_obs).max(1).values
                    # # Compute the expected Q values
                    # expected_state_action_values = (next_state_values * self.GAMMA) + reward_to_use
                    
                    # # Compute Huber loss
                    # criterion = nn.SmoothL1Loss()
                    # loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

                    # # Optimize the model
                    # self.optimizer.zero_grad()
                    # loss.backward()
                    # # In-place gradient clipping
                    # torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
                    # self.optimizer.step()
                    
                    # target_net_state_dict = self.target_net.state_dict()
                    # policy_net_state_dict = self.policy_net.state_dict()
                    # for key in policy_net_state_dict:
                    #     target_net_state_dict[key] = policy_net_state_dict[key]*self.TAU + target_net_state_dict[key]*(1-self.TAU)
                    # self.target_net.load_state_dict(target_net_state_dict)
                    
                    break
                
                
                if self.eps_threshold < 0.0502:
                    break
            if self.eps_threshold < 0.0502:
                    break
                
    """"""

    
    
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
                # print(self.policy_net(state))

                return self.policy_net(state).max(1).indices.view(1, 1).to(self.device)
        else:
            return torch.tensor([[random.choice([0,1])]], device=self.device, dtype=torch.int64)
        
        
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
        
        # print(self.y)
        
        # print(self.policy_net.state_dict())
        
        ### code I have adapted from elsewhere
            
        if len(self.memory) < self.BATCH_SIZE:
            return
        
        
        
        
        

        transitions = self.memory.sample(self.BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))
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
        action_batch = torch.cat(batch.action)
        # print(action_batch)
        reward_batch = torch.cat(batch.reward)
        
        # print(state_batch)
        
        # input()

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
     
        
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        
        # if self.y % 1000 == 0:
        #     print(state_action_values)
            
        #     with open("policy_net.txt", 'a') as f:
        #         f.write(str(self.y) + "\n")
        #         f.write(str(self.policy_net.state_dict()) + "\n\n")
                
                
        #     with open("target_net.txt", 'a') as f:
        #         f.write(str(self.y) + "\n")
        #         f.write(str(self.target_net.state_dict()) + "\n\n")
        
        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1).values
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        
        
        # print(f"non final mask = {non_final_mask}")
        # print(f"non_final_next_states = {non_final_next_states}")
        # print(f"state_batch = {state_batch}")
        # print(f"action_batch = {action_batch}")
        # print(f"reward_batch = {reward_batch}")
        # print(f"state_action_values = {state_action_values}")
        
        
        
        next_state_values = torch.zeros(self.BATCH_SIZE, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch
        
        
        
        # print(self.policy_net.state_dict())


        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()
        
        #####################################################################
        # output_batch = model(train_batch)           # compute model output
        # loss = loss_fn(output_batch, labels_batch)  # calculate loss

        # optimizer.zero_grad()  # clear previous gradients
        # loss.backward()        # compute gradients of all variables wrt loss

        # optimizer.step()  
        
        
        # print(self.policy_net.state_dict())
        
        
        
    def optim_test(self):
        
        # print(self.y)
        
        # print(self.policy_net.state_dict())
        
        ### code I have adapted from elsewhere
        if len(self.memory) < self.BATCH_SIZE:
            return
        transitions = self.memory.sample(self.BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))
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
        action_batch = torch.cat(batch.action)
        # print(action_batch)
        reward_batch = torch.cat(batch.reward)
        
        # print(state_batch)
        
        # input()

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        
        if self.y % 1000 == 0:
            with open(self.folder_path + "\\" + self.TRAINING_RESULTS + ".txt", 'a') as f:
                f.write(f"{state_action_values}\n\n\n\n")
        
        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1).values
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        
        
        # print(f"non final mask = {non_final_mask}")
        # print(f"non_final_next_states = {non_final_next_states}")
        # print(f"state_batch = {state_batch}")
        # print(f"action_batch = {action_batch}")
        # print(f"reward_batch = {reward_batch}")
        # print(f"state_action_values = {state_action_values}")
        
        
        
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
            test_train = input("Test_or_train")
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
    
    
    
    
    
m = threading.Thread(target=main, name="main")
p = threading.Thread(target=go, name="print")
j = threading.Thread(target=joins, name="joins")


# m.start()
# p.start()
# j.start()

# params = {"lr":0.001, "low_reward":0, "high_reward":1, "reward":0.5, "mid_low_reward":0, "test_train" :"train", "eps_decay":10000000, "layers" : 2}
# name = f"pairwise-lr_{params['lr']}-eps_{params['eps_decay']}-layers_{params['layers']}-8"
# load = f"pairwise-lr_{params['lr']}-eps_{params['eps_decay']}-layers_{params['layers']}-7"
# new_load = "load"
# params["save"] = name
# params["load"] = load
# params["new_load"] = new_load
# runner = Runner(**params, des = str(params) + "eps from 20" )
# t11 = threading.Thread(target=runner.run, name=name)
# t11.start()


# params = {"lr":0.0001, "low_reward":0, "high_reward":1, "reward":0.5, "mid_low_reward":0, "test_train" :"train", "eps_decay":10000000, "layers" : 2}
# name = f"pairwise-lr_{params['lr']}-eps_{params['eps_decay']}-layers_{params['layers']}-8"
# load = f"pairwise-lr_{params['lr']}-eps_{params['eps_decay']}-layers_{params['layers']}-7"
# new_load = "load"
# params["save"] = name
# params["load"] = load
# params["new_load"] = new_load
# runner = Runner(**params, des = str(params)+ "eps from 20")
# t12 = threading.Thread(target=runner.run, name=name)
# t12.start()

# params = {"lr":0.00001, "low_reward":0, "high_reward":1, "reward":0.5, "mid_low_reward":0, "test_train" :"train", "eps_decay":10000000, "layers" : 2}
# name = f"pairwise-lr_{params['lr']}-eps_{params['eps_decay']}-layers_{params['layers']}-8"
# load = f"pairwise-lr_{params['lr']}-eps_{params['eps_decay']}-layers_{params['layers']}-7"
# new_load = "load"
# params["save"] = name
# params["load"] = load
# params["new_load"] = new_load
# runner = Runner(**params, des = str(params)+ "eps from 20")
# t13 = threading.Thread(target=runner.run, name=name)
# t13.start()

# params = {"lr":0.000001, "low_reward":0, "high_reward":1, "reward":0.5, "mid_low_reward":0, "test_train" :"train", "eps_decay":10000000, "layers" : 2}
# name = f"pairwise-lr_{params['lr']}-eps_{params['eps_decay']}-layers_{params['layers']}-8"
# load = f"pairwise-lr_{params['lr']}-eps_{params['eps_decay']}-layers_{params['layers']}-7"
# new_load = "load"
# params["save"] = name
# params["load"] = load
# params["new_load"] = new_load
# runner = Runner(**params, des = str(params)+ "eps from 20")
# t14 = threading.Thread(target=runner.run, name=name)
# t14.start()


def main():
    params = {}
    name = f"score test"
    load = f""
    new_load = "new"
    params["save"] = name
    params["load"] = load
    params["new_load"] = new_load
    runner = Runner(**params, des = str(params), test_train="train")
    target=runner.run()
    
# main()


# t11.join()
# t12.join()
# t13.join()
# t14.join()
# t15.join()

# m.join()
# p.join()
# j.join()

# runner = Runner(0, 0, 1, 1, "Duration", 1, "relu",  "load", "test", "des", "res", "new", eps_decay= 10000)

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

# j = 0
# for eps in [1000000, 1000, 10000, 100000]:
#     for bs in [32, 64, 128, 512, 2048]:
#         for me in [10000, 100000, 1000000, 1000000]:
#             for tau in [0.5, 0.05, 0.005, 0.0005, 0.00005, 0.000005]:
#                 for lr in [0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001]:
#                     for gamma in [0.01, 0.4, 0.8, 0.99]:
                        
#                         j +=1
                        
#                         params = {"low_reward":0, "high_reward":1, "reward":0.5, "mid_low_reward":0, "test_train" :"train", "eps_decay":eps, "layers" : 1, "activation": "lrelu", "gamma": gamma, "tau": tau, "batch_size": bs, "mem_size" : me}
#                         name = "param_testing_" + str(j)
#                         runner = Runner(eps_decay=eps, batch_size=bs, mem_size=me, tau=tau, lr=lr, gamma=gamma, des = str(params), save=name)
#                         runner.trial()



# r1 = Runner()
# r2 = Runner()

# obs = r1.env.reset()
# r2.env.arr = copy.copy(r1.env.arr)

# r1.env.arr[0] = 69

# print(r1.env.arr)
# print(r2.env.arr)
    
    
