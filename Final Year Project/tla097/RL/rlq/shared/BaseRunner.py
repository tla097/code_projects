import math
from pathlib import Path
import pickle
import random
import torch
from datetime import datetime

from RL.rlq.shared.ReplayMemory import ReplayMemory, Transition
from RL.rlq.shared.BaseEnv import BaseEnv

import torch.nn as nn

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


import os


class BaseRunner():
    def __init__(self, tv = 0, ask= 0, mini_batch=1, added_comparison= 1, comparison= "Duration", layers =1, activation= "relu", load = "", save= "test", des= "des", results = "res", new_load = "new", gamma = 0.99, reward = 0.5, eps_decay = 10000, lr = 1e-4, test_train="test", num_actions= 10, file = "default", truncation_env = False, env = "an", penalty = -1,start = 0.9):
        self.device = torch.device(str("cuda") if torch.cuda.is_available() else "cpu")
        
        self.start_time = time.time()
        
        self.base_env = BaseEnv()
        
        self.incorrect = self.correct = self.counter = self.same = self.steps_done = self.eps_threshold = 0
        self.count = 0
        self.total_steps = 0
        self.memory = ReplayMemory(10000)
        self.BATCH_SIZE = 128
        gamma = float(gamma)
        self.GAMMA = gamma
        self.EPS_START = start
        self.EPS_END = 0.05
        self.EPS_DECAY = eps_decay
        self.TAU = 0.005
        self.LR = lr
        self.test_train = test_train
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
        
        self.save_frequency = 1000
        
        self.q = deque(maxlen=10)
    
    def run(self):
        self.start_time = time.time()
        test_train = " "
        while test_train not in ["test", "train"]:
            test_train = self.test_train
            
        if test_train == "test":
            self.test(False)
        else:
            self.full_train()
    
    
    def load_models(self):
        with open(self.folder_path + "/target.pkl", "rb") as p:
            self.target_net = pickle.load(p)
                    
        with open(self.folder_path + "/policy.pkl", "rb") as p:
            self.policy_net = pickle.load(p)
            
        with open(self.folder_path + "/optimiser.pkl", "rb") as p:
            self.optimizer = pickle.load(p)
            
        self.target_net.to(self.device)
        self.policy_net.to(self.device)
        self.optimiser.to(self.device)     
        
        
    def save_models(self):
        with open(self.folder_path + "/policy.pkl", "wb") as p:
            pickle.dump(self.policy_net, p)
        with open(self.folder_path + "/target.pkl", "wb") as p:
            pickle.dump(self.policy_net, p)
        with open(self.folder_path + "/optimiser.pkl", "wb") as p:
            pickle.dump(self.optimizer, p)
                  
                
                
    def pre_test(self, mid_run):
        if not mid_run:
            if self.ask:
                self.SAVE = input(f"Which file are you testing? - {self.terminal_values}: ")
                
            self.base_runner.load_models(self)
            
            path = self.font_folder_path + self.SAVE +"/tests/_post_tests.txt"
            
            with open(path, 'a') as f:
                f.write(f"\nTime = {str(datetime.utcfromtimestamp(self.start_time).date()):}\n")
        else:
            path = self.folder_path +"/tests/_in_computation.txt"
        return path
    
    
    def pre_train(self):
        
        if not Path(self.front_folder_path).exists():
            print(f"The file {self.front_folder_path} does not exist - creating folder.")
            Path(self.front_folder_path).mkdir(parents=True, exist_ok=True)
            
            
        if self.ask:
            self.SAVE = input(f"Where are you saving the model- ensure that if it is in a folder that file exists?- {self.terminal_values}: ")
        else:
            self.front_folder_path += self.file  + "/"
            
            if not Path(self.front_folder_path).exists():
                print(f"The file {self.front_folder_path} does not exist - creating folder.")
                Path(self.front_folder_path).mkdir(parents=True, exist_ok=True)
            
            
        self.folder_path = self.front_folder_path + self.SAVE
        
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
                                
            self.load_models()
        
        with open(self.folder_path + "/results/" + self.TRAINING_RESULTS + ".txt", 'a') as f:
                f.write(f"{self.SAVE}---------------------------------------------")
                f.write(f"\n\nTime = {str(datetime.utcfromtimestamp(self.start_time).date()):}\n")
                
                
        
            
        with open(self.folder_path +"/tests/_in_computation.txt", 'a') as f:
            f.write(f"\n\nTime = {str(datetime.utcfromtimestamp(self.start_time).date()):}")
            
            
            
    def train_round(self, rounds, save_location, env, state0, save = True):
        
        exit = False
        print(self.eps_threshold)
        print((f"ROUND {rounds}\n"))
        
        if save:
            print(save_location)
            with open(save_location, 'a') as f:
                f.write(f"ROUND {rounds}\n")
        
        state = torch.tensor(state0, dtype=torch.float32, device=self.device).unsqueeze(0)
        episode_rewards = []
        episode_averages = []
        k = 0
        while True: 
            k += 1
            self.total_steps +=1
            
            action = self.select_action(state)
                            
            observation, reward, done, truncated = env.step(action)

            reward_to_use = torch.tensor([reward], dtype=torch.float32, device=self.device)
            action_to_use = action

            if observation is not None:
                observation = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
                self.memory.push(state, action_to_use, observation, reward_to_use)
                state = observation.to(self.device)
            else:
                self.memory.push(state, action_to_use, observation, reward_to_use)
                done = True

            episode_rewards.append(reward)

            if k <= 100:
                episode_averages.append(sum(episode_rewards)/k)
            else:
                episode_averages.append(sum(episode_rewards[-100:])/100)
            self.optimize_model()
            
            
            target_net_state_dict = self.target_net.state_dict()
            policy_net_state_dict = self.policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*self.TAU + target_net_state_dict[key]*(1-self.TAU)
            self.target_net.load_state_dict(target_net_state_dict)
                
                
            if self.total_steps % self.save_frequency == 0:
                # self.save_models()
                pass
            
                
            
            if self.total_steps % self.test_frequency == 0:
                exit = self.full_test(True, stage=rounds, test_env=self.test_env)
                
                
                
            if done:
                avg= sum(episode_averages[-k:])/k
                curr_time=time.time() - self.start_time
                if save:
                    with open(save_location, 'a') as f:
                        f.write(f"average = {avg:.10f}. Time = {curr_time:.4f} - tuncated = {truncated}\n")
                    
                    with open(save_location, 'a') as f:
                        f.write(f"DONE WITH {k} steps- Truncated = {truncated}\n\n")
                    print(f"DONE WITH {k} steps")
                
                # self.save_models()
                pass
                    
                break
            
            if exit:
                return exit
            
            
            
    def full_train(self):
        exit = False
        self.pre_train()
        env = self.env
        state0 = self.env.reset()
        
        self.full_test(test_env=env)
        for rounds in range(0, self.num_rounds + 1):
            env = self.env
            state0 = self.env.reset()
            exit = self.train_round(rounds, self.folder_path + "/results/_" + self.TRAINING_RESULTS + ".txt", env, state0 = state0)
            if exit: 
                return exit

                    
                    
                    
    def full_test(self, mid_run= True, k= -1, stage = "", test_env = None):
        exit = False
        path = self.pre_test(mid_run=mid_run)
        
        print(f"Testing now: file = {self.SAVE}\n")
        correct, actual_count, total_count = self.test(test_env = test_env, state0=test_env.reset(test=True))
        mae = self.base_env.calculate_accuracy(test_env)
        npra = self.base_env.calc_normalised_rank_point_average(test_env)
        afpd = self.base_env.calc_APFD(test_env)
        
        string1 = f"\nROUND {stage} - total steps = {self.total_steps} - {self.SAVE} - round steps = {actual_count}   "
        string2 = f""
        string3 = f"correct = {correct/total_count:.5f} - mae = {mae}, AFPD = {afpd}, NPRA = {npra}"

        string = string1 + string2+ string3
        time_taken = time.time() - self.start_time
        
        if self.total_steps > 1000 * 100: 
            if sum(self.q) / 10 > correct/total_count:
                exit = True
        self.q.append(correct/total_count)
        
        
        if k == -1:
            k = ""
        with open(path, "a") as f:
            f.write(f"{string} - ")
            f.write(f"time taken = {time_taken}")
            
        with open(self.folder_path + "/results.txt", 'a') as f:
            f.write(f"{string2}")
            f.write(f"time taken = {time_taken}")
            
        print(f"Test Complete\n{string}")
        
        return exit
        
                
    
    def test(self,test_env = None, state0 = None):
        state = torch.tensor(state0, dtype=torch.float32, device=self.device).unsqueeze(0)
        total_count = 0
        correct = 0
        actual_count = 0
        # with open("tes1.txt", "a") as f:
        #         f.write("pivot\n")   
        #         f.write(str(test_env.pivot))
        #         f.write("\n")
        # with open("tes2.txt", "a") as f:
        #     f.write("pivot\n")   
        #     f.write(str(test_env.pivot))
        #     f.write("\n")
        while True:
            actual_count += 1
            total_count += 1
            
            action = self.policy_net(state).max(1).indices.view(1, 1)
            # with open("tes1.txt", "a") as f:
            #     f.write(str(test_env.state))
            #     f.write("\n")
            # with open("tes2.txt", "a") as f:
            #     f.write(str(test_env.state))
            #     f.write("\n")
                
                
            # with open("ARR1.txt", "a") as f:
            #     f.write(str(test_env.observation))
            #     f.write("\n")
            # with open("ARR2.txt", "a") as f:
            #     f.write(str(test_env.observation))
            #     f.write("\n")
            state, reward, done, truncated = test_env.step(action, test=True)
            
            if truncated:
                total_count = self.num_actions
                break
            
            state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            correct += reward
            if done:
                break
            
            
        with open("tes1.txt", "a") as f:
            f.write("\n")
        with open("tes2.txt", "a") as f:
            f.write("\n")
        return correct, actual_count, total_count
 
    
    def optimize_model(self):
        
        if len(self.memory) < self.BATCH_SIZE:
            return
        
        transitions = self.memory.sample(self.BATCH_SIZE)
        batch = Transition(*zip(*transitions))
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=self.device, dtype=torch.bool)

        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        # print(action_batch)
        reward_batch = torch.cat(batch.reward)
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(self.BATCH_SIZE, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch

        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()
        
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
            c = [random.choice(range(self.num_actions))]
            return torch.tensor([c], device=self.device, dtype=torch.long)
    







# import RL.rlq.shared.DQN
from RL.rlq.shared.DQN import DQN, DQN3
from RL.rlq.shared.ReplayMemory import ReplayMemory
from RL.rlq.shared.ReplayMemory import Transition
from RL.rlq.base_RL_with_shared_runner.listwise.listwise_env import ListwiseEnv
from RL.rlq.base_RL_with_shared_runner.listwise.listwise_env_truncation_trial import ListwiseEnv as ListwiseEnvTrunc

import threading
import os



        


    
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



# params = {"test_train" :"train", "num_actions" : 10, "eps_decay" : 10000}
# name = f"test"
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


    
    
