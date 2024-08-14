from typing import Any, Optional, Tuple, Union

import numpy as np
import gym
from gym import spaces
import pandas as pd
from sklearn import preprocessing
import random
import copy

class ListwiseEnv(gym.Env):
    def __init__(self, num_actions = 100, env = "an", penalty = -1):
        super(ListwiseEnv, self).__init__()
        
        
        self.penaltly = penalty
        file = ""
        if env == "an":
            file = "data/all_normed_logged6.csv"
        elif env == "orig":
            file = "data/my_data_mutual_info3P3.csv"
        elif env == "log5":
            file = "data/normalised_5_log_listwise_copy.csv"
        else:
            print("wrong file input")
            input()
        
        
        excel = pd.read_csv(file)
        self.column_names = ['Cycle', 'Duration', 'last_run', 'times_ran', 'Failure_Percentage', 'my_maturity', 'Month', 'Rank', 'Verdict']
    
        # Name,CalcPrio,Verdict,DurationGroup,TimeGroup,Quarter,Num_Previous_Execution,Maturity_Level,Rank_score,Original_Index,Rank,GapInRun,InSameCycle,CycleRun,last_result,gap_cats,Duration_normalized,Cycle_normalized,Month_normalized,Failure_Percentage_normalized,last_run_normalised_log,times_ran_normalized,my_maturity_normalized

        self.num_actions = num_actions

        excel = excel[self.column_names]
        
        self.excel = excel
        
        self.reward_range = (0,1)
        
        # print(excel)
        
        
        
        # self.observation_len = self.no_rank.shape[1]

        
        # min = df_copy.min(axis = 0).to_numpy()
        # min2 = np.concatenate((min, min))
        # max = df_copy.max(axis = 0).to_numpy()
        # max2 = np.concatenate((max, max))
        
        # self.observation_space = spaces.Box(low=min2, high=max2)
        # self.action_space = spaces.Discrete(2)
        
        
    def reset(self, test = False, needed = True, obs = None):
        self.rank = 0
        self.obs_to_send = []
        
        self.correct_guesses = 0
        
        if needed:
            self.dict_excel = self.excel.to_dict(orient='records')
            self.arr = self.dict_excel
            self.arr = self.arr[:5606]
            random.shuffle(self.arr)
            if test:
                self.arr = self.arr[4000:5606]
            else:
                self.arr = self.arr[:4000]
                
                
            self.arr = random.sample(self.arr, self.num_actions)
                
            self.arr.sort(key=lambda x:x['Rank'])
            for i, x in  enumerate(self.arr): x['ID'] = i
            self.optimal = copy.deepcopy(self.arr)
            random.shuffle(self.arr)
            
            self.num_fails = 0
            l = len(self.arr)
            for k in range(l):
                self.num_fails += self.arr[k]["Verdict"]
            
            self.arr = random.sample(self.arr, self.num_actions)
            
            
        
            df = pd.DataFrame(self.arr)
            sorted_df = df.sort_values(by='Rank')
            sorted_df["Rank"] = range(1, self.num_actions + 1)
            sorted_df["dummy"] = [0]*self.num_actions
            
            
            self.observation = sorted_df.to_dict(orient='records')
            random.shuffle(self.observation)
            
            observation = []
        
            for el in self.observation:
                    for key, value in el.items():
                        if key != "Rank" and key !='ID' and key != 'Verdict':
                            observation.append(value)
                            
            return observation
        else:
            self.observation = obs    
        
    def render(self, mode='human'):
        pass
        
        
    def calculate_reward(self, action,dummies=  None):
        
        
        if dummies is not None:
            if dummies[action]:
                return -100
            else:
                optimal_rank = self.observation[action]["Rank"]
                # print(f"opop {optimal_rank}")
                norm_op_rank = optimal_rank/self.num_actions
                norm_action = action/self.num_actions
                # print((1 - (norm_action - norm_op_rank)**2).item())
                return (1 - (norm_action - norm_op_rank)**2).item()
            
            
        else:
            if self.observation[action]["dummy"] > 0:
                return self.penaltly
            else:
                optimal_rank = self.observation[action]["Rank"]
                # print(f"opop {optimal_rank}")
                norm_op_rank = optimal_rank/self.num_actions
                norm_action = action/self.num_actions
                # print((1 - (norm_action - norm_op_rank)**2).item())
                return (1 - (norm_action - norm_op_rank)**2).item()
            
        
    def step(self, action, dummies = None, test = False):
        
        # print(dummies)
        done = False
        truncated = False

        reward = self.calculate_reward(action, dummies)
        
        if reward != self.penaltly:
            self.correct_guesses +=1  
           
        else:
            if test:
                return None, -1, True, True
            
        if self.correct_guesses == self.num_actions:
            done = True
        # elif self.observation[action]["dummy"] == 0:
        self.observation[action]["dummy"] = 1
                
        
        to_send = []
        for el in self.observation:
            for key, value in el.items():
                if key != "Rank" and key !='ID' and key != 'Verdict':
                    to_send.append(value)
                    
        return to_send, reward, done, False
            # self.reset()
            

        

        
        # print(action)
        
        # if dummies is not None:
            
        #     if self.rank < self.num_actions - 1 and dummies[action] == 0:
        #         dummies[action] = 1
        #         self.rank = self.rank + 1
        
        #     elif self.rank == self.num_actions-1:
        #         done = True
        #     to_send = []
            
        #     if reward == -100:
        #         to_send = None
        #         truncated = True
        #         # self.reset()
        #     else:
        #         for index, el in enumerate(self.observation):
        #             for key, value in el.items():
        #                 if key != "Rank":
        #                     if key == "dummy":
        #                         to_send.append(dummies[index])
        #                     else: 
        #                         to_send.append(value)
                            
        #     # print(to_send)
                        
                
        #     return to_send, reward, done, truncated, dummies
            
        # else:
            
        #     if reward == -100:
        #         return None, -100, False, True
        #     if self.rank < self.num_actions - 1 and self.observation[action]["dummy"] == 0:
        #         self.observation[action]["dummy"] = 1
        #         self.rank = self.rank + 1
            
        #     elif self.rank == self.num_actions-1:
        #         done = True
        #         # self.reset()
                
                
        #     to_send = []
            
        #     # print(f"{self.rank}")
            
        #     # print(self.observation)
            
        #     for el in self.observation:
        #         for key, value in el.items():
        #             if key != "Rank":
        #                 to_send.append(value)
                        
            
                
        #     return to_send, reward, done, False
                
        
        
        
        # return list(ob1) + list(ob2), reward, done, {}
        
        
# test = PointWiseEnv()
# test.reset()
