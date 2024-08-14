from typing import Any, Optional, Tuple, Union

import numpy as np
import gym
from gym import spaces
import pandas as pd
from sklearn import preprocessing
import random
import copy

class PairWiseEnv(gym.Env):
    def __init__(self, reward_val=0.5, low_reward=0, high_reward=1, mid_low_reward = 0, env = "an", num_actions = 2, penalty = -1, test_length = 50):
        super(PairWiseEnv, self).__init__()
        
        self.test_length = test_length
        
        
        # excel = pd.read_csv("normalised_4_log.csv")
        # # excel = pd.read_csv("classification/ranking/my_data_mutual_info3P3.csv")
        
        # column_names = ['last_run_normalised_log', 'my_maturity_normalized', 'Cycle_normalized', 'GapInRun_normalised_log', 'last_result', 'Month', 'Failure_Percentage', 'times_ran_normalized',
        #                         'Verdict', 'Duration']
        
        file = ""
        if env == "an":
            file = "data/all_normed_logged6.csv"
        elif env == "orig":
            file = "data/my_data_mutual_info3P3.csv"
        elif env == "log4":
            file = "data/normalised_4_log_copy.csv"
        else:
            print("wrong file input")
            input()
            
        
        
        excel = pd.read_csv(file)
        column_names = ['last_run', 'my_maturity', 'Cycle', 'GapInRun', 'last_result', 'Month', 'Failure_Percentage', 'times_ran',
                                'Verdict', 'Duration']

        excel = excel[column_names]
        
        self.excel = excel
        
        self.reward_val = float(reward_val)
        
        self.low_reward = float(low_reward)
        
        self.high_reward = high_reward
        
        self.mid_low_reward = mid_low_reward
        
        print(reward_val)
        
        
        self.reward_range = (0,1)
        
        
        df_copy = copy.deepcopy(excel)
        del df_copy["Verdict"]
        
        self.observation_len = df_copy.shape[1]
        self.observation_len = self.observation_len * 2

        
        min = df_copy.min(axis = 0).to_numpy()
        min2 = np.concatenate((min, min))
        max = df_copy.max(axis = 0).to_numpy()
        max2 = np.concatenate((max, max))
        
        self.observation_space = spaces.Box(low=min2, high=max2)
        self.action_space = spaces.Discrete(2)
        
        
    def reset(self, test = False, arr = None):
        
        if arr is not None:
            self.arr = arr
            
        else:
            self.dict_excel = self.excel.to_dict(orient='records')
            self.arr = self.dict_excel
            
            self.arr = self.arr[:5000]
            random.shuffle(self.arr)
            
            if test:
                self.arr = self.arr[4000:5000]
                self.arr = random.sample(self.arr, self.test_length)
                
            else:
                self.arr = self.arr[:4000]
                self.arr = random.sample(self.arr, 50)
            # self.arr = random.sample(self.arr, 50)
        
        
        
        self.optimal = sorted(self.arr, key=lambda x: (-x['Verdict'], x['Duration']))
        for i, x in  enumerate(self.optimal): x['ID'] = i
        self.optimal = copy.deepcopy(self.optimal)
        
        self.num_fails = 0
        l = len(self.arr)
        for k in range(l):
            self.num_fails += self.arr[k]["Verdict"]
        
        
        
        self.low = 0
        self.high = len(self.arr) - 1
        
        
        # Create an auxiliary stack
        self.size = self.high - self.low + 1
        self.stack = [0] * (self.size)

        # initialize top of stack
        self.top = -1

        # push initial values of l and h to stack
        self.top = self.top + 1
        self.stack[self.top] = self.low
        self.top = self.top + 1
        self.stack[self.top] =self. high
        
        # Pop h and l
        self.high = self.stack[self.top]
        self.top = self.top - 1
        self.low = self.stack[self.top]
        self.top = self.top - 1
        
        self.smallest_index = ( self.low - 1 )
        self.pivot = self.arr[self.high]
        self.list_counter = self.low
        self.state = (self.arr[self.list_counter]), (self.pivot)
        
        
        st1 = [value for key, value in self.arr[self.list_counter].items() if key != "Verdict" and key != 'ID']
        st2 = [value for key, value in self.pivot.items() if key != "Verdict" and key != 'ID']
        
        self.observation = self.arr
        return list(st1) + list(st2)
        
        
    def calculate_reward(self, action):
        st1 = self.state[0]
        st2 = self.state[1]
        
        if st1 is None:
            return 0
        
        if action:
            if st1["Verdict"] == st2["Verdict"]:
                if st1["Duration"] < st2["Duration"]:
                    return self.high_reward
                else:
                    return self.mid_low_reward
            elif st1["Verdict"] > st2["Verdict"]:
                return self.high_reward
            else:
                return self.low_reward
        else:
            if st1["Verdict"] == st2["Verdict"]:
                if st1["Duration"] > st2["Duration"]:
                    return self.high_reward
                else:
                    return self.mid_low_reward
            elif st1["Verdict"] < st2["Verdict"]:
                return self.high_reward
            else:
                return self.low_reward
        
        # if action:
        #     if st1["Verdict"] > st2["Verdict"]:
        #         return self.high_reward
        #     else:
        #         return self.low_reward
        # else:
        #     if st1["Verdict"] < st2["Verdict"]:
        #         return self.high_reward
        #     else:
        #         return self.low_reward
            
            
    # trains to do more and more steps
    # have a penalty every step to ensure this does not happen
    
    # maybe reward of 1 for correct but - 0.5 every step to speed everything up
    # reward of 1 0.5 -100
            
        
    def step(self, action, test= False):
        
        done = False
        self.state = (self.observation[self.list_counter], self.pivot)
        
        # print(f"state {self.state}\n\n")
        
        
        reward = self.calculate_reward(action)
        
        if action:
            # increment index of smaller element
            self.smallest_index = self.smallest_index+1
            self.observation[self.smallest_index],self.observation[self.list_counter] = self.observation[self.list_counter],self.observation[self.smallest_index]
            
        
        if self.list_counter < self.high - 1:
            self.list_counter +=1
        else:
            self.observation[self.smallest_index+1],self.observation[self.high] = self.observation[self.high],self.observation[self.smallest_index+1]
            self.pivot_index = self.smallest_index+1
            
            if self.pivot_index-1 > self.low:
                self.top = self.top + 1
                self.stack[self.top] = self.low
                self.top = self.top + 1
                self.stack[self.top] = self.pivot_index - 1

            # If there are elements on right side of pivot,
            # then push right side to stack
            if self.pivot_index+1 < self.high:
                self.top = self.top + 1
                self.stack[self.top] = self.pivot_index + 1
                self.top = self.top + 1
                self.stack[self.top] = self.high
                
            
            
            if self.top < 0:
                # return [1000]*18, 10000, True, {}
                done = True
            else:
                
                # Pop h and l
                self.high = self.stack[self.top]
                self.top = self.top - 1
                self.low = self.stack[self.top]
                self.top = self.top - 1

                # Set pivot element at its correct position in
                # sorted array
                
                ######################################
                
                self.smallest_index = ( self.low - 1 )
                self.pivot = self.observation[self.high]
                
                self.list_counter = self.low
        
        ob1 = [value for key, value in self.observation[self.list_counter].items() if key != "Verdict" and key != 'ID']
        ob2 = [value for key, value in self.pivot.items() if key != "Verdict" and key != 'ID']
        
        
        return list(ob1) + list(ob2), reward, done, {}

