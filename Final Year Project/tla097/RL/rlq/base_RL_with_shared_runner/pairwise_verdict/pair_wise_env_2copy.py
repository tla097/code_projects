from typing import Any, Optional, Tuple, Union

import numpy as np
import gym
from gym import spaces
import pandas as pd
from sklearn import preprocessing
import random
import copy

class CIPairWiseEnv(gym.Env):
    def __init__(self, reward_val):
        super(CIPairWiseEnv, self).__init__()
        
        
        excel = pd.read_csv("RL/rlq/my_data_mutual_info3P3.csv")
        column_names = ['Verdict', 'Duration']

        excel = excel[column_names]
        
        self.excel = excel
        
        self.reward_val = float(reward_val)
        
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
        
        
    def reset(self):
        
        self.dict_excel = self.excel.to_dict(orient='records')
        self.arr = self.dict_excel
        
        self.arr = self.arr[:5606]
        self.arr = random.sample(self.arr, 500)
        
        
        
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
        
        
        st1 = [value for key, value in self.arr[self.list_counter].items() if key != "Verdict"]
        st2 = [value for key, value in self.pivot.items() if key != "Verdict"]
        return list(st1) + list(st2)
    
    def reset_env(self):
        self.dict_excel = self.excel.to_dict(orient='records')
        self.arr = self.dict_excel
        
        self.arr = self.arr[:5606]
        self.arr = random.sample(self.arr, 500)
        
        
        
        self.low = 0
        self.high = len(self.arr) - 1
        self.top = -1
        # self.quicksort_stack = []
        
        # print(self.arr)

        
        
        
        # print(self.arr)
        
        size = self.high - self.low + 1
        self.quicksort_stack = [0] * (size)
    
        
        self.top = self.top + 1
        self.quicksort_stack[self.top] = self.low
        self.top = self.top + 1
        self.quicksort_stack[self.top] = self.high
        
        self.smallest_index = ( self.low - 1 )
        
        self.pivot = self.arr[self.high]
        self.pivot_index = self.high
        self.list_counter = self.low
        
        self.done = False
        
        return 
        
        
    def render(self, mode='human'):
        pass
    
    
    def calculate_reward_env(self):
        ob1 = self.state[0]
        ob2 = self.state[1]
        
        if ob1 is None:
            return 0
        
        if ob1["Verdict"] == ob2["Verdict"]:
            if ob1["Duration"] > ob2["Duration"]:
                return 0.5
            else:
                return 0
        elif ob1["Verdict"] < ob2["Verdict"]:
            return 1
        else:
            return 0    
        
    def calculate_reward(self, action):
        st1 = self.state[0]
        st2 = self.state[1]
        
        if st1 is None:
            return 0
        
        if action:
            if st1["Verdict"] == st2["Verdict"]:
                if st1["Duration"] < st2["Duration"]:
                    return self.reward_val
                else:
                    return 0
            elif st1["Verdict"] > st2["Verdict"]:
                return 1
            else:
                return 0
        else:
            if st1["Verdict"] == st2["Verdict"]:
                if st1["Duration"] > st2["Duration"]:
                    return self.reward_val
                else:
                    return 0
            elif st1["Verdict"] < st2["Verdict"]:
                return 1
            else:
                return 0
            
    def step_env(self):
        
        done = False
        
        self.state = self.arr[self.list_counter], self.pivot
        ob1, ob2 = self.state
        
        action = 0
        
        if ob1["Verdict"] == ob2["Verdict"]:
            if ob1["Duration"] < ob2["Duration"]:
                action = 0.5
            else:
                action = 0
        elif ob1["Verdict"] > ob2["Verdict"]:
            action = 1
        else:
            action = 0 
            
        reward = self.calculate_reward_env()
        
        print(reward)
                
        if action:
            self.smallest_index = self.smallest_index+1
            self.arr[self.smallest_index],self.arr[self.list_counter] = self.arr[self.list_counter],self.arr[self.smallest_index]
        
        self.list_counter = self.list_counter + 1
        
        if self.list_counter >= self.high:
            self.arr[self.smallest_index+1],self.arr[self.high] = self.arr[self.high],self.arr[self.smallest_index+1]
            self.pivot_index = self.smallest_index + 1
            
            
            if self.pivot_index-1 > self.low:
                self.top = self.top + 1
                self.quicksort_stack[self.top] = self.low
                self.top = self.top + 1
                self.quicksort_stack[self.top] = self.pivot_index - 1
    
            # # If there are elements on right side of pivot,
            # # then push right side to stack
            if self.pivot_index+1 < self.high:
                self.top = self.top + 1
                self.quicksort_stack[self.top] = self.pivot_index + 1
                self.top = self.top + 1
                self.quicksort_stack[self.top] = self.high
                
            if self.top < 0:
                done = True
            
            else:
                self.high = self.quicksort_stack[self.top]
                self.top = self.top - 1
                self.low = self.quicksort_stack[self.top]
                self.top = self.top - 1
                
                self.smallest_index = ( self.low - 1 )
                self.pivot = self.arr[self.high]
                self.list_counter = self.low
                
        return True, reward, done, {}
            
        
    def step(self):
        
        done = False
        
        action = 0
        self.state = (self.arr[self.list_counter], self.pivot)
        
        if self.arr[self.list_counter]["Verdict"] == self.pivot["Verdict"]:
            if self.arr[self.list_counter]["Duration"] < self.pivot["Duration"]:
                action = self.reward_val
            else:
                # input()
                action = 0
        elif self.arr[self.list_counter]["Verdict"] > self.pivot["Verdict"]:
            action = 1
        else:
            action =  0
        
        # print(self.state)
        
        reward = self.calculate_reward()
        
        print(reward)
        
        if action:
            # increment index of smaller element
            self.smallest_index = self.smallest_index+1
            self.arr[self.smallest_index],self.arr[self.list_counter] = self.arr[self.list_counter],self.arr[self.smallest_index]
            
        
        if self.list_counter < self.high - 1:
            self.list_counter +=1
        else:
            self.arr[self.smallest_index+1],self.arr[self.high] = self.arr[self.high],self.arr[self.smallest_index+1]
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
                self.pivot = self.arr[self.high]
                
                self.list_counter = self.low
        
        ob1 = [value for key, value in self.arr[self.list_counter].items() if key != "Verdict"]
        ob2 = [value for key, value in self.pivot.items() if key != "Verdict"]
        
        
        return list(ob1) + list(ob2), reward, done, {}
    
    
    
    
    
if 0.5:
    print(4)
    
env = CIPairWiseEnv(0.5)
env.reset_env()
while True:
    s,r,done, _ = env.step_env()
    if done:
        break
print(env.arr)
    
    

