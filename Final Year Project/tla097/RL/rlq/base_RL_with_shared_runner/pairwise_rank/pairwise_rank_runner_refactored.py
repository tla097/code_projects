import copy
from datetime import datetime
import os
import pickle
import queue
import time
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
import torch.optim as optim

from RL.rlq.shared.DQN import DQN, DQN3
from RL.rlq.base_RL_with_shared_runner.pairwise_rank.pair_wise_rank_env import PairWiseRankEnv

from RL.rlq.shared.BaseRunner import BaseRunner

import os
from RL.rlq.shared.BaseRunner import BaseRunner

class PairWiseRankRunner(BaseRunner):
    def __init__(self, tv = 0, ask= 0, mini_batch=1, added_comparison= 1, comparison= "Duration", layers =1, activation= "relu", load = "", save= "test", des= "des", results = "res", new_load = "n", gamma = 0.99, reward = 0.5, eps_decay = 10000, lr = 1e-4, low_reward = 0, high_reward = 1, mid_low_reward = 0, test_train = "test", batch_size = 128, mem_size = 10000, tau = 0.005, start= 0.9, test_frequency = 20, test_length = 50, env = "an", file = "default" ):
        super().__init__(tv, ask, mini_batch, added_comparison, comparison, layers, activation, load, save, des, results, new_load, gamma, reward, eps_decay, lr, test_train, 2, file, False, env, 0)
        self.n_actions = 2
        self.num_actions = 2
        self.env = PairWiseRankEnv(reward_val=reward, low_reward=low_reward, high_reward=high_reward, mid_low_reward = mid_low_reward, env = env, test_length=test_length)
        self.test_env = PairWiseRankEnv(reward_val=reward, low_reward=low_reward, high_reward=high_reward, mid_low_reward = mid_low_reward, env = env, test_length=test_length)
        n_observations = self.env.observation_len
        self.prev_tests = deque(maxlen=5) 
        self.previous_average = 0
        self.test_frequency = test_frequency
        self.test_length = test_length
        self.file = file
        self.front_folder_path = "RL/rlq/models/pair_wise_rank/_"
        
        self.policy_net = DQN3(n_observations, self.n_actions, layers, activation)
        self.target_net = DQN3(n_observations, self.n_actions, layers, activation)
        self.policy_net.to(self.device)
        self.target_net.to(self.device)
        
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net = self.target_net.to(self.device)
        
        
        self.target_net.train()
        self.policy_net.train()
        
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.LR, amsgrad=True)
        
        self.num_rounds = 200000
        
        self.test_frequency = test_frequency