from RL.rlq.shared.BaseRunner import BaseRunner
from RL.rlq.base_RL_with_shared_runner.listwise.listwise_env import ListwiseEnv
from RL.rlq.base_RL_with_shared_runner.listwise.listwise_env_truncation_trial import ListwiseEnv as ListwiseEnvTrunc
from RL.rlq.shared.DQN import DQN3
import torch.optim as optim

class ListWiseRunner(BaseRunner):
    def __init__(self, tv=0, ask=0, mini_batch=1, added_comparison=1, comparison="Duration", layers=0, activation="relu", load="", save="test", des="des", results="res", new_load="new", gamma=0.99, reward=0.5, eps_decay=10000, lr=0.0001, test_train="test", num_actions=100, file="default", truncation_env=False, env="an", penalty=-1, start = 0.9):
        super().__init__(tv, ask, mini_batch, added_comparison, comparison, layers, activation, load, save, des, results, new_load, gamma, reward, eps_decay, lr, test_train, num_actions, file, truncation_env, env, penalty, start=start) 
        
        self.num_actions= num_actions
        self.num_rounds = 200000000
        
        n_actions = num_actions
        n_observations = num_actions*8
        
        self.layers = layers
        self.activation = activation
        
        self.policy_net = DQN3(n_observations, n_actions, layers, activation)
        self.policy_net = self.policy_net.to(self.device)
        
        self.target_net = DQN3(n_observations, n_actions, layers, activation)        
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net = self.target_net.to(self.device)
                
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.LR, amsgrad=True)
        
        
        if truncation_env:
            self.env = ListwiseEnvTrunc(num_actions=num_actions, env = env, penalty = penalty)
            self.test_env= ListwiseEnvTrunc(num_actions=num_actions, env = env, penalty = penalty)
        else:
            self.env = ListwiseEnv(num_actions=num_actions, env = env)
            self.test_env= ListwiseEnv(num_actions=num_actions, env = env)
        self.file = file
        self.front_folder_path = "RL/rlq/models/list_wise/_"
        
        self.test_frequency = 1000