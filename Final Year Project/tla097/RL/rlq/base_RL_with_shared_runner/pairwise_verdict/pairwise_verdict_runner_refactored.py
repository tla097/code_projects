from RL.rlq.shared.BaseRunner import BaseRunner
from RL.rlq.base_RL_with_shared_runner.pairwise_verdict.pair_wise_env_2 import PairWiseEnv
from RL.rlq.shared.DQN import DQN3
import torch.optim as optim


class PairWiseVerdictRunner(BaseRunner):
    def __init__(self, tv = 0, ask= 0, mini_batch=1, added_comparison= 1, comparison= "Duration", layers =1, activation= "relu", load = "", save= "test", des= "des", results = "res", new_load = "n", gamma = 0.99, reward = 1, eps_decay = 10000, lr = 1e-4, low_reward = 0, high_reward = 1, mid_low_reward = 0, test_train = "test", batch_size = 128, mem_size = 10000, tau = 0.005, start= 0.9, file = "default", env = "an", test_length= 50, test_frequency = 10000):
        super().__init__(tv, ask, mini_batch, added_comparison, comparison, layers, activation, load, save, des, results, new_load, gamma, reward, eps_decay, lr, test_train, 2, file, False, env, 0)  
        self.env = PairWiseEnv(reward_val=reward, low_reward=low_reward, high_reward=high_reward, mid_low_reward = mid_low_reward, env = env, test_length=100)
        self.test_env = PairWiseEnv(reward_val=reward, low_reward=low_reward, high_reward=high_reward, mid_low_reward = mid_low_reward, env = env,test_length=100)
        self.n_actions = 2
        self.num_actions = 2
        n_observations = self.env.observation_len
        self.file = file
        self.front_folder_path = "RL/rlq/models/pairwise/_"
        self.num_rounds = 20000
        
        self.test_train = test_train
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
        
        self.test_frequency = test_frequency
        self.test_length = test_length
        
        
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.LR, amsgrad=True)