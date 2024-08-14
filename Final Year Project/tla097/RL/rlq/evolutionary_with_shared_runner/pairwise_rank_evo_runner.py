from RL.rlq.shared.memory_nodes import Memory_Nodes_Stack
from shared_evo_runner import Evo_runner
from RL.rlq.base_RL_with_shared_runner.pairwise_rank.pair_wise_rank_env import PairWiseRankEnv
from RL.rlq.shared.DQN import DQN3
import torch.optim as optim
from collections import deque


class PairWiseRankRunner(Evo_runner):
    def __init__(self, tv=0, ask=0, mini_batch=1, added_comparison=2, comparison="Duration", layers =0 , activation="relu", load="", save="test", des="des", results="res", new_load="new", gamma=0.99, reward=0.5, eps_decay=10000, lr=0.0001, test_train="test", memory: Memory_Nodes_Stack = None, name="test", num_actions=10, fitness=-100, parent_sd="", num_events=100, random_mutation_rate=0.5, num_samples=100, random_mutation_rate_threshold=0.05, uniform_low=0.95, uniform_high=1.05, low_reward = 0, high_reward = 1, mid_low_reward = 0, batch_size = 128, mem_size = 10000, tau = 0.005, start= 0.9, test_frequency = 20, test_length = 50, env = "an", file = "default"):
        super().__init__(tv, ask, mini_batch, added_comparison, comparison, layers, activation, load, save, des, results, new_load, gamma, reward, eps_decay, lr, test_train, memory, name, num_actions, fitness, parent_sd, num_events, random_mutation_rate, num_samples, random_mutation_rate_threshold, uniform_low, uniform_high)
                
        self.n_actions = 2
        self.num_actions = 2
        self.env = PairWiseRankEnv(reward_val=reward, low_reward=low_reward, high_reward=high_reward, mid_low_reward = mid_low_reward, env = env)
        self.test_env = PairWiseRankEnv(reward_val=reward, low_reward=low_reward, high_reward=high_reward, mid_low_reward = mid_low_reward, env = env)
        n_observations = self.env.observation_len
        self.prev_tests = deque(maxlen=5) 
        self.previous_average = 0
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
        
        self.num_rounds_ran= 20
        
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.LR, amsgrad=True)