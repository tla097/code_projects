import time
import torch
from RL.rlq.shared.memory_nodes import Memory_Nodes_Stack
from shared_evo_runner import Evo_runner
from RL.rlq.base_RL_with_shared_runner.listwise.listwise_env import ListwiseEnv
from RL.rlq.base_RL_with_shared_runner.listwise.listwise_env_truncation_trial import ListwiseEnv as ListwiseEnvTrunc
from RL.rlq.shared.DQN import DQN3
import torch.optim as optim


from shared_evo_runner import Evo_runner

class ListwiseEvoEnvRunner(Evo_runner):
    def __init__(self, tv=0, ask=0, mini_batch=1, added_comparison=2, comparison="Duration", layers=0, activation="relu", load="", save="test", des="des", results="res", new_load="new", gamma=0.99, reward=0.5, eps_decay=10000, lr=0.0001, test_train="test", memory: Memory_Nodes_Stack = None, name="test", num_actions=100, fitness=-100, parent_sd="", num_events=100, random_mutation_rate=0.5, num_samples=100, random_mutation_rate_threshold=0.05, uniform_low=0.95, uniform_high=1.05, file="default", truncation_env=False, env="an", penalty=-1, test_length = 100):
        super().__init__(tv, ask, mini_batch, added_comparison, comparison, layers, activation, load, save, des, results, new_load, gamma, reward, eps_decay, lr, test_train, memory, name, num_actions, fitness, parent_sd, num_events, random_mutation_rate, num_samples, random_mutation_rate_threshold, uniform_low, uniform_high)
        
        self.num_actions= num_actions
        self.num_rounds_ran= 150
        n_actions = num_actions
        self.n_actions= num_actions
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
        
        
        
    def run(self, gen = ""):
        
        print(self.num_actions)
        for epoch in range(self.num_rounds_ran):
            k = 0
            episode_rewards = []
            episode_averages = []
            
            state = self.envs[epoch][0]
            self_observation = self.envs[epoch][1]
            self.env.reset(arr=self_observation)
            # state = torch.tensor(self.env.reset(), dtype=torch.float32, device=self.device, requires_grad=True).unsqueeze(0)
            state = torch.tensor(state, dtype=torch.float32, device=self.device, requires_grad=True).unsqueeze(0)
            self.start_time= time.time()
            
            
            self.dummies = [0] * self.num_actions
            
            while True:
                k +=1
                self.total_steps += 1
                action = self.select_action(state)
                observation, reward, done, truncated, dummies = self.env.step(action, dummies=self.dummies)
                # observation, reward, done, truncated = self.env.step(action)
                
                reward_to_use = torch.tensor([reward], device=self.device)
                action_to_use = action
                
                self.dummies = dummies
                if truncated:
                    done = True
                else:
                    # observation[action.item() * 8 + 7] = 1
                    observation = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
                
                # Store the transition in memory
                self.memory.push(state, action_to_use, observation, reward_to_use)
                state = observation    
                
                episode_rewards.append(reward)
                
                
                
                
                if self.total_steps <= 100:
                    episode_averages.append(sum(episode_rewards)/k)
                else:
                    episode_averages.append(sum(episode_rewards[-100:])/100)
                    
                if self.total_steps % 200 == 0:
                    episode_averages = episode_averages[-100:]
                    
                if self.total_steps % 100 == 0:
                    avg= sum(episode_averages[-100:])/100
                    curr_time=time.time() - self.start_time
                    with open(self.folder_path + "/results.txt", 'a') as f:
                        f.write(f"average = {avg:.10f}. Time = {curr_time:.4f}\n")


                # print(1)
                # Perform one step of the optimization (on the policy network)
                self.optimize_model()
                
                # Soft update of the target network's weights
                # θ′ ← τ θ + (1 −τ )θ′
                target_net_state_dict = self.target_net.state_dict()
                policy_net_state_dict = self.policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key]*self.TAU + target_net_state_dict[key]*(1-self.TAU)
                self.target_net.load_state_dict(target_net_state_dict)
                    
                if done:
                    with open(self.folder_path + "/results.txt", 'a') as f:
                        f.write(f"SORTED WITH {k} steps\n\n")
                    print(f"SORTED WITH {k} steps")
                    
                    self.save_models()
                    
                    break
                
        self.calculate_fitness(gen=gen)