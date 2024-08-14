import gym
import numpy as np


# test = np.zeros((5, 5, 5))
# print(test[2,3])
# t = (2,3)
# print(test[t])
# input()

# # state = [0,0,0,0]

# # q_table = np.zeros((20,20,20,20,2))

# # print(q_table(state))
# input()

# Create the CartPole environment
env = gym.make('CartPole-v1')

# Set random seed for reproducibility
np.random.seed(42)

# Define discretization parameters
num_bins = [20, 20, 20, 20]  # number of bins for each dimension of the state space

# Initialize Q-table
state_space_size = np.prod(num_bins)
action_space_size = env.action_space.n
q_table  = {}
cart_pos = {}
cart_vel = {}
pole_ang = {}
pole_ang_vel = {}
action = {}


# q_table = np.zeros((state_space_size, action_space_size))
q_table = np.zeros((20,20,20,20,2))


# Define a function to discretize the state
def discretize_state(state):
    state_min = env.observation_space.low
    state_max = env.observation_space.high
    state_bins = [np.linspace(state_min[i], state_max[i], num=num_bins[i]) for i in range(len(num_bins))]
    discretized_state = [0,] * len(num_bins)
    
    for i in range(len(num_bins)):
        discretized_state[i] = np.digitize(state[i], state_bins[i]) - 1
        # print(discretized_state[i])
    
    return tuple(discretized_state)


# Hyperparameters
learning_rate = 0.8
discount_factor = 0.95
exploration_prob = 0.2
num_episodes = 1000

# Q-learning algorithm
for episode in range(num_episodes):
    state = discretize_state(env.reset()[0])
    # print(f"state {state}")
    total_reward = 0

    while True:
        # Choose action using epsilon-greedy policy
        if np.random.uniform(0, 1) < exploration_prob:
            action = env.action_space.sample()  # Explore
        else:
            action = np.argmax(q_table[state])
            # print(q_table[state])# Exploit
        
        step_result = env.step(action) 
        
        # Discretize the next state
        next_state, reward, done = step_result[:3]
        # print(next_state)
        next_state = discretize_state(next_state)

        # Update Q-value using the Q-learning update rule
        q_table[state][action] += learning_rate * (
            reward + discount_factor * np.max(q_table[state]) - q_table[state][action]
        )


        state = next_state
        total_reward += reward

        if done:
            break

    # Print total reward for this episode
    print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}")

# Test the trained model
total_rewards = []
for _ in range(10):
    state = discretize_state(env.reset()[0])
    # print(f"state {state}")
    episode_reward = 0
    while True:
        # print(q_table[state])
        action = np.argmax(q_table[state])
        print(f"Action {action}")
        step_result = env.step(action) 
        
        # Discretize the next state
        next_state, reward, done = step_result[:3]
        # print(next_state)
        next_state = discretize_state(next_state)
        
        
        episode_reward += reward
        next_state = discretize_state(next_state)
        state = next_state
        if done:
            break
    total_rewards.append(episode_reward)

# Print average total reward over 10 test episodes
print(f"Average Total Reward over 10 Test Episodes: {np.mean(total_rewards)}")

# Close the environment
env.close()
