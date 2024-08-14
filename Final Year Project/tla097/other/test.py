
import random

def generate_similar_data_with_random_increase(num_samples, start_parent_fitness, end_parent_fitness):
    data = ""
    parent_fitness = start_parent_fitness
    
    for _ in range(num_samples):
        child_fitness = random.uniform(0.45, 0.55)
        data += f"parent fitness {parent_fitness} - child fitness {child_fitness}\n"
        fitness_increment = random.uniform(0.0001, 0.01)  # Adding randomness to the increment
        parent_fitness = min(parent_fitness + fitness_increment, end_parent_fitness)  # Ensure it doesn't exceed end_parent_fitness
    
    return data

# Generate similar data with increasing parent fitness and randomness
num_samples = 100
start_parent_fitness = 0.5
end_parent_fitness = 0.7
similar_data_with_random_increase = generate_similar_data_with_random_increase(num_samples, start_parent_fitness, end_parent_fitness)

# Print the generated data with increasing parent fitness and randomness
print(similar_data_with_random_increase)