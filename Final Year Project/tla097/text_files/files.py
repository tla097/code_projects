# # import os
# # def remove_colons(directory):
# #     for root, dirs, files in os.walk(directory):
# #         # Rename directories
# #         for d in dirs:
# #             if ':' in d:
# #                 new_d = d.replace(':', '')
# #                 old_path = os.path.join(root, d)
# #                 new_path = os.path.join(root, new_d)
# #                 os.rename(old_path, new_path)
# #                 print(f"Renamed directory: {old_path} -> {new_path}")
# #         # Rename files
# #         for filename in files:
# #             if ':' in filename:
# #                 new_filename = filename.replace(':', '')
# #                 old_path = os.path.join(root, filename)
# #                 new_path = os.path.join(root, new_filename)
# #                 os.rename(old_path, new_path)
# #                 print(f"Renamed file: {old_path} -> {new_path}")

# # # Replace 'path_to_directory' with the path to the root directory of your file structure
# # # remove_colons('path_to_directory')

# # remove_colons('/data/private/tla097/tla097/RL')

import matplotlib.pyplot as plt
import numpy as np

# Data
classifiers = ['Gradient Boosted', 'SVM', 'Random Forest', 'Naive Bayes']
scores = [0.9816370384840789, 0.5690564563391287, 0.9835905450283259, 0.614573158820082]
times = [844.172189950943, 1478.1122269630432, 385.283056974411, 0.12885355949401855]
cv_scores = [0.8265286188708733, 0.9663996874389529]
cv_times = [26.32169270515442, 8.901706218719482]

# Plotting
fig, ax1 = plt.subplots()

# Bar graph for scores
bar_width = 0.35
index = np.arange(len(classifiers))
bar1 = ax1.bar(index, scores, bar_width, color='orange', label='Score')

# Secondary axis for running time
ax2 = ax1.twinx()
bar2 = ax2.bar(index + bar_width, times, bar_width, color='blue', alpha=0.5, label='Time')

# X-axis ticks and labels
ax1.set_xlabel('Classifier')
ax1.set_xticks(index + bar_width / 2)
ax1.set_xticklabels(classifiers)

# Y-axis labels
ax1.set_ylabel('Score', color='orange')
ax2.set_ylabel('Time (s)', color='blue')

# Adding cross-validation scores and times
ax1.scatter(index + bar_width / 2, cv_scores, color='black', label='CV Score', zorder=5)
ax2.scatter(index + bar_width / 2, cv_times, color='black', label='CV Time', zorder=5)

# Annotations for cross-validation points
for i, txt in enumerate(cv_scores):
    ax1.annotate(f'{txt:.2f}', (index[i] + bar_width / 2, cv_scores[i]), textcoords="offset points", xytext=(0,10), ha='center')
for i, txt in enumerate(cv_times):
    ax2.annotate(f'{txt:.2f}', (index[i] + bar_width / 2, cv_times[i]), textcoords="offset points", xytext=(0,10), ha='center')

# Legend
bars = [bar1, bar2]
ax1.legend(bars, [bar.get_label() for bar in bars])

plt.show()





# # # Function to parse the data from the text file
# # def parse_data(file_path):
# #     parent_fitness = []
# #     child_fitness = []

# #     with open(file_path, 'r') as file:
# #         for line in file:
# #             line = line.strip().split(' - ')
# #             parent_fitness.append(float(line[0].split()[-1]))
# #             child_fitness.append(float(line[1].split()[-1]))

# #     return parent_fitness, child_fitness

# # # Function to create a bar chart
# # def create_bar_chart(parent_fitness, child_fitness):
# #     avg_parent_fitness = sum(parent_fitness) / len(parent_fitness)
# #     avg_child_fitness = sum(child_fitness) / len(child_fitness)

# #     categories = ['Parent Fitness', 'Child Fitness']
# #     values = [avg_parent_fitness, avg_child_fitness]

# #     plt.bar(categories, values, color=['blue', 'orange'])
# #     plt.xlabel('Fitness Type')
# #     plt.ylabel('Average Fitness')
# #     plt.title('Average Child Fitness vs Average Parent Fitness')
# #     plt.show()

# # # Path to the text file containing the data
# # file_path = "test_children.txt"
# # # Parse data from the text file
# # parent_fitness, child_fitness = parse_data(file_path)

# # # Create and display the bar chart
# # create_bar_chart(parent_fitness, child_fitness)



def chlid_vs_parent_ret_better(file_path):
    parent_fitness = []
    child_fitness = []

    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip().split('-')
            
            p =float(line[0].split()[-1])
            c= float(line[1].split()[-1])
            
            if p >= c:
                parent_fitness.append(1)
            else:
                child_fitness.append(1)

    avg_parent_fitness = sum(parent_fitness)
    avg_child_fitness = sum(child_fitness)
    
    
    return avg_parent_fitness, avg_child_fitness
# Function to parse the data from the text file
def chlid_vs_parent_ret(file_path):
    parent_fitness = []
    child_fitness = []

    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip().split('-')
            if float(line[0].split()[-1]) != 0:
                parent_fitness.append(float(line[0].split()[-1]))
            else:
                parent_fitness.append(0)
                
            
            if float(line[1].split()[-1])!=0:
                child_fitness.append(float(line[1].split()[-1]))
            else:
                child_fitness.append(0)

    avg_parent_fitness = sum(parent_fitness) / len(parent_fitness)
    avg_child_fitness = sum(child_fitness) / len(child_fitness)
    
    
    return avg_parent_fitness, avg_child_fitness

    # categories = ['Parent Fitness', 'Child Fitness']
    # values = [avg_parent_fitness, avg_child_fitness]

    # plt.bar(categories, values, color=['blue', 'orange'])
    # plt.xlabel('Fitness Type')
    # plt.ylabel('Average Fitness')
    # plt.title('Average Child Fitness vs Average Parent Fitness')
    # plt.show()
    
def chlid_vs_parent(file_path):
    parent_fitness = []
    child_fitness = []

    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip().split('-')
            parent_fitness.append(float(line[0].split()[-1]))
            child_fitness.append(float(line[1].split()[-1]))

    avg_parent_fitness = sum(parent_fitness) / len(parent_fitness)
    avg_child_fitness = sum(child_fitness) / len(child_fitness)

    categories = ['Parent Fitness', 'Child Fitness']
    values = [avg_parent_fitness, avg_child_fitness]

    plt.bar(categories, values, color=['blue', 'orange'])
    plt.xlabel('Fitness Type')
    plt.ylabel('Average Fitness')
    plt.title('Average Child Fitness vs Average Parent Fitness')
    plt.show()
    
def child_sets(files, title):
    names = []
    par = []
    child = []
    for file, name in files:
          p, c = chlid_vs_parent_ret(file)
          print(c)
          names.append(name)
          par.append(p)
          child.append(c)

    # Calculate the width for each bar
    bar_width = 0.35
    index = np.arange(len(names))

    # Plotting the bars
    bars1 = plt.bar(index, par, bar_width, label='Average Parent Fitness')
    bars2 = plt.bar(index + bar_width, child, bar_width, label='Child Fitness')
    
    
    # Add value labels to each bar
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            plt.annotate('{:.3g}'.format(height),  # Format to 3 significant figures
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3),  # 3 points vertical offset
                     textcoords="offset points",
                     ha='center', va='bottom')

    # Customizing the plot
    plt.ylabel('Fitness')
    plt.title('Parent and Child fitness using ' + title)
    plt.xticks(index + bar_width / 2, names)
    plt.legend()

    # Show plot
    plt.tight_layout()
    plt.show()
    
    
    
def child_sets_better(files, title):
    names = []
    par = []
    child = []
    for file, name in files:
          p, c = chlid_vs_parent_ret_better(file)
          names.append(name)
          par.append(p)
          child.append(c)

    # Calculate the width for each bar
    bar_width = 0.35
    index = np.arange(len(names))

    # Plotting the bars
    bars1 = plt.bar(index, par, bar_width, label='Number of parents better then their chlild')
    bars2 = plt.bar(index + bar_width, child, bar_width, label='Number of children better than thier parent')
    
    
    # Add value labels to each bar
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            plt.annotate('{:.3g}'.format(height),  # Format to 3 significant figures
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3),  # 3 points vertical offset
                     textcoords="offset points",
                     ha='center', va='bottom')

    # Customizing the plot
    plt.ylabel('Fitness')
    plt.title('Number of Parents fitter than their Children  using ' + title)
    plt.xticks(index + bar_width / 2, names)
    plt.legend()

    # Show plot
    plt.tight_layout()
    plt.show()
    
    
    






# Function to count the number of lines with "BETTER" and without "BETTER"
def count_better_vs_not(file_path):
    with open(file_path, 'r') as file:
        better_count = 0
        not_better_count = 0
        for line in file:
            if "BETTER" in line:
                better_count += 1
            else:
                not_better_count += 1

    categories = ['Better', 'Not Better']
    values = [better_count, not_better_count]

    plt.bar(categories, values, color=['blue', 'orange'])
    plt.xlabel('Type')
    plt.ylabel('Count')
    plt.title('Number of Better vs Not Better Lines')

def count_better_vs_not(file_path):
    with open(file_path, 'r') as file:
        better_count = 0
        not_better_count = 0
        for line in file:
            if "BETTER" in line:
                better_count += 1
            else:
                not_better_count += 1

    categories = ['Better', 'Not Better']
    values = [better_count, not_better_count]
    
    return values
    
    ## Add labels to each bar
    # for i, v in enumerate(values):
    #     plt.text(i, v + 0.1, str(v), ha='center')

    # plt.show()

# # Count the number of lines with "BETTER" and without "BETTER"
# better_count, not_better_count = count_better_vs_not(file_path)

# # Create and display the bar chart
# create_bar_chart(better_count, not_better_count)


# import re

# sets = [("RL/rlq/models/evo/_aNewTests/1/pw_mut_none_cross_n_ar/child_comparison.txt", "pairwise verdict") , ("RL/rlq/models/evo/_aNewTests/1/pwr_mut_none_cross_n_ar/child_comparison.txt", "pairwse rank"),( r"RL/rlq/models/evo/_aNewTests/1/lw_mut_none_cross_n_ar/child_comparison.txt", "listwise")]
# child_sets(sets, "naive arithmetic crossover")

# name = "arithmetic crossover"

# dict = {"naive arithmetic crossover" : "nar", "naive cut crossover" :"ncut", "arithmetic crossover" : "ar", "cut crossover" : "cut", "random mutation" : "r", "availability mutation": "av"}
    
    
# for name, a in dict.items():
    
#     sets = [("children_parents/pwv" + a + ".txt", "pairwise verdict"), ("children_parents/pwr" + a + ".txt", "pairwise rank"), ("children_parents/lw" + a + ".txt", "listwise")]
#     child_sets(sets, name)
#     child_sets_better(sets, name)

# Function to parse the data from the text file
def correct_over_steps(file_path):
    steps = []
    correct_values = []

    with open(file_path, 'r') as file:
        for line in file:
            match = re.search(r'total steps = (\d+).*correct = (\d+\.\d+)', line)
            if match:
                total_steps = int(match.group(1))
                correct_value = float(match.group(2))
                steps.append(total_steps)
                correct_values.append(correct_value)
                
                


        # Create line chart
        plt.plot(steps, correct_values, marker='o', linestyle='-')

        # Add labels and title
        plt.xlabel('Total Steps')
        plt.ylabel('Correct Value')
        plt.title('Correct Value vs Total Steps')

        # Show the plot
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
        
def item_over_steps(item, file_path, title):
    steps = []
    correct_values = []

    with open(file_path, 'r') as file:
        for line in file:
            match = re.search(rf'total steps = (\d+).*{item} = (\d+\.\d+)', line)
            if match:
                total_steps = int(match.group(1))
                correct_value = float(match.group(2))
                steps.append(total_steps)
                correct_values.append(correct_value)
                
                


        # Create line chart
        plt.plot(steps, correct_values, marker='o', linestyle='-')

        # Add labels and title
        plt.xlabel('Total Steps')
        plt.ylabel(item)
        plt.title(f'{item} vs Total Steps ' + title)

        # Show the plot
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
        

   
def item_vs_item(item, file_path1, file_path2):
    steps1 = []
    steps2 = []
    correct_values1 = []
    correct_values2 = []

    with open(file_path1, 'r') as file:
        for line in file:
            line = line.strip()
            if line:
                match = re.search(rf'total steps = (\d+).*{item} = (\d+\.\d+)', line)
                if match:
                    total_steps = int(match.group(1))
                    correct_value = float(match.group(2))
                    steps1.append(total_steps)
                    correct_values1.append(correct_value)
                    
                    
        # print(correct_values1)
                
        with open(file_path2, 'r') as file:
            for line in file:
                line = line.strip()

                if line:
                    match = re.search(rf'total steps = (\d+).*{item} = (\d+\.\d+)', line)
                    if match:
                        total_steps = int(match.group(1))
                        correct_value = float(match.group(2))
                        steps2.append(total_steps)
                        correct_values2.append(correct_value)
                        
        # print(correct_values2)
                
                


        # Create line chart
        plt.plot(steps1, correct_values1, marker='o', linestyle='-', label = '1')
        plt.plot(steps2, correct_values2, marker='o', linestyle='-',label = '2')

        # Add labels and title
        plt.xlabel('Total Steps')
        plt.ylabel(item)
        plt.title(f'{item} vs Total Steps')

        # Show the plot
        plt.grid(True)
        plt.tight_layout()
        plt.show()  
        
        
# item_vs_item("NPRA", "fitness_RL.txt", "fitness_calc_test.txt")   
        
def correct_vs_correct(file_path1, file_path2):
    steps1 = []
    steps2 = []
    correct_values1 = []
    correct_values2 = []

    with open(file_path1, 'r') as file:
        for line in file:
            line = line.strip()
            if line:
                match = re.search(r'total steps = (\d+).*correct = (\d+\.\d+)', line)
                if match:
                    total_steps = int(match.group(1))
                    correct_value = float(match.group(2))
                    steps1.append(total_steps)
                    correct_values1.append(correct_value)
                    
                    
        # print(correct_values1)
                
        with open(file_path2, 'r') as file:
            for line in file:
                line = line.strip()

                if line:
                    match = re.search(r'total steps = (\d+).*correct = (\d+\.\d+)', line)
                    if match:
                        total_steps = int(match.group(1))
                        correct_value = float(match.group(2))
                        steps2.append(total_steps)
                        correct_values2.append(correct_value)
                        
        # print(correct_values2)
                
                


        # Create line chart
        plt.plot(steps1, correct_values1, marker='o', linestyle='-', label = '1')
        plt.plot(steps2, correct_values2, marker='o', linestyle='-',label = '2')

        # Add labels and title
        plt.xlabel('Total Steps')
        plt.ylabel('Correct Value')
        plt.title('Correct Value vs Total Steps')

        # Show the plot
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
        


# Function to parse the data from the text file
def correct_vs_data(file_path):
    steps = []
    correct_values = []
    npra_values = []

    with open(file_path, 'r') as file:
        for line in file:
            match = re.search(r'total steps = (\d+).*correct = (\d+\.\d+).*NPRA = (\d+\.\d+)', line)
            if match:
                total_steps = int(match.group(1))
                correct_value = float(match.group(2))
                npra_value = float(match.group(3))
                steps.append(total_steps)
                correct_values.append(correct_value)
                npra_values.append(npra_value)
                
                


    # Create line chart
    plt.plot(steps, correct_values, marker='o', linestyle='-', label='Correct Value')
    plt.plot(steps, npra_values, marker='o', linestyle='-', label='NPRA Value')

    # Add labels and title
    plt.xlabel('Total Steps')
    plt.ylabel('Value')
    plt.title('Correct Value and NPRA Value vs Total Steps')
    plt.legend()

    # Show the plot
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return steps, correct_values, npra_values



def chunk_item_at_a_time(item, file_name):
    
    steps = []
    items = []
    
    # Open the file
    with open(file_name, 'r') as file:
        lines = file.readlines()

    # Initialize a flag to check if we're in the first chunk
    chunk = []

    # Iterate over the lines of the file
    for line in lines:
        line = line.strip()
        if line:
            # Check if we're in the first chunk
            if "Time ="  not in line:
                chunk.append(line)
                
                
    for line in chunk:
        if line:
            match = re.search(rf'total steps = (\d+).*{item} = (\d+\.\d+)', line)
            if match:
                step = int(match.group(1))
                it = float(match.group(2))
                steps.append(step)
                items.append(it)
                
                
    stepits = zip(steps, items)
    zipped_array = stepits
    sorted_array = sorted(zipped_array, key=lambda x: x[0])
    # print(sorted_array)
    
    unzipped_steps, unzipped_items = zip(*sorted_array)
    
                        
                        
    return unzipped_steps, unzipped_items



def sum_item(item, file_path):
    steps = []
    correct_values = []

    with open(file_path, 'r') as file:
        for line in file:
            match = re.search(rf'total steps = (\d+).*{item} = (\d+\.\d+)', line)
            if match:
                total_steps = int(match.group(1))
                correct_value = float(match.group(2))
                steps.append(total_steps)
                correct_values.append(correct_value)
                
    return sum(correct_values)/len(correct_values)



def parse_log_file(log_text_path, phrase):
    with open(log_text_path, 'r', encoding='utf-8') as file:
        log_text = file.read()

    # Regular expression pattern to extract values
    pattern = re.compile(rf"{phrase}\s*=\s*([\d.]+)")

    # Find all matches of the pattern in the log text
    matches = pattern.findall(log_text)

    # Convert matched values to float and return as a list
    return [float(match) for match in matches]

# Example usage:
log_text_path = "RL/rlq/models/list_wise/_env-comp/an_trunc_False/tests/_in_computation.txt"
phrase_to_find = "total steps"
mae_values = parse_log_file(log_text_path, phrase_to_find)
print(f"{phrase_to_find} values:", mae_values)
# print(sum_item("correct", "RL/rlq/models/list_wise/_env-comp/an_trunc_False/tests/_in_computation.txt"))
# print(sum_item("correct", "RL/rlq/models/list_wise/_env-comp/orig_trunc_False/tests/_in_computation.txt"))
# print(sum_item("correct", "RL/rlq/models/list_wise/_env-comp/log5_trunc_False/tests/_in_computation.txt"))

# print(342342)





def file_at_a_time():
    envs = ['an', 'log4','orig']
    all = []
    cols = {'an':'blue', 'log4' : 'orange', 'orig' : 'green'}
    for env in envs:
        steps = []
        correct_values = []
        for i in range(5):
            # with open(f"RL/rlq/models/pairwise/default/{str(i)}verd_pw_env=_{env}/tests/_in_computation.txt", 'r') as file:
            with open(f"RL/rlq/models/pairwise/default/{str(i)}verd_pw_env=_{env}/tests/_in_computation.txt", 'r') as file:
                for line in file:
                    match = re.search(rf'total steps = (\d+).*{"correct"} = (\d+\.\d+)', line)
                    if match:
                        total_steps = int(match.group(1))
                        correct_value = float(match.group(2))
                        steps.append(total_steps)
                        correct_values.append(correct_value)
                        all.append(correct_value)
            total_steps, correct_values = zip(*sorted(zip(steps, correct_values)))        
            plt.plot(steps, correct_values, marker='o', linestyle= '-', color = cols[env], label = env + str(i))
            steps = []
            correct_values = []
        
                        
                        
        # print(sum(all)/len(all))
        all = []

    # Create line chart
    # plt.plot(all[0][0], all[0][1], marker='o', linestyle= '-', label = 'normalisation [0-1]')
    # plt.plot(all[1][0], all[1][1], marker='o', linestyle='-', label = 'normalisation [0-10]')
    # plt.plot(all[2][0], all[2][1], marker='o', linestyle='-', label = 'original')

    # Add labels and title
    plt.xlabel('Total Steps')
    plt.ylabel("Correct")
    plt.title(f'Correct vs Steps vs differing normalisation strategies for Pairwise Verdict')

    # Show the plot
    plt.grid(True)
    plt.tight_layout()
    plt.show()  
                
    return sum(correct_values)



# file_at_a_time()
        
    
                
                

def triple_comp(item, file1, file2, file3):
    
    steps1, items1 = chunk_item_at_a_time(item, file1)
    steps2, items2 = chunk_item_at_a_time(item, file2)
    steps3, items3 = chunk_item_at_a_time(item, file3)
    
    # print(sum(items1)/len(items1))
    # print(sum(items2)/len(items2))
    # print(sum(items3)/len(items3))

    # Create line chart
    plt.plot(steps1, items1, marker='o', linestyle='-', label = 'normalisation [0-1]')
    plt.plot(steps2, items2, marker='o', linestyle='-', label = 'normalisation [0-10]')
    plt.plot(steps3, items3, marker='o', linestyle='-', label = 'original')

    # Add labels and title
    plt.xlabel('Total Steps')
    plt.ylabel(item)
    plt.title(f'{item.capitalize()} vs Steps vs differing normalisation strategies for Listwise')
    
    
    # Add legend
    plt.legend()

    # Show the plot
    plt.grid(True)
    plt.tight_layout()
    plt.show()  
    
    
# triple_comp("correct", "RL/rlq/models/list_wise/_env-comp/an_trunc_False/tests/_in_computation.txt", "RL/rlq/models/list_wise/_env-comp/log5_trunc_False/tests/_in_computation.txt", "C:/Users/tomar/OneDrive - University of Birmingham/Documents/Uni/Project/git/secondProj/tla097/RL/rlq/models/list_wise/_env-comp/orig_trunc_False/tests/_in_computation.txt")

    



# "C:\Users\tomar\OneDrive - University of Birmingham\Documents\Uni\Project\git\secondProj\tla097\RL\rlq\models\list_wise\_env-comp"
# "RL\rlq\models\list_wise\_env-comp\an_trunc_False"
# "RL\rlq\models\list_wise\_env-comp\log5_trunc_False"
# "RL\rlq\models\list_wise\_env-comp\orig_trunc_False"


def old_file_type(file_name):
    # Initialize lists to store data
    time_taken = []
    correct_values = []
    
    file_name  = file_name.replace("\\", "/")
    # Regular expression patterns
    time_pattern = re.compile(r"time taken = (\d+\.\d+)")
    correct_pattern = re.compile(r"correct = (\d+\.\d+)")

    # Read the file
    with open(file_name, "r") as file:
        for line in file:
            time_match = re.match(time_pattern, line)
            if time_match:
                time_taken.append(float(time_match.group(1)))
            
            correct_match = re.match(correct_pattern, line)
            if correct_match:
                correct_values.append(float(correct_match.group(1)))

    # Plot the line graph
    plt.plot(time_taken, correct_values, marker='o', linestyle='-')
    plt.xlabel('Time Taken')
    plt.ylabel('Correct Values')
    plt.title('Correct Values vs Time Taken')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    
    
    
def model_comp(file, title):
    ts = parse_log_file(file, "total steps")
    AFPDs = parse_log_file(file, "AFPD")
    NPRAs = parse_log_file(file, "NPRA")
    corrects = parse_log_file(file, "correct")
    
    plt.plot(ts, AFPDs, marker='o', linestyle='-', label = 'AFPD')
    plt.plot(ts, NPRAs, marker='o', linestyle='-', label = 'NPRA')
    plt.plot(ts, corrects, marker='o', linestyle='-', label = 'Average Reward')
    
    plt.legend()
    
    plt.xlabel('Total Steps')
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    
    
    

# model_comp("pwr1000.txt", "Pairwise Rank 1000 length perfomance")





import matplotlib.pyplot as plt

def old_lw(file_name):
    # Lists to store average and time data
    averages = []
    times = []

    file_name = file_name.replace("\\", "/")

    # Open the file and read line by line
    with open(file_name, 'r') as file:
        # Read the first line
        line = file.readline()
        while line:
            # Check if the line contains "average" and "Time"
            if "average" in line and "Time" in line:
                # Extract average and time values from the current line
                average = float(line.split("average = ")[1].split(".")[0]  +"." +  line.split("average = ")[1].split(".")[1])
                time = float(line.split("Time = ")[1].split()[0])

                # Read the next line to get "steps" information
                next_line = file.readline()
                if "steps" in next_line:
                    steps = int(next_line.split("DONE WITH ")[1].split()[0])
                    # Calculate adjusted average
                    adjusted_average = (average * steps) / 100
                    averages.append(adjusted_average)
                    times.append(time)

            # Read the next line
            line = file.readline()

    # # Plotting the graph
    # print(averages)
    
    plt.plot(times, averages, marker='o', linestyle='none')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Average Reward')
    plt.title('Average Reward Over Time - Listwise')
    plt.grid(True)
    plt.show()
    
    
    
    
def return_items(item, file_path):
    steps = []
    correct_values = []

    # Define regular expression pattern to extract relevant data
    pattern = r"ROUND (\d+) - total steps = (\d+) - .*?round steps = (\d+).*?correct = (\d+\.\d+).*?time taken = (\d+\.\d+)"

    # Lists to store extracted data
    rounds = []
    total_steps = []
    round_steps = []
    correct_values = []
    time_taken = []
    steps = []

    # Read data from the text file
    with open(file_path, "r") as file:
        lines = file.readlines()

    # Iterate through lines and extract data using regular expressions
    for line in lines:
        match = re.search(pattern, line)
        if match:
            total_steps.append(int(match.group(2)))
            step  = int(match.group(3))
            correct = float(match.group(4))
            time_taken.append(float(match.group(5)))
            # try:
            #     correct = (correct*step) / 100
            # except:
            #     correct = 0
            # correct_values.append(correct)
            # steps.append(step)
            correct_values.append(correct)
            steps.append(step)
            
            
            stepits = zip(total_steps, correct_values)
            zipped_array = stepits
            sorted_array = sorted(zipped_array, key=lambda x: x[0])
            # print(sorted_array)
            
            unzipped_steps, unzipped_items = zip(*sorted_array)
                    
            # print(steps)
            
            
            

                
                
                
    return unzipped_items, unzipped_steps


def zippp(steps, items):
    stepits = zip(steps, items)
    zipped_array = stepits
    sorted_array = sorted(zipped_array, key=lambda x: x[0])
    return zip(*sorted_array)


def give_data(file, needed = False, sum_time = False):
    corrects = parse_log_file(file, "correct")
    if needed:
        rs = np.array(rs)
        npcor = np.array(corrects)
        corrects = (rs * npcor) / 100
    
    
    
    AFPDs = parse_log_file(file, "AFPD")
    NPRAs = parse_log_file(file, "NPRA")
    steps = parse_log_file(file, "total steps")
    times = parse_log_file(file, "time taken")
    
    steps, corrects = zippp(steps, corrects)
    steps, AFPDs = zippp(steps, AFPDs)
    steps, NPRAs = zippp(steps, NPRAs)
    steps, times = zippp(steps, times)
    
    
    times_ret = times[len(times)- 1]  
    if sum_time:
        times_ret = sum(times)
        
    
    return corrects[len(corrects)  - 1], AFPDs[len(AFPDs)-1], NPRAs[len(NPRAs)- 1], steps[len(times) - 1], times[len(times)- 1]  





# def read_gen(f):
#     with open(f, 'r') as file:
#         # Move the cursor to the end of the file
#         file.seek(0, 2)  # 2 indicates the end of the file

#         # Find the position of the last newline character
#         position = file.tell()
#         while position >= 0:
#             file.seek(position)
#             # Read the last character
#             char = file.read(1)
#             # Check if it's a newline character
#             if char == '\n':
#                 break
#             # Move the cursor to the previous character
#             position -= 1

#     # Now, position is at the beginning of the last line
#     # Read and print the last line
#     last_line = file.readline()
#     print("Last line:", last_line)
    
    
# read_gen()
#     with open(file, "r") as f:
#         lines = f.readlines()
#     # Define regular expression pattern to extract data
#     generation_pattern = r"generation (\d+)"
#     time_pattern = r"time = ([\d.]+)"

#     # Extract data using regular expressions
#     generation_match = re.search(generation_pattern, line)
#     time_match = re.search(time_pattern, line)

#     if generation_match and time_match:
#         generation = int(generation_match.group(1))
#         time = float(time_match.group(1))

#         # Print the extracted data
#         print("Generation:", generation)
#         print("Time:", time)
#     else:
#         print("Data not found in the given line.")

import re
def get_gen(search_time, file):
    f = file + r"/time.txt"
    with open(f, "r") as fl:
        lines = fl.readlines()
    
    for line in lines:
        # Define regular expression pattern to extract data
        generation_pattern = r"generation (\d+)"
        time_pattern = r"time = ([\d.]+)"

        # Extract data using regular expressions
        generation_match = re.search(generation_pattern, line)
        time_match = re.search(time_pattern, line)

        if generation_match and time_match:
            generation = int(generation_match.group(1))
            time = float(time_match.group(1))

        if time >= search_time:
            return generation
        

def get_line_data(file, gen):
    
    file = file + "/best_thread/fitness.txt"
    print(file)
    with open(file,"r") as f:
        lines = f.readlines()
        
        
    time = 0

    for line in lines:
        if line.strip():
            print(line)
            input()
            generation_pattern = r"generation = (\d+)"
            generation = re.search(generation_pattern, line).group(1)
            generation = float(generation)
            
            print(generation)
            print(gen)
            

            # print(time_taken)
            if generation >= gen:
                # Define regular expressions to extract data
                generation_pattern = r"generation = (\d+)"
                total_steps_pattern = r"total steps = (\d+)"
                round_steps_pattern = r"round steps = (\d+)"
                correct_pattern = r"correct = ([\d.]+)"
                mae_pattern = r"mae = ([\d.]+)"
                afpd_pattern = r"AFPD = ([\d.]+)"
                npra_pattern = r"NPRA = ([\d.]+)"
                

                # Extract data using regular expressions
                generation = re.search(generation_pattern, line).group(1)
                total_steps = re.search(total_steps_pattern, line).group(1)
                round_steps = re.search(round_steps_pattern, line).group(1)
                correct = re.search(correct_pattern, line).group(1)
                mae = re.search(mae_pattern, line).group(1)
                afpd = re.search(afpd_pattern, line).group(1)
                npra = re.search(npra_pattern, line).group(1)

                # Convert extracted strings to appropriate data types
                generation = int(generation)
                total_steps = int(total_steps)
                round_steps = int(round_steps)
                correct = float(correct)
                mae = float(mae)
                afpd = float(afpd)
                npra = float(npra)
                return correct, afpd, npra, total_steps
            else:
                print("wrong")

    # Print the extracted data
    # print("Generation:", generation)
    # print("Total Steps:", total_steps)
    # print("Round Steps:", round_steps)
    # print("Correct:", correct)
    # print("MAE:", mae)
    # print("AFPD:", afpd)
    # print("NPRA:", npra)
    # print("Time Taken:", time_taken)
    
    
    

def run(file, time):
    gen = get_gen(time, file)
    print(gen)
    print(get_line_data(file, gen))
    
    
# run(r"RL/rlq/models/evo/_aNewTests/1/pwr2_mut_none_cross_ar-1000", 3000)
# input("")
    
    
# print(give_data(r"RL\rlq\models\list_wise\_env-comp\long_lw0\tests\_in_computation.txt"))

def triple_triple_comp(file1, file2, file3, item, title1, title2, title3, bittitle):
    # items1 = parse_log_file(file1, item)
    # items2 = parse_log_file(file2, item)
    items3 = parse_log_file(file3, item)
    # steps1= parse_log_file(file1, "total steps")
    # steps2= parse_log_file(file2, "total steps")
    steps3= parse_log_file(file3, "total steps")
    rs = parse_log_file(file3, 'round steps')
    
    rs = np.array(rs)
    npcor = np.array(items3)
    npcorr = (rs * npcor) / 100
    
    # steps1, items1 = zippp(steps1, items1)
    # steps2, items2 = zippp(steps2, items2)
    steps3, items3 = zippp(steps3, npcorr)
    
    # print(sum(items1)/len(items1))
    # print(sum(items2)/len(items2))
    print(sum(items3)/len(items3))
    # print(items1[len(items1)-1])
    # print(items2[len(items2)-1])
    print(items3[len(items3)-1])
    
    
    
    # Create line chart
    # plt.plot(steps1, items1, marker='o', linestyle='-', label = title1)
    # plt.plot(steps2, items2, marker='o', linestyle='-', label = title2)
    plt.plot(steps3, items3, marker='o', linestyle='-', label = title3)

    # Add labels and title
    plt.xlabel('Total Steps')
    plt.title(bittitle)
    
    
    # Add legend
    # plt.legend()

    # Show the plot
    plt.grid(True)
    plt.tight_layout()
    plt.show()  
    
   
    
f = ""



with open("tes.txt", "r") as file:
    correctness_values = []
    
    # Iterate through each line in the file
    for line in file:
        # Split the line into tokens using "-" as the separator
        tokens = line.split(" ")
        correctness_values.append(float(tokens[18]))

# Calculate the average correctness
average_correctness = sum(correctness_values) / len(correctness_values)

print("Average Correctness:", average_correctness)

input()
    
# triple_triple_comp(f, f, r"RL\rlq\models\list_wise\_attempt_5\0_trunc=_False_env=_an\tests\_in_computation.txt", "correct", f, f, "Average epidose reward", "Correct vs Steps Listwise 100 inputs")
    
    
# triple_comp()
    
    
    
    
# model_comp("PWR100.txt", "Pairwise Rank 1000 length perfomance")

# model_comp("test_children.txt", "Listwise Agent with an Input of 10")

# triple_comp("correct", "test_children.txt", "", "", "" )

# read through and stop when steps = 

# Test the function
# old_lw('your_file_name.txt')


# Test the function
# old_lw('your_file_name.txt')

    
    
# old_lw(r"RL\rlq\models\1file_to_tidy\tidy2\_lw_eps_10000000_lr_0.0001\results\_res.txt")


# from collections import deque
# d = deque(maxlen=10)

# p = parse_log_file(r"RL\rlq\models\list_wise\_attempt_5\0_trunc=_False_env=_an\tests\_in_computation.txt", "correct")

def index_less_than_last_10_avg_before(arr):
    # Check if the array has at least 11 elements
    if len(arr) < 11:
        raise ValueError("Array must have at least 11 elements")

    result_index = -1  # Default result index

    print(arr)
    # Iterate over the array starting from the 10th index
    for i in range(20, len(arr) - 1):
        print(arr[i-10:i])
        print(sum(arr[i-20:i]) / 20)
        # Calculate the average of the last 10 values before the current index
        last_10_avg = sum(arr[i-20:i]) / 20

        # Check if the current value is less than the average of the last 10 values before the index
        if arr[i + 1] < last_10_avg:
            result_index = i
            break  # Exit loop if a value is found

    return result_index
# index = index_less_than_last_10_avg_before(p)




# item_over_steps("correct", "test_children.txt", "listwise 10 inputs")
# item_over_steps("correct", "", "listwise 10 inputs")

# triple_triple_comp()
# from collections inp

# chlid_vs_parent()