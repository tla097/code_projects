import sys
import numpy as np
import pandas as pd
import time
import datetime
import matplotlib.pyplot as plt
    
    
def calc_entropy(var, bins):
    to_bin = np.histogram(var, bins)[0]
    # print(to_bin)
    return entropy(to_bin)

def calc_joint_entropy(var1, var2, bins):
    to_bin = np.histogram2d(var1, var2, bins)[0]
    return entropy(to_bin)

def entropy(to_bin):
    #calculates the probability of each bin
    probs = to_bin / np.sum(to_bin)
    
    #cannot have a zero probabilty bin- removes it if so
    probs = probs[np.nonzero(probs)]

    #does the entropy calculation
    entropy = - np.sum(probs * np.log2(probs))
    
    return entropy


def mutual_info(var1, var2, bin1, bin2):
    H_1 = calc_entropy(var1, bin1)
    H_2 = calc_entropy(var2, bin2)
    H_12 = calc_joint_entropy(var1, var2, bin1 * bin2)
    
    MI = H_1 + H_2 - H_12
    return MI

def graph(x_axis, y_axis):
    
    plt.bar(x_axis, y_axis, width= 0.5)
    plt.xlabel("features")
    plt.ylabel("mutual information with verdict")
    plt.title("Bar chart of mutual information between differing features and verdict")
    
    plt.show()
    
    
def calculate_maturity(excel, BINS):
    gap = excel["GapInRun"].to_numpy()
    gap_cats = np.array(list(map(lambda x : 0 if x < 1 else (1 if x < 2 else 2), gap)))    
        
    # print(dur_cats)
    # both = np.where(added<=1, 1, 0)
        
    my_maturity = gap_cats
    
    verdict = excel["Verdict"].to_numpy()
    
    mut = mutual_info(verdict, my_maturity, 2, BINS)
    
    
    
    from datetime import datetime
    now = datetime.now()
    current_time = now.strftime("%d/%m/%y %H:%M:%S")
    
    
    file_path = "c:/Users/tomar/OneDrive - University of Birmingham/Documents/Uni/Project/git/tla097/new_maturity.txt"

    note = f"gap"
    
    with open(file_path, "a") as file:
        file.write(f"{current_time} : {note} -> {mut}\n")
        
    return mut
    

def calculate_duration_group(verdict, duration_col):
    
    # print(duration_col)
    
    
    # print(mutual_info(duration_col, verdict, 50, 2, 5000))
    
    from datetime import datetime
    now = datetime.now()
    current_time = now.strftime("%d/%m/%y %H:%M:%S")
    
    
    file_path = "c:/Users/tomar/OneDrive - University of Birmingham/Documents/Uni/Project/git/tla097/duration_group.txt"

    
    with open(file_path, "a") as file:
        file.write(f"{current_time}\n")
    
    for i in range(1, 200):
        # "DurationGroup": 0.0008616250595219199
        # try puttinng duration in 3 seperate bins
        mut = mutual_info(duration_col, verdict, i, 2, 5000)
        
        note = f"{i} bins for duration"
        with open(file_path, "a") as file:
            file.write(f"{note} -> {mut}\n")
            if i == 199:
                file.write("\n\n")
        
        
    # file_path2 = "c:/Users/tomar/OneDrive - University of Birmingham/Documents/Uni/Project/git/tla097/duration_col.txt"

    # # Write the NumPy array to a text file
    # # np.savetxt(file_path2, duration_col, delimiter=',', fmt='%.10f')

    # # print(f"NumPy array has been written to {file_path2}")
    
    
    # print("Written")


def make_file(converted_dict, BINS):
    
    import json
    from datetime import datetime
    now = datetime.now()
    current_time = now.strftime("%d/%m/%y %H:%M:%S")
    
     # Specify the file path
    file_path = "c:/Users/tomar/OneDrive - University of Birmingham/Documents/Uni/Project/git/tla097/output.txt"

    # Write the dictionary to a text file in JSON format
    
    note = F"currently {BINS} bins for all+ new strategy for binning bin1 x bin2 + my maturity - without target features"
    with open(file_path, "a") as file:
        file.write(f"{current_time} : {note}\n")
        json.dump(converted_dict, file)
        file.write("\n")

    print(f"Dictionary has been written to {file_path}")


def calculate_all_mutuals(excel,BINS):
    verdict = excel["Verdict"].to_numpy()
    
    time_ran =  [time.mktime(datetime.datetime.strptime(element, "%d/%m/%Y %H:%M").timetuple()) for element in excel["LastRun"].to_numpy()]
    mutual_informations = {}
    column_names = ["Duration", "Cycle", "Month", "Failure_Percentage", "Num_Previous_Execution","CycleRun"]    
    
    gap = excel["GapInRun"].to_numpy()
    gap_cats = np.array(list(map(lambda x : 0 if x < 1 else (1 if x < 2 else 3), gap)))
    
    
    
    # todo next: find the mutual information for the previous results by using the immedate previous
    for column_name in column_names:
        column = excel[column_name].to_numpy()
        mutual_informations[column_name] = mutual_info(verdict, column, 2, BINS)
        
    
    
    last_results = [int(element[1]) for element in excel["LastResults"].to_numpy()]
    
    mutual_informations["LastRun"] = mutual_info(verdict, time_ran, 2, BINS)
    mutual_informations["LastResults"] = mutual_info(verdict, last_results, 2, BINS)
    mutual_informations["gap_cats"] = mutual_info(verdict, gap_cats , 2, BINS)
    
    
    total_mut = sum(mutual_informations.values())
    
    for key, val in mutual_informations.items():
        mutual_informations[key] = val/total_mut
    
    muts = sorted(mutual_informations.items(), key=lambda x: x[1], reverse=True)
    
    return dict([x for x in muts]) #if x[1] >=0])

def calc_changing_mutuals(excel, BINS, mutual_informations):
    verdict = excel["Verdict"].to_numpy()
    column_names = ["Quarter", "Failure_Percentage", "Num_Previous_Execution", "Maturity_Level","Rank_score", "Rank"]    
    
    # todo next: find the mutual information for the previous results by using the immedate previous
    for column_name in column_names:
        column = excel[column_name].to_numpy()
        mutual_informations[column_name] = mutual_info(verdict, column, 2, BINS)
        
    
    
    arr = [int(element[1]) for element in excel["LastResults"].to_numpy()]
    time_ran =  [time.mktime(datetime.datetime.strptime(element, "%d/%m/%Y %H:%M").timetuple()) for element in excel["LastRun"].to_numpy()]
    
    
    mutual_informations["LastRan"] = mutual_info(verdict, time_ran, 2, BINS)
    mutual_informations["LastResults"] = mutual_info(verdict, arr, 2, BINS)
    
    muts = sorted(mutual_informations.items(), key=lambda x: x[1], reverse=True)
    
    return dict([x for x in muts]) #if x[1] >=0])
    
    
    
    

def calc_cols(excel):
    feature_columns = {}
    # column_names = ["Duration", "Cycle", "DurationGroup", "TimeGroup", "Month", "Failure_Percentage", "Num_Previous_Execution", "Maturity_Level","CycleRun"]    
    column_names = ["Duration", "Cycle", "Month", "Failure_Percentage", "Num_Previous_Execution","CycleRun"]    

    
    
    for column_name in column_names:
        column = excel[column_name].to_numpy()
        feature_columns[column_name] = column 
        
    last_results = [int(element[1]) for element in excel["LastResults"].to_numpy()]
    time_ran =  [time.mktime(datetime.datetime.strptime(element, "%d/%m/%Y %H:%M").timetuple()) for element in excel["LastRun"].to_numpy()]
    gap = excel["GapInRun"].to_numpy()
    gap_cats = np.array(list(map(lambda x : 0 if x < 1 else (1 if x < 2 else 3), gap)))
    tg = np.array(list(map(lambda x : 0 if x < 7 else (1 if x < 9 else (2 if x < 12 else 3)), feature_columns["Duration"])))
    
    cycle_run = feature_columns["CycleRun"]
    dur = gap_cats * feature_columns["CycleRun"]
    n_gap = np.zeros(len(gap))
    # print(n_gap)
    
    # print(gap_cats)
    # print(cycle_run)
    
    n_gap = np.array(list(map(lambda x,y: 0 if x == 0 and y == 0 else (1 if x == 0 or y == 0 else (2 if x == 1 and y == 1 else 3)), gap_cats, cycle_run)))
    
    
    
    # for i in range(len(gap)):
    #     # print(91209013)
    #     # print(gap)
    #     if gap_cats[i] == 0 and cycle_run[i] == 0:
    #         n_gap[i] = 0
    #     elif gap_cats[i] == 0 or cycle_run[i] == 0:
    #         n_gap = 1
    #     elif gap_cats[i] == 1 and cycle_run[i] == 1:
    #         n_gap = 2
    #     elif gap_cats[i] > 1 and cycle_run[i] == 1:
    #         n_gap = 3
    #     elif gap_cats[i] == 1 and cycle_run[i] > 1:
    #         n_gap = 4
    #     elif gap_cats[i] > 1 and cycle_run[i] > 1:
    #         n_gap = 5
            
    
    # isc = feature_columns["InSameCycle"]
    
    # print(n_gap)
    
    feature_columns["LastRun"] = time_ran
    feature_columns["LastResults"] = last_results
    feature_columns["gap_cats"] = gap_cats
    # feature_columns["mytg"] = tg  
    # feature_columns["cyclerun x gapcats"] = n_gap
    
    return feature_columns

    

def shannon_entropy(A, mode="auto", verbose=False):
    """
    https://stackoverflow.com/questions/42683287/python-numpy-shannon-entropy-array
    """
    A = np.asarray(A)

    # Determine distribution type
    if mode == "auto":
        condition = np.all(A.astype(float) == A.astype(int))
        if condition:
            mode = "discrete"
        else:
            mode = "continuous"
    if verbose:
        print(mode, file=sys.stderr)
    # Compute shannon entropy
    pA = A / A.sum()
    # Remove zeros
    pA = pA[np.nonzero(pA)[0]]
    if mode == "continuous":
        return -np.sum(pA*np.log2(A))  
    if mode == "discrete":
        return -np.sum(pA*np.log2(pA))  

def mutual_information(x,y, mode="auto", normalized=False):
    """
    I(X, Y) = H(X) + H(Y) - H(X,Y)
    https://stackoverflow.com/questions/20491028/optimal-way-to-compute-pairwise-mutual-information-using-numpy
    """
    x = np.asarray(x)
    y = np.asarray(y)
    # Determine distribution type
    if mode == "auto":
        condition_1 = np.all(x.astype(float) == x.astype(int))
        condition_2 = np.all(y.astype(float) == y.astype(int))
        if all([condition_1, condition_2]):
            mode = "discrete"
        else:
            mode = "continuous"

    H_x = shannon_entropy(x, mode=mode)
    H_y = shannon_entropy(y, mode=mode)
    H_xy = shannon_entropy(np.concatenate([x,y]), mode=mode)

    # Mutual Information
    I_xy = H_x + H_y - H_xy
    if normalized:
        return I_xy/np.sqrt(H_x*H_y)
    else:
        return  I_xy
        


def main():
    
    
    BINS = 17500
    print(f"running {BINS}")
 
    excel = pd.read_csv(r"C:\Users\tomar\Documents\Project\git\DataMineWithGap.csv")
    
    # muts = calculate_all_mutuals(excel, BINS)
    
    df = calc_cols(excel)
    verdict = excel["Verdict"].to_numpy()
    totals = {}
    
    for key, val in df.items():
        totals[key] = mutual_information(verdict, val)
        
    total = sum(totals.values())
    
    totals_div = {}
    
    for key, val in totals.items():
        totals_div[key] = totals[key]/total
        
        
    print(totals_div)
    
    
    # print(muts)
    # make_file(muts, 17500)
    




main()

