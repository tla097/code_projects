import copy
from matplotlib import pyplot as plt
from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import mutual_info_regression, f_regression
 
import numpy as np
import pandas as pd
import time
import datetime


def calculate_all_mutuals(excel):
    

    feature_columns = {}
    # column_names = ["Duration", "Cycle", "DurationGroup", "TimeGroup", "Month", "Failure_Percentage", "Num_Previous_Execution", "Maturity_Level","CycleRun"]    
    # column_names = ["Duration", "Cycle", "Month", "Failure_Percentage", "Num_Previous_Execution","CycleRun", "CalcPrio"]    
    column_names = ["Duration","LastRun","LastResults","Cycle","Month","Failure_Percentage","Num_Previous_Execution"]
    
    
    for column_name in column_names:
        column = excel[column_name].to_numpy()
        feature_columns[column_name] = column
        
    time_ran =  [time.mktime(datetime.datetime.strptime(element, "%d/%m/%Y %H:%M").timetuple()) for element in excel["LastRun"].to_numpy()]
    feature_columns["LastRun"] = time_ran
    
    last_results = [int(element[1]) for element in excel["LastResults"].to_numpy()]
    feature_columns["LastResults"] = last_results
    
    gap = excel["GapInRun"].to_numpy()
    feature_columns["gap"] = gap
    
    # gap_cats = np.array(list(map(lambda x : 0 if x < 1 else (1 if x < 2 else 3), gap)))
    # feature_columns["gap_cats"] = gap_cats

    
    
    """    
    
    print(last_results)
    time_ran =  [time.mktime(datetime.datetime.strptime(element, "%d/%m/%Y %H:%M").timetuple()) for element in excel["LastRun"].to_numpy()]
    
    gap_cats = np.array(list(map(lambda x : 0 if x < 1 else (1 if x < 2 else 3), gap)))
        
    cycle_run = feature_columns["CycleRun"]

    n_gap = np.zeros(len(gap))
    print(n_gap)
    
    print(gap_cats)
    print(cycle_run)
    
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
    
    feature_columns["gap_cats"] = gap_cats
    
    # feature_columns["CalcPrio"] = excel["CalcPrio"]
    
    """
    df = pd.DataFrame.from_dict(feature_columns)
    matur = calc_maturity(excel, df)
    df["my_maturity"] = matur
    
    
    return df

def calc_maturity(excel, df):
    
    # df = calculate_all_mutuals(excel)

    
    MOST_TIMES_RAN = 1186
    LEAST_TIMES_RAN = 4
    
    """
    TO FIND MOST AND LEAST
    last_results = excel["LastResults"].to_numpy()
    
    leng = [len(x) for x in last_results]
    
    print(sorted(leng, reverse=True)[0])"""
    
    # many times ran
    # ran at the end
    # high failure precentage
    # compare the max and min num of ran
    month = excel["Month"].to_numpy()
    fp = excel["Failure_Percentage"].to_numpy()
    
    last_results = excel["LastResults"].to_numpy()
    times_ran = np.array([len(x) for x in last_results])
    
    in_same_cycle = excel["InSameCycle"]
    cycle = excel["Cycle"]
    gap_in_run = excel["GapInRun"]
    gap_cats = np.array(list(map(lambda x : 0 if x < 1 else (1 if x < 2 else 3), gap_in_run)))
    
    matu1 = (month  *( in_same_cycle + 1))/ (gap_cats + 1) + fp * ((times_ran - LEAST_TIMES_RAN)/(MOST_TIMES_RAN))
    df1 = copy.deepcopy(df)
    df1["my_mat"] = matu1
    
    verdict = excel["Verdict"]
    feature_names, mutual_info_list = mat(df1,verdict)
    
    calculation = "(month* (in_same_cycle + 1))/ (gap_cats + 1) + fp * ((times_ran - LEAST_TIMES_RAN)/(MOST_TIMES_RAN))"
    
    st = ""
    for col in range(len(mutual_info_list)):
        st += f"{feature_names[col]}:{mutual_info_list[col]}\n"
    
    with open("classification\maturity_trials.txt", "a") as f:
        f.write(f"calculation = {calculation}\n{st}\n\n\n")
        
    return matu1                     
                    
        
    
    # time_dif = MOST_TIMES_RAN - LEAST_TIMES_RAN
    # for x in range(length):
    #     if month[x] < 5:
    #         if (times_ran[x] - LEAST_TIMES_RAN)/time_dif < 0.34:
    #             if fp[x] < 0.34:
    #                 matu[x] = 1
    #             elif fp[x] < 0.67:
    #                 matu[x] = 2
    #             else:
    #                 matu[x] = 3
    #         elif (times_ran[x] - LEAST_TIMES_RAN)/time_dif < 0.67:
    #             if fp[x] < 0.34:
    #                 matu[x] = 2
    #             elif fp[x] < 0.67:
    #                 matu[x] = 3
    #             else:
    #                 matu[x] = 4
    #         else:
    #             if fp[x] < 0.34:
    #                 matu[x] = 3
    #             elif fp[x] < 0.67:
    #                 matu[x] = 4
    #             else:
    #                 matu[x] = 5
    #     elif month[x] < 9:
    #         if (times_ran[x] - LEAST_TIMES_RAN)/time_dif < 0.34:
    #             if fp[x] < 0.34:
    #                 matu[x] = 4
    #             elif fp[x] < 0.67:
    #                 matu[x] = 5
    #             else:
    #                 matu[x] = 6
    #         elif (times_ran[x] - LEAST_TIMES_RAN)/time_dif < 0.67:
    #             if fp[x] < 0.34:
    #                 matu[x] = 5
    #             elif fp[x] < 0.67:
    #                 matu[x] = 6
    #             else:
    #                 matu[x] = 7
    #         else:
    #             if fp[x] < 0.34:
    #                 matu[x] = 6
    #             elif fp[x] < 0.67:
    #                 matu[x] = 7
    #             else:
    #                 matu[x] = 8 
    #     else:
    #         if (times_ran[x] - LEAST_TIMES_RAN)/time_dif < 0.34:
    #             if fp[x] < 0.34:
    #                 matu[x] = 7
    #             elif fp[x] < 0.67:
    #                 matu[x] = 8
    #             else:
    #                 matu[x] = 9
    #         elif (times_ran[x] - LEAST_TIMES_RAN)/time_dif < 0.67:
    #             if fp[x] < 0.34:
    #                 matu[x] = 8
    #             elif fp[x] < 0.67:
    #                 matu[x] = 9
    #             else:
    #                 matu[x] = 10
    #         else:
    #             if fp[x] < 0.34:
    #                 matu[x] = 9
    #             elif fp[x] < 0.67:
    #                 matu[x] = 10
    #             else:
    #                 matu[x] = 11
    

    
    
    
    
def mat(X_train, y_train):
    mutual_info = mutual_info_regression(X_train, y_train)
    
    k = 15  # Set the number of top features you want to select

    best_features = np.argsort(mutual_info)[::-1][:k]

    selected_features = X_train.columns[best_features]

    X_mi_selected = X_train[selected_features]
    
    print("\nSelected features based on mutual information:")

    print(X_mi_selected.columns)
    
    plt.figure(figsize=(10, 6))

    plt.barh(X_mi_selected.columns, mutual_info[best_features])

    st = ""
    for col in range(len(X_mi_selected.columns)):
        st += f"{X_mi_selected.columns[col]}:{mutual_info[best_features][col]}\n"
        
    return X_mi_selected.columns, mutual_info[best_features]
        
def graph(feature_names, mutual_info_list):
               
    plt.xlabel('Mutual Information Score')

    plt.ylabel('Feature Names')

    plt.title('Mutual Information for Feature Selection')

    plt.show()
    
    # Bar Chart
    
    print("Bar Chart ____")
    
    # Create the figure and axis

    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create the horizontal bar chart

    bars = ax.barh(feature_names, mutual_info_list)
    
    # Set the labels and title

    ax.set_xlabel('Mutual Information Score')

    ax.set_ylabel('Feature Names')

    ax.set_title('Mutual Information for Feature Selection')



    # Annotate scores on each bar

    for bar, score in zip(bars, mutual_info_list):

        width = bar.get_width()

        label_x_pos = width + 0.01  # Adjust the position for the score

        ax.text(label_x_pos, bar.get_y() + bar.get_height() / 2, f'{score:.6f}', ha='center', va='center')
    
    plt.show()



excel = pd.read_csv(r"C:\Users\tomar\Documents\Project\git\dataMineWithGap.csv")

X_train = calculate_all_mutuals(excel)
y_train = excel["Verdict"].to_numpy()

# calc_maturity(excel)

feature_names, mutual_info_list = mat(X_train, y_train)
# graph(feature_names, mutual_info_list)

# calc_maturity(excel)

        
    
    




